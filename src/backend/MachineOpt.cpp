// MachineOpt.cpp - see MachineOpt.h
//
// Three stages live here:
//   1. peephole()  - pre-register-allocation, block-local cleanups over the
//      virtual-register code (dead/self moves, x+0 / x^0 / x-x folding).
//   2. schedule()  - pre-register-allocation load hoisting within barrier-
//      delimited straight-line segments (in-order pipeline latency).
//   3. removeRedundantMoves() - post-register-allocation "SecondPeep": once
//      colours are baked in, register moves that colouring made redundant
//      (self / ping-pong / dead-consecutive / move-into-next-store) become
//      visible and are eliminated.  Mirrors the reference SysY backend's
//      RemoveRedundantMovePass which runs after its graph allocator.
#include "MachineOpt.h"

#include <algorithm>
#include <unordered_map>

#include "Machine.h"

namespace sakura {
namespace backend {

namespace {

// ---------------------------------------------------------------------------
// shared operand helpers
// ---------------------------------------------------------------------------

// All virtual-register reads of one instruction: the explicit a/b operands,
// the return value of Ret, and every Call argument.  (slot defs are handled
// separately by the caller when needed.)
static void usesOf(const MInst &m, std::vector<int> &out) {
  OpSlots s = slots(m);
  for (int i = 0; i < 3; ++i)
    if (s.use[i] >= 0) out.push_back(s.use[i]);
  if (m.op == MOp::Ret && m.a >= 0) out.push_back(m.a);
  if (m.op == MOp::Call)
    for (const MArg &a : m.args) out.push_back(a.vreg);
}

static bool isPureDef(const MInst &m) {
  if (m.op == MOp::Call || m.op == MOp::Ret || m.op == MOp::Jmp ||
      m.op == MOp::BrNz || m.op == MOp::BrZ || m.op == MOp::BrCmp ||
      m.op == MOp::Prologue || m.op == MOp::Nop)
    return false;
  switch (m.op) {
  case MOp::Sw: case MOp::Fsw: case MOp::SwF: case MOp::SwG:
  case MOp::FswF: case MOp::FswG: case MOp::StkArgSw: case MOp::StkArgFsw:
    return false; // memory / call-frame side effects
  default:
    break;
  }
  return slots(m).def >= 0; // pure when it only computes a definition
}

// ---------------------------------------------------------------------------
// stage 1: block-local peephole (virtual registers)
// ---------------------------------------------------------------------------

size_t peepholeBlock(MBlock &blk,
                     const std::unordered_map<int, int> &uses) {
  std::vector<MInst> &v = blk.instrs;
  std::vector<MInst> out;
  out.reserve(v.size());
  size_t gone = 0;
  for (MInst &m : v) {
    bool keep = true;

    // self-move / self fmv are no-ops
    if ((m.op == MOp::MoveX || m.op == MOp::MoveF) && m.dst >= 0 &&
        m.dst == m.a) {
      keep = false;
    }
    // dead pure definition (nothing anywhere reads this virtual register)
    else if (isPureDef(m) && m.dst >= 0 && uses.count(m.dst) == 0) {
      keep = false;
    }
    // x + 0 / x ^ 0 become plain moves
    else if ((m.op == MOp::IAddI || m.op == MOp::IXorI) && m.imm == 0 &&
             m.dst >= 0) {
      if (m.dst == m.a) {
        keep = false; // x + 0 in place
      } else {
        m.op = MOp::MoveX; // dst = a
        m.imm = 0;
      }
    }
    // x - x / x ^ x are the constant zero
    else if ((m.op == MOp::ISub || m.op == MOp::IXor) && m.dst >= 0 &&
             m.a == m.b) {
      m.op = MOp::Li;
      m.a = -1;
      m.b = -1;
      m.imm = 0;
    }
    if (!keep) {
      ++gone;
      continue;
    }
    out.push_back(std::move(m));
  }
  v.swap(out);
  return gone;
}

// ---------------------------------------------------------------------------
// stage 2: scheduling - hoist register-addressed loads within their segment
// ---------------------------------------------------------------------------

bool isBarrier(MOp op) {
  switch (op) {
  case MOp::Prologue:
  case MOp::Ret:
  case MOp::Jmp:
  case MOp::BrNz:
  case MOp::BrZ:
  case MOp::BrCmp:
  case MOp::Call:
  case MOp::Sw: case MOp::Fsw: case MOp::SwF: case MOp::SwG:
  case MOp::FswF: case MOp::FswG:
  case MOp::StkArgSw: case MOp::StkArgFsw:
  case MOp::SpillLw: case MOp::SpillFlw: case MOp::SpillSw:
  case MOp::SpillFsw:
    return true;
  default:
    return false;
  }
}

bool isHoistableLoad(MOp op) {
  // Only register-addressed integer/float loads.  Frame/global fused loads
  // read sp- or symbol-relative memory whose writer may live outside the
  // segment, so they stay put.
  return op == MOp::Lw || op == MOp::Flw;
}

// can `load` (at index `j`) move one position up past `prev` (at j-1)?
bool canPass(const MInst &load, const MInst &prev) {
  if (isBarrier(prev.op)) return false;
  if (isHoistableLoad(prev.op)) return true; // reads commute
  OpSlots ps = slots(prev);
  // must not pass the definition of its own address register
  return ps.def < 0 || ps.def != load.a;
}

size_t scheduleBlock(MBlock &blk) {
  std::vector<MInst> &v = blk.instrs;
  size_t moved = 0;
  for (size_t i = 1; i < v.size(); ++i) {
    if (!isHoistableLoad(v[i].op)) continue;
    size_t j = i;
    while (j > 0 && canPass(v[j], v[j - 1])) {
      std::swap(v[j], v[j - 1]);
      --j;
      ++moved;
    }
  }
  return moved;
}

// ---------------------------------------------------------------------------
// stage 3 (post-register-allocation): redundant-move elimination
// ---------------------------------------------------------------------------
//
// After RegisterAllocator every register operand is a physical number.  X
// and F are separate files that share the 0..31 numbering, so all checks
// below are file-aware (slots() carries the file of every def/use).

static bool isMoveOp(MOp op) { return op == MOp::MoveX || op == MOp::MoveF; }
static bool isMoveF(const MInst &m) { return m.op == MOp::MoveF; }

// stores whose value operand (`a`) is read from a register
static bool isValueStoreOp(MOp op) {
  switch (op) {
  case MOp::Sw: case MOp::Fsw:
  case MOp::SwF: case MOp::FswF:
  case MOp::SwG: case MOp::FswG:
    return true;
  default:
    return false;
  }
}

// Read counts of each physical register in its own file.  `readsX`/`readsF`
// are indexed by physical register number (1..31; x0/zero never reads).
static void bumpPhysReads(const MInst &m, std::vector<int> &rx,
                          std::vector<int> &rf) {
  OpSlots s = slots(m);
  auto bump = [&](RFile f, int r) {
    if (r < 1 || r >= 32) return;
    (f == RF_F ? rf : rx)[(size_t)r]++;
  };
  for (int k = 0; k < 3; ++k) bump(s.useFile[k], s.use[k]);
  if (m.op == MOp::Ret && m.a >= 0)
    bump(m.ty == Type::F32 ? RF_F : RF_X, m.a);
  if (m.op == MOp::Call)
    for (const MArg &a : m.args)
      bump(a.ty == Type::F32 ? RF_F : RF_X, a.vreg);
}

// Does `next` fully redefine the move's destination (same file, no read of
// it) so the move can never be observed?
static bool nextRedefinesDst(const MInst &mv, const MInst &next) {
  if (isMoveF(mv) != isMoveF(next)) return false; // file mismatch
  // never delete across a call: its dst register is shared, and it may read
  // the value through an argument slot.
  if (next.op == MOp::Call || next.op == MOp::Ret || next.op == MOp::Jmp ||
      next.op == MOp::BrNz || next.op == MOp::BrZ || next.op == MOp::BrCmp)
    return false;
  OpSlots ns = slots(next);
  RFile f = isMoveF(mv) ? RF_F : RF_X;
  if (ns.defFile != f || ns.def != mv.dst) return false;
  for (int k = 0; k < 3; ++k)
    if (ns.useFile[k] == f && ns.use[k] == mv.dst) return false;
  return true;
}

// One block of the post-RA pass, iterated to a fixpoint so that removing one
// move can expose another (e.g. after ping-pong / dead-consecutive removal).
// `rx`/`rf` are the *initial* function-wide physical read counts, taken
// before any instruction is removed; deletions can only shrink real counts,
// so treating them as an upper bound is conservative and never unsafe.
size_t postPeepholeBlock(MBlock &blk, const std::vector<int> &rx,
                         const std::vector<int> &rf) {
  std::vector<MInst> &v = blk.instrs;
  size_t gone = 0;
  bool changed = true;
  for (int guard = 0; guard < 16 && changed; ++guard) {
    changed = false;
    for (size_t i = 0; i < v.size();) {
      MInst &m = v[i];
      bool removed = false;
      if (isMoveOp(m.op) && m.dst >= 1 && m.dst < 32 && m.a >= 1 &&
          m.a < 32) {
        if (m.a == m.dst) {
          // self-move (colouring coalesced a distinct src/dst pair)
          v.erase(v.begin() + (long)i);
          removed = true;
        } else if (i > 0 && isMoveOp(v[i - 1].op) &&
                   v[i - 1].dst == m.a && v[i - 1].a == m.dst &&
                   isMoveF(v[i - 1]) == isMoveF(m)) {
          // ping-pong:  mv a,b ; mv b,a   ->  the second is a no-op
          v.erase(v.begin() + (long)i);
          removed = true;
        } else if (i + 1 < v.size() &&
                   nextRedefinesDst(m, v[i + 1])) {
          // dead move: the very next instruction overwrites dst
          v.erase(v.begin() + (long)i);
          removed = true;
        } else if (i + 1 < v.size() && isValueStoreOp(v[i + 1].op) &&
                   isMoveF(v[i + 1]) == isMoveF(m)) {
          // mv rd,rs ; store rd, ... : when rd has *no other reader* in the
          // whole function than this store, the store can take rs directly.
          // The check uses the initial function-wide read counts (deletions
          // can only shrink real counts, so this is conservative, never
          // unsafe).  The store reads rd as its value operand (`a`); if rd is
          // also its address base (`b`) the count below would already be >=2.
          const std::vector<int> &reads = isMoveF(m) ? rf : rx;
          MInst &st = v[i + 1];
          if (st.a == m.dst && m.a >= 1 && m.a < 32 &&
              reads[(size_t)m.dst] == 1) {
            st.a = m.a; // store rs instead of rd
            v.erase(v.begin() + (long)i);
            removed = true;
          }
        }
      }
      if (removed) {
        ++gone;
        changed = true;
        // stay at the same index: the next instruction shifted into `i`
      } else {
        ++i;
      }
    }
  }
  return gone;
}

} // namespace

// ---------------------------------------------------------------------------
// MachineOptimizer methods
// ---------------------------------------------------------------------------

size_t MachineOptimizer::peephole(std::vector<MachineFunc> &fns) {
  size_t p = 0;
  for (MachineFunc &f : fns) {
    // function-wide use counts first, so a pure definition is only removed
    // when no block anywhere reads it.
    std::unordered_map<int, int> uses;
    auto bump = [&](const MInst &m) {
      std::vector<int> us;
      usesOf(m, us);
      for (int v : us)
        if (v >= 0) uses[v]++;
    };
    for (const MBlock &blk : f.blocks)
      for (const MInst &m : blk.instrs) bump(m);

    for (MBlock &blk : f.blocks) p += peepholeBlock(blk, uses);
  }
  return p;
}

size_t MachineOptimizer::schedule(std::vector<MachineFunc> &fns) {
  size_t s = 0;
  for (MachineFunc &f : fns)
    for (MBlock &blk : f.blocks) s += scheduleBlock(blk);
  return s;
}

size_t MachineOptimizer::removeRedundantMoves(std::vector<MachineFunc> &fns) {
  size_t total = 0;
  for (MachineFunc &f : fns) {
    // physical read counts before any deletion (see postPeepholeBlock)
    std::vector<int> rx(32, 0), rf(32, 0);
    for (const MBlock &blk : f.blocks)
      for (const MInst &m : blk.instrs) bumpPhysReads(m, rx, rf);
    for (MBlock &blk : f.blocks) total += postPeepholeBlock(blk, rx, rf);
  }
  return total;
}

} // namespace backend
} // namespace sakura
