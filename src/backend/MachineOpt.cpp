// MachineOpt.cpp - see MachineOpt.h
//
// Pre-register-allocation stages over the virtual-register machine code:
//   1. peephole()  - block-local cleanups (dead/self moves, x+0 / x^0 / x-x).
//   2. schedule()  - load hoisting within barrier-delimited segments.
//   3. blockLocalCse() - block-local CSE (duplicate pure computations and
//      redundant loads from an identical address).  Mirrors the reference
//      SysY backend's BlockLocalCSE.
//   4. licm()      - machine-level loop-invariant code motion: hoists pure
//      (la/li / address arithmetic / integer/float) instructions out of
//      natural loops into their preheader.  Mirrors the reference SysY
//      backend's LICM.
//   5. removeRedundantMoves() - post-register-allocation "SecondPeep": once
//      colours are baked in, register moves that colouring made redundant
//      (self / ping-pong / dead-consecutive / move-into-next-store) become
//      visible and are eliminated.  Mirrors the reference SysY backend's
//      RemoveRedundantMovePass which runs after its graph allocator.
#include "MachineOpt.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "Machine.h"

#include "../midend/ir/IR.h" // Cond (Eq/Ne .. Ge) for compare canonicalisation

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

namespace {

// ---------------------------------------------------------------------------
// stage 3: block-local common-subexpression elimination (virtual registers)
// ---------------------------------------------------------------------------

// Pure, side-effect-free computation ops (each writes one virtual register)
// eligible for CSE.  Memory ops are not here: loads go through a separate
// map that respects store/call ordering, stores/calls act as clobbers.
static bool isPureCseOp(MOp op) {
  switch (op) {
  case MOp::Li: case MOp::LiWide: case MOp::LiF:
  case MOp::LeaFrame: case MOp::LeaGlobal:
  case MOp::MoveX: case MOp::MoveF:
  case MOp::IAdd: case MOp::IAddI: case MOp::ISub: case MOp::INeg:
  case MOp::IMul: case MOp::IDiv: case MOp::IRem: case MOp::Mulh:
  case MOp::SllI: case MOp::SrlI: case MOp::SraI:
  case MOp::Shl64I: case MOp::Shr64I: case MOp::Sar64I:
  case MOp::IXor: case MOp::IXorI:
  case MOp::ISlt: case MOp::ISlti: case MOp::ISltu: case MOp::ISltiu:
  case MOp::ISltuZ: case MOp::ICmp: case MOp::FCmp:
  case MOp::FAdd: case MOp::FSub: case MOp::FMul: case MOp::FDiv:
  case MOp::FNeg:
  case MOp::I2F: case MOp::F2I: case MOp::FMvWX:
    return true;
  default:
    return false;
  }
}

static bool isLoadOp(MOp op) {
  switch (op) {
  case MOp::Lw: case MOp::Flw:
  case MOp::LwF: case MOp::FlwF:
  case MOp::LwG: case MOp::FlwG:
    return true;
  default:
    return false;
  }
}

// Instructions that may write the memory a load could be reading: stores,
// outbound stack-argument stores and calls.  Seeing one invalidates every
// pending load-CSE entry in the block.
static bool isMemClobberOp(MOp op) {
  switch (op) {
  case MOp::Call:
  case MOp::Sw: case MOp::Fsw:
  case MOp::SwF: case MOp::FswF:
  case MOp::SwG: case MOp::FswG:
  case MOp::StkArgSw: case MOp::StkArgFsw:
  case MOp::SpillSw: case MOp::SpillFsw:
    return true;
  default:
    return false;
  }
}

// --- canonical instruction identity keys (exact text match) ---------------
static std::string key2(const char *t, int a, int b) {
  return std::string(t) + "," + std::to_string(a) + "," + std::to_string(b);
}
static std::string key1(const char *t, int a) {
  return std::string(t) + "," + std::to_string(a);
}
// commutative operand pair: keep a canonical (min,max) order so `a+b` and
// `b+a` produce the same key.
static std::string keyC(const char *t, int a, int b) {
  return a <= b ? key2(t, a, b) : key2(t, b, a);
}

// Canonical identity of a pure computation ("" when not foldable).
static std::string pureKey(const MInst &m) {
  switch (m.op) {
  case MOp::Li:        return key1("li", m.imm);
  case MOp::LiWide:    return "liw," + std::to_string(m.cst);
  case MOp::LiF:       return key1("lif", m.imm);
  case MOp::LeaFrame:  return key1("lf", m.imm);
  case MOp::LeaGlobal: return "lg," + m.sym;
  case MOp::MoveX:     return key1("mx", m.a);
  case MOp::MoveF:     return key1("mf", m.a);
  case MOp::IAdd:      return keyC("ia", m.a, m.b);
  case MOp::IAddI:     return key2("iai", m.a, m.imm);
  case MOp::ISub:      return key2("is", m.a, m.b);
  case MOp::INeg:      return key1("in", m.a);
  case MOp::IMul:      return keyC("im", m.a, m.b);
  case MOp::IDiv:      return key2("id", m.a, m.b);
  case MOp::IRem:      return key2("ir", m.a, m.b);
  case MOp::Mulh:      return keyC("mh", m.a, m.b);
  case MOp::SllI:      return key2("sll", m.a, m.imm);
  case MOp::SrlI:      return key2("srl", m.a, m.imm);
  case MOp::SraI:      return key2("sra", m.a, m.imm);
  case MOp::Shl64I:    return key2("sll64", m.a, m.imm);
  case MOp::Shr64I:    return key2("srl64", m.a, m.imm);
  case MOp::Sar64I:    return key2("sra64", m.a, m.imm);
  case MOp::IXor:      return keyC("ix", m.a, m.b);
  case MOp::IXorI:     return key2("ixi", m.a, m.imm);
  case MOp::ISlt:      return key2("sl", m.a, m.b);
  case MOp::ISlti:     return key2("sli", m.a, m.imm);
  case MOp::ISltu:     return key2("slu", m.a, m.b);
  case MOp::ISltiu:    return key2("slui", m.a, m.imm);
  case MOp::ISltuZ:    return key1("slz", m.a);
  case MOp::ICmp: {
    bool comm = m.imm == (int)Cond::Eq || m.imm == (int)Cond::Ne;
    return (comm ? keyC("ic", m.a, m.b) : key2("ic", m.a, m.b)) + "," +
           std::to_string(m.imm);
  }
  case MOp::FCmp: {
    bool comm = m.imm == (int)Cond::Eq || m.imm == (int)Cond::Ne;
    return (comm ? keyC("fc", m.a, m.b) : key2("fc", m.a, m.b)) + "," +
           std::to_string(m.imm);
  }
  case MOp::FAdd:      return keyC("fa", m.a, m.b);
  case MOp::FSub:      return key2("fs", m.a, m.b);
  case MOp::FMul:      return keyC("fm", m.a, m.b);
  case MOp::FDiv:      return key2("fd", m.a, m.b);
  case MOp::FNeg:      return key1("fn", m.a);
  case MOp::I2F:       return key1("i2f", m.a);
  case MOp::F2I:       return key1("f2i", m.a);
  case MOp::FMvWX:     return key1("fmv", m.a);
  default:             return std::string();
  }
}

// Identity of a load: identical address only (same op + addressing).  For a
// register load the address register must not have been rewritten since the
// first load; for fused loads the offset/symbol pin the address.
static std::string loadKey(const MInst &m) {
  switch (m.op) {
  case MOp::Lw:  return "lw," + std::to_string(m.a) + "," +
                        std::to_string(m.imm);
  case MOp::Flw: return "flw," + std::to_string(m.a) + "," +
                        std::to_string(m.imm);
  case MOp::LwF: return "lwf," + std::to_string(m.imm);
  case MOp::FlwF:return "flwf," + std::to_string(m.imm);
  case MOp::LwG: return "lwg," + m.sym + "," + std::to_string(m.imm);
  case MOp::FlwG:return "flwg," + m.sym + "," + std::to_string(m.imm);
  default:       return std::string();
  }
}

static void renameUsesIn(MInst &m, int from, int to) {
  if (m.a == from) m.a = to;
  if (m.b == from) m.b = to;
  if (m.op == MOp::Call)
    for (MArg &arg : m.args)
      if (arg.vreg == from) arg.vreg = to;
}

// Rewrite every *use* (never a def) of virtual register `from` to `to` in the
// whole function.  Only called when `from` has a single definition (the one
// being eliminated), so there is no def site to rewrite.
static void replaceVregUses(MachineFunc &f, int from, int to) {
  if (from == to) return;
  for (MBlock &blk : f.blocks)
    for (MInst &m : blk.instrs) renameUsesIn(m, from, to);
}

// Block-local CSE on one straight-line block.
//
//   * pure computations never alias, so their entries survive to the end of
//     the block;
//   * load entries are discarded at every store / call;
//   * an entry is only reused while none of its *operand* registers has been
//     redefined since the entry was made.  (The destination can never be
//     redefined: only single-definition virtual registers are CSE'd.)
//
// When a duplicate is found all uses of the duplicate's destination (the only
// def in the function) are redirected to the first computation, and the
// duplicate is deleted.  Returns the number of instructions eliminated.
static size_t blockCseBlock(MachineFunc &f, MBlock &blk,
                            const std::unordered_map<int, int> &defs) {
  std::vector<MInst> &v = blk.instrs;
  // key -> { first def vreg, step of the first def }
  std::unordered_map<std::string, std::pair<int, int>> pure, loads;
  std::unordered_map<int, int> lastDef; // vreg -> step of its last kept def
  int step = 0;
  size_t gone = 0;

  auto opsStable = [&](const MInst &m, int atStep) {
    if (m.a >= 0) {
      auto it = lastDef.find(m.a);
      if (it != lastDef.end() && it->second > atStep) return false;
    }
    if (m.b >= 0) {
      auto it = lastDef.find(m.b);
      if (it != lastDef.end() && it->second > atStep) return false;
    }
    return true;
  };

  for (size_t i = 0; i < v.size();) {
    MInst &m = v[i];
    const int dst = m.dst;
    const bool singleDef =
        dst >= 0 && defs.count(dst) && defs.at(dst) == 1;

    if (isMemClobberOp(m.op)) loads.clear();

    bool fold = false;
    int foldTo = -1;
    if (singleDef) {
      if (isLoadOp(m.op)) {
        std::string key = loadKey(m);
        auto it = loads.find(key);
        if (it != loads.end() && opsStable(m, it->second.second)) {
          fold = true;
          foldTo = it->second.first;
        } else {
          loads[key] = {dst, step};
        }
      } else if (isPureCseOp(m.op)) {
        std::string key = pureKey(m);
        if (!key.empty()) {
          auto it = pure.find(key);
          if (it != pure.end() && opsStable(m, it->second.second)) {
            fold = true;
            foldTo = it->second.first;
          } else {
            pure[key] = {dst, step};
          }
        }
      }
    }

    if (fold) {
      replaceVregUses(f, dst, foldTo);
      v.erase(v.begin() + (long)i);
      ++gone;
      continue; // same index now holds the next instruction; step does not move
    }
    if (dst >= 0) lastDef[dst] = step; // (no self-invalidation: entries keyed by
                                       //  this dst were just inserted above)
    ++step;
    ++i;
  }
  return gone;
}

// ---------------------------------------------------------------------------
// stage 4: machine-level LICM (hoist loop-invariant pure computations)
// ---------------------------------------------------------------------------

static bool isTerminatorOp(MOp op) {
  return op == MOp::Ret || op == MOp::Jmp || op == MOp::BrNz ||
         op == MOp::BrZ || op == MOp::BrCmp;
}

struct CfgEdges {
  std::vector<std::vector<int>> succ, pred;
};

static CfgEdges buildCfg(const MachineFunc &f) {
  const int n = (int)f.blocks.size();
  CfgEdges e;
  e.succ.assign((size_t)n, {});
  e.pred.assign((size_t)n, {});
  std::unordered_map<std::string, int> labelMap;
  for (int i = 0; i < n; ++i) labelMap[f.blocks[(size_t)i].name] = i;
  for (int i = 0; i < n; ++i) {
    const std::vector<MInst> &v = f.blocks[(size_t)i].instrs;
    if (v.empty()) continue;
    const MInst &last = v.back();
    switch (last.op) {
    case MOp::Ret:
      break; // no successors
    case MOp::Jmp: {
      auto it = labelMap.find(last.sym);
      if (it != labelMap.end()) e.succ[(size_t)i].push_back(it->second);
      break;
    }
    case MOp::BrNz:
    case MOp::BrZ:
    case MOp::BrCmp: {
      auto it = labelMap.find(last.sym);
      if (it != labelMap.end()) e.succ[(size_t)i].push_back(it->second);
      if (i + 1 < n) e.succ[(size_t)i].push_back(i + 1); // fall-through
      break;
    }
    default:
      if (i + 1 < n) e.succ[(size_t)i].push_back(i + 1); // fall-through
      break;
    }
  }
  for (int i = 0; i < n; ++i)
    for (int s : e.succ[(size_t)i]) e.pred[(size_t)s].push_back(i);
  return e;
}

// dom[b][a] == true  <=>  a dominates b (b is 0-based block index).
static std::vector<std::vector<char>>
computeDom(const CfgEdges &e, int n) {
  std::vector<std::vector<char>> dom((size_t)n, std::vector<char>((size_t)n, 1));
  for (int a = 0; a < n; ++a) dom[0][(size_t)a] = 0; // entry dominates only itself
  dom[0][0] = 1;
  bool changed = true;
  while (changed) {
    changed = false;
    for (int b = 1; b < n; ++b) {
      std::vector<char> nd;
      if (e.pred[(size_t)b].empty()) {
        nd.assign((size_t)n, 0); // unreachable: only itself
        nd[(size_t)b] = 1;
      } else {
        nd.assign((size_t)n, 1);
        for (int p : e.pred[(size_t)b])
          for (int a = 0; a < n; ++a) nd[(size_t)a] = nd[(size_t)a] && dom[(size_t)p][(size_t)a];
        nd[(size_t)b] = 1; // a block always dominates itself
      }
      if (nd != dom[(size_t)b]) {
        dom[(size_t)b].swap(nd);
        changed = true;
      }
    }
  }
  return dom;
}

// A natural loop discovered from its header: the set of blocks that can reach
// one of the latches (back-edge sources dominated by the header) without
// passing back through the header.
static std::vector<char> collectLoopNodes(const CfgEdges &e, int n, int header,
                                          const std::vector<std::vector<char>> &dom) {
  std::vector<char> in((size_t)n, 0);
  std::vector<int> work;
  for (int b = 0; b < n; ++b)
    for (int s : e.succ[(size_t)b])
      if (s == header && dom[(size_t)b][(size_t)header]) work.push_back(b);
  in[(size_t)header] = 1;
  while (!work.empty()) {
    int x = work.back();
    work.pop_back();
    if (in[(size_t)x]) continue;
    in[(size_t)x] = 1;
    for (int p : e.pred[(size_t)x])
      if (p != header && !in[(size_t)p]) work.push_back(p);
  }
  return in;
}

// Hoist loop-invariant pure instructions out of the natural loop whose header
// dominates each latch, into the loop preheader.  Returns instructions moved.
static size_t licmMachineFn(MachineFunc &f) {
  const int n = (int)f.blocks.size();
  if (n < 2) return 0;
  CfgEdges e = buildCfg(f);
  std::vector<std::vector<char>> dom = computeDom(e, n);

  // Find natural loops (header h dominates latch b for every back edge b->h)
  // and order them by ascending size so inner loops hoist before outer ones.
  struct Loop {
    int header;
    int pre; // preheader block index, or -1 when there is none
    std::vector<char> inLoop;
  };
  std::vector<Loop> loops;
  {
    std::vector<std::pair<int, int>> order; // (size, loops-index)
    for (int h = 0; h < n; ++h) {
      bool isHeader = false;
      for (int b = 0; b < n; ++b)
        for (int s : e.succ[(size_t)b])
          if (s == h && dom[(size_t)b][(size_t)h]) isHeader = true;
      if (!isHeader) continue;
      std::vector<char> in = collectLoopNodes(e, n, h, dom);
      // Preheader: a predecessor of h outside the loop.  Loops entered only
      // through the header itself (e.g. the entry block) get none and are
      // skipped - there is nowhere safe to hoist to.
      int pre = -1;
      for (int p : e.pred[(size_t)h])
        if (!in[(size_t)p]) {
          pre = p;
          break;
        }
      if (pre < 0) continue;
      int cnt = 0;
      for (int b = 0; b < n; ++b) cnt += in[(size_t)b];
      loops.push_back({h, pre, std::move(in)});
      order.push_back({cnt, (int)loops.size() - 1});
    }
    std::sort(order.begin(), order.end());
    std::vector<Loop> sorted;
    sorted.reserve(loops.size());
    for (auto &pr : order) sorted.push_back(std::move(loops[(size_t)pr.second]));
    loops.swap(sorted);
  }

  size_t total = 0;
  for (Loop &lp : loops) {
    const int pre = lp.pre;
    const std::vector<char> &inLoop = lp.inLoop;
    // Repeated rounds: after one invariant moves to the preheader its result
    // may enable hoisting of a second one that used it as an operand.
    for (int round = 0; round < 8; ++round) {
      // current def/use summary over the whole function
      std::unordered_map<int, int> defs, owner, uses;
      for (int b = 0; b < n; ++b) {
        for (const MInst &m : f.blocks[(size_t)b].instrs) {
          if (m.dst >= 0) {
            ++defs[m.dst];
            owner[m.dst] = b;
          }
          std::vector<int> us;
          usesOf(m, us);
          for (int u : us)
            if (u >= 0) ++uses[u];
        }
      }

      // collect candidates (pure ops whose def is inside the loop, operands
      // defined outside the loop in blocks that dominate the preheader)
      std::vector<MInst> cands;
      std::vector<std::pair<int, int>> candPos;
      for (int b = 0; b < n; ++b) {
        if (!inLoop[(size_t)b]) continue;
        const std::vector<MInst> &v = f.blocks[(size_t)b].instrs;
        for (size_t j = 0; j < v.size(); ++j) {
          const MInst &m = v[j];
          if (m.dst < 0 || !isPureCseOp(m.op)) continue;
          auto dIt = defs.find(m.dst);
          auto oIt = owner.find(m.dst);
          auto uIt = uses.find(m.dst);
          if (dIt == defs.end() || dIt->second != 1) continue;
          if (oIt == owner.end() || oIt->second != b) continue;
          if (uIt == uses.end() || uIt->second == 0) continue; // keep dead code
          bool ok = true;
          for (int op : {m.a, m.b}) {
            if (op < 0) continue;
            auto fd = defs.find(op);
            auto fo = owner.find(op);
            if (fd == defs.end() || fd->second != 1 || fo == owner.end()) {
              ok = false; // undefined or multi-def (phi-style) operand
              break;
            }
            if (!dom[(size_t)pre][(size_t)fo->second]) {
              ok = false; // def does not dominate the preheader
              break;
            }
          }
          if (!ok) continue;
          cands.push_back(m);
          candPos.push_back({b, (int)j});
        }
      }
      if (cands.empty()) break;

      // remove the candidates from their blocks (descending so indices hold)
      for (auto it = candPos.rbegin(); it != candPos.rend(); ++it) {
        std::vector<MInst> &v = f.blocks[(size_t)it->first].instrs;
        v.erase(v.begin() + it->second);
      }
      // and splice them into the preheader, just before its terminator
      std::vector<MInst> &pv = f.blocks[(size_t)pre].instrs;
      size_t ins = pv.size();
      for (size_t q = 0; q < pv.size(); ++q)
        if (isTerminatorOp(pv[q].op)) {
          ins = q;
          break;
        }
      for (size_t q = 0; q < cands.size(); ++q)
        pv.insert(pv.begin() + (long)(ins + q), std::move(cands[q]));
      total += cands.size();
    }
  }
  return total;
}

} // namespace

// ---------------------------------------------------------------------------
// stage 0: constant-operand strength reduction (runs first, pre-RA)
// ---------------------------------------------------------------------------
//
// Rewrites integer div / rem / mul with a *constant* operand (recognised as
// its unique `li` definition) into short sequences of shifts and multiply-
// high, avoiding the multi-cycle `divw`/`remw`:
//
//   x * c       c == 0 -> 0;  c == +-1 -> x / -x;
//               |c| == 2^k -> shifts;   |c| == 2^a + 2^b -> shift-add
//   x / c       |c| == 1 -> x / -x;  |c| == 2^k -> round-toward-zero shift
//               otherwise magic-number division (signed 64-bit mulh).
//   x % c       |c| == 1 -> 0;   else via the same quotient: x - q*|c|.
//
// The magic-number arithmetic mirrors the reference SysY SRFixedPass exactly
// and has been exhaustively verified: for every |c| in [3, 2^31) the 64-bit
// sequence  q = ((x*2^32)*magic >> 64) >> shift (+1 when x < 0)  equals the
// C truncating quotient for all x in int32.  `magic` is always positive and
// < 2^33; the multiply-high is performed on 64-bit registers so magic is
// materialised with a full 64-bit `li` (MOp::LiWide).
namespace {

struct DivMagic {
  int64_t magic = 0;
  int shift = 0;
};

// Hacker's-Delight magic multiplier for a positive divisor (SysY layout).
DivMagic magicFor(int64_t d) {
  const uint64_t two32 = 1ULL << 32;
  uint64_t ad = (uint64_t)d;
  uint64_t anc = two32 - 1 - (two32 - 1) % ad;
  int p = 32;
  uint64_t q1 = two32 / anc, r1 = two32 % anc;
  uint64_t q2 = two32 / ad, r2 = two32 % ad;
  uint64_t delta;
  do {
    ++p;
    q1 *= 2;
    r1 *= 2;
    if (r1 >= anc) {
      ++q1;
      r1 -= anc;
    }
    q2 *= 2;
    r2 *= 2;
    if (r2 >= ad) {
      ++q2;
      r2 -= ad;
    }
    delta = ad - r2;
  } while (q1 < delta || (q1 == delta && r1 == 0));
  return {static_cast<int64_t>(q2 + 1), p - 32};
}

// Sequence builder for one rewritten instruction.  All intermediates are
// fresh virtual registers; the final definition lands on `D` (the original
// instruction's destination) so every existing use stays valid.
class SrCtx {
public:
  int32_t next = 0;
  int32_t D = -1;
  int line = 0;
  std::vector<MInst> seq;

  int32_t fresh() { return next++; }
  void push(MInst m) {
    m.line = line;
    m.ty = Type::I32;
    seq.push_back(std::move(m));
  }
  int32_t uop(MOp op, int32_t a, int32_t imm) {
    MInst m;
    m.op = op;
    m.dst = fresh();
    m.a = a;
    m.imm = imm;
    push(m);
    return m.dst;
  }
  int32_t bop(MOp op, int32_t a, int32_t b) {
    MInst m;
    m.op = op;
    m.dst = fresh();
    m.a = a;
    m.b = b;
    push(m);
    return m.dst;
  }
  int32_t li32(int32_t v) {
    MInst m;
    m.op = MOp::Li;
    m.dst = fresh();
    m.imm = v;
    push(m);
    return m.dst;
  }
  int32_t liWide(int64_t v) {
    MInst m;
    m.op = MOp::LiWide;
    m.dst = fresh();
    m.cst = v;
    push(m);
    return m.dst;
  }
  void fin(MOp op, int32_t a, int32_t imm = 0) {
    MInst m;
    m.op = op;
    m.dst = D;
    m.a = a;
    m.imm = imm;
    push(m);
  }
  void finB(MOp op, int32_t a, int32_t b) {
    MInst m;
    m.op = op;
    m.dst = D;
    m.a = a;
    m.b = b;
    push(m);
  }
};

// x / (2^k), truncating towards zero.  Returns the quotient in a temp vreg.
int32_t pow2Quot(SrCtx &c, int32_t x, int k) {
  int32_t sgn = c.uop(MOp::Sar64I, x, 63);      // -1 / 0
  int32_t bias = c.uop(MOp::Shr64I, sgn, 64 - k); // (2^k-1) when x < 0
  int32_t xa = c.bop(MOp::IAdd, x, bias);       // canonical i32
  return c.uop(MOp::SraI, xa, k);
}

// magic quotient trunc(x / m), m >= 3 non-power-of-two; returns in a temp.
int32_t magicQuot(SrCtx &c, int32_t x, int64_t m) {
  DivMagic dm = magicFor(m);
  int32_t mreg = c.liWide(dm.magic);
  int32_t t1 = c.uop(MOp::Shl64I, x, 32); // x << 32 (64-bit)
  int32_t h = c.bop(MOp::Mulh, t1, mreg); // high 64 of (x*2^32)*magic
  int32_t q0 = c.uop(MOp::Sar64I, h, dm.shift);
  int32_t sgn = c.uop(MOp::Sar64I, x, 63);
  return c.bop(MOp::ISub, q0, sgn); // +1 when x < 0, canonical i32
}

// Build the replacement for instruction `m` (dst = D) when the constant
// operand equals `c` and the other operand is `x`.  Returns false when no
// rewrite applies for this (op, c).
bool buildRewrite(SrCtx &sr, MOp op, int32_t x, int64_t cst) {
  int64_t cc = cst;
  int64_t absC = cc < 0 ? -cc : cc; // cc == INT32_MIN: 2^31, safe in int64
  switch (op) {
  // ---------------- IMul ----------------
  case MOp::IMul: {
    if (cc == 0) {
      sr.fin(MOp::Li, -1, 0); // Li writes imm; a unused
      return true;
    }
    if (cc == 1) {
      sr.finB(MOp::MoveX, x, -1);
      return true;
    }
    if (cc == -1) {
      sr.finB(MOp::INeg, x, -1);
      return true;
    }
    int k = -1;
    if ((absC & (absC - 1)) == 0) { // power of two, includes absC==2^31
      k = 0;
      int64_t v = absC;
      while (v > 1) {
        v >>= 1;
        ++k;
      }
    }
    if (k >= 0) {
      if (cc > 0) {
        sr.fin(MOp::SllI, x, k);
      } else {
        int32_t t = sr.uop(MOp::SllI, x, k);
        sr.finB(MOp::INeg, t, -1);
      }
      return true;
    }
    // |c| = 2^a + 2^b (odd part has two set bits): shift-add strength-reduce
    uint64_t v = (uint64_t)absC;
    if (v > 1 && (v & (v - 1)) != 0) {
      int tz = __builtin_ctzll(v);
      uint64_t odd = v >> tz;
      if (__builtin_popcountll(odd) == 2) {
        int b0 = __builtin_ctzll(odd);
        odd &= odd - 1;
        int b1 = __builtin_ctzll(odd);
        int sh0 = tz + b0;
        int sh1 = tz + b1;
        if (sh1 <= 31) { // both shifts fit a slliw
          int32_t t0 = sr.uop(MOp::SllI, x, sh0);
          int32_t t1 = sr.uop(MOp::SllI, x, sh1);
          if (cc > 0) {
            sr.finB(MOp::IAdd, t0, t1);
          } else {
            int32_t s = sr.bop(MOp::IAdd, t0, t1);
            sr.finB(MOp::INeg, s, -1);
          }
          return true;
        }
      }
    }
    return false; // keep the mulw
  }
  // ---------------- IDiv ----------------
  case MOp::IDiv: {
    if (cc == 0) return false; // div-by-zero: keep the hardware divw
    if (absC == 1) {
      if (cc == 1) {
        sr.finB(MOp::MoveX, x, -1);
      } else {
        sr.finB(MOp::INeg, x, -1);
      }
      return true;
    }
    if ((absC & (absC - 1)) == 0 && absC > 1) { // |c| = 2^k, 1 <= k <= 31
      int k = 0;
      int64_t v = absC;
      while (v > 1) {
        v >>= 1;
        ++k;
      }
      if (cc > 0) {
        int32_t sgn = sr.uop(MOp::Sar64I, x, 63);
        int32_t bias = sr.uop(MOp::Shr64I, sgn, 64 - k);
        int32_t xa = sr.bop(MOp::IAdd, x, bias);
        sr.fin(MOp::SraI, xa, k);
      } else {
        int32_t q = pow2Quot(sr, x, k);
        sr.finB(MOp::INeg, q, -1);
      }
      return true;
    }
    // non-power-of-two |c| >= 3 (works for even divisors too, verified)
    if (cc > 0) {
      // final q writes straight to D: append magic steps manually
      DivMagic dm = magicFor(absC);
      int32_t mreg = sr.liWide(dm.magic);
      int32_t t1 = sr.uop(MOp::Shl64I, x, 32);
      int32_t h = sr.bop(MOp::Mulh, t1, mreg);
      int32_t q0 = sr.uop(MOp::Sar64I, h, dm.shift);
      int32_t sgn = sr.uop(MOp::Sar64I, x, 63);
      sr.finB(MOp::ISub, q0, sgn);
    } else {
      int32_t q = magicQuot(sr, x, absC);
      sr.finB(MOp::INeg, q, -1);
    }
    return true;
  }
  // ---------------- IRem ----------------
  case MOp::IRem: {
    if (cc == 0) return false; // rem-by-zero: keep the hardware remw
    if (absC == 1) {
      sr.fin(MOp::Li, -1, 0);
      return true;
    }
    int k = -1;
    if ((absC & (absC - 1)) == 0 && absC > 1) {
      k = 0;
      int64_t v = absC;
      while (v > 1) {
        v >>= 1;
        ++k;
      }
    }
    if (k >= 0) {
      // remainder is taken against |c| regardless of the divisor's sign
      int32_t q = pow2Quot(sr, x, k);
      int32_t qk = sr.uop(MOp::SllI, q, k);
      sr.finB(MOp::ISub, x, qk);
      return true;
    }
    // magic: q = trunc(x / |c|), then rem = x - q*|c|
    int32_t q = magicQuot(sr, x, absC);
    int32_t mreg = absC <= INT32_MAX ? sr.li32((int32_t)absC)
                                     : sr.liWide(absC);
    int32_t qm = sr.bop(MOp::IMul, q, mreg);
    sr.finB(MOp::ISub, x, qm);
    return true;
  }
  default:
    return false;
  }
}

size_t strengthReduceFn(MachineFunc &f) {
  // vreg -> value, but only for vregs with *exactly one* definition in the
  // whole function, that definition being a Li/LiWide.  This matters because
  // instruction selection's phi-copy lowering *reuses* the phi's vreg across
  // the copies in every predecessor (e.g. an entry `li v0, 0` followed by a
  // back-edge `mv v0, v1`), so a vreg is not single-assignment in general.
  // Trusting the first Li of such a vreg would misread a loop-carried value
  // as a compile-time constant (e.g. turning the loop's `x * 2` into `0`).
  std::unordered_map<int, int> defCount;
  std::unordered_map<int, int64_t> constVal; // value when def is Li/LiWide
  for (const MBlock &blk : f.blocks)
    for (const MInst &m : blk.instrs) {
      if (m.dst < 0) continue;
      ++defCount[m.dst];
      if (m.op == MOp::Li) constVal[m.dst] = (int64_t)m.imm;
      else if (m.op == MOp::LiWide) constVal[m.dst] = m.cst;
    }
  std::unordered_map<int, int64_t> constDef;
  for (const auto &kv : defCount)
    if (kv.second == 1) {
      auto it = constVal.find(kv.first);
      if (it != constVal.end()) constDef[kv.first] = it->second;
    }

  int nextReg = 0;
  auto upd = [&](int32_t r) {
    if (r >= 0 && r + 1 > nextReg) nextReg = r + 1;
  };
  for (const MBlock &blk : f.blocks)
    for (const MInst &m : blk.instrs) {
      upd(m.dst);
      upd(m.a);
      upd(m.b);
      for (const MArg &a : m.args) upd(a.vreg);
    }

  size_t total = 0;
  for (MBlock &blk : f.blocks) {
    std::vector<MInst> &v = blk.instrs;
    for (size_t i = 0; i < v.size();) {
      MInst &m = v[i];
      int64_t c = 0;
      bool haveC = false;
      int32_t x = -1;
      if ((m.op == MOp::IDiv || m.op == MOp::IRem) && m.b >= 0) {
        auto it = constDef.find(m.b);
        if (it != constDef.end()) {
          c = it->second;
          haveC = true;
          x = m.a;
        }
      } else if (m.op == MOp::IMul && m.dst >= 0) {
        // either operand may be the constant (multiplication is commutative)
        auto it = constDef.find(m.a);
        if (it != constDef.end()) {
          c = it->second;
          haveC = true;
          x = m.b;
        } else {
          it = constDef.find(m.b);
          if (it != constDef.end()) {
            c = it->second;
            haveC = true;
            x = m.a;
          }
        }
      }
      if (!haveC || x < 0) {
        ++i;
        continue;
      }
      SrCtx ctx;
      ctx.next = nextReg;
      ctx.D = m.dst;
      ctx.line = m.line;
      if (!buildRewrite(ctx, m.op, x, c)) {
        nextReg = ctx.next; // harmless; nothing emitted
        ++i;
        continue;
      }
      // splice the sequence over the original instruction
      size_t n = ctx.seq.size();
      nextReg = ctx.next;
      std::vector<MInst> repl = std::move(ctx.seq);
      v.erase(v.begin() + (long)i);
      v.insert(v.begin() + (long)i, repl.begin(), repl.end());
      ++total;
      i += n; // the emitted ops are never themselves div/rem/mul-by-const
    }
  }
  return total;
}

} // namespace

// ---------------------------------------------------------------------------
// MachineOptimizer methods
// ---------------------------------------------------------------------------

size_t MachineOptimizer::strengthReduction(std::vector<MachineFunc> &fns) {
  size_t total = 0;
  for (MachineFunc &f : fns) total += strengthReduceFn(f);
  return total;
}

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

size_t MachineOptimizer::blockLocalCse(std::vector<MachineFunc> &fns) {
  size_t total = 0;
  for (MachineFunc &f : fns) {
    // Function-wide definition counts: a candidate is only folded when its
    // destination virtual register has exactly one definition in the whole
    // function (the one being eliminated), so redirecting every use is safe.
    std::unordered_map<int, int> defs;
    for (const MBlock &blk : f.blocks)
      for (const MInst &m : blk.instrs)
        if (m.dst >= 0) ++defs[m.dst];
    for (MBlock &blk : f.blocks) total += blockCseBlock(f, blk, defs);
  }
  return total;
}

size_t MachineOptimizer::licm(std::vector<MachineFunc> &fns) {
  size_t total = 0;
  for (MachineFunc &f : fns) total += licmMachineFn(f);
  return total;
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
