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
#include <cstdio>
#include <cstdlib>
#include <functional>
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
  case MOp::SdF:
    return false; // memory / call-frame side effects
  default:
    break;
  }
  return slots(m).def >= 0; // pure when it only computes a definition
}

// ---------------------------------------------------------------------------
// stage 0: zero-fill coalescing (virtual registers, before `peephole`)
// ---------------------------------------------------------------------------
//
// A `T x[16] = {};` local arrives as one `Li 0` per element plus one `SwF`
// per element: 03_sort1's radixSort has three 16-element arrays, i.e. 48 `li`
// + 48 `sw` (192 bytes of frame), where clang emits 8 `sd zero, off(sp)`.
// Two rewrites close most of that gap.  Both are keyed on the stored value
// being the *constant* zero, which is what makes them sound: an 8-byte store
// of x0 writes two zero words, exactly what the two 4-byte x0 stores did.
//
//   1. a `SwF` whose value is provably zero stores x0 directly instead of the
//      register holding `li 0`.  The `li` is then left with no readers, and
//      the `peephole` round that follows deletes it.
//   2. two such stores to adjacent 4-byte slots, the first of them 8-byte
//      aligned, fuse into a single `SdF` (`sd zero, off(sp)`).
//
// Alignment: the address is `sp + localBase + imm`.  `sp` is 16-byte aligned
// and `localBase` is a sum of align8'd / 8-multiple terms (layoutFrame in
// Asm.cpp), so `imm % 8 == 0` is exactly the condition for an 8-byte store
// and is checked here.
//
// Only frame-addressed integer stores take part.  A register-addressed `Sw`
// offers no provable adjacency between its two address vregs, a `SwG` would
// need the symbol's alignment, and a float store can never hold the integer
// constant zero.
//
// Does `m` materialise the integer constant zero into its destination?
static bool isZeroMaterialisation(const MInst &m) {
  if (m.dst < 0) return false;
  if (m.op == MOp::Li && m.imm == 0) return true;
  if (m.op == MOp::LiWide && m.cst == 0) return true;
  return false;
}

size_t zeroFillBlock(MBlock &blk) {
  std::vector<MInst> &v = blk.instrs;
  size_t changed = 0;

  // Straight-line walk: `liveZero` holds the registers whose *most recent*
  // definition in this block was proved to be zero.  Program order makes this
  // sound with no dominance reasoning, and it covers both shapes the front-end
  // produces: one `li 0` per store (straight out of ISel) and a single shared
  // `li 0` for a whole run (after the CSE rounds).
  std::unordered_set<int> liveZero;
  for (MInst &m : v) {
    if (isZeroMaterialisation(m))
      liveZero.insert(m.dst);
    else if (m.dst >= 0)
      liveZero.erase(m.dst); // redefined to something unknown
    if (m.op == MOp::SwF && m.a >= 0 && liveZero.count(m.a)) {
      m.a = -1; // x0
      ++changed;
    }
  }

  // Rewrite every provably-zero frame store to use x0 as its source.  Fusing
  // the resulting runs is a separate step (fuseZeroStoresBlock): the constant
  // materialisations that this leaves dead still sit between the stores and
  // must be removed first (dropDeadZeroMaterialisations).
  return changed;
}

// Drop the `Li 0` / `LiWide 0` definitions that the rewrite above left with no
// readers.  This is what makes the fusion possible: while each store carried
// its own `li`, no two stores were adjacent.
size_t dropDeadZeroMaterialisations(MachineFunc &f) {
  std::unordered_map<int, int> uses;
  for (const MBlock &blk : f.blocks)
    for (const MInst &m : blk.instrs) {
      std::vector<int> us;
      usesOf(m, us);
      for (int v : us)
        if (v >= 0) ++uses[v];
    }
  size_t gone = 0;
  for (MBlock &blk : f.blocks) {
    std::vector<MInst> out;
    out.reserve(blk.instrs.size());
    for (MInst &m : blk.instrs) {
      if (isZeroMaterialisation(m) && uses.count(m.dst) == 0) {
        ++gone;
        continue;
      }
      out.push_back(std::move(m));
    }
    blk.instrs.swap(out);
  }
  return gone;
}

size_t fuseZeroStoresBlock(MBlock &blk) {
  std::vector<MInst> &v = blk.instrs;
  std::vector<MInst> out;
  out.reserve(v.size());
  size_t changed = 0;
  // greedy, left to right: a run at offsets 8,12,16,20,... becomes `sd` at 8
  // and at 16.  A run starting at 4 keeps its first word as a `sw` and fuses
  // from 8 on, which is the alignment-correct choice ((4,8) would need a
  // 4-mod-8 address).  A pair with anything in between is left alone.
  for (size_t i = 0; i < v.size();) {
    const MInst &m = v[i];
    if (m.op == MOp::SwF && m.a < 0 && m.imm % 8 == 0 && i + 1 < v.size() &&
        v[i + 1].op == MOp::SwF && v[i + 1].a < 0 &&
        v[i + 1].imm == m.imm + 4 && v[i + 1].ty == m.ty) {
      MInst sd = m;
      sd.op = MOp::SdF;
      sd.b = -1;
      out.push_back(std::move(sd));
      i += 2;
      ++changed;
      continue;
    }
    out.push_back(std::move(v[i]));
    ++i;
  }
  v.swap(out);
  return changed;
}

size_t zeroFillFn(MachineFunc &f) {
  size_t changed = 0;
  for (MBlock &blk : f.blocks) changed += zeroFillBlock(blk);
  changed += dropDeadZeroMaterialisations(f);
  for (MBlock &blk : f.blocks) changed += fuseZeroStoresBlock(blk);
  return changed;
}

// ---------------------------------------------------------------------------
// stage 1: block-local peephole (virtual registers)
// ---------------------------------------------------------------------------

size_t peepholeBlock(MBlock &blk,
                     const std::unordered_map<int, int> &uses,
                     const std::unordered_set<int> &wordRegs) {
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
    else if ((m.op == MOp::IXorI || (m.op == MOp::IAddI &&
              (m.ty == Type::Ptr || wordRegs.count(m.a)))) && m.imm == 0 &&
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
  case MOp::SdF:
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
  // Keep loads in order: moving a later load above a whole run of earlier
  // loads lengthens all their live ranges without hiding any extra latency.
  if (isHoistableLoad(prev.op)) return false;
  OpSlots ps = slots(prev);
  // Machine IR is no longer SSA (phi copies can redefine a vreg).  Check
  // RAW, WAW and WAR, including a read of the load destination by `prev`.
  if (ps.def >= 0 && (ps.def == load.a || ps.def == load.dst)) return false;
  for (int r : ps.use)
    if (r >= 0 && r == load.dst) return false;
  return true;
}

static void transferLive(const MInst &m, std::vector<char> &live) {
  OpSlots s = slots(m);
  if (s.def >= 0) live[(size_t)s.def] = 0;
  for (int r : s.use)
    if (r >= 0) live[(size_t)r] = 1;
  for (const MArg &a : m.args)
    if (a.vreg >= 0) live[(size_t)a.vreg] = 1;
}

size_t scheduleBlock(MBlock &blk, const std::vector<char> &liveOut,
                     const std::vector<char> &isFloat) {
  std::vector<MInst> &v = blk.instrs;
  // A boundary holds the exact live set before that instruction. A legal
  // adjacent swap changes only the boundary between the two instructions.
  std::vector<std::vector<char>> boundary(v.size() + 1, liveOut);
  for (size_t i = v.size(); i-- > 0;) {
    boundary[i] = boundary[i + 1];
    transferLive(v[i], boundary[i]);
  }
  auto pressure = [&](const std::vector<char> &live, bool fp) {
    int count = 0;
    for (size_t r = 0; r < live.size(); ++r)
      count += live[r] && (isFloat[r] != 0) == fp;
    return count;
  };
  size_t moved = 0;
  for (size_t i = 1; i < v.size(); ++i) {
    if (!isHoistableLoad(v[i].op)) continue;
    size_t j = i;
    // A short latency window is enough for an in-order target.  Unbounded
    // hoisting used to keep an entire unrolled loop's loads live together.
    constexpr size_t latencyWindow = 2;
    while (j > 0 && i - j < latencyWindow && canPass(v[j], v[j - 1])) {
      auto between = boundary[j + 1];
      transferLive(v[j - 1], between);
      bool increasesPeak = false;
      for (bool fp : {false, true}) {
        int oldPeak = std::max({pressure(boundary[j - 1], fp),
                               pressure(boundary[j], fp),
                               pressure(boundary[j + 1], fp)});
        if (pressure(between, fp) > oldPeak) increasesPeak = true;
      }
      if (increasesPeak) break;
      std::swap(v[j], v[j - 1]);
      boundary[j] = std::move(between);
      --j;
    }
    moved += j != i;
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

// stores whose value operand (`a`) is read from a register.  `SdF` is
// deliberately absent: its source is always x0, and the "move into the next
// store" fold below must never retarget it (a full 64-bit register there
// would store the wrong high word).
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
  if (m.op == MOp::Call)
    for (const MArg &a : m.args)
      bump(a.ty == Type::F32 ? RF_F : RF_X, a.vreg);
}

// Does `next` fully redefine the move's destination (same file, no read of
// it) so the move can never be observed?
static bool nextRedefinesDst(const MInst &mv, const MInst &next) {
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
                   slots(v[i + 1]).useFile[0] == (isMoveF(m) ? RF_F : RF_X)) {
          // mv rd,rs ; store rd, ... : when rd has *no other reader* in the
          // whole function than this store, the store can take rs directly.
          // The check uses the initial function-wide read counts (deletions
          // can only shrink real counts, so this is conservative, never
          // unsafe).  The store reads rd as its value operand (`a`); if rd is
          // also its address base (`b`) the count below would already be >=2.
          // Use the store's operand file, not isMoveF(store): a store is not
          // a MoveF even when its value is floating. Xn and Fn may have the
          // same number while containing unrelated values (or an address).
          const std::vector<int> &reads = isMoveF(m) ? rf : rx;
          MInst &st = v[i + 1];
          if (st.a == m.dst && m.a >= 1 && m.a < 32 &&
              reads[(size_t)m.dst] == 1) {
            st.a = m.a; // store rs instead of rd
            v.erase(v.begin() + (long)i);
            removed = true;
          }
        } else if (i >= 1 && (removed = [&]() -> bool {
          // Loop-latch (or phi-arm) copies:  ...; op (dst=rs); mv rd, rs; ...
          // When rs is read nowhere but this move, the move is pure renaming:
          // point the earlier definition at rd instead.  Safe because nothing
          // sits between the definition and the move that could read rd (the
          // instruction is adjacent), and rd's old value has already been
          // consumed by earlier uses in the block (the new def kills it only
          // at this later point).  `op` may itself read rd (e.g. the canonical
          //  addiw rs, rd, 1;  mv rd, rs  latch): read-before-write is fine.
          MInst &m = v[i];
          MInst &p = v[i - 1];
          if (!isMoveOp(m.op)) return false;
          if (m.dst < 1 || m.dst > 31 || m.a < 1 || m.a > 31) return false;
          if (m.dst == m.a) return false;
          bool fileF = isMoveF(m);
          OpSlots ps = slots(p);
          if (ps.def != m.a || ps.defFile != (fileF ? RF_F : RF_X))
            return false; // previous must define the move's source
          if (p.op == MOp::Call || p.op == MOp::Ret || p.op == MOp::Jmp ||
              p.op == MOp::BrNz || p.op == MOp::BrZ || p.op == MOp::BrCmp)
            return false;
          const std::vector<int> &reads = fileF ? rf : rx;
          if (reads[(size_t)m.a] != 1) return false;
          p.dst = m.dst; // rename the definition to skip the move
          v.erase(v.begin() + (long)i);
          return true;
        }())) {
        }
      } // if (isMoveOp(...)) chain
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

// Propagate copies only inside their block, with explicit invalidation for
// non-SSA phi copies. Confined destinations avoid extending two independent
// live ranges into successor blocks. Calls use virtual operands here, so the
// allocator still sees every value that must survive the call.
static size_t propagateCopies(MachineFunc &f) {
  std::unordered_map<int, size_t> useBlock;
  std::unordered_map<int, int> defs;
  std::unordered_set<int> nonlocal;
  for (size_t b = 0; b < f.blocks.size(); ++b)
    for (const auto &m : f.blocks[b].instrs) {
      if (m.dst >= 0) ++defs[m.dst];
      std::vector<int> reads; usesOf(m, reads);
      for (int r : reads) {
        auto it = useBlock.emplace(r, b);
        if (!it.second && it.first->second != b) nonlocal.insert(r);
      }
    }
  size_t changed = 0;
  for (size_t b = 0; b < f.blocks.size(); ++b) {
    std::unordered_map<int, int> copies;
    for (auto &m : f.blocks[b].instrs) {
      OpSlots sl = slots(m);
      auto rewrite = [&](int &r) {
        auto it = copies.find(r);
        if (it != copies.end() && r != it->second) { r = it->second; ++changed; }
      };
      if (sl.use[0] >= 0) rewrite(m.a);
      if (sl.use[1] >= 0) rewrite(m.b);
      if (m.op == MOp::Ret) rewrite(m.a);
      for (auto &arg : m.args) rewrite(arg.vreg);
      if (sl.def >= 0) {
        for (auto it = copies.begin(); it != copies.end();)
          if (it->first == sl.def || it->second == sl.def) it = copies.erase(it);
          else ++it;
      }
      if ((m.op == MOp::MoveX || m.op == MOp::MoveF) && m.a >= 0 &&
          defs[m.dst] == 1 && !nonlocal.count(m.dst) && useBlock[m.dst] == b)
        copies[m.dst] = m.a;
    }
  }
  return changed;
}

// In word arithmetic (x << k) + c == (x + c/2^k) << k modulo 2^32.
// Reuse an already available (x + c/2^k); do not introduce a new temporary.
// This removes the add from neighbouring scaled offsets such as i*4-4.
static size_t reuseScaledOffsets(MachineFunc &f) {
  size_t changed = 0;
  std::unordered_map<int, int> uses;
  for (const auto &b : f.blocks) for (const auto &m : b.instrs) {
    std::vector<int> rs; usesOf(m, rs); for (int r : rs) ++uses[r];
  }
  for (auto &b : f.blocks) {
    std::unordered_map<int, MInst> available;
    for (auto &m : b.instrs) {
      if (m.op == MOp::IAddI && m.ty == Type::I32 && available.count(m.a)) {
        const auto shift = available.at(m.a);
        if (shift.op == MOp::SllI && shift.imm > 0 && shift.imm < 31 &&
            uses[m.a] == 1 && int64_t(m.imm) % (int64_t(1) << shift.imm) == 0) {
          int small = int64_t(m.imm) / (int64_t(1) << shift.imm);
          for (const auto &entry : available) {
            const auto &add = entry.second;
            if (add.op != MOp::IAddI || add.ty != Type::I32 ||
                add.a != shift.a || add.imm != small) continue;
            m.op = MOp::SllI; m.a = entry.first; m.imm = shift.imm;
            ++changed; break;
          }
        }
      }
      OpSlots s = slots(m);
      if (s.def >= 0) {
        for (auto it = available.begin(); it != available.end();) {
          OpSlots old = slots(it->second);
          bool kill = it->first == s.def;
          for (int r : old.use) kill |= r == s.def;
          if (kill) it = available.erase(it); else ++it;
        }
        if ((m.op == MOp::SllI || m.op == MOp::IAddI) && m.dst != m.a)
          available[m.dst] = m;
      }
    }
  }
  return changed;
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
  case MOp::Mulhu: case MOp::Mul64:
  case MOp::SllI: case MOp::SrlI: case MOp::SraI:
  case MOp::Shl64I: case MOp::Shr64I: case MOp::Sar64I:
  case MOp::IXor: case MOp::IXorI:
  case MOp::IAnd: case MOp::IAndI:
  case MOp::IOr:
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
  case MOp::SdF:
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
  case MOp::Mulhu:     return keyC("mhu", m.a, m.b);
  case MOp::Mul64:     return keyC("m64", m.a, m.b);
  case MOp::SllI:      return key2("sll", m.a, m.imm);
  case MOp::SrlI:      return key2("srl", m.a, m.imm);
  case MOp::SraI:      return key2("sra", m.a, m.imm);
  case MOp::Shl64I:    return key2("sll64", m.a, m.imm);
  case MOp::Shr64I:    return key2("srl64", m.a, m.imm);
  case MOp::Sar64I:    return key2("sra64", m.a, m.imm);
  case MOp::IXor:      return keyC("ix", m.a, m.b);
  case MOp::IXorI:     return key2("ixi", m.a, m.imm);
  case MOp::IAnd:      return keyC("an", m.a, m.b);
  case MOp::IAndI:     return key2("ani", m.a, m.imm);
  case MOp::IOr:       return keyC("or", m.a, m.b);
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
    if (v.empty()) {
      if (i + 1 < n) e.succ[(size_t)i].push_back(i + 1);
      continue;
    }
    // ISel may emit a conditional branch *followed by* an unconditional jump
    // at the end of a block, neither of whose targets is the layout
    // fall-through.  Every trailing branch contributes a successor, so scan
    // the whole terminator run rather than only `v.back()`; treating such a
    // block as a plain `jmp` would silently drop the conditional edge and
    // corrupt the dominance/loop analysis (LICM then hoists out of a loop
    // into a block that does not dominate it).  Same shape as RA's buildCFG.
    size_t k = v.size();
    while (k > 0 && isTerminatorOp(v[k - 1].op)) --k;
    if (k == v.size()) {
      // No terminator at all: plain fall-through (or a `Ret`, which stops).
      if (v.back().op != MOp::Ret && i + 1 < n)
        e.succ[(size_t)i].push_back(i + 1);
      continue;
    }
    for (size_t t = k; t < v.size(); ++t)
      if (v[t].op != MOp::Ret) {
        auto it = labelMap.find(v[t].sym);
        if (it != labelMap.end()) e.succ[(size_t)i].push_back(it->second);
      }
    if (v.back().op != MOp::Jmp && v.back().op != MOp::Ret && i + 1 < n)
      e.succ[(size_t)i].push_back(i + 1); // fall-through
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

// ---------------------------------------------------------------------------
// stage 4b: global (dominator-tree) CSE of constant / address materialisations
// ---------------------------------------------------------------------------

// Instructions whose result is a pure function of the instruction itself: no
// register operand can be redefined under it, so a *dominating* definition is
// always a legal replacement.  Restricting the pass to these makes it correct
// without any operand-redefinition tracking.
static bool isConstMatOp(MOp op) {
  switch (op) {
  case MOp::Li: case MOp::LiWide: case MOp::LiF:
  case MOp::LeaFrame: case MOp::LeaGlobal:
    return true;
  default:
    return false;
  }
}

// Machine LICM hoists loop-invariant `li`/`la` materialisations into the loop
// preheader, and every inlined copy of a loop body brings its own, so several
// identical `la A` end up in the *same* preheader block.  Block-local CSE runs
// before LICM and therefore never sees them together; and it cannot fold a
// duplicate living in a different block at all.  This is the machine-level
// counterpart of LLVM's MachineCSE: walk the dominator tree keeping every
// materialisation seen on the way down, and redirect a redundant copy to the
// dominating definition.  Measured on 01_mm1: main went from 15 `la` to 3,
// which is what finally lets the register allocator keep the hot scalars (the
// loop bound, the three array bases) in registers instead of reloading them
// from the spill slots on every loop test.
static size_t globalCseFn(MachineFunc &f) {
  const int n = (int)f.blocks.size();
  if (n < 2) return 0;
  CfgEdges e = buildCfg(f);
  std::vector<std::vector<char>> dom = computeDom(e, n);

  // Only single-definition vregs can be folded by redirecting every use: a vreg
  // with several defs (ISel reuses a phi's vreg across predecessor copies)
  // would still be defined at the other sites.
  std::unordered_map<int, int> defs;
  for (const MBlock &b : f.blocks)
    for (const MInst &m : b.instrs)
      if (m.dst >= 0) ++defs[m.dst];

  // Immediate dominators (from the full dominator sets) so the tree can be
  // walked top-down: every block is visited after the blocks that dominate it.
  std::vector<int> idom((size_t)n, -1);
  for (int b = 1; b < n; ++b) {
    int best = -1;
    for (int a = 1; a < n; ++a) {
      if (a == b || !dom[(size_t)b][(size_t)a]) continue;
      if (best < 0 || dom[(size_t)a][(size_t)best]) best = a;
    }
    idom[(size_t)b] = best;
  }
  std::vector<std::vector<int>> kids((size_t)n);
  for (int b = 1; b < n; ++b)
    if (idom[(size_t)b] >= 0) kids[(size_t)idom[(size_t)b]].push_back(b);

  std::unordered_map<std::string, int> avail; // key -> dominating vreg
  size_t gone = 0;
  std::vector<char> visited((size_t)n, 0);

  std::function<void(int)> walk = [&](int b) {
    if (visited[(size_t)b]) return; // unreachable block: no idom chain
    visited[(size_t)b] = 1;
    std::vector<std::string> added;
    std::vector<std::pair<int, int>> folds; // {redundant def, keep def}
    for (MInst &m : f.blocks[(size_t)b].instrs) {
      if (m.dst < 0 || !isConstMatOp(m.op)) continue;
      auto d = defs.find(m.dst);
      if (d == defs.end() || d->second != 1) continue;
      std::string key = pureKey(m);
      if (key.empty()) continue;
      auto it = avail.find(key);
      if (it != avail.end() && it->second != m.dst) {
        folds.push_back({m.dst, it->second});
      } else {
        avail[key] = m.dst;
        added.push_back(key);
      }
    }
    for (auto &pr : folds) {
      replaceVregUses(f, pr.first, pr.second);
      ++gone;
    }
    if (!folds.empty()) {
      std::unordered_set<int> drop;
      for (auto &pr : folds) drop.insert(pr.first);
      auto &v = f.blocks[(size_t)b].instrs;
      for (size_t i = 0; i < v.size();)
        if (v[i].dst >= 0 && drop.count(v[i].dst))
          v.erase(v.begin() + (long)i);
        else
          ++i;
    }
    for (int k : kids[(size_t)b]) walk(k);
    for (const std::string &k : added) avail.erase(k);
  };
  walk(0);
  for (int b = 0; b < n; ++b) walk(b); // unreachable blocks, if any
  return gone;
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
      // A real preheader must dominate the header, otherwise executing it
      // does not guarantee the hoisted value has been computed on every path
      // into the loop.  The dominance check also rejects the degenerate case
      // of an unreachable/self-only loop, where the analysis below would
      // otherwise place a definition on a path that never reaches the use.
      if (!dom[(size_t)h][(size_t)pre]) continue;
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
  bool dbg = std::getenv("SAKU_LICM_DEBUG") != nullptr;
  for (Loop &lp : loops) {
    const int pre = lp.pre;
    const std::vector<char> &inLoop = lp.inLoop;
    if (dbg)
      fprintf(stderr, "[licm] %s header=%s pre=%s\n", f.name.c_str(),
              f.blocks[(size_t)lp.header].name.c_str(),
              f.blocks[(size_t)pre].name.c_str());

    bool loopHasCall = false;
    for (int b = 0; b < n && !loopHasCall; ++b) {
      if (!inLoop[(size_t)b]) continue;
      for (const MInst &m : f.blocks[(size_t)b].instrs)
        if (m.op == MOp::Call) { loopHasCall = true; break; }
    }
    if (!std::getenv("SAKU_OLD_LICM_SPLIT") && loopHasCall) continue;

    // ---- split inline global addressing into an explicit preheader `la` ----
    // A global scalar access is a single `LwG sym` / `SwG sym` with the symbol
    // baked in, so the writer has to materialise `la t0, sym` afresh at every
    // use: three instructions per access, recomputed on every iteration, and
    // invisible to the round below because there is no address *instruction*
    // to hoist.  Splitting it into an explicit `LeaGlobal` plus the plain
    // register-addressed access lets the rounds below move the address into
    // the preheader (where `globalCse` then collapses the duplicates) and
    // leaves one instruction per access in the loop body.  This is LLVM's
    // RISC-V addressing model, where the `auipc`/`pcrel_hi` half lives in a
    // callee-saved base register instead of being re-materialised per access.
    //
    // Only done inside loops: in straight-line code the split is neutral (the
    // address still has to be computed once) and would only add live ranges.
    {
      int maxReg = -1;
      for (const MBlock &b : f.blocks)
        for (const MInst &m : b.instrs) {
          if (m.dst > maxReg) maxReg = m.dst;
          for (int r : {m.a, m.b})
            if (r > maxReg) maxReg = r;
        }
      int32_t nextReg = maxReg + 1;
      std::unordered_map<std::string, int32_t> symReg;
      std::vector<MInst> addrs;
      for (int b = 0; b < n; ++b) {
        if (!inLoop[(size_t)b]) continue;
        for (MInst &m : f.blocks[(size_t)b].instrs) {
          bool isG = (m.op == MOp::LwG || m.op == MOp::SwG ||
                      m.op == MOp::FlwG || m.op == MOp::FswG);
          if (!isG || m.imm != 0) continue; // keep a non-zero displacement
          int32_t addr;
          auto it = symReg.find(m.sym);
          if (it == symReg.end()) {
            addr = nextReg++;
            MInst lg;
            lg.op = MOp::LeaGlobal;
            lg.dst = addr;
            lg.sym = m.sym;
            lg.line = m.line;
            addrs.push_back(lg);
            symReg[m.sym] = addr;
          } else {
            addr = it->second;
          }
          bool isF = (m.op == MOp::FlwG || m.op == MOp::FswG);
          bool store = (m.op == MOp::SwG || m.op == MOp::FswG);
          if (store) {
            m.op = isF ? MOp::Fsw : MOp::Sw;
            m.b = addr; // a stays the value being stored
          } else {
            m.op = isF ? MOp::Flw : MOp::Lw;
            m.a = addr;
            m.b = -1;
          }
          m.sym.clear();
          m.imm = 0;
          ++total;
        }
      }
      if (!addrs.empty()) {
        std::vector<MInst> &pv = f.blocks[(size_t)pre].instrs;
        size_t ins = pv.size();
        for (size_t q = 0; q < pv.size(); ++q)
          if (isTerminatorOp(pv[q].op)) {
            ins = q;
            break;
          }
        pv.insert(pv.begin() + (long)ins, addrs.begin(), addrs.end());
      }
    }
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
//               otherwise magic-number division (signed 64-bit mul).
//   x % c       |c| == 1 -> 0;   else via the same quotient: x - q*|c|.
//
// The magic-number arithmetic mirrors the reference SysY SRFixedPass exactly
// and has been exhaustively verified: for every |c| in [3, 2^31) the sequence
//
//     q = floor(x*magic / 2^(32 + shift)) + (x < 0)
//
// equals the C truncating quotient for all x in int32, where `magic` is the
// Hacker's-Delight magic (always positive, materialised with a full 64-bit
// `li`, MOp::LiWide) and `32 + shift` folds the `>> 32` of the multiply-high
// into the magic's own shift.  See magicQuotTo for the two ways that product
// is evaluated - a `mul` when the magic fits in 32 bits, else a `mulh` - the
// former being LLVM's RISC-V shape for a constant i32 sdiv.
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

// Unsigned magic multiplier (Hacker's Delight, fig. 10-10).  Unlike `magicFor`
// this keeps M inside 32 bits, so the quotient of an unsigned division is
//
//     q = (x * M) >> (32 + s)
//
// Here x is a signed i32 proven non-negative, rather than an arbitrary u32.
// Its 32x32 product with M fits signed i64, so RV64 `mul` followed by `srli`
// suffices. There is no arithmetic shift or sign correction. LICM/CSE can
// hoist the magic materialization out of the loop.
struct UMagic {
  uint32_t M = 0;
  int s = 0;
  bool add = false; // the product needs the "(x + high) / 2" average variant
};

UMagic magicu(uint32_t d) {
  UMagic r;
  uint32_t nc = 0xFFFFFFFFu - ((0u - d) % d);
  int p = 31;
  uint32_t q1 = 0x80000000u / nc, r1 = 0x80000000u - q1 * nc;
  uint32_t q2 = 0x7FFFFFFFu / d, r2 = 0x7FFFFFFFu - q2 * d;
  uint32_t delta = 0;
  do {
    ++p;
    if (r1 >= nc - r1) {
      q1 = 2 * q1 + 1;
      r1 = 2 * r1 - nc;
    } else {
      q1 = 2 * q1;
      r1 = 2 * r1;
    }
    if (r2 + 1 >= d - r2) {
      if (q2 >= 0x7FFFFFFFu) r.add = true;
      q2 = 2 * q2 + 1;
      r2 = 2 * r2 + 1 - d;
    } else {
      if (q2 >= 0x80000000u) r.add = true;
      q2 = 2 * q2;
      r2 = 2 * r2 + 1;
    }
    delta = d - 1 - r2;
  } while (p < 64 && (q1 < delta || (q1 == delta && r1 == 0)));
  r.M = q2 + 1;
  r.s = p - 32;
  return r;
}

// Divisors whose unsigned magic needs the average variant AND a zero shift
// would need an undefined `>> -1`, and the plain variant must keep `32 + s`
// inside the 0..63 shift-amount range (the same limit LLVM's `srli` sequence
// has).  Anything else keeps the signed sequence, which is never longer than
// the corresponding unsigned form in those cases.
bool haveUnsignedQuot(uint32_t d) {
  UMagic um = magicu(d);
  if (um.add) return um.s >= 1;
  return um.s <= 31;
}

// Sequence builder for one rewritten instruction.  All intermediates are
// fresh virtual registers; the final definition lands on `D` (the original
// instruction's destination) so every existing use stays valid.
class SrCtx {
public:
  int32_t next = 0;
  int32_t D = -1;
  int32_t dstVreg() const { return D; }
  int line = 0;
  // Set when every read of D is a test of D against zero (BrZ / BrNz /
  // ISltuZ), i.e. the *value* of the computation is never observed, only
  // whether it is zero.  `x % 2^k` in that position is exactly a test of the
  // low k bits of x, so it collapses to one shift (see the IRem case).
  bool zeroOnly = false;
  // When non-zero, the `% 2^k` result is only ever compared (Eq/Ne) against
  // small positive constants, so the whole quotient/remainder expansion
  // collapses onto `x & andMask` - a one-instruction bit test (see the IRem
  // case for why the sign bit is part of the mask).
  int64_t andMask = 0;
  // The dividend x is known to be in [0, 2^31) (ISel's range analysis).  A
  // constant division is then an *unsigned* division and can use the much
  // shorter multiply/shift magic sequence instead of the signed one.
  bool xNonNeg = false;
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
  // As finB, but the result goes to an explicit vreg (or a fresh one for
  // `dst < 0`) instead of the instruction being rewritten.
  int32_t finTo(MOp op, int32_t a, int32_t b, int32_t dst) {
    MInst m;
    m.op = op;
    m.dst = dst >= 0 ? dst : fresh();
    m.a = a;
    m.b = b;
    push(m);
    return m.dst;
  }
};

// (2^k - 1) when x < 0, else 0: the round-towards-zero correction added before
// an arithmetic shift right by k, i.e. `((int64)x >> 63) >>> (64 - k)`.
//
// For k == 1 the bias is just the sign bit, `x >>> 63`, which saves
// materialising the all-ones sign mask first.  That is the same
// `(x >>s 63) >>> 63` -> `x >>> 63` simplification LLVM's DAGCombiner makes,
// and it sits on the critical path of every `x % 2` / `x / 2` - e.g. `rotl1`
// in crypto-1, which is a tight `temp = temp * 2 + temp % 2` loop.
int32_t signBias(SrCtx &c, int32_t x, int k) {
  if (k == 1) return c.uop(MOp::Shr64I, x, 63);
  int32_t sgn = c.uop(MOp::Sar64I, x, 63);        // -1 / 0
  return c.uop(MOp::Shr64I, sgn, 64 - k);         // (2^k-1) when x < 0
}

// x / (2^k), truncating towards zero.  Returns the quotient in a temp vreg.
int32_t pow2Quot(SrCtx &c, int32_t x, int k) {
  int32_t bias = signBias(c, x, k);
  int32_t xa = c.bop(MOp::IAdd, x, bias);       // canonical i32
  return c.uop(MOp::SraI, xa, k);
}

// magic quotient trunc(x / m) for m >= 3 non-power-of-two.  The result lands
// in `dst` when that is a real vreg, else in a fresh one (the div/rem
// rewrites want the last step to define the rewritten instruction directly).
//
// Both shapes compute the same thing, `floor(x*M / 2^(32+s)) + (x < 0)`, which
// is the C truncating quotient - the identity the reference SRFixedPass uses
// and which has been exhaustively verified for every |c| in [3, 2^31):
//
//   * magic < 2^32: this is LLVM's RISC-V sequence for a constant i32 sdiv -
//     one plain 64-bit `mul` of the sign-extended dividend with the magic, one
//     arithmetic shift that folds the `>> 32` of the multiply-high into the
//     magic's own shift, and the `t >>> 63` round-towards-zero bias.  The
//     product stays inside int64 because |x| < 2^31 and M < 2^32.  `mul` is
//     much cheaper than `mulh` - a multi-instruction `muls2_i64` expansion in
//     QEMU's TCG, and multi-cycle on real cores.
//
//   * magic >= 2^32 (a 33-bit magic, which a few divisors such as 97 need):
//     `x * M` would overflow int64, so take the high half of the 128-bit
//     product instead, `mulh(x << 32, M)`, which is the same value.
//
// The last step is a `w` form (addw/subw) so the quotient is left
// sign-extended, the invariant every downstream i32 consumer relies on.
int32_t magicQuotTo(SrCtx &c, int32_t x, int64_t m, int32_t dst = -1) {
  DivMagic dm = magicFor(m);
  int32_t mreg = c.liWide(dm.magic);
  if (static_cast<uint64_t>(dm.magic) < (1ULL << 32)) {
    int32_t t = c.bop(MOp::Mul64, x, mreg);     // sext64(x) * magic
    int32_t q0 = c.uop(MOp::Sar64I, t, 32 + dm.shift);
    int32_t bias = c.uop(MOp::Shr64I, t, 63);   // 1 when the product is < 0
    return c.finTo(MOp::IAdd, q0, bias, dst);
  }
  int32_t t1 = c.uop(MOp::Shl64I, x, 32);       // x in the high half
  int32_t h = c.bop(MOp::Mulh, t1, mreg);       // high 64 of (x*2^32)*magic
  int32_t q0 = c.uop(MOp::Sar64I, h, dm.shift);
  int32_t sgn = c.uop(MOp::Sar64I, x, 63);      // -1 when x < 0
  return c.finTo(MOp::ISub, q0, sgn, dst);
}

int32_t magicQuot(SrCtx &c, int32_t x, int64_t m) {
  return magicQuotTo(c, x, m);
}

// Complete product x * M for a non-negative signed i32 dividend. The caller
// extracts the high 32 bits (and any extra magic shift) from this i64 result.
int32_t unsignedMulHi(SrCtx &sr, int32_t x, uint32_t M) {
  // This path is only used for signed i32 values proven non-negative, so
  // x <= INT32_MAX and M <= UINT32_MAX. Their product fits signed i64.
  // A plain RV64 multiply therefore yields the same complete 32x32 product
  // as mulhu(x << 32, M << 32), without the shift or a multiply-high.
  int32_t mreg = sr.liWide((int64_t)M);
  return sr.bop(MOp::Mul64, x, mreg);
}

// floor(x / d) for x *known non-negative*, with the result left in a fresh
// vreg.  `haveUnsignedQuot(d)` must hold.
int32_t unsignedQuot(SrCtx &sr, int32_t x, uint32_t d) {
  UMagic um = magicu(d);
  int32_t h = unsignedMulHi(sr, x, um.M); // x * M
  if (!um.add)
    return sr.uop(MOp::Shr64I, h, 32 + um.s); // 64-bit srli
  // q = (t + (x - t) / 2) >> (s-1) where t = (x*M) >> 32.  `t <= x`, so the
  // half-difference form keeps the i32 sum from overflowing.
  int32_t t = sr.uop(MOp::Shr64I, h, 32);
  int32_t diff = sr.bop(MOp::ISub, x, t);
  int32_t half = sr.uop(MOp::SrlI, diff, 1);
  int32_t avg = sr.bop(MOp::IAdd, t, half);
  return sr.uop(MOp::SrlI, avg, um.s - 1);
}

// Same, but the final shift writes straight to D (the rewritten
// instruction's destination) instead of a fresh register.
void unsignedQuotFin(SrCtx &sr, int32_t x, uint32_t d) {
  UMagic um = magicu(d);
  int32_t h = unsignedMulHi(sr, x, um.M);
  if (!um.add) {
    sr.fin(MOp::Shr64I, h, 32 + um.s);
    return;
  }
  int32_t t = sr.uop(MOp::Shr64I, h, 32);
  int32_t diff = sr.bop(MOp::ISub, x, t);
  int32_t half = sr.uop(MOp::SrlI, diff, 1);
  int32_t avg = sr.bop(MOp::IAdd, t, half);
  sr.fin(MOp::SrlI, avg, um.s - 1);
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
          // A shift by zero is the identity: use the operand itself instead
          // of emitting a `slliw x, x, 0` no-op (odd multipliers like 5 =
          // 4 + 1 otherwise pay an extra instruction on every use).
          int32_t t0 = sh0 == 0 ? x : sr.uop(MOp::SllI, x, sh0);
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
        int32_t bias = signBias(sr, x, k);
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
      // A known non-negative dividend makes this an unsigned division, which
      // the unsigned magic computes in two instructions instead of the six the
      // signed sequence spends (see magicu).  Divisors needing the average
      // variant keep the signed path, which is never longer.
      if (sr.xNonNeg && haveUnsignedQuot((uint32_t)absC)) {
        unsignedQuotFin(sr, x, (uint32_t)absC);
        return true;
      }
      // final q writes straight to D (see magicQuotTo for the two shapes)
      magicQuotTo(sr, x, absC, sr.dstVreg());
    } else {
      int32_t q;
      if (sr.xNonNeg && haveUnsignedQuot((uint32_t)absC))
        q = unsignedQuot(sr, x, (uint32_t)absC);
      else
        q = magicQuot(sr, x, absC);
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
      // `(x % 2^k) == c` for a small positive c is a bit test, and nothing
      // else in the program looks at the remainder's *value* (the caller has
      // verified every read).  Since the sign decides whether the remainder is
      // c or c-2^k, folding the sign bit into the mask makes a single `and`
      // exact for both signs:
      //
      //   x % 2^k == c   <=>   (x & (SIGN | (2^k-1))) == c     (0 < c < 2^k)
      //
      // for x >= 0 the sign bit is clear and the low k bits are compared; for
      // x < 0 the sign bit is set and the value can never equal the small
      // positive c, which is exactly right because a negative remainder is
      // never c.  This is the fold InstCombine performs, and it is what
      // collapses the bit-at-a-time kernels (`crc1`, `huffman`, `crypto`):
      // their inner loops spend six instructions per `a % 2` today, one after.
      if (sr.andMask != 0) {
        int32_t mask = sr.liWide(sr.andMask);
        sr.finB(MOp::IAnd, x, mask);
        return true;
      }
      // `(x % 2^k) == 0` (also `!= 0`) is a divisibility test: it is true
      // exactly when the low k bits of x are clear, which `x << (32-k)`
      // reports as "result == 0".  When the remainder is never used for its
      // value, this single shift replaces the whole quotient+subtract
      // expansion - and, because the pattern's canonical home is the guard of
      // a loop body, it takes a 6-instruction parity test per iteration down
      // to one.  Only the zero-test sense has to be preserved, which the
      // caller guarantees via `zeroOnly`.
      if (sr.zeroOnly && k >= 1 && k <= 31) {
        sr.fin(MOp::SllI, x, 32 - k);
        return true;
      }
      // remainder is taken against |c| regardless of the divisor's sign
      //
      // `x - (x>>k<<k)`: the shift pair only clears the low k bits of the
      // biased dividend, so `(u >>s k) << k` folds into a single `and` with
      // the complement mask whenever that mask fits `andi`'s 12-bit
      // immediate.  LLVM's DAGCombiner performs the same
      // `sra`/`shl` -> `and` fold; it takes an instruction off every
      // `x % 2^k` whose value is actually observed.
      if (k >= 1 && k <= 11) {
        int32_t bias = signBias(sr, x, k);
        int32_t xa = sr.bop(MOp::IAdd, x, bias);
        int32_t qk = sr.uop(MOp::IAndI, xa, -(int32_t)(1u << k));
        sr.finB(MOp::ISub, x, qk);
        return true;
      }
      int32_t q = pow2Quot(sr, x, k);
      int32_t qk = sr.uop(MOp::SllI, q, k);
      sr.finB(MOp::ISub, x, qk);
      return true;
    }
    // magic: q = trunc(x / |c|), then rem = x - q*|c|.  When x is known
    // non-negative the quotient comes from the two-instruction multiply/shift
    // sequence instead of the signed one (see magicu).
    int32_t q;
    if (sr.xNonNeg && haveUnsignedQuot((uint32_t)absC))
      q = unsignedQuot(sr, x, (uint32_t)absC);
    else
      q = magicQuot(sr, x, absC);
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

  // The (unique) definition of every single-assignment virtual register, used
  // to confirm that a tested value really is a `% 2^k` leftover, and to see
  // through the `xori` that instruction selection emits for `x == C`.
  std::unordered_map<int, const MInst *> uniqDef;
  for (const MBlock &blk : f.blocks)
    for (const MInst &m : blk.instrs) {
      if (m.dst < 0) continue;
      auto d = defCount.find(m.dst);
      if (d != defCount.end() && d->second == 1) uniqDef[m.dst] = &m;
    }

  // vreg -> "every read of it is a comparison against zero".  Recognising the
  // `(x % 2^k) == 0` parity test needs to know that the remainder's value is
  // never looked at, only its zero-ness; Branches are the canonical consumer
  // (`bne d, zero, skip` guarding a loop body / conditional store).
  std::unordered_map<int, int> reads;    // total reads per vreg
  std::unordered_map<int, int> zeroCmp;  // reads that test against zero
  for (const MBlock &blk : f.blocks)
    for (const MInst &m : blk.instrs) {
      std::vector<int> us;
      usesOf(m, us);
      bool isZeroTest = false;
      int tested = -1;
      if (m.op == MOp::BrZ || m.op == MOp::BrNz) {
        isZeroTest = true;
        tested = m.a;
      } else if (m.op == MOp::ISltuZ) { // dst = (0 <u a) == (a != 0)
        isZeroTest = true;
        tested = m.a;
      } else if ((m.op == MOp::BrCmp || m.op == MOp::ICmp) && m.b < 0 &&
                 (m.imm == (int)Cond::Eq || m.imm == (int)Cond::Ne)) {
        // ISel folds `cmpi eq/ne x, 0` into a direct branch against the zero
        // register, leaving the compare-to-zero implicit in `b == -1`.
        isZeroTest = true;
        tested = m.a;
      }
      for (int r : us) {
        if (r < 0) continue;
        ++reads[r];
        if (isZeroTest && r == tested) ++zeroCmp[r];
      }
    }
  std::unordered_map<int, char> zeroOnly;
  for (const auto &kv : reads)
    if (kv.second > 0 && zeroCmp[kv.first] == kv.second)
      zeroOnly[kv.first] = 1;

  // vreg -> the constants every read of it compares against, when those reads
  // are Eq/Ne comparisons.  Combined with the read count below this identifies
  // a `% 2^k` whose *value* is never observed, only whether it equals one of a
  // few small constants.
  std::unordered_map<int, std::vector<int64_t>> cmpConsts;
  for (const MBlock &blk : f.blocks)
    for (const MInst &m : blk.instrs) {
      if (m.op != MOp::ICmp && m.op != MOp::BrCmp) continue;
      if (m.imm != (int)Cond::Eq && m.imm != (int)Cond::Ne) continue;
      if (m.a == m.b) continue;
      // `cmp x, zero` (b == -1) is the zero test, already covered by the
      // divisibility fold; it records no constant here.  There is one shape
      // that does carry a constant, though: instruction selection lowers
      // `x == C` with a 12-bit C to `xori t, x, C; cmp t, zero`, so look
      // through the `xori` and record C against the value it tested.  Without
      // this the bit-test fold below would stop firing for `% 2^k == c`
      // precisely because that cheaper compare spelling was chosen.
      if (m.b == -1) {
        auto ud = m.a >= 0 ? uniqDef.find(m.a) : uniqDef.end();
        // The xor's numeric result changes for negative dividends. Looking
        // through it is legal only when *all* its uses observe zero-ness;
        // an additional arithmetic/store use must retain the true remainder.
        if (ud != uniqDef.end() && ud->second->op == MOp::IXorI &&
            ud->second->a >= 0 && zeroOnly.count(m.a))
          cmpConsts[ud->second->a].push_back((int64_t)ud->second->imm);
        continue;
      }
      auto ca = m.a >= 0 ? constDef.find(m.a) : constDef.end();
      auto cb = m.b >= 0 ? constDef.find(m.b) : constDef.end();
      if (m.b >= 0 && ca != constDef.end()) cmpConsts[m.b].push_back(ca->second);
      if (m.a >= 0 && cb != constDef.end()) cmpConsts[m.a].push_back(cb->second);
    }

  // remainder vreg -> { dividend, mask } for the `and`-form bit test described
  // at the IRem case of buildRewrite.
  std::unordered_map<int, std::pair<int32_t, int64_t>> bitTest;
  for (const auto &kv : cmpConsts) {
    auto rd = reads.find(kv.first);
    if (rd == reads.end() || rd->second != (int)kv.second.size())
      continue; // some read looks at the value itself: `and` would not match
    auto df = uniqDef.find(kv.first);
    if (df == uniqDef.end()) continue;
    const MInst &ir = *df->second;
    if (ir.op != MOp::IRem || ir.a < 0 || ir.b < 0) continue;
    auto cv = constDef.find(ir.b);
    if (cv == constDef.end()) continue;
    int64_t absC = cv->second < 0 ? -cv->second : cv->second;
    if (absC < 2 || (absC & (absC - 1)) != 0) continue;
    int k = 0;
    for (int64_t v = absC; v > 1; v >>= 1) ++k;
    if (k < 1 || k > 31) continue;
    bool ok = true;
    for (int64_t c : kv.second)
      if (!(c > 0 && c < ((int64_t)1 << k))) ok = false;
    if (!ok) continue; // a non-positive / out-of-range constant needs the
                       // sign-adjusted mask, which this fold does not build
    int64_t mask = (int64_t)(int32_t)(0x80000000u | ((1u << (unsigned)k) - 1u));
    bitTest[kv.first] = {ir.a, mask};
  }

  int nextReg = 0;
  // A/B switch for the `% 2^k == c` bit-test fold (see the IRem case).
  static const bool noBitTest = std::getenv("SAKU_NO_BITTEST") != nullptr;
  // A/B switch for unsigned-magic division of a known
  // non-negative dividend (see magicu / the IDiv case).
  static const bool noUnsignedDiv = std::getenv("SAKU_NO_UDIV") != nullptr;
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
      ctx.zeroOnly = m.dst >= 0 && zeroOnly.count(m.dst) != 0;
      ctx.xNonNeg = !noUnsignedDiv && f.nonNeg.count(x) != 0;
      if (m.op == MOp::IRem && m.dst >= 0 && !noBitTest) {
        auto bt = bitTest.find(m.dst);
        if (bt != bitTest.end() && bt->second.first == x)
          ctx.andMask = bt->second.second;
      }
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

namespace {

// ---------------------------------------------------------------------------
// stage: pre-register-allocation copy folding (virtual-register coalescing)
// ---------------------------------------------------------------------------
//
// A copy `mv d, s` is pure renaming.  When `s` is defined by a side-effect-free
// instruction that nothing else reads, that definition can write `d` directly
// and the copy disappears - the same effect as coalescing `d` and `s` onto one
// register, but decided on virtual registers where the "read exactly once"
// property is exact.  (After register allocation a temporary shares a physical
// scratch register with many unrelated temporaries, so that property is gone;
// see postPeepholeBlock, which can only catch artificial cases.)
//
// The canonical target is a loop-latch induction update:
//
//      addiw t, iv, 4          addiw iv, iv, 4
//      mv    iv, t        ->   (copy deleted)
//
// Loop-carried phi copies emitted by ISel at the end of every predecessor have
// exactly this shape, so folding them here removes one move per loop iteration
// and frees a register.  The definition is only moved to `d` when no
// instruction between it and the copy reads or redefines `d`, so the value `d`
// had before the copy could not have been observed in between.
static size_t coalesceMovesFn(MachineFunc &f) {
  size_t gone = 0;
  bool changed = true;
  // Function-wide read counts are refreshed each round; within a round they
  // only ever over-count (a fold removes one reader), which is safe: a fold is
  // taken only when the source's sole reader is the copy, so a stale count can
  // never enable a rewrite whose source is read elsewhere.
  for (int guard = 0; guard < 16 && changed; ++guard) {
    changed = false;
    std::unordered_map<int, int> uses;
    for (const MBlock &b : f.blocks)
      for (const MInst &m : b.instrs) {
        std::vector<int> us;
        usesOf(m, us);
        for (int r : us)
          if (r >= 0) uses[r]++;
      }

    for (MBlock &blk : f.blocks) {
      std::vector<MInst> &v = blk.instrs;
      for (size_t i = 0; i < v.size(); ++i) {
        MInst &m = v[i];
        if (!isMoveOp(m.op) || m.dst < 0 || m.a < 0) continue;
        if (m.dst == m.a) { // self copy
          v.erase(v.begin() + (long)i);
          --i;
          ++gone;
          changed = true;
          continue;
        }
        int s = m.a, d = m.dst;
        RFile file = isMoveF(m) ? RF_F : RF_X;
        // `s` must be read by nothing but this copy, otherwise renaming its
        // definition to `d` would strand the other readers.
        auto uit = uses.find(s);
        if (uit == uses.end() || uit->second != 1) continue;

        for (size_t j = i; j-- > 0;) {
          MInst &p = v[j];
          // Control flow ends the straight-line window.  A call does *not*:
          // the rewrite is purely a virtual-register renaming (`s`'s only
          // reader is the copy, so retargeting the defining instruction at `d`
          // cannot strand a reader), and the redefinition / read-of-`d` checks
          // below still see every call argument through usesOf().  Bailing at
          // a call costs a register per loop-carried value updated before the
          // call, exactly the values a loop body containing a call already has
          // to keep in a callee-saved register.
          if (p.op == MOp::Ret || p.op == MOp::Jmp ||
              p.op == MOp::BrNz || p.op == MOp::BrZ || p.op == MOp::BrCmp ||
              p.op == MOp::Prologue)
            break;
          OpSlots ps = slots(p);
          bool defD = ps.defFile == file && ps.def == d;
          if (ps.defFile == file && ps.def == s) {
            // Definition of `s`: retarget it at `d` and drop the copy.  `p`
            // may itself read `d` (the canonical `addiw rs, rd, 1; mv rd, rs`
            // latch): one instruction reads its destination before writing
            // it, so that read observes the same value as before.  Nothing
            // between `p` and the copy may read `d` (the scan below would
            // have bailed), and `s`'s only reader is the copy, so no use is
            // left dangling.
            p.dst = d;
            v.erase(v.begin() + (long)i);
            --i;
            ++gone;
            changed = true;
            break;
          }
          if (defD) break; // `d` is redefined before the copy: bail
          // Any read of `d` strictly between the definition and the copy
          // would observe `d` earlier than it is written today.
          std::vector<int> us;
          usesOf(p, us);
          bool readsD = false;
          for (int r : us)
            if (r == d) { readsD = true; break; }
          if (readsD) break;
        }
      }
    }
  }
  return gone;
}

} // namespace

namespace {

// ---------------------------------------------------------------------------
// stage: post-register-allocation loop rotation by block layout
// ---------------------------------------------------------------------------
//
// The mid-end `loop-rotate` pass already turns `while (cond) { body }` inside
// out, but it only accepts loops whose body entry and exit blocks sit directly
// next to the header / latch ("layout gate"), and the block order left behind
// by the structured lowering - especially once the unroller has added a
// guard + remainder pair - usually violates that.  Those loops keep the
// two-branch shape:
//
//   ^H: <cond>; BrZ c, ^E     ; continues by falling into ^B
//   ^B: ...body...
//   ^L: j ^H                  ; unconditional back edge: a taken branch
//
// After register allocation nothing binds the layout any more (every operand
// is a physical register, every target is a label), so the header can simply
// be sunk to its latch.  The CFG is deliberately *not* changed: the ^L -> ^H
// edge merely becomes a fall-through, and only the terminators whose meaning
// depends on the physical order are reshaped:
//
//   ^P: ... ; j ^H            ; ^P used to fall through into ^H
//   ^B: ...body...
//   ^L: ...body tail...       ; `j ^H` deleted: falls through into ^H
//   ^H: <cond>; BrNz c, ^B    ; taken back into the body every iteration
//        j ^E                 ; reached on loop exit only
//   ^X: ...                   ; whatever followed ^L before
//
// One taken branch per iteration disappears; the added `j ^E` is fetched but
// never executed while the loop runs, and is free when ^E already follows ^H.
// Every natural loop with a single back edge is considered, one rotation per
// round so the CFG/layout analysis below always sees the updated order.
static bool isCondBranchOp(MOp op) {
  return op == MOp::BrNz || op == MOp::BrZ || op == MOp::BrCmp;
}

static Cond invertCond(Cond c) {
  switch (c) {
  case Cond::Eq: return Cond::Ne;
  case Cond::Ne: return Cond::Eq;
  case Cond::Lt: return Cond::Ge;
  case Cond::Ge: return Cond::Lt;
  case Cond::Le: return Cond::Gt;
  case Cond::Gt: return Cond::Le;
  }
  return c;
}

// Does `p` fall through into the block that physically follows it?  An empty
// block, a block whose last instruction is not a branch, and a block ending in
// a conditional branch all fall through; only `j` and `ret` stop fall-through.
static bool blockFallsThrough(const MachineFunc &f, int p) {
  const std::vector<MInst> &v = f.blocks[(size_t)p].instrs;
  if (v.empty()) return true;
  if (!isTerminatorOp(v.back().op)) return true;
  return isCondBranchOp(v.back().op);
}

// Rotate at most one loop of `f`; true when one was rotated.
static bool rotateOneLoopByLayout(MachineFunc &f) {
  const int n = (int)f.blocks.size();
  if (n < 4) return false;
  const CfgEdges e = buildCfg(f);
  const std::vector<std::vector<char>> dom = computeDom(e, n);
  std::unordered_map<std::string, int> labelMap;
  for (int i = 0; i < n; ++i) labelMap[f.blocks[(size_t)i].name] = i;

  for (int h = 1; h < n; ++h) { // block 0 carries the prologue: never move it
    // Exactly one back edge: the loop must have a single latch, otherwise only
    // one of them could turn into a fall-through and the layout would stay
    // ambiguous for the others.
    std::vector<int> latches;
    for (int b = 0; b < n; ++b)
      for (int s : e.succ[(size_t)b])
        if (s == h && dom[(size_t)b][(size_t)h]) latches.push_back(b);
    if (latches.size() != 1) continue;
    const int L = latches[0];
    if (L == h || L + 1 == h) continue; // already adjacent: nothing to gain
    // The latch must be a plain `j ^H` and nothing else.
    const std::vector<MInst> &lv = f.blocks[(size_t)L].instrs;
    if (lv.empty() || lv.back().op != MOp::Jmp ||
        lv.back().sym != f.blocks[(size_t)h].name)
      continue;
    {
      size_t lk = lv.size();
      while (lk > 0 && isTerminatorOp(lv[lk - 1].op)) --lk;
      if (lv.size() - lk != 1) continue; // cond + jmp: not a plain back edge
    }
    // The header must end in one conditional branch, optionally followed by an
    // explicit jump for the arm that is not the layout fall-through.
    std::vector<MInst> &hv = f.blocks[(size_t)h].instrs;
    if (hv.empty()) continue;
    size_t k = hv.size();
    while (k > 0 && isTerminatorOp(hv[k - 1].op)) --k;
    const size_t runLen = hv.size() - k;
    if (runLen == 0 || !isCondBranchOp(hv[k].op)) continue;
    int t1 = -1, t2 = -1;
    if (runLen == 1) { // taken target + layout fall-through
      if (h + 1 >= n) continue;
      auto it = labelMap.find(hv.back().sym);
      if (it == labelMap.end()) continue;
      t1 = it->second;
      t2 = h + 1;
    } else if (runLen == 2 && hv[k + 1].op == MOp::Jmp) {
      auto i1 = labelMap.find(hv[k].sym);
      auto i2 = labelMap.find(hv[k + 1].sym);
      if (i1 == labelMap.end() || i2 == labelMap.end()) continue;
      t1 = i1->second;
      t2 = i2->second;
    } else {
      continue;
    }
    if (t1 == t2) continue;
    const std::vector<char> inLoop = collectLoopNodes(e, n, h, dom);
    const bool in1 = inLoop[(size_t)t1] != 0, in2 = inLoop[(size_t)t2] != 0;
    if (in1 == in2) continue; // both stay in / both leave: not a simple header
    const int B = in1 ? t1 : t2; // body entry (in the loop)
    const int E = in1 ? t2 : t1; // loop exit  (outside the loop)
    if (B == h || E == h || E == L) continue;
    // Why this is worth doing at all: the rotation turns the loop's back edge
    // from an unconditional `j ^H` (which ends the translation block in an
    // exit-to-dispatch) into a *conditional* branch at the end of the body,
    // which the TCG can chain directly to itself.  Measured on the QEMU guest
    // with the 01_mm3 inner loop that difference alone is ~8x (940ms -> 125ms),
    // far more than the single branch the rotation nominally saves.
    // Where does control go after the header once ^B is no longer next?  The
    // block that followed ^L, unless ^E lands exactly there (then it is the
    // exit fall-through and no jump is needed).
    const std::string exitName = f.blocks[(size_t)E].name;
    const std::string afterL =
        (L + 1 < n) ? f.blocks[(size_t)(L + 1)].name : std::string();
    const bool exitFallsThrough = (afterL == exitName);

    if (std::getenv("SAKU_ROTATE_DEBUG"))
      std::fprintf(stderr,
                   "[layout-rotate] %s: header %s latch %s body %s exit %s"
                   " (exitFallsThrough=%d, shape=%s)\n",
                   f.name.c_str(), f.blocks[(size_t)h].name.c_str(),
                   f.blocks[(size_t)L].name.c_str(), f.blocks[(size_t)B].name.c_str(),
                   exitName.c_str(), (int)exitFallsThrough,
                   runLen == 2 ? "cond+jmp" : "cond");

    // ---- rewrite (no bail-out past this point) --------------------------
    const std::string hName = f.blocks[(size_t)h].name;
    // The loop continues when the edge to ^B is taken; retarget the condition
    // at it, inverting the predicate when ^B was the not-taken arm.
    MInst &cond = hv[k];
    if (t1 != B) {
      switch (cond.op) {
      case MOp::BrZ: cond.op = MOp::BrNz; break;
      case MOp::BrNz: cond.op = MOp::BrZ; break;
      case MOp::BrCmp: cond.imm = (int32_t)invertCond((Cond)cond.imm); break;
      default: break;
      }
    }
    cond.sym = f.blocks[(size_t)B].name;
    if (runLen == 2) hv.pop_back(); // explicit exit jump is subsumed now
    if (!exitFallsThrough) {
      MInst j;
      j.op = MOp::Jmp;
      j.sym = exitName;
      j.line = cond.line;
      hv.push_back(j);
    }
    // The block before ^H used to reach it by falling through (or by a
    // conditional branch whose fall-through was ^H); it now needs an explicit
    // jump, because ^H leaves this slot.
    const int P = h - 1;
    if (blockFallsThrough(f, P)) {
      MInst j;
      j.op = MOp::Jmp;
      j.sym = hName;
      j.line = 0;
      f.blocks[(size_t)P].instrs.push_back(j);
    }
    // Sink the header to just after the latch.
    MBlock moved = std::move(f.blocks[(size_t)h]);
    f.blocks.erase(f.blocks.begin() + h);
    const int Lpos = (L > h) ? L - 1 : L;
    f.blocks.insert(f.blocks.begin() + (long)(Lpos + 1), std::move(moved));
    // The latch's back edge is now the physical fall-through into ^H.
    std::vector<MInst> &nlv = f.blocks[(size_t)Lpos].instrs;
    if (!nlv.empty() && nlv.back().op == MOp::Jmp && nlv.back().sym == hName)
      nlv.pop_back();
    return true;
  }
  return false;
}

static size_t rotateLoopsByLayoutFn(MachineFunc &f) {
  size_t done = 0;
  const int n = (int)f.blocks.size();
  for (int round = 0; round < n; ++round)
    if (rotateOneLoopByLayout(f)) ++done;
    else break;
  return done;
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

// Greedily join high-weight CFG edges into disjoint layout chains. Unlike the
// mid-end order, physical-register MIR can place a use before its definition
// in the file: execution still follows the same CFG. No block can move across
// an implicit edge here; the caller has already materialized all fall-throughs.
static void placeMachineBlocks(MachineFunc &f) {
  const int n = (int)f.blocks.size();
  if (n < 3 || n > 2048 || blockFallsThrough(f, n - 1)) return;
  const auto cfg = buildCfg(f);
  const auto dom = computeDom(cfg, n);
  std::vector<int> depth(n, 0), calls(n, 0);
  for (int h = 0; h < n; ++h) {
    bool loop = false;
    for (int p : cfg.pred[h]) loop |= dom[p][h];
    if (!loop) continue;
    const auto members = collectLoopNodes(cfg, n, h, dom);
    for (int i = 0; i < n; ++i) depth[i] += members[i];
  }
  std::unordered_map<std::string, int> labels;
  for (int i = 0; i < n; ++i) {
    labels[f.blocks[i].name] = i;
    for (const auto &m : f.blocks[i].instrs) calls[i] += m.op == MOp::Call;
  }
  auto frequency = [&](int i) { return uint64_t(1) << std::min(3 * depth[i], 24); };
  auto probability = [&](int from, int to) {
    if (cfg.succ[from].size() == 1) return 100;
    bool exit = false;
    for (int s : cfg.succ[from]) exit |= depth[s] < depth[from];
    return exit ? (depth[to] < depth[from] ? 10 : 90) : 50;
  };
  struct Edge { int from, to; uint64_t weight; };
  std::vector<Edge> edges;
  for (int i = 0; i < n; ++i) for (int s : cfg.succ[i]) {
    if (s == 0 || s == i) continue;
    const uint64_t tie = (s == i + 1 ? 4 : 0) + std::min(2, calls[i] + calls[s]);
    edges.push_back({i, s, frequency(i) * probability(i, s) * 16 + tie});
  }
  std::stable_sort(edges.begin(), edges.end(), [](const Edge &a, const Edge &b) {
    return a.weight > b.weight;
  });
  std::vector<std::vector<int>> chains(n);
  std::vector<int> owner(n);
  for (int i = 0; i < n; ++i) { chains[i].push_back(i); owner[i] = i; }
  for (const auto &edge : edges) {
    const int a = owner[edge.from], b = owner[edge.to];
    if (a == b || chains[a].back() != edge.from || chains[b].front() != edge.to) continue;
    for (int block : chains[b]) { chains[a].push_back(block); owner[block] = a; }
    chains[b].clear();
  }
  std::vector<int> order;
  std::vector<char> placed(n, 0);
  int chain = owner[0];
  while ((int)order.size() < n) {
    for (int block : chains[chain]) order.push_back(block);
    placed[chain] = 1;
    int next = -1;
    // Continue with a connected remaining chain when possible, otherwise keep
    // the original order of disconnected/cold chains for deterministic ties.
    for (const auto &edge : edges)
      if (edge.from == order.back() && !placed[owner[edge.to]] &&
          chains[owner[edge.to]].front() == edge.to) { next = owner[edge.to]; break; }
    if (next < 0)
      for (int i = 0; i < n; ++i)
        if (!placed[owner[i]]) { next = owner[i]; break; }
    if (next < 0) break;
    chain = next;
  }
  auto cost = [&](const std::vector<int> &layout) {
    uint64_t weighted = 0;
    size_t jumps = 0;
    for (int p = 0; p < n; ++p) {
      const int i = layout[p], next = p + 1 < n ? layout[p + 1] : -1;
      const auto &code = f.blocks[i].instrs;
      if (code.empty() || code.back().op != MOp::Jmp) continue;
      const int target = labels.at(code.back().sym);
      if (target == next) continue;
      bool conditional = code.size() >= 2 && isCondBranchOp(code[code.size() - 2].op);
      if (conditional && labels.at(code[code.size() - 2].sym) == next) continue;
      ++jumps;
      weighted += frequency(i) * (conditional ? probability(i, target) : 100);
    }
    return std::make_pair(weighted, jumps);
  };
  std::vector<int> original(n);
  for (int i = 0; i < n; ++i) original[i] = i;
  const auto before = cost(original), after = cost(order);
  // Estimates are not a profile. Require a strict cost improvement and no
  // static jump increase before accepting the candidate; keep a good existing
  // loop rotation when the greedy chains cannot improve it.
  if (after.first >= before.first || after.second > before.second) return;
  std::vector<MBlock> reordered;
  reordered.reserve(n);
  for (int i : order) reordered.push_back(std::move(f.blocks[i]));
  f.blocks = std::move(reordered);
}

// Final branch folding after layout/rotation. Keep every fall-through explicit
// while deleting or joining blocks; otherwise erasing a block can silently
// redirect its physical predecessor. Registers and instructions inside a
// joined block keep their order, so this is also safe after allocation.
static size_t finalizeBlockLayoutFn(MachineFunc &f) {
  auto branchCount = [&]() {
    size_t count = 0;
    for (const auto &b : f.blocks) for (const auto &m : b.instrs)
      count += m.op == MOp::Jmp || isCondBranchOp(m.op);
    return count;
  };
  const size_t before = branchCount();
  for (size_t i = 0; i + 1 < f.blocks.size(); ++i)
    if (blockFallsThrough(f, (int)i)) {
      MInst jump; jump.op = MOp::Jmp; jump.sym = f.blocks[i + 1].name;
      f.blocks[i].instrs.push_back(std::move(jump));
    }

  bool changed = true;
  while (changed) {
    changed = false;
    // Forward branch-only blocks, including the empty blocks normalized above.
    // A self-loop is observable (it does not terminate), so never erase it.
    for (size_t i = 1; i < f.blocks.size(); ++i) {
      const auto &b = f.blocks[i];
      if (b.instrs.size() != 1 || b.instrs[0].op != MOp::Jmp ||
          b.instrs[0].sym == b.name) continue;
      const auto name = b.name, target = b.instrs[0].sym;
      for (auto &pred : f.blocks) for (auto &m : pred.instrs)
        if ((m.op == MOp::Jmp || isCondBranchOp(m.op)) && m.sym == name)
          m.sym = target;
      f.blocks.erase(f.blocks.begin() + i);
      changed = true;
      break;
    }
    if (changed) continue;

    const auto cfg = buildCfg(f);
    std::unordered_map<std::string, size_t> labels;
    for (size_t i = 0; i < f.blocks.size(); ++i) labels[f.blocks[i].name] = i;
    for (size_t i = 0; i < f.blocks.size(); ++i) {
      auto &code = f.blocks[i].instrs;
      if (code.empty() || code.back().op != MOp::Jmp ||
          (code.size() > 1 && isTerminatorOp(code[code.size() - 2].op))) continue;
      auto target = labels.find(code.back().sym);
      if (target == labels.end()) continue;
      const size_t j = target->second;
      if (j == 0 || j == i || cfg.pred[j].size() != 1 || cfg.pred[j][0] != (int)i)
        continue;
      // Join a single-successor/single-predecessor chain even when the earlier
      // layout separated it. This also combines update + latch-test blocks.
      code.pop_back();
      const auto &tail = f.blocks[j].instrs;
      code.insert(code.end(), tail.begin(), tail.end());
      f.blocks.erase(f.blocks.begin() + j);
      changed = true;
      break;
    }
  }

  if (!std::getenv("SAKU_NO_MACHINE_PLACE")) placeMachineBlocks(f);
  for (size_t i = 0; i < f.blocks.size(); ++i) {
    auto &code = f.blocks[i].instrs;
    if (code.empty() || code.back().op != MOp::Jmp) continue;
    const std::string next = i + 1 < f.blocks.size() ? f.blocks[i + 1].name : "";
    if (code.size() >= 2 && isCondBranchOp(code[code.size() - 2].op)) {
      auto &branch = code[code.size() - 2];
      if (branch.sym == code.back().sym) {
        // Both outcomes have the same successor: no condition is required.
        code.erase(code.end() - 2);
      } else if (branch.sym == next) {
        branch.sym = code.back().sym;
        if (branch.op == MOp::BrCmp) branch.imm = (int64_t)invertCond((Cond)branch.imm);
        else branch.op = branch.op == MOp::BrZ ? MOp::BrNz : MOp::BrZ;
        code.pop_back();
        continue;
      }
    }
    if (code.back().sym == next) code.pop_back();
  }
  const size_t after = branchCount();
  return before > after ? before - after : 0;
}

size_t MachineOptimizer::zeroFill(std::vector<MachineFunc> &fns) {
  size_t total = 0;
  for (MachineFunc &f : fns) total += zeroFillFn(f);
  return total;
}

size_t MachineOptimizer::peephole(std::vector<MachineFunc> &fns) {
  size_t p = 0;
  for (MachineFunc &f : fns) {
    if (!std::getenv("SAKU_NO_COPY_PROP")) p += propagateCopies(f);
    if (!std::getenv("SAKU_NO_SCALE_REUSE")) p += reuseScaledOffsets(f);
    std::unordered_set<int> wordRegs;
    std::unordered_set<int> wideRegs;
    for (const auto &b : f.blocks) for (const auto &m : b.instrs) {
      if (m.dst < 0 || slots(m).defFile != RF_X) continue;
      bool word = m.op == MOp::Li || m.op == MOp::Lw || m.op == MOp::LwF ||
        m.op == MOp::LwG || m.op == MOp::SllI || m.op == MOp::SrlI || m.op == MOp::SraI ||
        m.op == MOp::IMul || m.op == MOp::IDiv || m.op == MOp::IRem ||
        m.op == MOp::ISub || m.op == MOp::INeg || m.op == MOp::F2I ||
        ((m.op == MOp::IAdd || m.op == MOp::IAddI) && m.ty == Type::I32);
      if (word) wordRegs.insert(m.dst); else wideRegs.insert(m.dst);
    }
    for (int r : wideRegs) wordRegs.erase(r);
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

    for (MBlock &blk : f.blocks) p += peepholeBlock(blk, uses, wordRegs);
  }
  return p;
}

size_t MachineOptimizer::schedule(std::vector<MachineFunc> &fns) {
  size_t s = 0;
  for (MachineFunc &f : fns) {
    int nreg = 0;
    for (const auto &b : f.blocks)
      for (const auto &m : b.instrs) {
        OpSlots sl = slots(m);
        nreg = std::max(nreg, sl.def + 1);
        for (int r : sl.use) nreg = std::max(nreg, r + 1);
        for (const auto &a : m.args) nreg = std::max(nreg, a.vreg + 1);
      }
    std::vector<char> isFloat((size_t)nreg, 0);
    for (const auto &b : f.blocks)
      for (const auto &m : b.instrs) {
        OpSlots sl = slots(m);
        if (sl.def >= 0) isFloat[(size_t)sl.def] = sl.defFile == RF_F;
        for (int i = 0; i < 3; ++i)
          if (sl.use[i] >= 0) isFloat[(size_t)sl.use[i]] = sl.useFile[i] == RF_F;
        for (const auto &a : m.args)
          if (a.vreg >= 0) isFloat[(size_t)a.vreg] = a.ty == Type::F32;
      }
    const CfgEdges cfg = buildCfg(f);
    std::vector<std::vector<char>> in(f.blocks.size(), std::vector<char>((size_t)nreg, 0));
    auto out = in;
    bool changed = true;
    while (changed) {
      changed = false;
      for (size_t bi = f.blocks.size(); bi-- > 0;) {
        std::vector<char> live((size_t)nreg, 0);
        for (int successor : cfg.succ[bi])
          for (int r = 0; r < nreg; ++r) live[(size_t)r] |= in[(size_t)successor][(size_t)r];
        out[bi] = live;
        const auto &code = f.blocks[bi].instrs;
        for (size_t i = code.size(); i-- > 0;) transferLive(code[i], live);
        if (live != in[bi]) {
          in[bi] = std::move(live);
          changed = true;
        }
      }
    }
    for (size_t bi = 0; bi < f.blocks.size(); ++bi)
      s += scheduleBlock(f.blocks[bi], out[bi], isFloat);
  }
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

size_t MachineOptimizer::globalCse(std::vector<MachineFunc> &fns) {
  size_t total = 0;
  for (MachineFunc &f : fns) total += globalCseFn(f);
  return total;
}

size_t MachineOptimizer::coalesceMoves(std::vector<MachineFunc> &fns) {
  size_t total = 0;
  for (MachineFunc &f : fns) total += coalesceMovesFn(f);
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

size_t MachineOptimizer::rotateLoopsByLayout(std::vector<MachineFunc> &fns) {
  size_t total = 0;
  for (MachineFunc &f : fns) total += rotateLoopsByLayoutFn(f);
  return total;
}

size_t MachineOptimizer::finalizeBlockLayout(std::vector<MachineFunc> &fns) {
  size_t total = 0;
  for (MachineFunc &f : fns) {
    size_t oldBlocks, removed;
    do {
      oldBlocks = f.blocks.size();
      removed = finalizeBlockLayoutFn(f);
      total += removed;
    } while (removed || f.blocks.size() != oldBlocks);
  }
  return total;
}

} // namespace backend
} // namespace sakura
