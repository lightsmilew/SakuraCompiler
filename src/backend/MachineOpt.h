// ---------------------------------------------------------------------------
// MachineOpt.h - block-local back-end peephole + scheduling passes.
//
// These run on the virtual-register machine IR produced by instruction
// selection, just before register allocation.  The stages are strictly
// semantics-preserving:
//
//   * peephole  - drops dead / self / zero-extending moves, folds x+0 / x^0
//     moves, x-x / x^x into li 0, removes Nops.
//   * schedule  - hoists register-addressed loads (lw/flw) as early as their
//     dependencies allow within each barrier-delimited straight-line segment,
//     giving the (in-order) pipeline an earlier start on memory latency.
//     Loads never cross a store / call / branch / prologue, so memory
//     ordering is preserved exactly.
//   * blockcse  - block-local common-subexpression elimination on the
//     virtual-register code: duplicate pure computations (constant / address
//     materialisation, integer/float arithmetic, compares) and redundant
//     loads from the same address are folded into the first occurrence.
//     Mirrors the reference SysY backend's BlockLocalCSE.
//   * licm      - loop-invariant code motion over the machine CFG: pure
//     instructions (la/li materialisation, address arithmetic, integer/float
//     ops on loop-invariant registers) are hoisted out of natural loops into
//     the preheader.  Mirrors the reference SysY backend's LICM (la/li
//     hoisting).  Memory loads are never hoisted, so the transform is
//     exception- and aliasing-safe by construction.
//
// A final stage runs *after* register allocation, once every register operand
// is physical: removeRedundantMoves() is a second peephole round whose whole
// purpose is eliminating the moves that colouring makes redundant (self /
// ping-pong / dead-consecutive moves, and a move feeding the very next store).
// Mirror of the reference SysY backend's SecondPeep (RemoveRedundantMovePass).
//
// Exposed separately so tests can verify each one fires; the RISCVBackend
// pipeline calls them in order and sums the totals for its pass report.
#pragma once

#include <cstddef>
#include <vector>

namespace sakura {
namespace backend {

struct MachineFunc;

// MachineOptimizer owns the machine-level optimisation stages.
class MachineOptimizer {
public:
  // ---- pre-register-allocation (operates on virtual registers) ----------

  // Strength-reduce integer div/rem/mul with a constant operand (recognised
  // as its unique `li` definition): power-of-two and magic-number division /
  // remainder, plus mul by 0 / +-1 / 2^k / 2^a+2^b into shifts and adds.
  // Returns the number of div/rem/mul instructions rewritten.
  size_t strengthReduction(std::vector<MachineFunc> &fns);

  // Drop dead/self/zero-extending moves and fold trivial arithmetic.
  // Returns the number of machine instructions eliminated.
  size_t peephole(std::vector<MachineFunc> &fns);

  // Hoist register-addressed loads within barrier-delimited straight-line
  // segments.  Returns the number of loads reordered.
  size_t schedule(std::vector<MachineFunc> &fns);

  // Block-local CSE over the virtual-register code (duplicate pure
  // computations + redundant loads from an identical address).  Returns the
  // number of instructions eliminated.
  size_t blockLocalCse(std::vector<MachineFunc> &fns);

  // Machine-level LICM: hoist loop-invariant pure instructions out of natural
  // loops into their preheader.  Returns the number of instructions hoisted.
  size_t licm(std::vector<MachineFunc> &fns);

  // ---- post-register-allocation (operates on physical registers) ---------

  // Second peephole: eliminate the redundant register moves that colouring
  // exposes - self moves (dst==src after coalescing), ping-pong moves,
  // dead-consecutive moves to the same destination and a move whose value is
  // consumed immediately by the next store.  Returns the number of
  // instructions eliminated.
  size_t removeRedundantMoves(std::vector<MachineFunc> &fns);
};

} // namespace backend
} // namespace sakura
