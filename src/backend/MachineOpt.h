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

  // Coalesce zero-fill frame stores: a run of `sw x0` zero initialisations
  // becomes `sd x0` pairs, and the constant materialisation they used is left
  // for the peephole round to delete.  Returns the number of rewrites.
  size_t zeroFill(std::vector<MachineFunc> &fns);

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

  // Global (dominator-tree) CSE of constant / address materialisations
  // (`li`/`la`/frame addresses).  Machine-level LICM hoists one copy of each
  // invariant materialisation per loop body, so the same `la A` can land in
  // the preheader several times once the inliner has duplicated a loop; this
  // folds those duplicates (and any duplicate across blocks) onto the
  // dominating definition.  It is the machine-level counterpart of LLVM's
  // MachineCSE, deliberately restricted to instructions with no register
  // operand so that a dominating definition is unconditionally a legal
  // replacement.  Returns the number of instructions eliminated.
  size_t globalCse(std::vector<MachineFunc> &fns);

  // Pre-register-allocation copy folding: retarget the definition that feeds a
  // register copy at the copy's destination when the source has no other
  // reader, so the copy disappears before colouring (the virtual-register form
  // of coalescing).  Loop-latch induction updates `addiw t, iv, 4; mv iv, t`
  // are the canonical case.  Returns the number of copies eliminated.
  size_t coalesceMoves(std::vector<MachineFunc> &fns);

  // ---- post-register-allocation (operates on physical registers) ---------

  // Second peephole: eliminate the redundant register moves that colouring
  // exposes - self moves (dst==src after coalescing), ping-pong moves,
  // dead-consecutive moves to the same destination and a move whose value is
  // consumed immediately by the next store.  Returns the number of
  // instructions eliminated.
  size_t removeRedundantMoves(std::vector<MachineFunc> &fns);

  // Post-register-allocation loop rotation by block layout: sink a loop header
  // to its latch so the unconditional back edge becomes the physical
  // fall-through and only the conditional branch is paid per iteration.  Only
  // the block order and the layout-dependent terminators change; the CFG and
  // every register/operand is untouched.  Returns the number of loops rotated.
  size_t rotateLoopsByLayout(std::vector<MachineFunc> &fns);
};

} // namespace backend
} // namespace sakura
