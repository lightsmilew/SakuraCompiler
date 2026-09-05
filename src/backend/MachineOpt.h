// MachineOpt.h - block-local back-end peephole + scheduling passes.
//
// These run on the virtual-register machine IR produced by instruction
// selection, just before register allocation (peephole() then schedule()).
// Both passes are strictly block-local and semantics-preserving:
//
//   * peephole  - drops dead / self / zero-extending moves, folds x+0 / x^0
//     moves, x-x / x^x into li 0, removes Nops.
//   * schedule  - hoists register-addressed loads (lw/flw) as early as their
//     dependencies allow within each barrier-delimited straight-line segment,
//     giving the (in-order) pipeline an earlier start on memory latency.
//     Loads never cross a store / call / branch / prologue, so memory
//     ordering is preserved exactly.
//
// A third stage runs *after* register allocation, once every register operand
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

// MachineOptimizer owns the three machine-level optimisation stages.
class MachineOptimizer {
public:
  // ---- pre-register-allocation (operates on virtual registers) ----------

  // Drop dead/self/zero-extending moves and fold trivial arithmetic.
  // Returns the number of machine instructions eliminated.
  size_t peephole(std::vector<MachineFunc> &fns);

  // Hoist register-addressed loads within barrier-delimited straight-line
  // segments.  Returns the number of loads reordered.
  size_t schedule(std::vector<MachineFunc> &fns);

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
