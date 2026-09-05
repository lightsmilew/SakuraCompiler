// ---------------------------------------------------------------------------
// Pipeline.cpp: RISCVBackend - the standard back-end pass sequence.
//
// run() hands the pure-cf module to the five stages and returns the emitted
// assembly; the driver never calls the individual stages itself.
// ---------------------------------------------------------------------------
#include "Pipeline.h"

namespace sakura {
namespace backend {

std::string RISCVBackend::run(ir::Module *mod, const BackendOptions &opts) {
  // ---- 1) instruction selection: pure-cf IR -> machine IR --------------
  // Throws if a structured op survived the mid-end (defensive re-check of
  // the verify-cf-only contract, in case a future driver bypasses it).
  std::vector<MachineFunc> fns = isel_.select(mod);

  // ---- 2) machine-level optimisation (peephole + scheduling) -----------
  // Runs on virtual registers before allocation; off for the -O0 baseline.
  size_t peephole = 0, scheduled = 0, postra = 0;
  if (opts.machineOpt) {
    peephole = opt_.peephole(fns);
    scheduled = opt_.schedule(fns);
  }

  // ---- 3) register allocation (graph-colouring + spills) ---------------
  ra_.allocate(fns);

  // ---- 4) second peephole after register allocation --------------------
  // Colouring exposes redundant moves (self / ping-pong / dead-consecutive /
  // move-into-next-store); this round removes them on physical registers.
  // Mirrors the reference SysY backend's SecondPeep (RemoveRedundantMovePass).
  if (opts.machineOpt) postra = opt_.removeRedundantMoves(fns);

  if (opts.stats)
    *opts.stats << "backend-peephole: " << peephole
                << " instructions eliminated\n"
                << "backend-schedule: " << scheduled << " loads hoisted\n"
                << "backend-postra-peephole: " << postra
                << " moves removed\n";

  // ---- 5) assembly emission ---------------------------------------------
  return writer_.write(mod, fns);
}

} // namespace backend
} // namespace sakura
