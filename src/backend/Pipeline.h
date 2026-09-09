// ---------------------------------------------------------------------------
// Pipeline.h: RISCVBackend - the back-end pipeline orchestrator.
//
// main (and any other driver) never assembles the back-end stages itself: it
// constructs one RISCVBackend and calls run() on the pure-cf module produced
// by the mid-end, receiving the final RISC-V assembly text.  Instruction
// selection, the machine peephole/scheduler, register allocation, the second
// (post-allocation) peephole and assembly emission all live inside this one
// class - mirroring the reference SysY compiler's RISCVBuilder.  Stages are
// added/ordered here, never in the driver:
//
//     pure-cf IR (from runMidEndPipeline)
//       InstructionSelector::select   (IR op -> RISC-V MInst on virtual regs)
//       MachineOptimizer::peephole    (block-local peephole; off at O0)
//       MachineOptimizer::schedule    (load hoisting within segments; off at O0)
//       MachineOptimizer::blockLocalCse (block-local CSE; off at O0)
//       MachineOptimizer::licm        (machine-level LICM; off at O0)
//       RegisterAllocator::allocate   (graph-colouring + spills)
//       MachineOptimizer::removeRedundantMoves (post-RA move elimination)
//       AssemblyWriter::write         (coloured machine IR + globals -> .s)
// ---------------------------------------------------------------------------
#pragma once

#include <iosfwd>
#include <string>

#include "../midend/ir/IR.h"
#include "Asm.h"
#include "ISel.h"
#include "MachineOpt.h"
#include "RA.h"

namespace sakura {
namespace backend {

// Options for RISCVBackend::run().
struct BackendOptions {
  // Run the machine-level peephole + load scheduler over the virtual-register
  // code before register allocation, and the second redundant-move peephole
  // after it (the -O0 baseline keeps the naive schedule).  Mirrors the
  // mid-end optimisation level.
  bool machineOpt = true;
  // When non-null, a per-stage report is written here (driven by the
  // --pass-stats flag), e.g. "backend-peephole: N instructions eliminated".
  std::ostream *stats = nullptr;
};

// RISCVBackend is the single high-level back-end entry point: it owns the
// stage objects (instruction selector, machine optimizer, register allocator,
// assembly writer) and runs the whole pipeline inside run().
class RISCVBackend {
public:
  // Lower `mod` (pure-cf, as verified by the mid-end's verify-cf-only pass) to
  // RISC-V assembly in one call and return the printable .s text.  `mod` is
  // read for globals only after instruction selection has consumed the
  // functions.  Stage totals (when opts.stats is set) are written there.
  std::string run(ir::Module *mod, const BackendOptions &opts = BackendOptions{});

private:
  InstructionSelector isel_;  // stage 1: cf IR -> machine IR (virtual regs)
  MachineOptimizer opt_;      // stages 2..5: pre-RA peephole/schedule/blockcse/licm + post-RA moves
  RegisterAllocator ra_;      // stage 6: colour virtual regs (with spills)
  AssemblyWriter writer_;     // stage 7: machine IR + globals -> .s
};

} // namespace backend
} // namespace sakura
