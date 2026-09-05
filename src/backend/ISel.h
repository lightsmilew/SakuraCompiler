// Instruction selection: lower the MLIR-style mid-end IR into RISC-V machine
// instructions on virtual registers.  Frame slots are 4-byte (SysY has only
// i32/f32 values); arrays use one contiguous region of 4-byte slots each.
//
// InstructionSelector is the class-based front of this stage: the driver
// constructs one and calls select() per module.  All per-function lowering
// state (virtual-register map, pointer/frame-slot records, cf.phi bookkeeping)
// lives inside the implementation in ISel.cpp; ISel.h only exposes the stage.
#pragma once

#include <vector>

#include "Machine.h"

namespace sakura {
namespace backend {

// Lower the pure-cf module produced by the mid-end pipeline to RISC-V machine
// instructions on virtual registers.
//
// cf.phi is not emitted as an instruction: the phi of a successor is written
// by register copies at the end of each predecessor (a parallel-copy schedule,
// so cyclic SSA merges such as register rotations stay correct), and each phi
// result reads that pre-assigned virtual register.  Compare-and-branch pairs
// are fused afterwards (transparent over the phi copies).  Globals are left
// for the AssemblyWriter to emit as data.
//
// The mid-end contract (only cf-dialect ops) is re-checked here so the
// boundary holds even if a future driver bypasses the pipeline.
class InstructionSelector {
public:
  // Lower every defined, non-library function of `mod` to machine form.
  // `mod` itself is only read for globals after this call.
  std::vector<MachineFunc> select(ir::Module *mod);
};

} // namespace backend
} // namespace sakura
