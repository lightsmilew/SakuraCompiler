// Instruction selection: lower the MLIR-style mid-end IR into RISC-V machine
// instructions on virtual registers.  Frame slots are 4-byte (SysY has only
// i32/f32 values); arrays use one contiguous region of 4-byte slots each.
#pragma once

#include <vector>
#include "Machine.h"

namespace sakura {
namespace backend {

// Lower a module's defined functions to machine form.  Leaves `globals` for the
// writer to emit as data.
std::vector<MachineFunc> runInstructionSelection(ir::Module *mod);

} // namespace backend
} // namespace sakura
