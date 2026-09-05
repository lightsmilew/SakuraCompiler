// RISC-V assembly writer: turns coloured machine functions plus the module's
// globals into a printable .s text.
//
// AssemblyWriter is the class-based front of the emission stage: the driver
// constructs one and calls write() once per module.  Frame layout, the
// per-call register-argument shuffle and the per-instruction text formatting
// are implementation details in Asm.cpp.
#pragma once

#include <string>
#include <vector>

#include "../midend/ir/IR.h"
#include "Machine.h"

namespace sakura {
namespace backend {

// Serialise coloured machine functions (physical register operands, after
// RegisterAllocator) and the module's global data into RISC-V assembly text.
class AssemblyWriter {
public:
  // Emit the .text section for every function in `fns` followed by the .data
  // section for the module's globals.  `mod` is read for globals only; the
  // functions were already consumed by InstructionSelector.
  std::string write(ir::Module *mod, std::vector<MachineFunc> &fns);
};

} // namespace backend
} // namespace sakura
