// RISC-V assembly writer: turns coloured machine functions plus the module's
// globals into a printable .s text.
#pragma once

#include <string>
#include <vector>

#include "../midend/ir/IR.h"
#include "Machine.h"

namespace sakura {
namespace backend {

std::string emitAssembly(ir::Module *mod, std::vector<MachineFunc> &fns);

} // namespace backend
} // namespace sakura
