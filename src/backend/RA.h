// Graph-colouring register allocator with spill support.
// Operates in place on machine functions: converts virtual registers to
// physical ones and records callee-saved usage.
#pragma once
#include <vector>
#include "Machine.h"

namespace sakura {
namespace backend {

void runRegisterAlloc(std::vector<MachineFunc> &fns);

} // namespace backend
} // namespace sakura
