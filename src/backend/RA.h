// Graph-colouring register allocator with spill support.
//
// RegisterAllocator is the class-based front of the allocation stage: the
// driver calls allocate() once per module.  It operates in place on the
// machine functions - converting virtual registers to physical ones, spilling
// when colouring fails - and records the callee-saved usage each function
// needs for its prologue/epilogue.  The per-function liveness/interference/
// colouring machinery is an implementation detail in RA.cpp.
#pragma once

#include <vector>

#include "Machine.h"

namespace sakura {
namespace backend {

// Colour every machine function's virtual registers to physical ones in
// place.  Values that cannot be coloured are spilled through explicit
// spill-slot loads/stores and re-coloured, so after a successful run all
// register operands hold physical register numbers (0..31).
class RegisterAllocator {
public:
  // Allocate `fns` in place; also fills each function's usedCSX/usedCSF and
  // hasCall fields consumed by the AssemblyWriter.
  void allocate(std::vector<MachineFunc> &fns);
};

} // namespace backend
} // namespace sakura
