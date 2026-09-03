// ---------------------------------------------------------------------------
// TensorLower.h: affine-layer pass that expands whole-tensor (tensor dialect)
// operations into affine.for loop nests.
//
// The IRBuilder keeps whole-tensor ops (tensor.addi / tensor.matmul / ...) as
// single region-free operations on buffers + static shapes.  This pass - part
// of the affine (top) layer of the mid-end - rewrites every such op into
// scalar memref/arith code inside canonical affine.for loops, so that the
// affine layer is a uniform sea of countable loops that later affine-level
// passes (unrolling, vectorisation, ...) can act on.  It runs between IR
// generation and the affine -> scf lowering; it is registered as the
// expand-tensor-ops pass in the mid-end PassManager (see Passes.cpp).
// ---------------------------------------------------------------------------
#pragma once

#include "../ir/IR.h"

namespace sakura {
namespace ir {

// Expand every tensor dialect operation in `mod` into affine.for loop nests.
void expandTensorOps(Module *mod);

} // namespace ir
} // namespace sakura
