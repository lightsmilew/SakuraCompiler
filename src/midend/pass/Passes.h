// ---------------------------------------------------------------------------
// Passes.h: the standard mid-end passes and their factories.
//
// Each factory returns a Pass with the correct inLayer/outLayer annotations,
// so composing the pipeline in the driver (or in tests) can never feed a pass
// the wrong layer:
//
//     layer            pass                          layer out
//     ---------------------------------------------------------------
//     affine   (top)   expand-tensor-ops             affine   (tensor -> affine)
//     affine            + affine optimisation passes   affine
//     affine            lower-affine-to-scf          scf      (affine.for -> scf.while)
//     scf               + scf optimisation passes      scf
//     scf               canonicalize-control-flow    cf       (scf.while -> flat cf)
//     cf                + cf optimisation passes       cf
//     cf                verify-cf-only               cf       (back-end gate)
//
// The three conversion passes are implemented in this directory; per-layer
// optimisation passes (unrolling / vectorising / fusion ...) plug into the
// same pipeline between the conversions.
// ---------------------------------------------------------------------------
#pragma once

#include <memory>

#include "PassManager.h"

namespace sakura {
namespace ir {

// expand-tensor-ops: expand every whole-tensor (tensor dialect) op into an
// affine.for loop nest.  Keeps the IR in the affine layer.
std::unique_ptr<Pass> makeExpandTensorOpsPass();

// lower-affine-to-scf: rewrite every affine.for into the generic scf.while
// form (affine layer -> scf layer).
std::unique_ptr<Pass> makeLowerAffineToScfPass();

// canonicalize-control-flow: flatten every structured loop (scf.while) into
// the flat cf CFG the back-end consumes (scf layer -> cf layer).
std::unique_ptr<Pass> makeCanonicalizeToCfPass();

// verify-cf-only: assert the module contains no tensor / affine / scf ops, so
// the back-end only ever receives the flat cf form.  Throws CompileError when
// the invariant is violated.  Should be the last pass of the pipeline.
std::unique_ptr<Pass> makeVerifyCfOnlyPass();

// True when `mod` holds no tensor/affine/scf ops (every opcode belongs to the
// cf / func / arith / memref dialects).  This is the mid-end's contract with
// the back-end.
bool isCfOnly(const Module &mod);

} // namespace ir
} // namespace sakura
