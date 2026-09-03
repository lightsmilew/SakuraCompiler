// Pipeline.cpp: the standard mid-end pass sequence, wrapped in one call.
//
// This is where the driver-facing pipeline is defined.  All IR lowering (and,
// in the future, per-layer optimisation) is registered here as passes on a
// PassManager; runMidEndPipeline() merely hands the module to the manager.
// Keep the driver free of pass plumbing: new passes are added in this file.
//
// See PassManager.h for the affine / scf / cf layer model and Passes.h for
// the individual passes.
#include "Pipeline.h"

#include <memory>

#include "PassManager.h"
#include "Passes.h"

namespace sakura {
namespace ir {

void runMidEndPipeline(Module &mod, const LayerView &view) {
  PassManager pm;

  // Dump the module whenever the pipeline enters a layer (IRBuilder output
  // and each lowering boundary).
  pm.setLayerPrinter(view);

  // ---- affine layer (top) -------------------------------------------------
  // IRBuilder output.  Expand every whole-tensor op into affine.for loop
  // nests so the layer is a uniform sea of countable loops.  Affine-level
  // optimisation passes (unrolling, vectorising, fusion, ...) slot in here,
  // still at the affine layer.
  pm.addPass(makeExpandTensorOpsPass());

  // affine -> scf: every affine.for becomes the generic scf.while.
  pm.addPass(makeLowerAffineToScfPass());

  // ---- scf layer (middle) -------------------------------------------------
  // Everything structured is now scf.while; only loops that cannot be
  // structured (return / break / continue inside) remain as flat cf.
  // Scf-level optimisation passes slot in here.

  // scf -> cf: flatten every structured loop to the flat CFG.
  pm.addPass(makeCanonicalizeToCfPass());

  // ---- cf layer (bottom) --------------------------------------------------
  // The only form the back-end ever consumes.  Cf-level optimisation passes
  // slot in here, before the back-end gate below.

  // Back-end gate: assert no tensor / affine / scf op survives, so the
  // module handed to the back-end is guaranteed pure cf.
  pm.addPass(makeVerifyCfOnlyPass());

  pm.run(mod);
}

} // namespace ir
} // namespace sakura
