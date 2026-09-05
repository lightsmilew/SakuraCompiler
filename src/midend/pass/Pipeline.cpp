// Pipeline.cpp: the standard mid-end pass sequence, wrapped in one call.
//
// This is where the driver-facing pipeline is defined.  All IR lowering and
// per-layer optimisation are registered here as passes on a PassManager;
// runMidEndPipeline() merely hands the module to the manager.  Keep the
// driver free of pass plumbing: new passes are added in this file.
//
// See PassManager.h for the affine / scf / cf layer model, Passes.h for the
// conversion passes and Opt.h for the optimisation passes.
#include "Pipeline.h"

#include <memory>
#include <ostream>

#include "Opt.h"
#include "PassManager.h"
#include "Passes.h"

namespace sakura {
namespace ir {

namespace {

// The O1/O2 optimisation pass sequence at one layer of the pipeline.  The
// passes are deliberately layer-agnostic (they recurse through regions), so
// the same "cleanup round" can be re-registered at the affine and scf
// layers.  Ordering inside a round matters: fold -> simplify -> forward ->
// eliminate, then a couple of repeats because each transformation exposes
// the next one.
void addCleanupRound(PassManager &pm, Layer l, bool aggressive) {
  pm.addPass(makeConstFoldPass(l));
  pm.addPass(makeAlgebraicPass(l));
  pm.addPass(makeConstFoldPass(l));
  pm.addPass(makeMemCsePass(l));
  pm.addPass(makeAlgebraicPass(l));
  pm.addPass(makeDeadCodePass(l));
  pm.addPass(makeConstFoldPass(l));
  if (aggressive) {
    pm.addPass(makeMemCsePass(l));
    pm.addPass(makeDeadCodePass(l));
  }
}

void addOptPipeline(PassManager &pm, OptLevel level) {
  bool aggr = level != OptLevel::O1;
  // ---- affine layer (top) -----------------------------------------------
  // IRBuilder output.  Expand every whole-tensor op into affine.for loop
  // nests first; then a cleanup round, the loop optimisations (LICM + full
  // unrolling of small constant-trip affine.for), then another cleanup.
  pm.addPass(makeExpandTensorOpsPass());
  if (level != OptLevel::O0) {
    addCleanupRound(pm, Layer::Affine, aggr);
    pm.addPass(makeLicmPass(Layer::Affine));
    pm.addPass(makeLoopUnrollPass(Layer::Affine));
    addCleanupRound(pm, Layer::Affine, aggr);
    pm.addPass(makeRemoveUnusedPass(Layer::Affine));
  }

  // affine -> scf: every affine.for becomes the generic scf.while.
  pm.addPass(makeLowerAffineToScfPass());

  // ---- scf layer (middle) -----------------------------------------------
  // Everything structured is now scf.while; run the scalar cleanup + LICM
  // once more on the generic loop form.
  if (level != OptLevel::O0) {
    addCleanupRound(pm, Layer::Scf, aggr);
    pm.addPass(makeLicmPass(Layer::Scf));
    addCleanupRound(pm, Layer::Scf, aggr);
  }

  // scf -> cf: flatten every structured loop to the flat CFG.
  pm.addPass(makeCanonicalizeToCfPass());

  // ---- cf layer (bottom) -------------------------------------------------
  if (level != OptLevel::O0) {
    addCleanupRound(pm, Layer::Cf, aggr);
    pm.addPass(makeCfgSimplifyPass());
    addCleanupRound(pm, Layer::Cf, aggr);
    // call-structure optimisation on the flat cf: inline small leaf helpers,
    // then turn self tail calls into loops.
    pm.addPass(makeInlineSmallPass());
    pm.addPass(makeTailRecElimPass());
    addCleanupRound(pm, Layer::Cf, aggr);
    // SSA register promotion: lift non-escaping scalar slots out of memory so
    // loop counters and temporaries live in registers.  The cleanup round
    // after it drops the now-dead slot allocas/loads/stores and folds away
    // any trivial phis.
    pm.addPass(makeMem2RegPass());
    addCleanupRound(pm, Layer::Cf, aggr);
    pm.addPass(makeRemoveUnusedPass(Layer::Cf));
  }

  // Back-end gate: assert no tensor / affine / scf op survives, so the
  // module handed to the back-end is guaranteed pure cf.
  pm.addPass(makeVerifyCfOnlyPass());
}

} // namespace

void runMidEndPipeline(Module &mod, const LayerView &view, OptLevel level,
                       std::ostream *stats) {
  PassManager pm;
  pm.setLayerPrinter(view);
  addOptPipeline(pm, level);
  pm.run(mod);
  if (stats) {
    for (const auto &r : pm.results())
      *stats << r.first << ": " << (r.second ? "changed" : "unchanged")
             << '\n';
  }
}

} // namespace ir
} // namespace sakura
