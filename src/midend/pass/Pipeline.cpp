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

#include <cstdlib>
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
  // The comparison / add-chain canonicalisations reshape flat address and
  // condition arithmetic; on the structured affine/scf layers they can fight
  // the loop form, so keep them to the canonical cf form.
  if (l == Layer::Cf) pm.addPass(makeNormalizeCmpPass(l));
  // Reassociate add/sub trees into a canonical linear combination (LLVM's
  // Reassociate): the inlined index / mask helpers expand into long chains of
  // `a - b + b - c` that only this rewrite collapses.  The algebraic pass that
  // follows clears up the `x + 0` shims it leaves behind.
  if (l == Layer::Cf && !getenv("SAKU_NO_REASSOC")) pm.addPass(makeReassocPass(l));
  pm.addPass(makeAlgebraicPass(l));
  pm.addPass(makeConstFoldPass(l));
  pm.addPass(makeMemCsePass(l));
  if (l == Layer::Cf) pm.addPass(makeAddChainPass(l));
  pm.addPass(makeRedundantStorePass(l));
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
    // Matmul ijk->ikj before LICM/unroll so the rewritten nests benefit from
    // both (LLVM LoopInterchange-style locality for array/matrix kernels).
    if (aggr) pm.addPass(makeMatmulIkjPass(Layer::Affine));
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
    // turn self tail calls into loops, then inline the general non-recursive
    // multi-block helpers (a recursion reduced to a flat loop above is now a
    // plain inlineable body).  A cfg-simplify follows so dead clone blocks
    // left by the splice are gone before mem2reg (which needs every non-entry
    // block reachable).
    pm.addPass(makeInlineSmallPass());
    pm.addPass(makeTailRecElimPass());
    pm.addPass(makeInlineGeneralPass());
    pm.addPass(makeCfgSimplifyPass());
    addCleanupRound(pm, Layer::Cf, aggr);
    // SSA register promotion: lift non-escaping scalar slots out of memory so
    // loop counters and temporaries live in registers.  The cleanup round
    // after it drops the now-dead slot allocas/loads/stores and folds away
    // any trivial phis.
    //
    // mem2reg needs a fully reachable CFG (dominance-based).  Its entry guard
    // bails out on any function that still carries an unreachable block
    // (constant folding in the cleanup round above can leave dead arms), so
    // those functions keep their memory form - correct, if less promoted.
    pm.addPass(makeMem2RegPass());
    addCleanupRound(pm, Layer::Cf, aggr);
    // Branchless if-conversion (LLVM's SimplifyCFG two-entry-PHI select).
    // The bit-at-a-time kernels are a dense chain of one-instruction `if`s;
    // after promotion each is a two-entry diamond whose phi the back-end would
    // lower to a branch plus register copies.  Turning it into `select` and
    // merging the diamond collapses the chain into a handful of ALU ops.  It
    // has to run after mem2reg (the phis are what it recognises) and before
    // ind-var / rotation (it removes the branches they would otherwise have to
    // work around); cfg-simplify afterwards merges the remains so the next
    // `if` becomes visible to the pass's own fixpoint loop.
    if (!getenv("SAKU_NO_IFCONV")) {
      pm.addPass(makeIfConvPass());
      pm.addPass(makeCfgSimplifyPass());
      addCleanupRound(pm, Layer::Cf, aggr);
    }
    // Flat-CFG LICM.  It has to run after the inliner (so a callee's array
    // parameters have been replaced by the caller's actual objects, which is
    // what makes the alias test decide `no-alias` for `C[i][j] = C[i][j] *
    // A[i][k] + B[k][j]`) and after mem2reg (so the loop-carried scalars are
    // already SSA and the addresses are plain gep chains).  It runs *before*
    // ind-var so the addresses are still `root + offset` and `addressOf` can
    // classify them.
    if (!getenv("SAKU_NO_CFLICM")) pm.addPass(makeCfLicmPass());
    addCleanupRound(pm, Layer::Cf, aggr);
    // Induction-variable strength reduction: rewrite `gep base, iv*K + C` into
    // a loop-carried pointer, so the inner loop bumps an address instead of
    // recomputing one (and the multiply leaves the critical path).  It needs
    // the promoted phi form, hence after mem2reg; the cleanup round that
    // follows removes the now-dead offset arithmetic.
    if (!getenv("SAKU_NO_INDVAR")) pm.addPass(makeIndVarPass(Layer::Cf));
    addCleanupRound(pm, Layer::Cf, aggr);
    // Loop rotation: move the loop test from the header to the latch so the
    // unconditional back-edge jump becomes the conditional one (one branch
    // per iteration instead of two).  Runs after ind-var so the pointer phis
    // are already in the header and rotate along with the others; cfg-simplify
    // afterwards folds a guard whose condition is statically known.
    if (!getenv("SAKU_NO_ROTATE")) pm.addPass(makeLoopRotatePass(Layer::Cf));
    // mem2reg strips the promoted loads/stores, which leaves many blocks
    // holding nothing but a jump; collapse those (and any identical-arm
    // cond_br) now that the phis are in place, then run whole-CFG dominance
    // CSE over the promoted phi/SSA values: folds the recomputations that
    // straight-line mem-cse has to give up on at joins / loop headers.
    pm.addPass(makeCfgSimplifyPass());
    pm.addPass(makeDomCsePass(Layer::Cf));
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
