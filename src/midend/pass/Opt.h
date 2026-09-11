// ---------------------------------------------------------------------------
// Opt.h: factories for the mid-end optimisation passes.
//
// These are the "general" optimisations shared by any compiler (ported from
// the reference SysY project into SakuraCompiler's IR + PassManager model).
// Each factory takes the Layer the pass must run at, so one logical pass can
// be registered on several layers of the pipeline (e.g. dead-code elimination
// after the tensor expansion, after affine->scf lowering, and on the final
// flat cf).
//
//   const-fold      - fold constant arith/cmp/cast, constant branches, gep(0)
//   algebraic       - algebraic simplification (x+0, x*1, x-x, not-not, ...)
//   normalize-cmp   - canonicalise comparisons (constants moved to the RHS,
//                     Gt/Ge mirrored to Lt/Le forms) so that the same
//                     condition written in different operand orders CSEs
//   add-chain       - reassociate single-use (x+c1)+c2 ... chains into one
//                     add of the accumulated constant (address arithmetic)
//   redundant-store - drop block-local dead stores (a later store to the same
//                     exact location overwrites an earlier one) and
//                     "store x,p where x == load p" store-backs
//   dead-code       - erase unused pure/load/gep, write-only allocas + stores
//   remove-unused   - delete uncalled user functions and unused globals
//   mem-cse         - block/EBB local: store->load forward, dead-store,
//                     redundant-load and pure-arithmetic CSE
//   dom-cse         - whole-CFG CSE of pure expressions along the dominator
//                     tree (folds what mem-cse's join-reset cannot see)
//   licm            - hoist loop-invariant pure ops / loads out of loops
//   loop-unroll     - fully unroll affine.for with a small constant trip count
//   matmul-ikj      - rewrite ijk matmul nests to cache-friendly ikj order
//   cfg-simplify    - flat-cf cleanup: const cond_br, empty/merge/unreachable
//   tail-rec-elim   - self tail calls -> loops (simple form)
//   inline-general  - inline non-recursive functions with more than one block
//                     (multi-block clone + return merge); mirrors the
//                     reference FunctionInliningPass
//   mem2reg         - promote non-escaping scalar alloca slots to SSA phis
//
// See also Passes.h for the conversion passes and Pipeline.h for where these
// slot in.
// ---------------------------------------------------------------------------
#pragma once

#include <memory>

#include "PassManager.h"

namespace sakura {
namespace ir {

std::unique_ptr<Pass> makeConstFoldPass(Layer l);
std::unique_ptr<Pass> makeAlgebraicPass(Layer l);
std::unique_ptr<Pass> makeDeadCodePass(Layer l);
std::unique_ptr<Pass> makeRemoveUnusedPass(Layer l);
std::unique_ptr<Pass> makeMemCsePass(Layer l);
std::unique_ptr<Pass> makeDomCsePass(Layer l);
std::unique_ptr<Pass> makeIndVarPass(Layer l);
std::unique_ptr<Pass> makeCfLicmPass();
std::unique_ptr<Pass> makeIfConvPass();
std::unique_ptr<Pass> makeLoopRotatePass(Layer l);
std::unique_ptr<Pass> makeLicmPass(Layer l);
std::unique_ptr<Pass> makeLoopUnrollPass(Layer l);
std::unique_ptr<Pass> makeMatmulIkjPass(Layer l);
std::unique_ptr<Pass> makeCfgSimplifyPass();
std::unique_ptr<Pass> makeTailRecElimPass();
std::unique_ptr<Pass> makeInlineSmallPass();
std::unique_ptr<Pass> makeMem2RegPass();

std::unique_ptr<Pass> makeNormalizeCmpPass(Layer l);
std::unique_ptr<Pass> makeReassocPass(Layer l);
std::unique_ptr<Pass> makeAddChainPass(Layer l);
std::unique_ptr<Pass> makeRedundantStorePass(Layer l);
std::unique_ptr<Pass> makeInlineGeneralPass();

} // namespace ir
} // namespace sakura
