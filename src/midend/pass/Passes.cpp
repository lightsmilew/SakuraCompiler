// ---------------------------------------------------------------------------
// Passes.cpp: the standard mid-end passes.
//
// The layered pipeline lives here as plain Pass objects instead of Module
// methods so that the driver only has to assemble passes and run the manager:
//
//   affine layer -> lower-affine-to-scf -> scf layer -> canonicalize-control-flow
//                -> cf layer -> verify-cf-only -> back-end
//
// IRBuilder keeps canonical counting loops as affine.for region ops (top
// layer); whole-tensor ops are expanded into affine.for loops by
// expand-tensor-ops (midend/pass/TensorLower.cpp) before any lowering.
// lower-affine-to-scf rewrites every affine.for into the generic scf.while
// form so the middle layer only contains scf.while + cf (plus the scalar
// memref/arith/func ops).  scf-level optimisation passes run after that, and
// canonicalize-control-flow then splices every structured loop into the flat
// cf form that the back-end consumes.  verify-cf-only closes the pipeline by
// checking that no structured / whole-tensor op survives to the back-end.
// ---------------------------------------------------------------------------
#include "Passes.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "TensorLower.h"

namespace sakura {
namespace ir {

// ===========================================================================
// expand-tensor-ops (affine layer)
// ===========================================================================
namespace {

class ExpandTensorOpsPass final : public Pass {
public:
  const char *name() const override { return "expand-tensor-ops"; }
  Layer inLayer() const override { return Layer::Affine; }

  bool run(Module &mod) override {
    expandTensorOps(&mod); // tensor dialect -> affine.for loop nests
    return false;
  }
};

} // namespace

std::unique_ptr<Pass> makeExpandTensorOpsPass() {
  return std::make_unique<ExpandTensorOpsPass>();
}

// ===========================================================================
// lower-affine-to-scf (affine layer -> scf layer)
//
// affine.for semantics (canonical ascending counting loop over a memory slot):
//     store lb, slot
//     while (slot < ub) { <bodyRegion>; slot += step }
// The scf.while rewrite keeps that shape: its cond region re-tests slot < ub
// and its body region re-seeds the slot in the increment block.
// ===========================================================================
namespace {

int &affineNameSeq() {
  static int seq = 0;
  return seq;
}

// Append a fresh instruction to `bb`; returns it (owned by the block).
Instruction *irAppend(BasicBlock *bb, Op op, Type ty, int line) {
  auto i = std::make_unique<Instruction>(op, ty);
  i->line = line;
  Instruction *p = i.get();
  bb->instrs.push_back(std::move(i));
  return p;
}

std::string freshAffineBlockName() {
  return "afl" + std::to_string(affineNameSeq()++);
}

// Rewrite one affine.for (whose nested regions have already been lowered)
// into an scf.while.  `af` is the block terminator of its containing block.
void convertAffineFor(Module *mod, Instruction *af) {
  BasicBlock *exit = static_cast<BasicBlock *>(af->ops[0]);
  Value *slot = af->ops[1];
  Value *ub = af->ops[3];
  int64_t step = af->step;
  int line = af->line;

  // increment block: slot += step, then loop back to the condition region
  auto inc = std::make_unique<BasicBlock>(freshAffineBlockName());
  BasicBlock *incP = inc.get();
  {
    Instruction *v2 = irAppend(incP, Op::Load, Type::I32, line);
    v2->ops = {slot};
    Instruction *v3 = irAppend(incP, Op::Add, Type::I32, line);
    v3->ops = {v2, mod->constInt((int32_t)step)};
    Instruction *st = irAppend(incP, Op::Store, Type::Void, line);
    st->ops = {v3, slot};
    irAppend(incP, Op::ScfYield, Type::Void, line);
  }

  // the original body blocks now jump to the increment block instead of
  // yielding the iteration
  for (auto &bb : af->bodyRegion) {
    for (auto &iu : bb->instrs) {
      Instruction *i = iu.get();
      if (i->op != Op::AffineYield) continue;
      i->op = Op::Br;
      i->cond = Cond::Eq;
      i->ops = {incP};
    }
  }
  af->bodyRegion.push_back(std::move(inc));

  // condition block: re-test slot < ub every iteration
  auto cond = std::make_unique<BasicBlock>(freshAffineBlockName());
  BasicBlock *condP = cond.get();
  {
    Instruction *v = irAppend(condP, Op::Load, Type::I32, line);
    v->ops = {slot};
    Instruction *ok = irAppend(condP, Op::ICmp, Type::I32, line);
    ok->cond = Cond::Lt;
    ok->ops = {v, ub};
    Instruction *sc = irAppend(condP, Op::ScfCondition, Type::Void, line);
    sc->ops = {ok, af->bodyRegion.front().get(), exit};
  }
  af->condRegion.push_back(std::move(cond));

  // reshape the op in place: affine.for -> scf.while { cond } do { body }
  af->op = Op::ScfWhile;
  af->ops = {exit};
  af->cond = Cond::Eq;
  af->step = 0;
}

// Recursively rewrite one block list.  The prologue `store lb, slot` is
// inserted right before the rewritten scf.while so entering the loop re-seeds
// the induction slot.
void lowerAffineList(Module *mod,
                     std::vector<std::unique_ptr<BasicBlock>> &list) {
  for (auto &b : list) {
    for (size_t i = 0; i < b->instrs.size(); ++i) {
      Instruction *inst = b->instrs[i].get();
      if (inst->op == Op::ScfWhile) {
        lowerAffineList(mod, inst->condRegion);
        lowerAffineList(mod, inst->bodyRegion);
        continue;
      }
      if (inst->op != Op::AffineFor) continue;
      // nested affine.for inside the body is lowered first
      lowerAffineList(mod, inst->bodyRegion);

      // seed the induction slot with lb before the loop runs
      auto st = std::make_unique<Instruction>(Op::Store, Type::Void);
      st->line = inst->line;
      st->ops = {inst->ops[2], inst->ops[1]}; // {lb, slot}
      b->instrs.insert(b->instrs.begin() + i, std::move(st));
      inst = b->instrs[i + 1].get(); // re-fetch after the insertion
      convertAffineFor(mod, inst);
      break; // a region op terminates its block
    }
  }
}

class LowerAffineToScfPass final : public Pass {
public:
  const char *name() const override { return "lower-affine-to-scf"; }
  Layer inLayer() const override { return Layer::Affine; }
  Layer outLayer() const override { return Layer::Scf; }

  bool run(Module &mod) override {
    for (auto &f : mod.functions()) {
      if (f->isLib || f->blocks.empty()) continue;
      lowerAffineList(&mod, f->blocks);
    }
    return true;
  }
};

} // namespace

std::unique_ptr<Pass> makeLowerAffineToScfPass() {
  return std::make_unique<LowerAffineToScfPass>();
}

// ===========================================================================
// canonicalize-control-flow (scf layer -> cf layer)
//
// Runs right before the back-end, which only understands the flat cf form.
// Every scf.while is spliced into flat cf:
//
//   * scf.while op     -> cf.br <cond-region entry>
//   * scf.condition %c -> cf.cond_br %c, <body-region entry>, <exit>
//   * scf.yield        -> cf.br <cond-region entry>
// ===========================================================================
namespace {

// Move every block of `src` into `dst` in layout order.  A block whose last
// instruction is scf.while is split: the loop's regions are spliced in right
// after it, and the loop's own terminator semantics are lowered to cf (above).
//
// The three BasicBlock* arguments describe the region currently being
// flattened (null at function level) so the terminator rewrites know their
// targets; nested scf.while ops supply their own as they are unwrapped.
void flattenBlocks(std::vector<std::unique_ptr<BasicBlock>> &src,
                   std::vector<std::unique_ptr<BasicBlock>> &dst,
                   BasicBlock *condEntry, BasicBlock *bodyEntry,
                   BasicBlock *exit) {
  for (auto &b : src) {
    Instruction *last = b->instrs.empty() ? nullptr : b->instrs.back().get();
    if (last && last->op == Op::ScfWhile) {
      auto op = std::move(b->instrs.back());
      b->instrs.pop_back();
      Instruction *w = op.get();
      BasicBlock *ce = w->condRegion.front().get();
      BasicBlock *be = w->bodyRegion.front().get();
      BasicBlock *ex = static_cast<BasicBlock *>(w->ops[0]);
      auto br = std::make_unique<Instruction>(Op::Br, Type::Void);
      br->line = w->line;
      br->ops = {ce};
      b->instrs.push_back(std::move(br));
      dst.push_back(std::move(b));
      flattenBlocks(w->condRegion, dst, ce, be, ex);
      flattenBlocks(w->bodyRegion, dst, ce, be, ex);
      continue;
    }
    if (last && last->op == Op::ScfCondition) {
      // ops: {condition, body-region entry, exit block}
      if (auto *ci = dynamic_cast<ConstantInt *>(last->ops[0])) {
        // Fold a statically known condition to a plain jump so the back-end
        // never sees a constant-armed cf.cond_br.
        auto br = std::make_unique<Instruction>(Op::Br, Type::Void);
        br->line = last->line;
        br->ops = {ci->v ? last->ops[1] : last->ops[2]};
        b->instrs.back() = std::move(br);
      } else {
        auto cb = std::make_unique<Instruction>(Op::CondBr, Type::Void);
        cb->line = last->line;
        cb->ops = {last->ops[0], last->ops[1], last->ops[2]};
        b->instrs.back() = std::move(cb);
      }
    } else if (last && last->op == Op::ScfYield) {
      auto br = std::make_unique<Instruction>(Op::Br, Type::Void);
      br->line = last->line;
      br->ops = {condEntry};
      b->instrs.back() = std::move(br);
    }
    dst.push_back(std::move(b));
  }
}

class CanonicalizeToCfPass final : public Pass {
public:
  const char *name() const override { return "canonicalize-control-flow"; }
  Layer inLayer() const override { return Layer::Scf; }
  Layer outLayer() const override { return Layer::Cf; }

  bool run(Module &mod) override {
    for (auto &f : mod.functions()) {
      if (f->isLib || f->blocks.empty()) continue;
      std::vector<std::unique_ptr<BasicBlock>> flat;
      flattenBlocks(f->blocks, flat, nullptr, nullptr, nullptr);
      f->blocks = std::move(flat);
    }
    return true;
  }
};

} // namespace

std::unique_ptr<Pass> makeCanonicalizeToCfPass() {
  return std::make_unique<CanonicalizeToCfPass>();
}

// ===========================================================================
// verify-cf-only (cf layer) - the mid-end's contract with the back-end.
// ===========================================================================
namespace {

bool opSurvivesToCf(Op o) {
  switch (dialectOf(o)) {
  case Dialect::Tensor:
  case Dialect::Affine:
  case Dialect::Scf:
    return false; // whole-tensor ops / structured loops must be gone
  default:
    return true; // cf / func / arith / memref stay
  }
}

class VerifyCfOnlyPass final : public Pass {
public:
  const char *name() const override { return "verify-cf-only"; }
  Layer inLayer() const override { return Layer::Cf; }

  bool run(Module &mod) override {
    for (auto &f : mod.functions()) {
      if (f->isLib) continue;
      for (auto &b : f->blocks) {
        for (auto &i : b->instrs) {
          if (opSurvivesToCf(i->op))
            continue;
          throw CompileError(
              "internal: '" + opName(i->op) +
                  "' survived the mid-end pipeline; the module handed to the "
                  "back-end must only contain flat cf (verify-cf-only, func '" +
                  f->name + "')",
              i->line);
        }
      }
    }
    return false;
  }
};

} // namespace

std::unique_ptr<Pass> makeVerifyCfOnlyPass() {
  return std::make_unique<VerifyCfOnlyPass>();
}

bool isCfOnly(const Module &mod) {
  for (auto &f : mod.functions()) {
    if (f->isLib) continue;
    for (auto &b : f->blocks)
      for (auto &i : b->instrs)
        if (!opSurvivesToCf(i->op)) return false;
  }
  return true;
}

} // namespace ir
} // namespace sakura
