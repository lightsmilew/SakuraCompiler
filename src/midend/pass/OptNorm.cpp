// ---------------------------------------------------------------------------
// OptNorm.cpp: comparison / commutative-operand canonicalisation
// (normalize-cmp).
//
// Ported (in spirit) from the reference project's NormalizationPass.  Two
// pieces of the canonical form live here:
//
//  * every Gt/Ge comparison is mirrored to its Lt/Le form by swapping the
//    operands (a > b  <=>  b < a), and an Eq/Ne comparison whose constant
//    operand happens to sit on the left is swapped so constants end up on the
//    right.  The whole pipeline then sees one spelling of every condition no
//    matter which direction the source wrote it, so mem-cse / dom-cse can
//    fold "i < n" together with "n > i".
//
//  * a comparison against a constant whose other side is an add/sub of a
//    constant is rewritten to compare the inner variable against an adjusted
//    bound:  (x + c1) < c2  <=>  x < (c2 - c1),  (x - c1) > c2  <=>  x > (c2
//    + c1), with the constant-on-the-left case mirrored by flipping the
//    predicate first.  This is what turns a loop bound written as `i < n - 1`
//    into `i < c` once n folds to a constant, and it reuses the constants
//    that add-chain reduction folds out of address expressions.
//
// Only int comparisons participate in the add/sub rewrites (float arithmetic
// cannot be re-associated).  Commutative arithmetic (add/mul/fadd/fmul) is
// canonicalised so a constant operand sits on the right.
// ---------------------------------------------------------------------------
#include "Opt.h"

#include "OptUtil.h"

namespace sakura {
namespace ir {

namespace {

class NormalizeCmpPass final : public Pass {
public:
  explicit NormalizeCmpPass(Layer l) : layer_(l) {}
  const char *name() const override { return "normalize-cmp"; }
  Layer inLayer() const override { return layer_; }

  bool run(Module &mod) override {
    bool any = false;
    for (auto &f : mod.functions()) {
      if (f->isLib) continue;
      if (rewrite(mod, f->blocks)) any = true;
    }
    return any;
  }

private:
  Layer layer_;

  bool rewrite(Module &mod, BlockList &list) {
    bool any = false;
    for (auto &bb : list)
      for (auto &iu : bb->instrs) {
        Instruction *inst = iu.get();
        if (inst->op == Op::ScfWhile) {
          any |= rewrite(mod, inst->condRegion);
          any |= rewrite(mod, inst->bodyRegion);
          continue;
        }
        if (inst->op == Op::AffineFor) {
          any |= rewrite(mod, inst->bodyRegion);
          continue;
        }
        if (normalizeOne(mod, inst)) any = true;
      }
    return any;
  }

  // (x +- c) classification: an int add/sub with exactly one constant
  // operand.  Sub only when the constant is the subtrahend (x - c).
  struct AddSub {
    bool ok = false;
    bool isSub = false;
    Value *var = nullptr;
    int32_t c = 0;
  };
  static AddSub matchAddSub(Value *v) {
    AddSub r;
    auto *inst = dynamic_cast<Instruction *>(v);
    if (!inst || inst->ops.size() < 2) return r;
    if (inst->op != Op::Add && inst->op != Op::Sub) return r;
    const ConstantInt *cl = asConstInt(inst->ops[0]);
    const ConstantInt *cr = asConstInt(inst->ops[1]);
    if (inst->op == Op::Add && (cl || cr) && !(cl && cr)) {
      r.ok = true;
      r.var = cl ? inst->ops[1] : inst->ops[0];
      r.c = cl ? cl->v : cr->v;
      return r;
    }
    if (inst->op == Op::Sub && !cl && cr) {
      r.ok = true;
      r.isSub = true;
      r.var = inst->ops[0];
      r.c = cr->v;
      return r;
    }
    return r;
  }

  static Cond swapDir(Cond c) { // predicate for the swapped operand order
    switch (c) {
    case Cond::Lt: return Cond::Gt;
    case Cond::Le: return Cond::Ge;
    case Cond::Gt: return Cond::Lt;
    case Cond::Ge: return Cond::Le;
    default: return c; // Eq / Ne are symmetric
    }
  }

  static bool isConst(Value *v) {
    return asConstInt(v) || asConstFloat(v);
  }

  bool normalizeOne(Module &mod, Instruction *inst) {
    Op op = inst->op;
    // Commutative arithmetic: keep a lone constant on the right.
    if ((op == Op::Add || op == Op::Mul || op == Op::FAdd || op == Op::FMul) &&
        inst->ops.size() == 2) {
      if (isConst(inst->ops[0]) && !isConst(inst->ops[1])) {
        std::swap(inst->ops[0], inst->ops[1]);
        return true;
      }
      return false;
    }
    if (op != Op::ICmp && op != Op::FCmp) return false;
    if (inst->ops.size() != 2) return false;

    bool changed = false;

    if (op == Op::ICmp) {
      // 1) pull a `var +- c` expression onto the left of a constant:
      //    (var +- c1) pred c2  =>  var pred' (adjusted), with pred' flipped
      //    when the expression originally sat on the right.
      AddSub el = matchAddSub(inst->ops[0]);
      AddSub er = matchAddSub(inst->ops[1]);
      const ConstantInt *cl = asConstInt(inst->ops[0]);
      const ConstantInt *cr = asConstInt(inst->ops[1]);
      if (cl && er.ok) {
        // c2 pred (var +- c1): swap the sides first, then fold the bound.
        int32_t adj = er.isSub ? cl->v + er.c : cl->v - er.c;
        inst->ops[0] = er.var;
        inst->ops[1] = mod.constInt(adj);
        inst->cond = swapDir(inst->cond);
        changed = true;
      } else if (cr && el.ok) {
        int32_t adj = el.isSub ? cr->v + el.c : cr->v - el.c;
        inst->ops[0] = el.var;
        inst->ops[1] = mod.constInt(adj);
        changed = true;
      }
    }

    // 2) canonical direction: Gt/Ge become Lt/Le by swapping operands;
    //    Eq/Ne put any constant on the right (symmetric predicates).
    bool swapLeft =
        inst->cond == Cond::Gt || inst->cond == Cond::Ge ||
        ((inst->cond == Cond::Eq || inst->cond == Cond::Ne) &&
         isConst(inst->ops[0]) && !isConst(inst->ops[1]));
    if (swapLeft) {
      std::swap(inst->ops[0], inst->ops[1]);
      inst->cond = swapDir(inst->cond);
      changed = true;
    }
    return changed;
  }
};

} // namespace

std::unique_ptr<Pass> makeNormalizeCmpPass(Layer l) {
  return std::make_unique<NormalizeCmpPass>(l);
}

} // namespace ir
} // namespace sakura
