// ---------------------------------------------------------------------------
// OptCall.cpp: small leaf-function inlining (inline-small).
//
// Inlines *single-basic-block leaf* functions into their callers at the flat
// cf layer: no recursion, no calls inside the callee, no branches, no
// regions, no alloca - just a straight-line value computation that ends in a
// return.  For such helpers the clone is a plain textual splice: the callee's
// scalar/ptr arguments are replaced by the call operands, the callee's
// return-value definition replaces the call result, and the call + return
// instructions disappear.  Non-leaf / multi-block / recursive / large callers
// are left alone, so the transform is trivially semantics-preserving.
//
// Only meaningful at the cf layer (after canonicalize-control-flow, where
// every remaining user function has already been flattened to plain blocks).
// ---------------------------------------------------------------------------
#include "Opt.h"

#include <unordered_map>
#include <unordered_set>

#include "OptUtil.h"

namespace sakura {
namespace ir {

namespace {

constexpr int kMaxCalleeInstrs = 20; // body straight-line instructions
constexpr int kMaxInlinePerCaller = 6;  // calls inlined into one function
constexpr int kMaxCallerGrowth = 48;    // extra instructions per caller

class InlineSmallPass final : public Pass {
public:
  const char *name() const override { return "inline-small"; }
  Layer inLayer() const override { return Layer::Cf; }

  bool run(Module &mod) override {
    bool any = false;
    bool grow = true;
    while (grow) { // inlining may expose further inlining opportunities
      grow = false;
      for (auto &f : mod.functions()) {
        if (f->isLib) continue;
        if (process(mod, *f)) {
          any = true;
          grow = true;
        }
      }
    }
    return any;
  }

private:
  // Is `callee` a single-block leaf small enough to inline?
  bool inlineable(Module &mod, Function *callee, Function *caller,
                  Instruction *call) const {
    if (callee->isLib || callee == caller) return false;
    if (callee->blocks.size() != 1) return false;
    if (call->ops.size() - 1 != callee->args.size()) return false;
    BasicBlock *B = callee->blocks[0].get();
    if (B->instrs.empty()) return false;
    if (B->instrs.back()->op != Op::Ret) return false; // must end in return
    int n = 0;
    for (auto &iu : B->instrs) {
      Instruction *i = iu.get();
      if (i->op == Op::Ret) continue;
      if (++n > kMaxCalleeInstrs) return false;
      switch (i->op) {
      case Op::Call:
      case Op::Alloca:
      case Op::ScfWhile:
      case Op::AffineFor:
      case Op::CondBr:
      case Op::Br:
        return false; // not a plain straight-line leaf
      default:
        break;
      }
    }
    return true;
  }

  bool process(Module &mod, Function &caller) {
    bool any = false;
    int inlineCount = 0;
    int budget = 0; // extra instructions this caller already absorbed
    // Restart the block/instruction walk after each inline: indexes shift.
    for (;;) {
      bool found = false;
      for (auto &bb : caller.blocks) {
        size_t ci = 0;
        while (ci < bb->instrs.size()) {
          Instruction *inst = bb->instrs[ci].get();
          if (inst->op != Op::Call) { ++ci; continue; }
          auto *callee = dynamic_cast<Function *>(inst->ops[0]);
          if (!callee || !inlineable(mod, callee, &caller, inst)) {
            ++ci;
            continue;
          }
          int cloneCost = (int)callee->blocks[0]->instrs.size();
          if (inlineCount >= kMaxInlinePerCaller ||
              budget + cloneCost > kMaxCallerGrowth) {
            ++ci;
            continue;
          }
          doInline(mod, *bb, ci, callee);
          inlineCount++;
          budget += cloneCost;
          any = true;
          found = true;
          break; // block list changed; rescan from the top of the function
        }
        if (found) break;
      }
      if (!found) break;
    }
    return any;
  }

  // Splice one clone of `callee` over the call at bb.instrs[ci].
  void doInline(Module &mod, BasicBlock &bb, size_t ci, Function *callee) {
    BasicBlock *B = callee->blocks[0].get();
    Instruction *call = bb.instrs[ci].get();

    // value the callee returns (clone of its ret operand, resolved below)
    Instruction *retI = B->instrs.back().get();
    bool hasRet = callee->ret != Type::Void && !retI->ops.empty();

    // clone every non-return body instruction, remapping arguments and
    // in-body defs.
    std::unordered_map<Instruction *, Instruction *> remap;
    std::vector<std::unique_ptr<Instruction>> clones;
    for (auto &iu : B->instrs) {
      Instruction *src = iu.get();
      if (src->op == Op::Ret) continue;
      auto c = std::make_unique<Instruction>(src->op, src->ty);
      c->cond = src->cond;
      c->elem = src->elem;
      c->n = src->n;
      c->step = src->step;
      c->shape = src->shape;
      c->line = src->line;
      Instruction *cp = c.get();
      remap[src] = cp;
      clones.push_back(std::move(c));
      // argument operands become the caller's argument values
      for (size_t oi = 0; oi < src->ops.size(); ++oi) {
        Value *o = src->ops[oi];
        if (auto *arg = dynamic_cast<Argument *>(o)) {
          cp->ops.push_back(call->ops[1 + arg->index]);
        } else if (auto *d = dynamic_cast<Instruction *>(o)) {
          auto it = remap.find(d);
          cp->ops.push_back(it != remap.end() ? (Value *)it->second : o);
        } else {
          cp->ops.push_back(o);
        }
      }
    }
    // in-body forward references are impossible (straight line, def before
    // use), so the single pass above already resolved every operand.

    // the returned value: an argument, a cloned def, or nothing
    Value *retVal = nullptr;
    if (hasRet) {
      Value *rv = retI->ops[0];
      if (auto *arg = dynamic_cast<Argument *>(rv)) {
        retVal = call->ops[1 + arg->index];
      } else if (auto *d = dynamic_cast<Instruction *>(rv)) {
        auto it = remap.find(d);
        retVal = it != remap.end() ? (Value *)it->second : rv;
      } else {
        retVal = rv; // constant / global
      }
    }

    // callers reading the call result now read the cloned return value
    if (call->hasResult() && retVal) replaceAllUses(mod, call, retVal);

    // erase the call and splice the clone in its place
    bb.instrs.erase(bb.instrs.begin() + (long)ci);
    size_t at = ci;
    for (auto &c : clones) {
      bb.instrs.insert(bb.instrs.begin() + (long)at++, std::move(c));
    }
  }
};

} // namespace

std::unique_ptr<Pass> makeInlineSmallPass() {
  return std::make_unique<InlineSmallPass>();
}

} // namespace ir
} // namespace sakura
