// ---------------------------------------------------------------------------
// OptChain.cpp: constant addition-chain reduction (add-chain).
//
// Ported from the reference project's AddChainReductionPass.  Address
// arithmetic tends to be built up one constant at a time -- a flattened array
// index such as a[(i + 1) + 2] reaches the IR as
//
//     %t1 = arith.addi %i, 1
//     %t2 = arith.addi %t1, 2     (%t1 is used exactly once: by %t2)
//     %off = arith.muli %t2, 4
//
// This pass recognises the single-use `(%x + c1) + c2` chains and reassociates
// them into one add of the accumulated constant (%x + (c1 + c2)), deleting the
// intermediate adds.  The reference uses the same shape to shorten GEP / index
// computation so the follow-up const-folding, LICM and CSE passes can see the
// flattened address.  A chain whose constants sum to zero collapses straight
// to the base value (%x + 0 == %x).  When the chain is long the shorter add
// count also removes a register/instruction dependency per element.
//
// Only *integer* additions take part (float reassociation would change the
// rounding).  Every intermediate add must be used exactly once, and only by
// the next add of the chain, so the deletion is trivially safe.
// ---------------------------------------------------------------------------
#include "Opt.h"

#include <functional>
#include <unordered_map>
#include <unordered_set>

#include "OptUtil.h"

namespace sakura {
namespace ir {

namespace {

class AddChainPass final : public Pass {
public:
  explicit AddChainPass(Layer l) : layer_(l) {}
  const char *name() const override { return "add-chain"; }
  Layer inLayer() const override { return layer_; }

  bool run(Module &mod) override {
    bool any = false;
    for (auto &f : mod.functions()) {
      if (f->isLib) continue;
      if (process(mod, *f)) any = true;
    }
    return any;
  }

private:
  Layer layer_;

  static void walkBlocks(BlockList &list,
                         const std::function<void(Instruction *)> &f) {
    std::function<void(BlockList &)> rec = [&](BlockList &l) {
      for (auto &bb : l)
        for (auto &iu : bb->instrs) {
          Instruction *inst = iu.get();
          f(inst);
          if (inst->op == Op::ScfWhile) {
            rec(inst->condRegion);
            rec(inst->bodyRegion);
          } else if (inst->op == Op::AffineFor) {
            rec(inst->bodyRegion);
          }
        }
    };
    rec(list);
  }

  bool process(Module &mod, Function &f) {
    bool any = false;
    bool again = true;
    while (again) { // a collapse can expose a longer chain above it
      again = false;

      // Function-wide single-use counts (SSA defs are function-local) plus,
      // for defs with exactly one use, that unique user.
      std::unordered_map<Instruction *, int> uses;
      std::unordered_map<Instruction *, Instruction *> onlyUser;
      walkBlocks(f.blocks, [&](Instruction *inst) {
        for (Value *o : inst->ops) {
          auto *d = dynamic_cast<Instruction *>(o);
          if (!d) continue;
          uses[d]++;
          if (uses[d] == 1) onlyUser[d] = inst;
          else onlyUser.erase(d);
        }
      });

      // Erasures are staged: the walk below rewrites chain heads in place and
      // records the (now single-use) interior adds for a phase-2 sweep, so an
      // erase never disturbs an iterator/pointer the scan still holds.
      std::unordered_set<Instruction *> eraseSet;
      bool localCh = false;

      auto classify = [](Instruction *add, Value *&nonConst,
                         const ConstantInt *&c) {
        // matches add x c / add c x only
        if (add->op != Op::Add || add->ops.size() != 2) return false;
        c = asConstInt(add->ops[0]);
        if (c) {
          nonConst = add->ops[1];
          return true;
        }
        c = asConstInt(add->ops[1]);
        if (c) {
          nonConst = add->ops[0];
          return true;
        }
        return false;
      };

      // Is `add` an *interior* node of a constant chain, i.e. its only user
      // is another constant add?  Such nodes are handled when their chain head
      // is reached (the head's user is something else).
      auto isInterior = [&](Instruction *add) {
        if (uses[add] != 1) return false;
        Instruction *user = onlyUser[add];
        if (!user || user->op != Op::Add || user->ops.size() != 2)
          return false;
        return asConstInt(user->ops[0]) || asConstInt(user->ops[1]);
      };

      // Textual walk.  For every chain head, grow the chain *backwards*
      // through single-use interior adds, then rewrite the head to
      // base + accumulatedConstant and retire the interior nodes.
      walkBlocks(f.blocks, [&](Instruction *inst) {
        if (localCh) return; // fixpoint re-run: stop at first rewrite
        if (inst->op != Op::Add || inst->ops.size() != 2) return;
        if (isInterior(inst)) return; // consumed by its own chain head
        Value *nc = nullptr;
        const ConstantInt *c = nullptr;
        if (!classify(inst, nc, c)) return;

        std::vector<Instruction *> chain;
        int64_t total = 0;
        Value *base = nullptr;

        Instruction *cur = inst;
        for (int depth = 0; depth < 16; ++depth) {
          if (!classify(cur, nc, c)) break;
          total += c->v;
          chain.push_back(cur);
          Instruction *def = dynamic_cast<Instruction *>(nc);
          if (!def || def->op != Op::Add || def->ops.size() != 2) {
            base = nc;
            break;
          }
          const ConstantInt *c2 = nullptr;
          Value *nc2 = nullptr;
          if (uses[def] != 1 || !classify(def, nc2, c2)) {
            base = nc;
            break;
          }
          if (std::find(chain.begin(), chain.end(), def) != chain.end()) {
            base = nc; // cycle guard (broken IR)
            break;
          }
          cur = def; // keep walking into the single-use add below
        }
        if (chain.size() < 2 || base == nullptr) return;

        Instruction *head = chain.front();
        // Reassociate the whole chain into one add at the head's position.
        if (total != 0) {
          head->ops = {base, mod.constInt((int32_t)total)};
        } else {
          // x + 0  ->  x
          replaceAllUses(mod, head, base);
          eraseSet.insert(head);
        }
        for (size_t i = 1; i < chain.size(); ++i)
          eraseSet.insert(chain[i]);
        localCh = true;
      });

      if (localCh) {
        // phase 2: erase retired adds (never the head when it was rewritten)
        std::function<void(BlockList &)> eraseRec = [&](BlockList &list) {
          for (auto &bb : list)
            for (auto it = bb->instrs.begin(); it != bb->instrs.end();) {
              if (eraseSet.count(it->get())) {
                it = bb->instrs.erase(it);
              } else {
                if ((*it)->op == Op::ScfWhile) {
                  eraseRec((*it)->condRegion);
                  eraseRec((*it)->bodyRegion);
                } else if ((*it)->op == Op::AffineFor) {
                  eraseRec((*it)->bodyRegion);
                }
                ++it;
              }
            }
        };
        eraseRec(f.blocks);
        any = true;
        again = true;
      }
    }
    return any;
  }
};

} // namespace

std::unique_ptr<Pass> makeAddChainPass(Layer l) {
  return std::make_unique<AddChainPass>(l);
}

} // namespace ir
} // namespace sakura
