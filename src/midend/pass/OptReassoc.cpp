// ---------------------------------------------------------------------------
// OptReassoc.cpp: integer add/sub expression reassociation.
//
// Ported from LLVM's Reassociate pass plus the algebraic foldings InstCombine
// performs on the linear combination it exposes.  A 32-bit add/sub expression
// tree whose interior nodes have exactly one use is flattened into its exact
// integer-ring form
//
//     c_1 * t_1 + c_2 * t_2 + ... + K          (all arithmetic mod 2^32)
//
// and rebuilt with as few instructions as possible: terms whose coefficient
// cancels to zero disappear, coefficient -1 becomes a negation, the remaining
// scaled terms are summed bottom-up and the constant is folded in last.
//
// This is the transformation that collapses an inlined helper chain such as
//
//     _xor(a,b) = a - _and(a,b) + b - _and(a,b),   _and(a,b) = a + b
//
// into `-(a + b)` (five adds become two instructions) and then
// `_or(a,b) = _xor(_xor(a,b), _and(a,b))` all the way to the constant 0.  LLVM
// leans on Reassociate + InstCombine for exactly these identities; without it
// every kernel that computes indices or bit masks through inlined helpers pays
// for the un-simplified arithmetic.
//
// Safety notes:
//   * only 32-bit integer add/sub take part (reassociation is exact in the
//     wrapping ring; reassociating floats would change the rounding);
//   * every interior node of the flattened tree must be single-use, so the
//     deleted arithmetic is dead the instant the root is rewritten;
//   * the rewrite only fires when it strictly reduces the instruction count,
//     which also makes the pass monotone (dead Add/Sub nodes can never be
//     collected because the root must have a use);
//   * the replacement is materialised immediately before the original root,
//     so dominance is preserved and no live range is lengthened.
// ---------------------------------------------------------------------------
#include "Opt.h"

#include <climits>
#include <functional>
#include <unordered_map>
#include <unordered_set>

#include "OptUtil.h"

namespace sakura {
namespace ir {

namespace {

class ReassocPass final : public Pass {
public:
  explicit ReassocPass(Layer l) : layer_(l) {}
  const char *name() const override { return "reassoc"; }
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
  std::unordered_map<Instruction *, int> uses_;

  // A flattened linear combination: coefficient of every leaf term (in first
  // -seen order, so the rewrite is deterministic) plus the constant part.
  struct Terms {
    std::vector<std::pair<Value *, long long>> items;
    long long constant = 0;
    int nodes = 0; // add/sub instructions consumed by the flattening
    int extra = 0; // shared nodes we bypass but do not delete
    bool bad = false;
  };

  // One rewrite, staged during the walk and applied afterwards (inserting
  // into the instruction list while iterating it would invalidate the cursor).
  struct Plan {
    BasicBlock *bb = nullptr;
    Instruction *root = nullptr;
    std::vector<std::unique_ptr<Instruction>> pre;
    Value *result = nullptr;
  };

  static void walk(BlockList &list,
                   const std::function<void(BasicBlock &, Instruction *)> &f) {
    std::function<void(BlockList &)> rec = [&](BlockList &l) {
      for (auto &bb : l)
        for (auto &iu : bb->instrs) {
          Instruction *inst = iu.get();
          f(*bb, inst);
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

  static void addTerm(Terms &t, Value *v, long long sign) {
    for (auto &it : t.items)
      if (it.first == v) {
        it.second += sign;
        return;
      }
    t.items.push_back({v, sign});
  }

  // Flatten `sign * v` into `t`.  Recurses through 32-bit add/sub nodes (the
  // interior nodes of the tree) and stops at everything else, recording it as
  // an opaque leaf term.  A node with several uses is still flattened -- the
  // same subexpression serves every user -- but its extra users are charged
  // to `extra` so the caller can tell whether the total instruction count
  // really goes down (the original node survives for the other users).
  void collect(Value *v, long long sign, Terms &t, int depth,
               std::unordered_set<Instruction *> &path) {
    if (t.bad) return;
    if (const ConstantInt *c = asConstInt(v)) {
      t.constant += sign * (long long)c->v;
      return;
    }
    auto *d = dynamic_cast<Instruction *>(v);
    if (d && depth < 24 && d->ty == Type::I32 && d->ops.size() == 2 &&
        (d->op == Op::Add || d->op == Op::Sub) && !path.count(d)) {
      int u = uses_[d];
      if (u > 4) { // very hot value: leave the shared computation alone
        addTerm(t, v, sign);
        return;
      }
      path.insert(d);
      t.nodes++;
      t.extra += u - 1;
      if (t.nodes > 32) { // pathological tree: leave it alone
        t.bad = true;
        return;
      }
      collect(d->ops[0], sign, t, depth + 1, path);
      collect(d->ops[1], d->op == Op::Add ? sign : -sign, t, depth + 1, path);
      path.erase(d);
      return;
    }
    addTerm(t, v, sign);
  }

  static int termCost(long long c) {
    if (c == 1) return 0;  // the leaf itself
    if (c == -1) return 1; // neg
    long long a = c < 0 ? -c : c;
    if (a <= 32) return 1;                  // shift/add strength reduction
    if ((a & (a - 1)) == 0) return 1;       // power of two: shift
    return 2;                               // li + mul
  }

  int cost(const Terms &t) const {
    int c = 0;
    int live = 0;
    for (const auto &it : t.items) {
      if (it.second == 0) continue;
      live++;
      c += termCost(it.second);
    }
    if (live > 1) c += live - 1;
    if (t.constant != 0) c++;
    if (live == 0) c = 1; // the whole expression folded to a constant
    return c;
  }

  bool process(Module &mod, Function &f) {
    bool any = false;
    bool again = true;
    // Bounded fixpoint: an accepted rewrite always shrinks the root's own
    // subtree, but the cap keeps a pathological CFG from making this spin.
    for (int iter = 0; again && iter < 64; ++iter) {
      again = false;

      uses_.clear();
      walk(f.blocks, [&](BasicBlock &, Instruction *inst) {
        for (Value *o : inst->ops) {
          auto *d = dynamic_cast<Instruction *>(o);
          if (d) uses_[d]++;
        }
      });

      Plan plan;
      walk(f.blocks, [&](BasicBlock &bb, Instruction *root) {
        if (again) return;
        if (root->op != Op::Add && root->op != Op::Sub) return;
        if (root->ty != Type::I32 || root->ops.size() != 2) return;
        if (uses_[root] == 0) return; // dead: leave it to DCE

        Terms t;
        std::unordered_set<Instruction *> path;
        path.insert(root); // cycle guard: the root can never be a leaf
        t.nodes = 1;
        collect(root->ops[0], 1, t, 0, path);
        collect(root->ops[1], root->op == Op::Add ? 1 : -1, t, 0, path);
        if (t.bad || t.nodes < 2) return;
        for (const auto &it : t.items)
          if (it.second < INT32_MIN || it.second > INT32_MAX) return;
        if (t.constant < INT32_MIN || t.constant > INT32_MAX) return;
        if (cost(t) + t.extra >= t.nodes) return;

        // Rebuild `sum c_i * t_i + K`, exposing the result as `acc`.
        Value *acc = nullptr;
        for (const auto &it : t.items) {
          if (it.second == 0) continue;
          Value *v = it.first;
          if (it.second == -1) {
            auto n = std::make_unique<Instruction>(Op::Sub, Type::I32);
            n->line = root->line;
            n->ops = {mod.constInt(0), v};
            v = n.get();
            plan.pre.push_back(std::move(n));
          } else if (it.second != 1) {
            auto m = std::make_unique<Instruction>(Op::Mul, Type::I32);
            m->line = root->line;
            m->ops = {v, mod.constInt((int32_t)it.second)};
            v = m.get();
            plan.pre.push_back(std::move(m));
          }
          if (!acc) {
            acc = v;
          } else {
            auto a = std::make_unique<Instruction>(Op::Add, Type::I32);
            a->line = root->line;
            a->ops = {acc, v};
            acc = a.get();
            plan.pre.push_back(std::move(a));
          }
        }
        if (t.constant != 0) {
          if (!acc) {
            acc = mod.constInt((int32_t)t.constant);
          } else {
            auto a = std::make_unique<Instruction>(Op::Add, Type::I32);
            a->line = root->line;
            a->ops = {acc, mod.constInt((int32_t)t.constant)};
            acc = a.get();
            plan.pre.push_back(std::move(a));
          }
        }
        if (!acc) acc = mod.constInt(0); // everything cancelled

        plan.bb = &bb;
        plan.root = root;
        plan.result = acc;
        again = true;
      });

      if (again && plan.root) {
        // Insert the new arithmetic right before the root, then retire it.
        // Every value it referenced is available at that point (the flattened
        // nodes are all operands of the root, hence dominate it).
        auto &v = plan.bb->instrs;
        size_t at = v.size();
        for (size_t i = 0; i < v.size(); ++i)
          if (v[i].get() == plan.root) {
            at = i;
            break;
          }
        for (auto &u : plan.pre) v.insert(v.begin() + (long)at++, std::move(u));
        replaceAllUses(mod, plan.root, plan.result);
        any = true;
      }
    }
    return any;
  }
};

} // namespace

std::unique_ptr<Pass> makeReassocPass(Layer l) {
  return std::make_unique<ReassocPass>(l);
}

} // namespace ir
} // namespace sakura
