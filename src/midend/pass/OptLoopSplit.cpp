// Index-set splitting: partition an affine loop into consecutive intervals
// where induction-dependent predicates are constant. Unlike statement loop
// distribution, the order of every executed iteration/effect stays unchanged.
// See docs/loop-split.md for the proof, limits and primary references.
#include "Opt.h"
#include "OptUtil.h"
#include <algorithm>
#include <cstdlib>
#include <functional>
#include <limits>
#include <set>

namespace sakura { namespace ir { namespace {
struct Form { int64_t k = 0, c = 0; };
struct Predicate {
  Instruction *branch;
  Form lhs, rhs;
  Cond cond;
  bool inverted = false;
};
bool add(int64_t a, int64_t b, int64_t &r) {
  if ((b > 0 && a > INT64_MAX - b) || (b < 0 && a < INT64_MIN - b)) return false;
  r = a + b; return true;
}
bool mul(int64_t a, int64_t b, int64_t &r) {
  if (!a || !b) { r = 0; return true; }
  if (a == INT64_MIN || b == INT64_MIN || std::abs(a) > INT64_MAX / std::abs(b)) return false;
  r = a * b; return true;
}
bool bounded(Form f, int64_t lo, int64_t hi) {
  int64_t a, b;
  return mul(f.k, lo, a) && mul(f.k, hi, b) &&
         add(a, f.c, a) && add(b, f.c, b) &&
         std::min(a, b) >= INT32_MIN && std::max(a, b) <= INT32_MAX;
}
bool form(Value *v, Value *slot, int64_t lo, int64_t hi, Form &f,
          const std::unordered_set<Value *> &body, int depth = 0) {
  if (depth > 32 || v->ty != Type::I32) return false;
  if (auto *c = asConstInt(v)) { f = {0, c->v}; return true; }
  auto *i = dynamic_cast<Instruction *>(v);
  if (!i || !body.count(i)) return false;
  if (i->op == Op::Load && i->ops.size() == 1 && i->ops[0] == slot) {
    f = {1, 0}; return true;
  }
  if (i->ops.size() != 2) return false;
  Form a, b;
  if (!form(i->ops[0], slot, lo, hi, a, body, depth + 1) ||
      !form(i->ops[1], slot, lo, hi, b, body, depth + 1)) return false;
  if (i->op == Op::Add || i->op == Op::Sub) {
    int sign = i->op == Op::Add ? 1 : -1;
    if (b.k == INT64_MIN || b.c == INT64_MIN ||
        !add(a.k, sign * b.k, f.k) || !add(a.c, sign * b.c, f.c)) return false;
  } else if (i->op == Op::Mul) {
    if (a.k && b.k) return false;
    if (!mul(a.k, b.c, f.k) || !mul(b.k, a.c, b.k) ||
        !add(f.k, b.k, f.k) || !mul(a.c, b.c, f.c)) return false;
  } else return false;
  // Check every intermediate, not only the final linear expression: i32
  // wrapping arithmetic cannot be treated as an integer affine function.
  return bounded(f, lo, hi);
}
bool compare(Cond c, int64_t a, int64_t b) {
  switch (c) {
  case Cond::Eq: return a == b; case Cond::Ne: return a != b;
  case Cond::Lt: return a < b; case Cond::Le: return a <= b;
  case Cond::Gt: return a > b; case Cond::Ge: return a >= b;
  }
  return false;
}
bool truth(const Predicate &p, int64_t i) {
  return compare(p.cond, p.lhs.k * i + p.lhs.c, p.rhs.k * i + p.rhs.c) != p.inverted;
}
bool privateSlot(Function &fn, Value *slot) {
  auto *a = dynamic_cast<Instruction *>(slot);
  if (!a || a->op != Op::Alloca || a->n != 1 || a->elem != Type::I32) return false;
  bool ok = true;
  walkInstrs(&fn, [&](BasicBlock &, Instruction &i) {
    for (size_t n = 0; n < i.ops.size(); ++n) if (i.ops[n] == slot)
      if (!((i.op == Op::Load && n == 0) || (i.op == Op::Store && n == 1) ||
            (i.op == Op::AffineFor && n == 1))) ok = false;
  });
  return ok;
}
// Recover a local constant seed; all other bounds remain runtime SSA values.
const ConstantInt *seed(Value *v, Value *slot, BlockList &list) {
  if (auto *c = asConstInt(v)) return c;
  auto *l = dynamic_cast<Instruction *>(v);
  if (!l || l->op != Op::Load || l->ops.size() != 1 || l->ops[0] != slot) return nullptr;
  for (auto &b : list) for (size_t n = 0; n < b->instrs.size(); ++n) if (b->instrs[n].get() == l) {
    size_t before = n;
    while (before) {
      auto &i = *b->instrs[--before];
      if (i.op == Op::Store && i.ops[1] == slot) return asConstInt(i.ops[0]);
      if (!i.bodyRegion.empty() || !i.condRegion.empty()) return nullptr;
    }
    return nullptr;
  }
  return nullptr;
}
bool bodySafe(Function &fn, Instruction &loop, size_t &size) {
  std::unordered_set<Value *> values;
  std::unordered_set<BasicBlock *> blocks;
  std::vector<Instruction *> temporaries;
  for (auto &b : loop.bodyRegion) blocks.insert(b.get());
  size = 0;
  for (auto &b : loop.bodyRegion) {
    if (b->instrs.empty()) return false;
    for (size_t n = 0; n < b->instrs.size(); ++n) {
      auto &i = *b->instrs[n]; values.insert(&i); ++size;
      if (size > 192 || !i.bodyRegion.empty() || !i.condRegion.empty() ||
          i.op == Op::Phi ||
          (i.op == Op::Store && i.ops[1] == loop.ops[1])) return false;
      if (i.op == Op::Alloca) {
        // Scalar temporaries have unobservable address identity when they
        // never escape and are definitely initialized on each iteration.
        if (i.n != 1 ||
            (i.elem != Type::I32 && i.elem != Type::F32)) return false;
        bool local = true;
        walkInstrs(&fn, [&](BasicBlock &, Instruction &use) {
          for (size_t q = 0; q < use.ops.size(); ++q) if (use.ops[q] == &i)
            if (!((use.op == Op::Load && q == 0) || (use.op == Op::Store && q == 1))) local = false;
        });
        if (!local) return false;
        temporaries.push_back(&i);
      }
      if (i.op == Op::Br || i.op == Op::CondBr || i.op == Op::AffineYield) {
        if (n + 1 != b->instrs.size()) return false;
        if (i.op == Op::Br && (i.ops.size() != 1 || !blocks.count(dynamic_cast<BasicBlock *>(i.ops[0])))) return false;
        if (i.op == Op::CondBr && (i.ops.size() != 3 ||
            !blocks.count(dynamic_cast<BasicBlock *>(i.ops[1])) ||
            !blocks.count(dynamic_cast<BasicBlock *>(i.ops[2])))) return false;
      } else if (i.op != Op::Call && i.op != Op::Store && !isPure(i.op)) return false;
    }
    auto op = b->instrs.back()->op;
    if (op != Op::Br && op != Op::CondBr && op != Op::AffineYield) return false;
  }
  bool escape = false;
  walkInstrs(&fn, [&](BasicBlock &b, Instruction &i) {
    if (!blocks.count(&b)) for (auto *o : i.ops) if (values.count(o)) escape = true;
  });
  if (escape) return false;
  // Exclude region-internal cycles: the domain is the outer counted loop.
  std::unordered_map<BasicBlock *, int> color;
  std::function<bool(BasicBlock *)> dfs = [&](BasicBlock *b) {
    if (color[b] == 1) return false;
    if (color[b] == 2) return true;
    color[b] = 1; auto &i = *b->instrs.back();
    if (i.op == Op::Br && !dfs(static_cast<BasicBlock *>(i.ops[0]))) return false;
    if (i.op == Op::CondBr && (!dfs(static_cast<BasicBlock *>(i.ops[1])) ||
                             !dfs(static_cast<BasicBlock *>(i.ops[2])))) return false;
    color[b] = 2; return true;
  };
  for (auto &b : loop.bodyRegion) if (!dfs(b.get())) return false;
  // Must-initialization over the acyclic CFG, including short-circuit boolean
  // slots initialized in different arms. Visit at most two states per block;
  // a read is legal only after a store on every path in that iteration.
  for (auto *slot : temporaries) {
    std::unordered_map<BasicBlock *, unsigned> seen;
    std::function<bool(BasicBlock *, bool)> initialized = [&](BasicBlock *b, bool state) {
      unsigned bit = state ? 2 : 1;
      if (seen[b] & bit) return true;
      seen[b] |= bit;
      for (auto &u : b->instrs) {
        if (u.get() == slot) state = false;
        if (u->op == Op::Store && u->ops[1] == slot) state = true;
        if (u->op == Op::Load && u->ops[0] == slot && !state) return false;
      }
      auto &t = *b->instrs.back();
      if (t.op == Op::Br) return initialized(static_cast<BasicBlock *>(t.ops[0]), state);
      if (t.op == Op::CondBr) return initialized(static_cast<BasicBlock *>(t.ops[1]), state) &&
                                    initialized(static_cast<BasicBlock *>(t.ops[2]), state);
      return true;
    };
    if (!initialized(loop.bodyRegion.front().get(), false)) return false;
  }
  return true;
}
std::unique_ptr<Instruction> copy(const Instruction &s) {
  auto i = std::make_unique<Instruction>(s.op, s.ty);
  i->cond = s.cond; i->elem = s.elem; i->n = s.n; i->step = s.step;
  i->shape = s.shape; i->line = s.line; i->name = s.name;
  return i;
}
BlockList specialize(Instruction &loop, const std::vector<Predicate> &preds, int64_t at,
                     const std::string &suffix) {
  BlockList out;
  std::unordered_map<Value *, Value *> map;
  for (auto &b : loop.bodyRegion) {
    auto clone = std::make_unique<BasicBlock>(b->name + suffix);
    map[b.get()] = clone.get(); out.push_back(std::move(clone));
  }
  for (size_t n = 0; n < out.size(); ++n) for (auto &s : loop.bodyRegion[n]->instrs) {
    auto c = copy(*s); map[s.get()] = c.get(); out[n]->instrs.push_back(std::move(c));
  }
  for (size_t n = 0; n < out.size(); ++n) for (size_t q = 0; q < out[n]->instrs.size(); ++q) {
    auto &s = *loop.bodyRegion[n]->instrs[q]; auto &c = *out[n]->instrs[q];
    for (auto *v : s.ops) c.ops.push_back(map.count(v) ? map.at(v) : v);
  }
  for (const auto &p : preds) {
    auto *b = static_cast<Instruction *>(map.at(p.branch));
    Value *target = b->ops[truth(p, at) ? 1 : 2]; b->op = Op::Br; b->ops = {target};
  }
  std::unordered_set<BasicBlock *> live;
  std::function<void(BasicBlock *)> visit = [&](BasicBlock *b) {
    if (!live.insert(b).second) return;
    auto &t = *b->instrs.back();
    if (t.op == Op::Br) visit(static_cast<BasicBlock *>(t.ops[0]));
    if (t.op == Op::CondBr) { visit(static_cast<BasicBlock *>(t.ops[1])); visit(static_cast<BasicBlock *>(t.ops[2])); }
  };
  visit(out.front().get());
  // No retained definition may refer to a discarded instruction. Valid SSA
  // dominance implies this, but verify before committing the transformation.
  std::unordered_set<Value *> dead;
  for (auto &b : out) if (!live.count(b.get())) for (auto &i : b->instrs) dead.insert(i.get());
  for (auto &b : out) if (live.count(b.get())) for (auto &i : b->instrs)
    for (auto *o : i->ops) if (dead.count(o)) return {};
  out.erase(std::remove_if(out.begin(), out.end(), [&](auto &b) { return !live.count(b.get()); }), out.end());
  // Merge unique-predecessor paths so LICM/unroll can see straight-line bodies.
  bool changed = true;
  while (changed) {
    changed = false; std::unordered_map<Value *, int> incoming;
    for (auto &b : out) {
      auto &t = *b->instrs.back();
      if (t.op == Op::Br) ++incoming[t.ops[0]];
      if (t.op == Op::CondBr) { ++incoming[t.ops[1]]; ++incoming[t.ops[2]]; }
    }
    for (auto &b : out) {
      auto &t = *b->instrs.back();
      if (t.op != Op::Br || incoming[t.ops[0]] != 1) continue;
      auto *dst = static_cast<BasicBlock *>(t.ops[0]);
      if (dst == out.front().get()) continue;
      b->instrs.pop_back();
      for (auto &i : dst->instrs) b->instrs.push_back(std::move(i));
      out.erase(std::find_if(out.begin(), out.end(), [&](auto &u) { return u.get() == dst; }));
      changed = true; break;
    }
  }
  // Drop the comparison/address temporaries made dead by specialization
  // before the subsequent unroller estimates the size of this body.
  changed = true;
  while (changed) {
    changed = false; std::unordered_map<Value *, size_t> uses;
    for (auto &b : out) for (auto &i : b->instrs) for (auto *v : i->ops) ++uses[v];
    for (auto &b : out) b->instrs.erase(std::remove_if(b->instrs.begin(), b->instrs.end(), [&](auto &i) {
      if (isPure(i->op) && !uses[i.get()]) { changed = true; return true; }
      return false;
    }), b->instrs.end());
  }
  return out;
}
class LoopSplitPass final : public Pass {
public:
  const char *name() const override { return "loop-split"; }
  Layer inLayer() const override { return Layer::Affine; }
  bool run(Module &m) override {
    bool changed = false; int seq = 0;
    for (auto &f : m.functions()) if (!f->isLib) changed |= walk(m, *f, f->blocks, seq);
    return changed;
  }
private:
  bool walk(Module &m, Function &fn, BlockList &list, int &seq) {
    bool changed = false;
    for (size_t n = 0; n < list.size(); ++n) {
      auto *b = list[n].get();
      for (size_t q = 0; q < b->instrs.size(); ++q) {
        auto *i = b->instrs[q].get();
        changed |= walk(m, fn, i->condRegion, seq);
        changed |= walk(m, fn, i->bodyRegion, seq);
        if (split(m, fn, list, n, q, seq)) { changed = true; break; }
      }
    }
    return changed;
  }
  bool split(Module &m, Function &fn, BlockList &list, size_t bi, size_t pos, int &seq) {
    auto *head = list[bi].get(); auto &loop = *head->instrs[pos];
    if (loop.op != Op::AffineFor || loop.ops.size() != 4 || loop.cond != Cond::Lt ||
        loop.step != 1 || loop.bodyRegion.empty() || pos + 1 != head->instrs.size() ||
        !privateSlot(fn, loop.ops[1])) return false;
    size_t size;
    if (!bodySafe(fn, loop, size)) return false;
    auto *lb = seed(loop.ops[2], loop.ops[1], list), *ub = asConstInt(loop.ops[3]);
    int64_t lo = lb ? lb->v : INT32_MIN, hi = ub ? int64_t(ub->v) - 1 : INT32_MAX - 1;
    if (lo > hi || (lb && ub && hi - lo + 1 < 8)) return false;
    std::vector<Predicate> preds; std::set<int64_t> cuts;
    std::unordered_set<Value *> body;
    for (auto &b : loop.bodyRegion) for (auto &i : b->instrs) body.insert(i.get());
    for (auto &b : loop.bodyRegion) {
      auto *branch = b->instrs.back().get();
      if (branch->op != Op::CondBr) continue;
      Value *v = branch->ops[0]; bool inverted = false; int depth = 0;
      auto *cmp = dynamic_cast<Instruction *>(v);
      while (cmp && cmp->op == Op::Not && cmp->ops.size() == 1 && depth++ < 8) {
        inverted = !inverted; v = cmp->ops[0]; cmp = dynamic_cast<Instruction *>(v);
      }
      Predicate p{branch, {}, {}, cmp ? cmp->cond : Cond::Eq, inverted};
      if (!cmp || cmp->op != Op::ICmp || cmp->ops.size() != 2 ||
          !form(cmp->ops[0], loop.ops[1], lo, hi, p.lhs, body) ||
          !form(cmp->ops[1], loop.ops[1], lo, hi, p.rhs, body)) continue;
      int64_t k, c;
      if (p.rhs.k == INT64_MIN || p.rhs.c == INT64_MIN ||
          !add(p.lhs.k, -p.rhs.k, k) || !add(p.lhs.c, -p.rhs.c, c) || !k) continue;
      if (p.cond == Cond::Eq || p.cond == Cond::Ne) {
        if (c == INT64_MIN || (-c == INT64_MIN && k == -1)) continue;
        if (-c % k == 0) {
          int64_t root = -c / k;
          if (root > lo && root <= hi) cuts.insert(root);
          if (root >= lo && root < hi) cuts.insert(root + 1);
        }
      } else if (truth(p, lo) != truth(p, hi)) {
        int64_t a = lo, b = hi;
        while (a + 1 < b) { int64_t mid = a + (b - a) / 2;
          if (truth(p, mid) == truth(p, lo)) a = mid; else b = mid;
        }
        cuts.insert(b);
      }
      preds.push_back(p);
    }
    if (cuts.empty() || cuts.size() > 7 || size * (cuts.size() + 1) > 1024) return false;
    std::vector<int64_t> points{lo}; points.insert(points.end(), cuts.begin(), cuts.end());
    std::vector<BlockList> bodies;
    std::string suffix = ".split" + std::to_string(++seq);
    for (size_t n = 0; n < points.size(); ++n) {
      bodies.push_back(specialize(loop, preds, points[n], suffix + "." + std::to_string(n)));
      if (bodies.back().empty()) return false;
    }
    // Each boundary is max(lb,min(cut,ub)); the final one is max(lb,ub).
    // Thus empty runtime domains preserve the original induction-slot seed.
    Value *lower = lb ? m.constInt(lb->v) : loop.ops[2];
    Value *upper = loop.ops[3], *slot = loop.ops[1], *exit = loop.ops[0]; int line = loop.line;
    std::vector<std::unique_ptr<Instruction>> pre;
    auto emit = [&](Op op, Type ty, std::vector<Value *> ops) {
      auto u = std::make_unique<Instruction>(op, ty); u->ops = std::move(ops); u->line = line;
      auto *v = u.get(); pre.push_back(std::move(u)); return v;
    };
    auto minmax = [&](Value *a, Value *b, bool maximum) -> Value * {
      auto *ca = asConstInt(a), *cb = asConstInt(b);
      if (ca && cb) return m.constInt(maximum ? std::max(ca->v, cb->v) : std::min(ca->v, cb->v));
      auto *cmp = emit(Op::ICmp, Type::I32, {a, b}); cmp->cond = Cond::Lt;
      return emit(Op::Select, Type::I32, {cmp, maximum ? b : a, maximum ? a : b});
    };
    std::vector<Value *> bounds{lower};
    for (int64_t cut : cuts) bounds.push_back(minmax(lower, minmax(m.constInt(int32_t(cut)), upper, false), true));
    bounds.push_back(minmax(lower, upper, true));
    BlockList siblings;
    for (size_t n = 1; n < bodies.size(); ++n) siblings.push_back(std::make_unique<BasicBlock>("split" + std::to_string(seq) + "." + std::to_string(n)));
    std::vector<std::unique_ptr<Instruction>> loops;
    for (size_t n = 0; n < bodies.size(); ++n) {
      auto u = std::make_unique<Instruction>(Op::AffineFor, Type::Void);
      u->ops = {n + 1 < bodies.size() ? siblings[n].get() : exit, slot, bounds[n], bounds[n + 1]};
      u->cond = Cond::Lt; u->step = 1; u->line = line; u->bodyRegion = std::move(bodies[n]);
      loops.push_back(std::move(u));
    }
    head->instrs.pop_back();
    for (auto &u : pre) head->instrs.push_back(std::move(u));
    head->instrs.push_back(std::move(loops[0]));
    for (size_t n = 1; n < loops.size(); ++n) siblings[n - 1]->instrs.push_back(std::move(loops[n]));
    list.insert(list.begin() + bi + 1, std::make_move_iterator(siblings.begin()), std::make_move_iterator(siblings.end()));
    return true;
  }
};
} // namespace
std::unique_ptr<Pass> makeLoopSplitPass() { return std::make_unique<LoopSplitPass>(); }
} } // namespace sakura::ir
