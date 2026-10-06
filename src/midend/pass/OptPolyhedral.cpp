// A bounded polyhedral scheduler for rectangular, two-dimensional affine
// bands. Domains, byte access maps and schedules remain explicit here; unknown
// aliasing, arithmetic wrap and unmodelled effects reject the whole band.
// See docs/polyhedral-layout.md for the legality proof and literature.
#include "Opt.h"
#include "OptUtil.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <numeric>

namespace sakura { namespace ir { namespace {
struct Form {
  int64_t c = 0;
  std::array<int64_t, 2> k{{0, 0}};
};
struct Band {
  Instruction *outer = nullptr, *inner = nullptr;
  BasicBlock *leaf = nullptr;
  std::array<Value *, 2> slots;
  std::array<int64_t, 2> lb, ub, count;
};
struct Access { Value *root; Form map; bool write; };

bool add(int64_t a, int64_t b, int64_t &out) {
  if ((b > 0 && a > INT64_MAX - b) || (b < 0 && a < INT64_MIN - b)) return false;
  out = a + b; return true;
}
bool multiply(int64_t a, int64_t b, int64_t &out) {
  if (!a || !b) { out = 0; return true; }
  if (a == INT64_MIN || b == INT64_MIN) return false;
  if (std::abs(a) > INT64_MAX / std::abs(b)) return false;
  out = a * b; return true;
}

bool bounded(const Form &f, const Band &b) {
  int64_t lo = f.c, hi = f.c;
  for (int d = 0; d < 2; ++d) {
    // Each accepted subexpression fits i32; its coefficients are therefore
    // bounded as well (both dimensions contain at least two points).
    int64_t x, y;
    if (!multiply(f.k[d], b.lb[d], x) || !multiply(f.k[d], b.ub[d] - 1, y) ||
        !add(lo, std::min(x, y), lo) || !add(hi, std::max(x, y), hi)) return false;
  }
  return lo >= INT32_MIN && hi <= INT32_MAX;
}
bool form(Value *v, const Band &b, Form &out, int depth = 0) {
  if (depth > 32) return false;
  if (auto *c = asConstInt(v)) { out.c = c->v; return true; }
  auto *i = dynamic_cast<Instruction *>(v);
  if (!i) return false;
  if (i->op == Op::Load && i->ops.size() == 1) {
    for (int d = 0; d < 2; ++d)
      if (i->ops[0] == b.slots[d]) { out.k[d] = 1; return true; }
    return false;
  }
  if (i->ops.size() != 2 ||
      (i->op != Op::Add && i->op != Op::Sub && i->op != Op::Mul)) return false;
  Form x, y;
  if (!form(i->ops[0], b, x, depth + 1) ||
      !form(i->ops[1], b, y, depth + 1)) return false;
  if (i->op == Op::Mul) {
    if (x.k[0] || x.k[1]) std::swap(x, y);
    if (x.k[0] || x.k[1]) return false;
    if (!multiply(x.c, y.c, out.c)) return false;
    for (int d = 0; d < 2; ++d) if (!multiply(x.c, y.k[d], out.k[d])) return false;
  } else {
    int s = i->op == Op::Add ? 1 : -1;
    if (y.c == INT64_MIN || !add(x.c, s * y.c, out.c)) return false;
    for (int d = 0; d < 2; ++d)
      if (y.k[d] == INT64_MIN || !add(x.k[d], s * y.k[d], out.k[d])) return false;
  }
  return bounded(out, b);
}
bool address(Value *v, const Band &b, Value *&root, Form &f, int depth = 0) {
  if (depth > 16) return false;
  auto *i = dynamic_cast<Instruction *>(v);
  if (i && i->op == Op::Gep && i->ops.size() == 2) {
    Form offset;
    if (!address(i->ops[0], b, root, f, depth + 1) ||
        !form(i->ops[1], b, offset)) return false;
    if (!add(f.c, offset.c, f.c)) return false;
    for (int d = 0; d < 2; ++d) if (!add(f.k[d], offset.k[d], f.k[d])) return false;
    return bounded(f, b);
  }
  if (dynamic_cast<GlobalVar *>(v) || dynamic_cast<Argument *>(v) ||
      (i && i->op == Op::Alloca)) { root = v; return true; }
  return false;
}
bool privateSlot(Function &fn, Value *slot) {
  auto *a = dynamic_cast<Instruction *>(slot);
  if (!a || a->op != Op::Alloca || a->n != 1 || a->elem != Type::I32) return false;
  bool ok = true;
  walkInstrs(&fn, [&](BasicBlock &, Instruction &i) {
    for (size_t p = 0; p < i.ops.size(); ++p) if (i.ops[p] == slot)
      if (!((i.op == Op::Load && p == 0) || (i.op == Op::Store && p == 1) ||
            (i.op == Op::AffineFor && p == 1))) ok = false;
  });
  return ok;
}
bool writesSlot(Instruction &i, Value *slot) {
  if ((i.op == Op::Store && i.ops[1] == slot) ||
      (i.op == Op::AffineFor && i.ops[1] == slot)) return true;
  for (auto *region : {&i.condRegion, &i.bodyRegion})
    for (auto &bb : *region) for (auto &u : bb->instrs) if (writesSlot(*u, slot)) return true;
  return false;
}
// mem-cse deliberately clears its memory map at structured exits. Recover a
// constant IV seed across a unique predecessor chain only, stopping at every
// operation that can modify this private, non-escaping slot.
const ConstantInt *bound(Value *v, Value *slot, const BlockList &list) {
  if (auto *c = asConstInt(v)) return c;
  auto *load = dynamic_cast<Instruction *>(v);
  if (!load || load->op != Op::Load || load->ops[0] != slot) return nullptr;
  BasicBlock *bb = nullptr; size_t pos = 0;
  for (auto &b : list) for (size_t p = 0; p < b->instrs.size(); ++p)
    if (b->instrs[p].get() == load) { bb = b.get(); pos = p; }
  std::vector<BasicBlock *> seen;
  while (bb) {
    if (std::find(seen.begin(), seen.end(), bb) != seen.end()) return nullptr;
    seen.push_back(bb);
    while (pos) {
      auto &i = *bb->instrs[--pos];
      if (i.op == Op::Store && i.ops[1] == slot) return asConstInt(i.ops[0]);
      if (writesSlot(i, slot)) return nullptr;
    }
    BasicBlock *prev = nullptr;
    for (auto &b : list) if (!b->instrs.empty()) {
      auto &t = *b->instrs.back(); bool edge = false;
      if (t.op == Op::Br || t.op == Op::AffineFor || t.op == Op::ScfWhile)
        edge = !t.ops.empty() && t.ops[0] == bb;
      if (t.op == Op::CondBr) edge = t.ops[1] == bb || t.ops[2] == bb;
      if (edge) { if (prev) return nullptr; prev = b.get(); }
    }
    bb = prev; if (bb) pos = bb->instrs.size();
  }
  return nullptr;
}
bool match(Function &fn, const BlockList &list, Instruction *outer, Band &b) {
  if (outer->op != Op::AffineFor || outer->ops.size() != 4 ||
      outer->bodyRegion.size() != 2) return false;
  BasicBlock *head = outer->bodyRegion[0].get(), *exit = outer->bodyRegion[1].get();
  if (head->instrs.empty() || exit->instrs.size() != 1 ||
      exit->instrs[0]->op != Op::AffineYield) return false;
  auto *inner = head->instrs.back().get();
  if (inner->op != Op::AffineFor || inner->ops.size() != 4 ||
      inner->ops[0] != exit || inner->bodyRegion.size() != 1) return false;
  b.outer = outer; b.inner = inner; b.leaf = inner->bodyRegion[0].get();
  for (int d = 0; d < 2; ++d) {
    auto *loop = d ? inner : outer;
    if (!privateSlot(fn, loop->ops[1])) return false;
    auto *lb = bound(loop->ops[2], loop->ops[1], d ? outer->bodyRegion : list);
    auto *ub = asConstInt(loop->ops[3]);
    if (loop->cond != Cond::Lt || loop->step != 1 || !lb || !ub) return false;
    b.slots[d] = loop->ops[1]; b.lb[d] = lb->v; b.ub[d] = ub->v;
    b.count[d] = b.ub[d] - b.lb[d];
    // Nonempty dimensions also preserve both induction slots' final values.
    if (b.count[d] < 2) return false;
  }
  if (b.slots[0] == b.slots[1]) return false;
  // The only allowed imperfect-nest prologue is the redundant inner-IV seed.
  for (size_t p = 0; p + 1 < head->instrs.size(); ++p) {
    auto *i = head->instrs[p].get();
    auto *c = i->ops.empty() ? nullptr : asConstInt(i->ops[0]);
    if (i->op != Op::Store || i->ops.size() != 2 || i->ops[1] != b.slots[1] ||
        !c || c->v != b.lb[1]) return false;
  }
  return !b.leaf->instrs.empty() && b.leaf->instrs.back()->op == Op::AffineYield;
}
bool accesses(const Band &b, std::vector<Access> &out) {
  for (auto &u : b.leaf->instrs) {
    auto &i = *u;
    if (i.op == Op::AffineYield) continue;
    if (i.op == Op::Load || i.op == Op::Store) {
      Type scalar = i.op == Op::Load ? i.ty : i.ops[0]->ty;
      if (scalar != Type::I32 && scalar != Type::F32) return false;
      Value *p = i.ops[i.op == Op::Store ? 1 : 0];
      if (p == b.slots[0] || p == b.slots[1]) {
        if (i.op == Op::Store) return false;
        continue;
      }
      Access a{nullptr, {}, i.op == Op::Store};
      if (!address(p, b, a.root, a.map)) return false;
      // i32/f32 accesses are four bytes; require alignment, so equality of
      // byte addresses also covers partial overlaps.
      if ((a.map.c % 4) || (a.map.k[0] % 4) || (a.map.k[1] % 4)) return false;
      out.push_back(a);
    } else if (!isPure(i.op) || i.op == Op::Alloca || i.op == Op::Phi) return false;
  }
  return !out.empty() && out.size() <= 64;
}

// Exact integer feasibility for uniform accesses in the two opposite-sign
// quadrants. A*x+B*y=C, 1<=x<=X, 1<=y<=Y, is the dependence polyhedron after
// substituting the displacement and selecting a schedule-reversing quadrant.
// Enumerating one bounded displacement coordinate (not iteration instances)
// keeps compilation predictable. Large unresolved domains conservatively fail.
bool conflict(int64_t a, int64_t b, int64_t c, int64_t x, int64_t y) {
  if (x > y) { std::swap(a, b); std::swap(x, y); }
  if (x > 4096) return true;
  for (int64_t d = 1; d <= x; ++d) {
    int64_t rem = c - a * d;
    if (!b) { if (!rem) return true; }
    else if (rem % b == 0 && rem / b >= 1 && rem / b <= y) return true;
  }
  return false;
}
bool permutable(const Band &band, const std::vector<Access> &acc) {
  for (size_t i = 0; i < acc.size(); ++i) for (size_t j = i; j < acc.size(); ++j) {
    const auto &a = acc[i], &b = acc[j];
    if (!a.write && !b.write) continue;
    if (a.root != b.root) {
      // Array arguments may alias any global or another argument. Distinct
      // named objects and this frame's allocas do not overlap.
      auto local = [](Value *v) { auto *p = dynamic_cast<Instruction *>(v);
        return p && p->op == Op::Alloca; };
      if (!local(a.root) && !local(b.root) &&
          (dynamic_cast<Argument *>(a.root) || dynamic_cast<Argument *>(b.root))) return false;
      continue;
    }
    if (a.map.k != b.map.k) return false;
    int64_t c = a.map.c - b.map.c, x = band.count[0] - 1, y = band.count[1] - 1;
    if (conflict(a.map.k[0], -a.map.k[1], c, x, y) ||
        conflict(-a.map.k[0], a.map.k[1], c, x, y)) return false;
  }
  return true;
}
Instruction *emit(BasicBlock *bb, Op op, Type ty, std::vector<Value *> ops = {}) {
  auto i = std::make_unique<Instruction>(op, ty); i->ops = std::move(ops);
  auto *p = i.get(); bb->instrs.push_back(std::move(i)); return p;
}
// Build one structured wrapper. Region exits always stay owned by the parent.
Instruction *wrapper(BasicBlock *head, Value *slot, Value *lb, Value *ub,
                     BasicBlock *exit, BlockList body, int64_t step = 1) {
  auto *f = emit(head, Op::AffineFor, Type::Void, {exit, slot, lb, ub});
  f->cond = Cond::Lt; f->step = step; f->bodyRegion = std::move(body); return f;
}
class PolyhedralPass final : public Pass {
public:
  const char *name() const override { return "polyhedral-schedule"; }
  Layer inLayer() const override { return Layer::Affine; }
  bool run(Module &m) override {
    bool changed = false; int seq = 0;
    for (auto &f : m.functions()) if (!f->isLib) changed |= walk(m, *f, f->blocks, seq);
    return changed;
  }
private:
  bool walk(Module &m, Function &fn, BlockList &list, int &seq) {
    bool changed = false;
    for (auto &bb : list) for (size_t p = 0; p < bb->instrs.size(); ++p) {
      auto *o = bb->instrs[p].get(); Band b;
      std::vector<Access> acc;
      if (match(fn, list, o, b) && accesses(b, acc) && permutable(b, acc)) {
        int64_t cost[2] = {0, 0}; bool stores = false;
        for (const auto &a : acc) {
          stores |= a.write;
          for (int d = 0; d < 2; ++d)
            cost[d] += std::min<int64_t>(std::abs(a.map.k[d]), 4096) * (a.write ? 2 : 1);
        }
        bool swap = stores && cost[0] * 2 < cost[1];
        int outerDim = swap ? 1 : 0, innerDim = 1 - outerDim;
        bool reuse = false;
        for (const auto &a : acc) if (!a.write && !a.map.k[outerDim] && a.map.k[innerDim]) reuse = true;
        bool tile = stores && reuse && b.count[0] >= 64 && b.count[1] >= 64 &&
                    !std::getenv("SAKU_NO_POLY_TILE");
        if (swap || tile) {
          // Extract the leaf before deleting the redundant seed/old wrappers.
          BlockList leaf = std::move(b.inner->bodyRegion);
          o->bodyRegion.clear();
          auto block = [&]() { return std::make_unique<BasicBlock>("poly" + std::to_string(seq++)); };
          auto head = block(), exit = block(); emit(exit.get(), Op::AffineYield, Type::Void);
          Value *outerSlot = b.slots[outerDim], *innerSlot = b.slots[innerDim];
          Value *olb = m.constInt((int32_t)b.lb[outerDim]), *oub = m.constInt((int32_t)b.ub[outerDim]);
          Value *ilb = m.constInt((int32_t)b.lb[innerDim]), *iub = m.constInt((int32_t)b.ub[innerDim]);
          if (!tile) {
            wrapper(head.get(), innerSlot, ilb, iub, exit.get(), std::move(leaf));
            o->ops[1] = outerSlot; o->ops[2] = olb; o->ops[3] = oub;
          } else {
            // Strip mine both dimensions, then interchange tile/point loops:
            // (i,j) -> (floor((i-lb)/32),floor((j-lb)/32),i,j).
            // Reject tile increments/additions that could wrap i32.
            if (b.ub[0] > INT32_MAX - 32 || b.ub[1] > INT32_MAX - 32) {
              wrapper(head.get(), innerSlot, ilb, iub, exit.get(), std::move(leaf));
              o->ops[1] = outerSlot; o->ops[2] = olb; o->ops[3] = oub;
            } else {
              auto alloc = [&](size_t pos) { auto u = std::make_unique<Instruction>(Op::Alloca, Type::Ptr);
                u->elem = Type::I32; u->n = 1; auto *v = u.get();
                bb->instrs.insert(bb->instrs.begin() + pos, std::move(u)); return v; };
              Value *ts0 = alloc(p++), *ts1 = alloc(p++);
              Value *t0 = emit(head.get(), Op::Load, Type::I32, {ts0});
              auto h1 = block(), e1 = block(), h2 = block(), e2 = block();
              emit(e1.get(), Op::AffineYield, Type::Void); emit(e2.get(), Op::AffineYield, Type::Void);
              Value *t1 = emit(h1.get(), Op::Load, Type::I32, {ts1});
              auto end = [&](Value *t, Value *ub) -> Value * {
                auto *sum = emit(h1.get(), Op::Add, Type::I32, {t, m.constInt(32)});
                auto *cmp = emit(h1.get(), Op::ICmp, Type::I32, {sum, ub}); cmp->cond = Cond::Lt;
                return emit(h1.get(), Op::Select, Type::I32, {cmp, sum, ub}); };
              Value *u0 = end(t0, oub), *u1 = end(t1, iub);
              wrapper(h2.get(), innerSlot, t1, u1, e2.get(), std::move(leaf));
              BlockList r2; r2.push_back(std::move(h2)); r2.push_back(std::move(e2));
              wrapper(h1.get(), outerSlot, t0, u0, e1.get(), std::move(r2));
              BlockList r1; r1.push_back(std::move(h1)); r1.push_back(std::move(e1));
              wrapper(head.get(), ts1, ilb, iub, exit.get(), std::move(r1), 32);
              o->ops[1] = ts0; o->ops[2] = olb; o->ops[3] = oub; o->step = 32;
            }
          }
          o->bodyRegion.push_back(std::move(head)); o->bodyRegion.push_back(std::move(exit));
          changed = true;
          continue; // Do not recursively reschedule generated bands.
        }
      }
      changed |= walk(m, fn, o->condRegion, seq);
      changed |= walk(m, fn, o->bodyRegion, seq);
    }
    return changed;
  }
};
} // namespace
std::unique_ptr<Pass> makePolyhedralPass() { return std::make_unique<PolyhedralPass>(); }
} } // namespace sakura::ir
