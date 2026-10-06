#include "midend/pass/Opt.h"
#include "midend/pass/OptUtil.h"
#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>
#include <unordered_map>

using namespace sakura::ir;
static void require(bool ok, const char *s) { if (!ok) throw std::runtime_error(s); }
static Instruction *emit(BasicBlock *bb, Op op, Type ty, std::vector<Value *> ops = {}) {
  auto i = std::make_unique<Instruction>(op, ty); i->ops = std::move(ops);
  auto *p = i.get(); bb->instrs.push_back(std::move(i)); return p;
}
static BasicBlock *block(BlockList &list, const char *name) {
  auto b = std::make_unique<BasicBlock>(name); auto *p = b.get(); list.push_back(std::move(b)); return p;
}
using Cell = std::pair<Value *, int64_t>;
using Memory = std::map<Cell, int32_t>;
// An independent structured-IR interpreter. Compare all original objects,
// including the observable induction slots, before/after the transformation.
static Memory execute(Function &fn) {
  Memory mem; std::unordered_map<Value *, int32_t> val;
  auto value = [&](Value *v) { if (auto *c = dynamic_cast<ConstantInt *>(v)) return c->v;
    require(val.count(v), "undefined SSA value"); return val.at(v); };
  std::function<Cell(Value *)> address = [&](Value *v) -> Cell {
    auto *i = dynamic_cast<Instruction *>(v);
    if (i && i->op == Op::Gep) { auto p = address(i->ops[0]); p.second += value(i->ops[1]); return p; }
    return {v, 0};
  };
  int fuel = 1000000;
  std::function<void(BasicBlock *)> run = [&](BasicBlock *b) {
    while (b) {
      require(--fuel > 0, "interpreter fuel exhausted"); BasicBlock *next = nullptr;
      for (auto &u : b->instrs) {
        auto &i = *u;
        switch (i.op) {
        case Op::Alloca: mem[{u.get(), 0}] = 0; break;
        case Op::Gep: break;
        case Op::Load: val[u.get()] = mem[address(i.ops[0])]; break;
        case Op::Store: mem[address(i.ops[1])] = value(i.ops[0]); break;
        case Op::Add: val[u.get()] = int32_t(int64_t(value(i.ops[0])) + value(i.ops[1])); break;
        case Op::Sub: val[u.get()] = int32_t(int64_t(value(i.ops[0])) - value(i.ops[1])); break;
        case Op::Mul: val[u.get()] = int32_t(int64_t(value(i.ops[0])) * value(i.ops[1])); break;
        case Op::ICmp: val[u.get()] = value(i.ops[0]) < value(i.ops[1]); break;
        case Op::Select: val[u.get()] = value(i.ops[value(i.ops[0]) ? 1 : 2]); break;
        case Op::AffineFor: {
          auto p = address(i.ops[1]); mem[p] = value(i.ops[2]); int32_t ub = value(i.ops[3]);
          while (mem[p] < ub) {
            require(--fuel > 0, "loop fuel exhausted"); run(i.bodyRegion.front().get());
            mem[p] = int32_t(int64_t(mem[p]) + i.step);
          }
          next = dynamic_cast<BasicBlock *>(i.ops[0]); break;
        }
        case Op::Br: next = dynamic_cast<BasicBlock *>(i.ops[0]); break;
        case Op::AffineYield: case Op::Ret: return;
        default: throw std::runtime_error("unsupported test operation");
        }
      }
      b = next;
    }
  };
  run(fn.entry()); return mem;
}
struct Fixture {
  Module m{"poly-test"}; Function *f; Instruction *outer, *inner;
  Value *i, *j, *array; BasicBlock *leaf;
  Fixture(int ni = 17, int nj = 19, int lb = 0) {
    f = m.addFunction("main", Type::I32, {}, false);
    array = m.addGlobal("a", Type::I32, 20000, false, false, {});
    auto *entry = block(f->blocks, "entry"), *exit = block(f->blocks, "exit");
    auto slot = [&]() { auto *s = emit(entry, Op::Alloca, Type::Ptr); s->n = 1; return s; };
    i = slot(); j = slot();
    outer = emit(entry, Op::AffineFor, Type::Void, {exit, i, m.constInt(lb), m.constInt(ni)});
    outer->cond = Cond::Lt; outer->step = 1;
    auto *head = block(outer->bodyRegion, "outer.body"), *yield = block(outer->bodyRegion, "outer.yield");
    emit(head, Op::Store, Type::Void, {m.constInt(lb), j});
    inner = emit(head, Op::AffineFor, Type::Void, {yield, j, m.constInt(lb), m.constInt(nj)});
    inner->cond = Cond::Lt; inner->step = 1;
    leaf = block(inner->bodyRegion, "leaf"); emit(yield, Op::AffineYield, Type::Void);
    emit(exit, Op::Ret, Type::Void, {m.constInt(0)});
  }
  Value *ptr(Value *root, int a, int b, int c = 0) {
    Value *x = emit(leaf, Op::Load, Type::I32, {i}), *y = emit(leaf, Op::Load, Type::I32, {j});
    x = emit(leaf, Op::Mul, Type::I32, {x, m.constInt(a)});
    y = emit(leaf, Op::Mul, Type::I32, {y, m.constInt(b)});
    Value *z = emit(leaf, Op::Add, Type::I32, {x, y});
    z = emit(leaf, Op::Add, Type::I32, {z, m.constInt(c)});
    return emit(leaf, Op::Gep, Type::Ptr, {root, z});
  }
  void store(Value *p, Value *v) { emit(leaf, Op::Store, Type::Void, {v, p}); }
  void finish() { emit(leaf, Op::AffineYield, Type::Void); }
  void compare(bool changed) {
    auto before = execute(*f);
    require(makePolyhedralPass()->run(m) == changed, "unexpected scheduling decision");
    auto after = execute(*f);
    for (const auto &v : before) require(after.at(v.first) == v.second, "loop transformation changed memory");
  }
};
static void interchange() {
  for (int lb : {-3, 0, 5}) {
    Fixture t(17, 19, lb);
    auto *p = t.ptr(t.array, 4, 128, 512);
    auto *old = emit(t.leaf, Op::Load, Type::I32, {p});
    auto *inc = emit(t.leaf, Op::Add, Type::I32, {old, t.m.constInt(3)});
    t.store(p, inc); t.finish(); Value *oldJ = t.j;
    t.compare(true); require(t.outer->ops[1] == oldJ, "unit-stride dimension not innermost");
    require(!makePolyhedralPass()->run(t.m), "schedule is not stable");
  }
}
static void tiling() {
  for (int ub : {64, 67, 97}) {
    Fixture t(ub, 71, 3);
    auto *input = t.m.addGlobal("b", Type::I32, 100, false, false, {});
    auto *v = emit(t.leaf, Op::Load, Type::I32, {t.ptr(input, 0, 4)});
    t.store(t.ptr(t.array, 512, 4), v); t.finish();
    t.compare(ub >= 67); // 64-3 points does not meet the profitability threshold.
    if (ub >= 67) require(t.outer->step == 32, "missing tile loop");
  }
}
static void dependenceOracle() {
  // Exhaustively enumerate the finite displacement domain independently of
  // the optimizer's one-coordinate integer feasibility test.
  for (int a : {-8, 0, 4, 8}) for (int b : {-128, 16, 128})
    for (int c : {-128, -124, -4, 0, 4, 124, 128}) {
      bool conflict = false;
      for (int di = -6; di <= 6; ++di) for (int dj = -6; dj <= 6; ++dj)
        if (di * dj < 0 && (a * di + b * dj == c || a * di + b * dj == 0)) conflict = true;
      Fixture t(7, 7);
      auto *old = emit(t.leaf, Op::Load, Type::I32, {t.ptr(t.array, a, b, 2048 + c)});
      auto *inc = emit(t.leaf, Op::Add, Type::I32, {old, t.m.constInt(1)});
      t.store(t.ptr(t.array, a, b, 2048), inc); t.finish();
      t.compare(!conflict && std::abs(a) * 2 < std::abs(b));
    }
}
static void boundTrace() {
  for (bool clobber : {false, true}) {
    Fixture t(17, 19, 5); auto *entry = t.f->entry();
    auto pre = std::make_unique<BasicBlock>("pre");
    // Move the slot declarations into the new entry, then initialize i before
    // a separate structured loop. Its exit contains the outer bound load.
    pre->instrs.push_back(std::move(entry->instrs[0]));
    pre->instrs.push_back(std::move(entry->instrs[1]));
    entry->instrs.erase(entry->instrs.begin(), entry->instrs.begin() + 2);
    auto *k = emit(pre.get(), Op::Alloca, Type::Ptr); k->n = 1;
    emit(pre.get(), Op::Store, Type::Void, {t.m.constInt(5), t.i});
    auto *loop = emit(pre.get(), Op::AffineFor, Type::Void,
                      {entry, k, t.m.constInt(0), t.m.constInt(4)});
    loop->cond = Cond::Lt; loop->step = 1;
    auto *body = block(loop->bodyRegion, "seed.body");
    if (clobber) emit(body, Op::Store, Type::Void, {t.m.constInt(6), t.i});
    emit(body, Op::AffineYield, Type::Void);
    auto load = std::make_unique<Instruction>(Op::Load, Type::I32); load->ops = {t.i};
    t.outer->ops[2] = load.get(); entry->instrs.insert(entry->instrs.begin(), std::move(load));
    t.f->blocks.insert(t.f->blocks.begin(), std::move(pre));
    t.store(t.ptr(t.array, 4, 128), t.m.constInt(1)); t.finish(); t.compare(!clobber);
  }
}
static void rejections() {
  { Fixture t; auto *v = emit(t.leaf, Op::Load, Type::I32, {t.ptr(t.array, 4, 128, 4-128)});
    t.store(t.ptr(t.array, 4, 128), v); t.finish(); t.compare(false); }
  { Fixture t; auto *s = t.m.addGlobal("sum", Type::I32, 1, false, false, {});
    auto *v = emit(t.leaf, Op::Load, Type::I32, {s});
    v = emit(t.leaf, Op::Add, Type::I32, {v, t.m.constInt(1)});
    t.store(s, v); t.store(t.ptr(t.array, 4, 128), v); t.finish(); t.compare(false); }
  { Fixture t; auto *v = emit(t.leaf, Op::Load, Type::I32, {t.ptr(t.array, 8, 128)});
    t.store(t.ptr(t.array, 4, 128), v); t.finish(); t.compare(false); }
  { Fixture t(0, 19); t.store(t.ptr(t.array, 4, 128), t.m.constInt(1)); t.finish(); t.compare(false); }
  { Fixture t; t.outer->step = 2; t.store(t.ptr(t.array, 4, 128), t.m.constInt(1)); t.finish(); t.compare(false); }
  { Fixture t; t.store(t.ptr(t.array, INT32_MAX-3, 128), t.m.constInt(1)); t.finish();
    require(!makePolyhedralPass()->run(t.m), "overflowing access accepted"); }
  { Fixture t; auto arg = std::make_unique<Argument>(); arg->ty = Type::Ptr;
    Value *p = arg.get(); t.f->args.push_back(std::move(arg));
    auto *v = emit(t.leaf, Op::Load, Type::I32, {t.ptr(p, 4, 128)});
    t.store(t.ptr(t.array, 4, 128), v); t.finish();
    require(!makePolyhedralPass()->run(t.m), "unknown alias accepted"); }
  { Fixture t; auto *callee = t.m.addFunction("opaque", Type::Void, {}, true);
    emit(t.leaf, Op::Call, Type::Void, {callee});
    t.store(t.ptr(t.array, 4, 128), t.m.constInt(1)); t.finish();
    require(!makePolyhedralPass()->run(t.m), "side-effecting call accepted"); }
}
static void layouts() {
  Module m("layout"); auto *f = m.addFunction("main", Type::I32, {}, false);
  auto *callee = m.addFunction("work", Type::Void, {}, true);
  auto *entry = block(f->blocks, "entry"), *cold = block(f->blocks, "cold");
  auto *hot = block(f->blocks, "hot"), *hot2 = block(f->blocks, "hot2"), *exit = block(f->blocks, "exit");
  emit(entry, Op::CondBr, Type::Void, {m.constInt(1), cold, hot});
  emit(cold, Op::Br, Type::Void, {exit}); emit(hot, Op::Call, Type::Void, {callee});
  emit(hot, Op::Br, Type::Void, {hot2}); emit(hot2, Op::Call, Type::Void, {callee});
  auto *value = emit(hot2, Op::Add, Type::I32, {m.constInt(4), m.constInt(5)});
  emit(hot2, Op::Br, Type::Void, {exit});
  auto *phi = emit(exit, Op::Phi, Type::I32, {cold, m.constInt(2), hot2, value});
  emit(exit, Op::Ret, Type::Void, {phi}); auto inputs = phi->ops;
  require(makeBlockLayoutPass()->run(m), "layout did not change");
  require(f->entry() == entry && f->blocks[1].get() == hot && f->blocks[2].get() == hot2,
          "connected call blocks not clustered");
  require(phi->ops == inputs, "phi identity changed");
  require(!makeBlockLayoutPass()->run(m), "layout is not deterministic/stable");
  // A non-Phi definition in a predecessor must be bound before its phi copy,
  // even though the copy's destination block may be placed earlier.
  Module d("def-order"); auto *g = d.addFunction("main", Type::I32, {}, false);
  auto *a = block(g->blocks, "entry"), *use = block(g->blocks, "use"), *def = block(g->blocks, "def");
  auto *v = emit(def, Op::Add, Type::I32, {d.constInt(1), d.constInt(2)});
  emit(def, Op::Br, Type::Void, {use}); emit(a, Op::Br, Type::Void, {def});
  emit(use, Op::Ret, Type::Void, {v});
  require(makeBlockLayoutPass()->run(d), "definition constraint layout not repaired");
  require(g->blocks[1].get() == def, "use placed before definition");
}
int main() {
  try { interchange(); tiling(); dependenceOracle(); boundTrace(); rejections(); layouts(); }
  catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
  std::cout << "polyhedral and block layout regressions passed\n";
}
