#include "midend/pass/Opt.h"
#include "midend/pass/OptUtil.h"
#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>
#include <unordered_map>
using namespace sakura::ir;
static void require(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
static Instruction *emit(BasicBlock *b, Op op, Type ty, std::vector<Value *> ops = {}) {
  auto u = std::make_unique<Instruction>(op, ty); u->ops = std::move(ops);
  auto *p = u.get(); b->instrs.push_back(std::move(u)); return p;
}
static BasicBlock *block(BlockList &bs, const char *name) {
  auto u = std::make_unique<BasicBlock>(name); auto *p = u.get(); bs.push_back(std::move(u)); return p;
}
static bool cmp(Cond c, int32_t a, int32_t b) {
  switch (c) {
  case Cond::Eq: return a == b; case Cond::Ne: return a != b;
  case Cond::Lt: return a < b; case Cond::Le: return a <= b;
  case Cond::Gt: return a > b; case Cond::Ge: return a >= b;
  } return false;
}
struct Result {
  std::map<std::pair<Value *, int64_t>, int32_t> memory;
  std::vector<int32_t> calls;
};
// Independent IR execution oracle, including effect order and final IV values.
static Result execute(Function &f, std::unordered_map<Value *, int32_t> val = {}) {
  Result r; int fuel = 100000;
  auto value = [&](Value *v) {
    if (auto *c = asConstInt(v)) return c->v;
    require(val.count(v), "undefined SSA value in split body"); return val.at(v);
  };
  std::function<std::pair<Value *, int64_t>(Value *)> address = [&](Value *v) {
    auto *i = dynamic_cast<Instruction *>(v);
    if (i && i->op == Op::Gep) { auto a = address(i->ops[0]); a.second += value(i->ops[1]); return a; }
    return std::make_pair(v, int64_t(0));
  };
  std::function<void(BasicBlock *)> run = [&](BasicBlock *b) {
    while (b) {
      require(--fuel > 0, "interpreter fuel exhausted"); BasicBlock *next = nullptr;
      for (auto &u : b->instrs) {
        auto &i = *u;
        switch (i.op) {
        case Op::Alloca: r.memory[address(u.get())] = 0; break;
        case Op::Gep: break;
        case Op::Load: val[&i] = r.memory[address(i.ops[0])]; break;
        case Op::Store: r.memory[address(i.ops[1])] = value(i.ops[0]); break;
        case Op::Add: val[&i] = int32_t(int64_t(value(i.ops[0])) + value(i.ops[1])); break;
        case Op::Sub: val[&i] = int32_t(int64_t(value(i.ops[0])) - value(i.ops[1])); break;
        case Op::Mul: val[&i] = int32_t(int64_t(value(i.ops[0])) * value(i.ops[1])); break;
        case Op::SRem: val[&i] = value(i.ops[0]) % value(i.ops[1]); break;
        case Op::ICmp: val[&i] = cmp(i.cond, value(i.ops[0]), value(i.ops[1])); break;
        case Op::Not: val[&i] = value(i.ops[0]) ^ 1; break;
        case Op::Select: val[&i] = value(i.ops[value(i.ops[0]) ? 1 : 2]); break;
        case Op::Call: for (size_t n = 1; n < i.ops.size(); ++n) r.calls.push_back(value(i.ops[n])); break;
        case Op::Br: next = static_cast<BasicBlock *>(i.ops[0]); break;
        case Op::CondBr: next = static_cast<BasicBlock *>(i.ops[value(i.ops[0]) ? 1 : 2]); break;
        case Op::AffineFor: {
          auto a = address(i.ops[1]); int32_t ub = value(i.ops[3]);
          r.memory[a] = value(i.ops[2]);
          while (r.memory[a] < ub) {
            require(--fuel > 0, "loop fuel exhausted"); run(i.bodyRegion.front().get());
            r.memory[a] = int32_t(int64_t(r.memory[a]) + i.step);
          }
          next = static_cast<BasicBlock *>(i.ops[0]); break;
        }
        case Op::AffineYield: case Op::Ret: return;
        default: throw std::runtime_error("unexpected oracle instruction");
        }
      }
      b = next;
    }
  };
  run(f.entry()); return r;
}
struct Fixture {
  Module m{"split"}; Function *f; Instruction *loop, *iv;
  BasicBlock *body, *exit; Value *sum; Function *callee;
  Fixture(int32_t lb = -12, int32_t ub = 23, bool dynamic = false) {
    f = m.addFunction("main", Type::I32, {}, false);
    callee = m.addFunction("effect", Type::Void, {}, true);
    sum = m.addGlobal("sum", Type::I32, 1, false, false, {});
    auto *entry = block(f->blocks, "entry"); exit = block(f->blocks, "exit");
    iv = emit(entry, Op::Alloca, Type::Ptr); iv->n = 1;
    Value *l = m.constInt(lb), *u = m.constInt(ub);
    if (dynamic) {
      for (int n = 0; n < 2; ++n) { auto a = std::make_unique<Argument>(); a->ty = Type::I32; a->index = n; f->args.push_back(std::move(a)); }
      l = f->args[0].get(); u = f->args[1].get();
    }
    loop = emit(entry, Op::AffineFor, Type::Void, {exit, iv, l, u}); loop->cond = Cond::Lt; loop->step = 1;
    body = block(loop->bodyRegion, "body"); emit(exit, Op::Ret, Type::Void, {m.constInt(0)});
  }
  Value *index(int k = 1, int c = 0) {
    Value *v = emit(body, Op::Load, Type::I32, {iv});
    if (k != 1) v = emit(body, Op::Mul, Type::I32, {v, m.constInt(k)});
    if (c) v = emit(body, Op::Add, Type::I32, {v, m.constInt(c)});
    return v;
  }
  void diamond(Cond cond, int threshold, int k = 1, int c = 0, bool reverse = false,
               bool invert = false, bool unknown = false) {
    Value *v = unknown ? emit(body, Op::Load, Type::I32, {sum}) : index(k, c);
    auto *p = emit(body, Op::ICmp, Type::I32,
                   reverse ? std::vector<Value *>{m.constInt(threshold), v} : std::vector<Value *>{v, m.constInt(threshold)});
    p->cond = cond; Value *test = invert ? emit(body, Op::Not, Type::I32, {p}) : p;
    auto *t = block(loop->bodyRegion, "true"), *e = block(loop->bodyRegion, "false"), *join = block(loop->bodyRegion, "join");
    emit(body, Op::CondBr, Type::Void, {test, t, e});
    for (auto *b : {t, e}) {
      auto *old = emit(b, Op::Load, Type::I32, {sum});
      auto *n = emit(b, Op::Mul, Type::I32, {old, m.constInt(3)});
      n = emit(b, Op::Add, Type::I32, {n, m.constInt(b == t ? 1 : 2)});
      n = emit(b, Op::SRem, Type::I32, {n, m.constInt(997)});
      emit(b, Op::Store, Type::Void, {n, sum});
      emit(b, Op::Call, Type::Void, {callee, n});
      emit(b, Op::Br, Type::Void, {join});
    }
    body = join;
  }
  void finish() { emit(body, Op::AffineYield, Type::Void); }
  void check(bool change, const std::vector<std::pair<int, int>> &ranges = {{0, 0}}) {
    std::unordered_set<Value *> locals;
    for (auto &b : loop->bodyRegion) for (auto &i : b->instrs) if (i->op == Op::Alloca) locals.insert(i.get());
    std::vector<Result> before;
    auto inputs = [&](std::pair<int,int> range) {
      std::unordered_map<Value *, int32_t> v;
      if (!f->args.empty()) { v[f->args[0].get()] = range.first; v[f->args[1].get()] = range.second; }
      return v;
    };
    for (auto p : ranges) before.push_back(execute(*f, inputs(p)));
    require(makeLoopSplitPass()->run(m) == change, "unexpected splitting decision");
    for (size_t n = 0; n < ranges.size(); ++n) {
      auto after = execute(*f, inputs(ranges[n]));
      for (auto &cell : before[n].memory) if (!locals.count(cell.first.first))
        require(after.memory.at(cell.first) == cell.second, "split changed memory/final IV");
      require(after.calls == before[n].calls, "split reordered effects/iterations");
    }
    require(!makeLoopSplitPass()->run(m), "splitting is not stable");
  }
};
static void predicates() {
  for (Cond c : {Cond::Eq, Cond::Ne, Cond::Lt, Cond::Le, Cond::Gt, Cond::Ge})
    for (int k : {-3, -1, 1, 2}) for (bool rev : {false, true}) for (bool inv : {false, true}) {
      Fixture t; t.diamond(c, 8, k, 2, rev, inv); t.finish(); t.check(true);
      size_t branches = 0; walkInstrs(t.f, [&](BasicBlock &, Instruction &i) { branches += i.op == Op::CondBr; });
      require(branches == 0, "proven predicates survived splitting");
    }
  { Fixture t; t.diamond(Cond::Lt, -3); t.diamond(Cond::Eq, 5); t.diamond(Cond::Ge, 11);
    t.diamond(Cond::Lt, 450, 1, 0, false, false, true); t.finish(); t.check(true); }
}
static void dynamic() {
  Fixture t(0, 0, true); t.diamond(Cond::Lt, -3); t.diamond(Cond::Eq, 5); t.diamond(Cond::Ge, 11); t.finish();
  std::vector<std::pair<int,int>> ranges;
  for (int l : {-20,-3,0,5,6,11,12,25}) for (int u : {-21,-3,0,5,6,11,12,30}) ranges.push_back({l,u});
  ranges.push_back({INT32_MIN,INT32_MIN+10}); ranges.push_back({INT32_MAX-10,INT32_MAX});
  t.check(true, ranges);
}
static void predecessorSeed() {
  for (bool branch : {false,true}) {
    Fixture t; auto *entry = t.f->entry();
    auto prefix = std::make_unique<BasicBlock>("seed.predecessor");
    prefix->instrs.push_back(std::move(entry->instrs[0])); entry->instrs.erase(entry->instrs.begin());
    emit(prefix.get(),Op::Store,Type::Void,{t.m.constInt(2),t.iv});
    emit(prefix.get(),Op::Br,Type::Void,{entry}); t.f->blocks.insert(t.f->blocks.begin(),std::move(prefix));
    auto load = std::make_unique<Instruction>(Op::Load,Type::I32); load->ops = {t.iv};
    auto otherStore = std::make_unique<Instruction>(Op::Store,Type::Void);
    otherStore->ops = {t.m.constInt(0),t.sum}; entry->instrs.insert(entry->instrs.begin(),std::move(otherStore));
    t.loop->ops[2] = load.get(); entry->instrs.insert(entry->instrs.begin()+1,std::move(load));
    if (branch) t.diamond(Cond::Lt,5);
    else emit(t.body,Op::Store,Type::Void,{t.index(),t.sum});
    t.finish(); t.check(branch);
  }
}
static void boundaries() {
  { Fixture t(INT32_MIN, INT32_MIN+20); t.diamond(Cond::Le, INT32_MIN+8); t.finish(); t.check(true); }
  { Fixture t(INT32_MAX-20, INT32_MAX); t.diamond(Cond::Eq, INT32_MAX-8); t.finish(); t.check(true); }
  { Fixture t(12, 3); t.diamond(Cond::Lt, 5); t.finish(); t.check(false); }
  { Fixture t(0, 6); t.diamond(Cond::Lt, 3); t.finish(); t.check(false); }
  { Fixture t; t.diamond(Cond::Lt, 500); t.finish(); t.check(false); }
  { Fixture t(0, 0, true); t.diamond(Cond::Lt, 5, 2); t.finish(); t.check(false, {{-8,23}}); }
  { Fixture t(INT32_MAX-20,INT32_MAX); t.diamond(Cond::Lt, -5, 1, 40); t.finish(); t.check(false); }
  { Fixture t; for (int n = 0; n < 8; ++n) t.diamond(Cond::Lt, n); t.finish(); t.check(false); }
}
static void rejectUnsafe() {
  { Fixture t; t.diamond(Cond::Lt, 5); emit(t.body, Op::Call, Type::Void, {t.callee,t.iv}); t.finish();
    require(!makeLoopSplitPass()->run(t.m), "escaped IV accepted"); }
  { Fixture t; t.diamond(Cond::Lt, 5); emit(t.body, Op::Store, Type::Void, {t.m.constInt(0),t.iv}); t.finish();
    require(!makeLoopSplitPass()->run(t.m), "modified IV accepted"); }
  { Fixture t; t.diamond(Cond::Lt, 5); emit(t.body, Op::Br, Type::Void, {t.loop->bodyRegion.front().get()});
    require(!makeLoopSplitPass()->run(t.m), "cyclic body accepted"); }
  { Fixture t; t.diamond(Cond::Lt, 5); auto *v = t.index(); t.finish(); emit(t.exit, Op::Store, Type::Void, {v,t.sum});
    require(!makeLoopSplitPass()->run(t.m), "escaping SSA value accepted"); }
  { Fixture t; t.diamond(Cond::Lt, 5); emit(t.body, Op::Alloca, Type::Ptr); t.finish();
    require(!makeLoopSplitPass()->run(t.m), "body allocation accepted"); }
  { Fixture t; auto *entry = t.f->entry(); auto old = std::make_unique<Instruction>(Op::Load,Type::I32);
    old->ops = {t.iv}; auto *p = old.get(); entry->instrs.insert(entry->instrs.begin()+1,std::move(old));
    t.diamond(Cond::Lt, 5); t.loop->bodyRegion[0]->instrs[1]->ops[0] = p; t.finish(); t.check(false); }
}
static void scalarTemporaries() {
  { Fixture t; auto *x = emit(t.body,Op::Alloca,Type::Ptr); x->n = 1;
    emit(t.body,Op::Store,Type::Void,{t.index(),x});
    t.diamond(Cond::Lt,5);
    auto *v = emit(t.body,Op::Load,Type::I32,{x}); emit(t.body,Op::Call,Type::Void,{t.callee,v});
    t.finish(); t.check(true); }
  for (bool both : {false,true}) {
    Fixture t; t.diamond(Cond::Lt,5);
    auto *x = emit(t.body,Op::Alloca,Type::Ptr); x->n = 1;
    size_t start = t.loop->bodyRegion.size(); t.diamond(Cond::Lt,9);
    for (size_t n : {start,start+1}) if (both || n == start) {
      auto &list = t.loop->bodyRegion[n]->instrs;
      auto store = std::make_unique<Instruction>(Op::Store,Type::Void);
      store->ops = {t.m.constInt(int(n-start)),x}; list.insert(list.end()-1,std::move(store));
    }
    auto *v = emit(t.body,Op::Load,Type::I32,{x}); emit(t.body,Op::Call,Type::Void,{t.callee,v});
    t.finish();
    if (both) t.check(true);
    else require(!makeLoopSplitPass()->run(t.m),"temporary initialized on only one path accepted");
  }
  { Fixture t; auto *x = emit(t.body,Op::Alloca,Type::Ptr); x->n = 1;
    emit(t.body,Op::Store,Type::Void,{t.m.constInt(0),x});
    emit(t.body,Op::Call,Type::Void,{t.callee,x}); t.diamond(Cond::Lt,5); t.finish();
    require(!makeLoopSplitPass()->run(t.m),"escaping temporary accepted"); }
  { Fixture t; auto *x = emit(t.body,Op::Alloca,Type::Ptr); x->n = 1;
    emit(t.body,Op::Load,Type::I32,{x});
    emit(t.body,Op::Store,Type::Void,{t.m.constInt(0),x}); t.diamond(Cond::Lt,5); t.finish();
    require(!makeLoopSplitPass()->run(t.m),"read before initialization accepted"); }
}
static void fullUnrollIV() {
  Fixture t(2,5); auto *v = t.index(); emit(t.body,Op::Store,Type::Void,{v,t.sum}); t.finish();
  auto before = execute(*t.f); require(makeLoopUnrollPass(Layer::Affine)->run(t.m),"small loop not unrolled");
  require(execute(*t.f).memory == before.memory,"full unroll lost final IV");
}
static void partialUnrollBounds() {
  Fixture t(0,0,true); auto *v = t.index();
  auto *old = emit(t.body,Op::Load,Type::I32,{t.sum});
  auto *n = emit(t.body,Op::Add,Type::I32,{old,t.m.constInt(1)});
  emit(t.body,Op::Store,Type::Void,{n,t.sum});
  emit(t.body,Op::Store,Type::Void,{v,t.sum}); t.finish();
  std::vector<std::pair<int,int>> ranges = {{0,-5},{7,3},{-20,-9},{-6,5},{3,17},
    {INT32_MIN,INT32_MIN+9},{INT32_MAX-9,INT32_MAX},{INT32_MIN+3,INT32_MIN}};
  std::vector<Result> before;
  auto args = [&](std::pair<int,int> r) { return std::unordered_map<Value *,int32_t>{{t.f->args[0].get(),r.first},{t.f->args[1].get(),r.second}}; };
  for (auto r : ranges) before.push_back(execute(*t.f,args(r)));
  require(makeLoopUnrollPass(Layer::Affine)->run(t.m),"dynamic loop not partially unrolled");
  for (size_t n = 0; n < ranges.size(); ++n)
    require(execute(*t.f,args(ranges[n])).memory == before[n].memory,"partial unroll changed empty/signed domain");
}
int main() {
  try { predicates(); dynamic(); predecessorSeed(); boundaries(); rejectUnsafe(); scalarTemporaries(); fullUnrollIV(); partialUnrollBounds(); }
  catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
  std::cout << "loop splitting regressions passed\n";
}
