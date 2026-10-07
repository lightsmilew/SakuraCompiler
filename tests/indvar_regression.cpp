#include "midend/pass/Opt.h"
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

using namespace sakura::ir;

static void require(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}

// Execute the scalar CFG independently of the optimizer. Phi inputs are
// evaluated simultaneously, and memory contents depend on the full address.
static int32_t execute(Function &f, int stride, int base = 4096) {
  std::unordered_map<Value *, int64_t> values;
  values[f.args[0].get()] = stride;
  values[f.args[1].get()] = base;
  auto value = [&](Value *v) -> int64_t {
    if (auto *c = dynamic_cast<ConstantInt *>(v)) return c->v;
    auto it = values.find(v);
    require(it != values.end(), "use before definition");
    return it->second;
  };
  BasicBlock *block = f.entry(), *pred = nullptr;
  for (int fuel = 0; fuel < 1000; ++fuel) {
    std::unordered_map<Value *, int64_t> phis;
    for (auto &u : block->instrs) {
      if (u->op != Op::Phi) continue;
      bool found = false;
      for (size_t i = 0; i + 1 < u->ops.size(); i += 2)
        if (u->ops[i] == pred) { phis[u.get()] = value(u->ops[i + 1]); found = true; }
      require(found, "missing phi predecessor");
    }
    for (auto &p : phis) values[p.first] = p.second;
    BasicBlock *next = nullptr;
    for (auto &u : block->instrs) {
      auto &in = *u;
      switch (in.op) {
      case Op::Phi: break;
      case Op::Add: values[u.get()] = int32_t(value(in.ops[0]) + value(in.ops[1])); break;
      case Op::Sub: values[u.get()] = int32_t(value(in.ops[0]) - value(in.ops[1])); break;
      case Op::Mul: values[u.get()] = int32_t(value(in.ops[0]) * value(in.ops[1])); break;
      case Op::SRem: values[u.get()] = int32_t(value(in.ops[0])) % int32_t(value(in.ops[1])); break;
      case Op::Gep: values[u.get()] = value(in.ops[0]) + int32_t(value(in.ops[1])); break;
      case Op::Select: values[u.get()] = value(in.ops[value(in.ops[0]) ? 1 : 2]); break;
      case Op::Load: values[u.get()] = int32_t(value(in.ops[0]) * 17 + 31); break;
      case Op::ICmp:
        switch (in.cond) {
        case Cond::Eq: values[u.get()] = value(in.ops[0]) == value(in.ops[1]); break;
        case Cond::Ne: values[u.get()] = value(in.ops[0]) != value(in.ops[1]); break;
        case Cond::Lt: values[u.get()] = value(in.ops[0]) < value(in.ops[1]); break;
        case Cond::Le: values[u.get()] = value(in.ops[0]) <= value(in.ops[1]); break;
        case Cond::Gt: values[u.get()] = value(in.ops[0]) > value(in.ops[1]); break;
        case Cond::Ge: values[u.get()] = value(in.ops[0]) >= value(in.ops[1]); break;
        }
        break;
      case Op::Br: next = dynamic_cast<BasicBlock *>(in.ops[0]); break;
      case Op::CondBr: next = dynamic_cast<BasicBlock *>(in.ops[value(in.ops[0]) ? 1 : 2]); break;
      case Op::Ret: return int32_t(value(in.ops[0]));
      default: throw std::runtime_error("unsupported test operation");
      }
    }
    require(next != nullptr, "unterminated block");
    pred = block; block = next;
  }
  throw std::runtime_error("loop failed to terminate");
}

static void check(int step, int otherStep, bool runtimeStride, bool stencil, bool quadratic = false) {
  Module m("indvar-test");
  Function *f = m.addFunction("test", Type::I32,
      {{Type::I32, false, "stride"}, {Type::Ptr, true, "base"}}, false);
  auto block = [&](const char *name) {
    f->blocks.push_back(std::make_unique<BasicBlock>(name));
    return f->blocks.back().get();
  };
  auto emit = [](BasicBlock *b, Op op, Type ty, std::vector<Value *> ops) {
    auto u = std::make_unique<Instruction>(op, ty); u->ops = std::move(ops);
    Instruction *p = u.get(); b->instrs.push_back(std::move(u)); return p;
  };
  BasicBlock *pre = block("pre"), *head = block("head"), *body = block("body");
  BasicBlock *left = block("continue"), *right = block("latch"), *exit = block("exit");
  Value *stride = runtimeStride ? (Value *)f->args[0].get() : m.constInt(8);
  if (runtimeStride) {
    auto *negative = emit(pre, Op::ICmp, Type::I32, {stride, m.constInt(0)});
    negative->cond = Cond::Lt;
    stride = emit(pre, Op::Select, Type::I32, {negative, m.constInt(-12), m.constInt(28)});
  }
  // An invariant expression with an embedded constant is shared across rows.
  auto *row = emit(pre, Op::Add, Type::I32, {stride, m.constInt(5)});
  emit(pre, Op::Br, Type::Void, {head});
  auto *iv = emit(head, Op::Phi, Type::I32, {});
  auto *sum = emit(head, Op::Phi, Type::I32, {});
  auto *cmp = emit(head, Op::ICmp, Type::I32, {iv, m.constInt(step > 0 ? 8 : 0)});
  cmp->cond = step > 0 ? Cond::Lt : Cond::Gt;
  emit(head, Op::CondBr, Type::Void, {cmp, body, exit});
  Value *total = sum;
  const int offsets[] = {-251004, -250000, -1000, -4, 0, 4, 1000, 250000};
  for (int displacement : offsets) {
    if (!stencil && displacement != 0) continue;
    auto *scaled = emit(body, Op::Mul, Type::I32, {iv, stride});
    if (quadratic) scaled = emit(body, Op::Mul, Type::I32, {scaled, iv});
    auto *shift = emit(body, Op::Add, Type::I32, {row, m.constInt(displacement)});
    auto *off = emit(body, Op::Add, Type::I32, {scaled, shift});
    auto *address = emit(body, Op::Gep, Type::Ptr, {f->args[1].get(), off});
    auto *loaded = emit(body, Op::Load, Type::I32, {address});
    total = emit(body, Op::Add, Type::I32, {total, loaded});
  }
  auto *odd = emit(body, Op::SRem, Type::I32, {iv, m.constInt(2)});
  emit(body, Op::CondBr, Type::Void, {odd, left, right});
  auto *a = emit(left, Op::Add, Type::I32, {iv, m.constInt(step)});
  emit(left, Op::Br, Type::Void, {head});
  auto *b = emit(right, Op::Add, Type::I32, {iv, m.constInt(otherStep)});
  emit(right, Op::Br, Type::Void, {head});
  iv->ops = {pre, m.constInt(step > 0 ? 0 : 8), left, a, right, b};
  sum->ops = {pre, m.constInt(0), left, total, right, total};
  emit(exit, Op::Ret, Type::Void, {sum});
  int32_t before[3]; int strides[] = {8, -12, 28};
  for (int i = 0; i < 3; ++i) before[i] = execute(*f, strides[i]);
  bool changed = makeIndVarPass(Layer::Cf)->run(m);
  require(changed == (step == otherStep && !quadratic), "unexpected eligibility of non-affine loop");
  for (int i = 0; i < 3; ++i)
    require(before[i] == execute(*f, strides[i]), "strength reduction changed memory addresses");
  size_t pointers = 0;
  for (auto &u : head->instrs) if (u->op == Op::Phi && u->ty == Type::Ptr) ++pointers;
  if (changed) require(pointers == (stencil ? 3u : 1u), "stencil pointer sharing exceeded budget");
}

static void checkIfConv(bool minMax) {
  Module m("ifconv-test");
  auto *f = m.addFunction("test", Type::I32,
      {{Type::I32, false, "x"}, {Type::I32, false, "y"}}, false);
  auto block = [&](const char *name) {
    f->blocks.push_back(std::make_unique<BasicBlock>(name)); return f->blocks.back().get();
  };
  auto emit = [](BasicBlock *b, Op op, Type ty, std::vector<Value *> ops) {
    auto u = std::make_unique<Instruction>(op, ty); u->ops = std::move(ops);
    auto *p = u.get(); b->instrs.push_back(std::move(u)); return p;
  };
  auto *entry = block("entry"), *left = block("left"), *right = block("right"), *merge = block("merge");
  auto *x = emit(entry, Op::Add, Type::I32, {f->args[0].get(), m.constInt(0)});
  auto *y = emit(entry, Op::Add, Type::I32, {f->args[1].get(), m.constInt(0)});
  Value *cond = x;
  if (minMax) {
    auto *cmp = emit(entry, Op::ICmp, Type::I32, {x, y}); cmp->cond = Cond::Lt; cond = cmp;
  }
  emit(entry, Op::CondBr, Type::Void, {cond, left, right});
  emit(left, Op::Br, Type::Void, {merge}); emit(right, Op::Br, Type::Void, {merge});
  auto *phi = emit(merge, Op::Phi, Type::I32, {left, y, right, x});
  emit(merge, Op::Ret, Type::Void, {phi});
  int inputs[][2] = {{-7, 3}, {3, -7}, {3, 3}, {0, -7}, {INT32_MIN, INT32_MAX}};
  int32_t before[5];
  for (int i = 0; i < 5; ++i) before[i] = execute(*f, inputs[i][0], inputs[i][1]);
  require(makeIfConvPass()->run(m) == !minMax, "incorrect target cost for min/max selection");
  for (int i = 0; i < 5; ++i)
    require(before[i] == execute(*f, inputs[i][0], inputs[i][1]), "if conversion changed result");
}

static void pointerEnd(int initial, int bound, int step, int coefficient,
                       bool observeIV, bool duplicate, bool expected, bool pointerExpected = true,
                       int displacement = 0, int factor = 1) {
  Module m("pointer-end");
  auto *f = m.addFunction("test", Type::I32,
      {{Type::I32, false, "unused"}, {Type::Ptr, true, "base"}}, false);
  auto block = [&](const char *name) {
    f->blocks.push_back(std::make_unique<BasicBlock>(name)); return f->blocks.back().get();
  };
  auto emit = [](BasicBlock *b, Op op, Type ty, std::vector<Value *> ops) {
    auto u = std::make_unique<Instruction>(op, ty); u->ops = std::move(ops);
    auto *p = u.get(); b->instrs.push_back(std::move(u)); return p;
  };
  auto *pre = block("pre"), *head = block("head"), *body = block("body"), *exit = block("exit");
  emit(pre, Op::Br, Type::Void, {head});
  auto *i = emit(head, Op::Phi, Type::I32, {});
  auto *j = duplicate ? emit(head, Op::Phi, Type::I32, {}) : i;
  auto *sum = emit(head, Op::Phi, Type::I32, {});
  auto *cmp = emit(head, Op::ICmp, Type::I32, {i, m.constInt(bound)});
  cmp->cond = step > 0 ? Cond::Lt : Cond::Gt;
  emit(head, Op::CondBr, Type::Void, {cmp, body, exit});
  auto *off = emit(body, Op::Mul, Type::I32, {j, m.constInt(coefficient)});
  auto *ptr = emit(body, Op::Gep, Type::Ptr, {f->args[1].get(), off});
  auto *load = emit(body, Op::Load, Type::I32, {ptr});
  auto *total = emit(body, Op::Add, Type::I32, {sum, load});
  auto *next = emit(body, Op::Add, Type::I32, {i, m.constInt(step)});
  i->ops = {pre, m.constInt(initial), body, next};
  if (duplicate) {
    auto *n = emit(body, Op::Add, Type::I32, {j, m.constInt(step*factor)});
    j->ops = {pre, m.constInt(initial*factor+displacement), body, n};
  }
  sum->ops = {pre, m.constInt(0), body, total};
  emit(body, Op::Br, Type::Void, {head});
  emit(exit, Op::Ret, Type::Void, {observeIV ? (Value *)i : sum});
  int32_t before[2];
  for (int t = 0; t < 2; ++t) before[t] = execute(*f, 0, t ? -100 : 4096);
  bool changed = makeIndVarPass(Layer::Cf)->run(m);
  require(changed == expected, "pointer end eligibility incorrect");
  for (int t = 0; t < 2; ++t)
    require(before[t] == execute(*f, 0, t ? -100 : 4096), "pointer end changed result/trip count");
  if (expected) {
    require((cmp->cond == Cond::Ne) == (!observeIV && pointerExpected), "counter elimination lost observable IV");
    size_t integers = 0;
    for (auto &u : head->instrs) integers += u->op == Op::Phi && u->ty == Type::I32;
    require(integers == (pointerExpected ? 1u : 3u), "redundant induction recurrence retained");
  }
}

static void objectBounds(bool conditionalAccess, bool unknownCaller) {
  Module m("object-bounds");
  auto *f = m.addFunction("test", Type::I32,
      {{Type::I32, false, "n"}, {Type::Ptr, true, "base"}}, false);
  auto block = [](Function *fn, const char *name) {
    fn->blocks.push_back(std::make_unique<BasicBlock>(name)); return fn->blocks.back().get();
  };
  auto emit = [](BasicBlock *b, Op op, Type ty, std::vector<Value *> ops) {
    auto u = std::make_unique<Instruction>(op, ty); u->ops = std::move(ops);
    auto *p = u.get(); b->instrs.push_back(std::move(u)); return p;
  };
  auto *pre=block(f,"pre"), *head=block(f,"head"), *body=block(f,"body");
  auto *access=block(f,"access"), *skip=block(f,"skip"), *latch=block(f,"latch"), *exit=block(f,"exit");
  emit(pre,Op::Br,Type::Void,{head});
  auto *i=emit(head,Op::Phi,Type::I32,{}), *sum=emit(head,Op::Phi,Type::I32,{});
  auto *cmp=emit(head,Op::ICmp,Type::I32,{i,f->args[0].get()}); cmp->cond=Cond::Lt;
  emit(head,Op::CondBr,Type::Void,{cmp,body,exit});
  if (conditionalAccess) {
    auto *odd=emit(body,Op::SRem,Type::I32,{i,m.constInt(2)});
    emit(body,Op::CondBr,Type::Void,{odd,access,skip});
  } else emit(body,Op::Br,Type::Void,{access});
  auto *off=emit(access,Op::Mul,Type::I32,{i,m.constInt(4)});
  auto *gep=emit(access,Op::Gep,Type::Ptr,{f->args[1].get(),off});
  auto *load=emit(access,Op::Load,Type::I32,{gep});
  auto *total=emit(access,Op::Add,Type::I32,{sum,load});
  emit(access,Op::Br,Type::Void,{latch});
  if (conditionalAccess) emit(skip,Op::Br,Type::Void,{latch});
  else emit(skip,Op::Ret,Type::Void,{m.constInt(0)});
  auto *merge=emit(latch,Op::Phi,Type::I32,conditionalAccess ? std::vector<Value *>{access,total,skip,sum}
                                                         : std::vector<Value *>{access,total});
  auto *next=emit(latch,Op::Add,Type::I32,{i,m.constInt(1)});
  emit(latch,Op::Br,Type::Void,{head});
  i->ops={pre,m.constInt(0),latch,next}; sum->ops={pre,m.constInt(0),latch,merge};
  emit(exit,Op::Ret,Type::Void,{sum});
  auto *main=m.addFunction("main",Type::I32,{},false); auto *entry=block(main,"entry");
  auto *data=m.addGlobal("data",Type::I32,64,false,false,{});
  emit(entry,Op::Call,Type::I32,{f,m.constInt(8),data}); emit(entry,Op::Ret,Type::Void,{m.constInt(0)});
  if (unknownCaller) {
    auto *caller=m.addFunction("unknown",Type::I32,{{Type::Ptr,true,"other"}},false);
    auto *bb=block(caller,"entry");
    emit(bb,Op::Call,Type::I32,{f,m.constInt(8),caller->args[0].get()});
    emit(bb,Op::Ret,Type::Void,{m.constInt(0)});
  }
  int32_t expected[9]; for(int n=0;n<=8;++n) expected[n]=execute(*f,n);
  bool changed=makeIndVarPass(Layer::Cf)->run(m);
  require(changed==!conditionalAccess,"conditional offset recurrence added latch work");
  for(int n=0;n<=8;++n) require(execute(*f,n)==expected[n],"object proof changed valid accesses");
  size_t pointers=0; for(auto &u:head->instrs) pointers+=u->op==Op::Phi && u->ty==Type::Ptr;
  require(pointers==(!conditionalAccess && !unknownCaller ? 1u : 0u),
          "object proof ignored an unknown caller or skipped access");
  if (conditionalAccess) {
    size_t integers=0; for(auto &u:head->instrs) integers+=u->op==Op::Phi && u->ty==Type::I32;
    require(integers==2,"skipped-access loop gained a live offset phi");
  }
}

int main() {
  try {
    check(1, 1, true, false);
    check(-1, -1, true, false);
    check(1, 2, true, false);
    check(1, 1, false, true);
    check(1, 1, true, true);
    check(1, 1, true, false, true);
    checkIfConv(true);
    checkIfConv(false);
    pointerEnd(0, 17, 1, 4, false, false, true);
    pointerEnd(0, 17, 3, 4, false, true, true);
    pointerEnd(1, 17, 1, 4, false, true, true, true, -1, 1);
    pointerEnd(0, 17, 1, 4, false, true, true, true, 32, -1);
    pointerEnd(-17, 0, 2, 4, false, false, true);
    pointerEnd(17, 0, -2, -4, false, false, true);
    pointerEnd(20, 17, 1, 4, false, false, true);
    pointerEnd(0, 17, 1, 4, true, false, true);
    pointerEnd(0, 4, 1, 1073741824, false, false, true, false);
    pointerEnd(INT32_MAX-2, INT32_MAX, 1, 4, false, false, true, false);
    objectBounds(false,false); objectBounds(true,false); objectBounds(false,true);
    std::cout << "induction regression tests passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n'; return 1;
  }
}
