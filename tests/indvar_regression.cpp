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
        values[u.get()] = in.cond == Cond::Lt ? value(in.ops[0]) < value(in.ops[1])
                                             : value(in.ops[0]) > value(in.ops[1]);
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
    std::cout << "induction regression tests passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n'; return 1;
  }
}
