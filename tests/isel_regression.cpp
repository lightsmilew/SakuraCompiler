#include "backend/ISel.h"
#include "midend/pass/Opt.h"
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

using namespace sakura::ir;
using namespace sakura::backend;

static void require(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}

static bool compare(Cond c, int64_t a, int64_t b) {
  switch (c) {
  case Cond::Eq: return a == b;
  case Cond::Ne: return a != b;
  case Cond::Lt: return a < b;
  case Cond::Le: return a <= b;
  case Cond::Gt: return a > b;
  case Cond::Ge: return a >= b;
  }
  throw std::runtime_error("invalid comparison");
}

// Execute selected virtual-register CFG, including fall-through and phi copies.
static int32_t execute(const MachineFunc &f, int32_t argument = 0) {
  std::unordered_map<int, int64_t> r;
  std::unordered_map<std::string, size_t> labels;
  for (size_t b = 0; b < f.blocks.size(); ++b) labels[f.blocks[b].name] = b;
  auto get = [&](int v) -> int64_t { return v < 0 ? 0 : r.at(v); };
  size_t bi = 0;
  for (int fuel = 0; fuel < 100; ++fuel) {
    size_t next = bi + 1;
    for (const auto &m : f.blocks.at(bi).instrs) {
      int64_t a = get(m.a), b = get(m.b);
      bool jump = false;
      switch (m.op) {
      case MOp::Prologue: break;
      case MOp::EntryInt: r[m.dst] = argument; break;
      case MOp::Li: r[m.dst] = m.imm; break;
      case MOp::MoveX: r[m.dst] = a; break;
      case MOp::IAddI: r[m.dst] = m.ty == Type::Ptr ? a + m.imm
                                                  : int64_t(int32_t(uint32_t(a) + uint32_t(m.imm))); break;
      case MOp::IAdd: r[m.dst] = m.ty == Type::Ptr ? a + b
                                                 : int64_t(int32_t(uint32_t(a) + uint32_t(b))); break;
      case MOp::LeaFrame: r[m.dst] = 0x123456780000LL + m.imm; break;
      case MOp::LeaGlobal: r[m.dst] = 0x123456780000LL; break;
      case MOp::Call:
        require(m.sym == "pointer_probe" && m.args.size() == 1, "unsupported call fixture");
        r[m.dst] = int32_t(get(m.args[0].vreg)); break;
      case MOp::SrlI: r[m.dst] = int32_t(uint32_t(a) >> m.imm); break;
      case MOp::IDiv: r[m.dst] = int32_t(a) / int32_t(b); break;
      case MOp::ISlti: r[m.dst] = a < m.imm; break;
      case MOp::ISlt: r[m.dst] = a < b; break;
      case MOp::ISltiu: r[m.dst] = uint64_t(a) < uint64_t(int64_t(m.imm)); break;
      case MOp::IXorI: r[m.dst] = a ^ int64_t(m.imm); break;
      case MOp::ICmp: r[m.dst] = compare(Cond(m.imm), a, b); break;
      case MOp::BrNz: jump = a != 0; break;
      case MOp::BrZ: jump = a == 0; break;
      case MOp::BrCmp: jump = compare(Cond(m.imm), a, b); break;
      case MOp::Jmp: jump = true; break;
      case MOp::Ret: return m.a < 0 ? m.imm : int32_t(a);
      default: throw std::runtime_error("unsupported selector fixture opcode");
      }
      if (jump) { next = labels.at(m.sym); break; }
    }
    bi = next;
  }
  throw std::runtime_error("selector fixture failed to terminate");
}

static BasicBlock *block(Function *f, const char *name) {
  f->blocks.push_back(std::make_unique<BasicBlock>(name));
  return f->blocks.back().get();
}

static Instruction *emit(BasicBlock *b, Op op, Type ty, std::vector<Value *> ops) {
  auto u = std::make_unique<Instruction>(op, ty);
  u->ops = std::move(ops);
  auto *p = u.get(); b->instrs.push_back(std::move(u)); return p;
}

static void branches() {
  for (Cond c : {Cond::Lt, Cond::Le, Cond::Gt, Cond::Ge, Cond::Eq, Cond::Ne}) {
    for (int bound : {-2048, -1, 0, 1, 2047, 10000}) {
      Module mod("branch");
      auto *f = mod.addFunction("test", Type::I32, {{Type::I32, false, "x"}}, false);
      auto *e = block(f, "entry"), *yes = block(f, "yes"), *no = block(f, "no");
      auto *cmp = emit(e, Op::ICmp, Type::I32, {f->args[0].get(), mod.constInt(bound)});
      cmp->cond = c;
      emit(e, Op::CondBr, Type::Void, {cmp, yes, no});
      emit(yes, Op::Ret, Type::Void, {mod.constInt(1)});
      emit(no, Op::Ret, Type::Void, {mod.constInt(0)});
      auto mf = InstructionSelector().select(&mod)[0];
      if ((bound == 1 && (c == Cond::Lt || c == Cond::Ge)) ||
          (bound == 0 && (c == Cond::Le || c == Cond::Gt))) {
        bool zeroBranch = false;
        for (const auto &m : mf.blocks[0].instrs) {
          zeroBranch |= m.op == MOp::BrCmp && m.b < 0;
          require(m.op != MOp::Li || m.imm != 1,
                  "zero threshold unnecessarily materialized one");
        }
        require(zeroBranch, "signed zero threshold lost direct branch");
      }
      if ((bound == -2048 || bound == -1 || bound == 2047) &&
          (c == Cond::Lt || c == Cond::Ge)) {
        bool immediateCompare = false;
        for (const auto &m : mf.blocks[0].instrs) {
          immediateCompare |= m.op == MOp::ISlti && m.imm == bound;
          require(m.op != MOp::Li || m.imm != bound,
                  "small bound increased comparison register pressure");
          require(m.op != MOp::IXorI || m.imm != 1,
                  "branch retained a redundant boolean inversion");
        }
        require(immediateCompare, "small bound lost immediate comparison");
      }
      for (int x : {INT32_MIN, -2049, -1, 0, 1, 2048, 10000, INT32_MAX})
        require(execute(mf, x) == compare(c, x, bound), "fused comparison changed result");
    }
  }
}

static void phiClobber() {
  Module mod("phi-clobber");
  auto *f = mod.addFunction("test", Type::I32, {}, false);
  auto *e = block(f, "entry"), *loop = block(f, "loop"), *exit = block(f, "exit");
  emit(e, Op::Br, Type::Void, {loop});
  auto *iv = emit(loop, Op::Phi, Type::I32, {});
  auto *next = emit(loop, Op::Add, Type::I32, {iv, mod.constInt(1)});
  auto *cmp = emit(loop, Op::ICmp, Type::I32, {iv, mod.constInt(3)});
  cmp->cond = Cond::Lt;
  emit(loop, Op::CondBr, Type::Void, {cmp, loop, exit});
  iv->ops = {e, mod.constInt(0), loop, next};
  emit(exit, Op::Ret, Type::Void, {next});
  require(execute(InstructionSelector().select(&mod)[0]) == 4,
          "fusion read phi operand after its back-edge copy");
}

static void swappedGuard() {
  Module mod("swapped-guard");
  auto *f = mod.addFunction("test", Type::I32, {}, false);
  auto *e = block(f, "entry"), *head = block(f, "head");
  auto *check = block(f, "check"), *latch = block(f, "latch"), *exit = block(f, "exit");
  emit(e, Op::Br, Type::Void, {head});
  auto *iv = emit(head, Op::Phi, Type::I32, {});
  auto *upper = emit(head, Op::ICmp, Type::I32, {mod.constInt(INT32_MAX), iv});
  upper->cond = Cond::Ge;
  emit(head, Op::CondBr, Type::Void, {upper, check, exit});
  auto *negative = emit(check, Op::ICmp, Type::I32, {iv, mod.constInt(0)});
  negative->cond = Cond::Lt;
  emit(check, Op::CondBr, Type::Void, {negative, exit, latch});
  auto *next = emit(latch, Op::Add, Type::I32, {iv, mod.constInt(1)});
  emit(latch, Op::Br, Type::Void, {head});
  iv->ops = {e, mod.constInt(INT32_MAX - 1), latch, next};
  auto *q = emit(exit, Op::SDiv, Type::I32, {iv, mod.constInt(2)});
  emit(exit, Op::Ret, Type::Void, {q});
  require(execute(InstructionSelector().select(&mod)[0]) == INT32_MIN / 2,
          "swapped inclusive guard incorrectly excluded overflow");
}

static void conditionalPhiEdges() {
  for (int mode = 0; mode < 4; ++mode) {
    Module mod("conditional-phi-edge");
    auto *f = mod.addFunction("test", Type::I32, {}, false);
    auto *e = block(f, "entry"), *loop = block(f, "loop"), *exit = block(f, "exit");
    emit(e, Op::Br, Type::Void, {loop});
    auto *iv = emit(loop, Op::Phi, Type::I32, {});
    auto *next = emit(loop, Op::Add, Type::I32,
                      {iv, mod.constInt(mode == 2 ? -1 : 1)});
    Value *condition;
    if (mode == 2) condition = iv;
    else if (mode == 3) condition = mod.constInt(0);
    else {
      auto *cmp = emit(loop, Op::ICmp, Type::I32, {iv, mod.constInt(3)});
      cmp->cond = Cond::Lt; condition = cmp;
    }
    emit(loop, Op::CondBr, Type::Void, {condition, loop, exit});
    iv->ops = {e, mod.constInt(mode >= 2 ? 3 : 0), loop, next};
    Value *result = mode == 2 ? next : iv;
    if (mode == 1) {
      auto *merge = emit(exit, Op::Phi, Type::I32, {loop, iv});
      result = merge;
    }
    emit(exit, Op::Ret, Type::Void, {result});
    require(execute(InstructionSelector().select(&mod)[0]) == (mode == 2 ? -1 : 3),
            "conditional phi copy overwrote a condition or exit value");
  }
}

static void constantInlining() {
  Module mod("constant-inlining");
  auto *dispatch = mod.addFunction("dispatch", Type::I32,
      {{Type::I32, false, "n"}}, false);
  auto *e = block(dispatch, "entry"), *one = block(dispatch, "one");
  auto *testTwo = block(dispatch, "test_two"), *two = block(dispatch, "two");
  auto *other = block(dispatch, "other");
  Value *chain = dispatch->args[0].get();
  for (int k = 0; k < 80; ++k)
    chain = emit(e, Op::Add, Type::I32, {chain, mod.constInt(1)});
  auto *c1 = emit(e, Op::ICmp, Type::I32, {dispatch->args[0].get(), mod.constInt(1)});
  c1->cond = Cond::Eq;
  emit(e, Op::CondBr, Type::Void, {c1, one, testTwo});
  emit(one, Op::Ret, Type::Void, {mod.constInt(10)});
  auto *c2 = emit(testTwo, Op::ICmp, Type::I32, {dispatch->args[0].get(), mod.constInt(2)});
  c2->cond = Cond::Eq;
  emit(testTwo, Op::CondBr, Type::Void, {c2, two, other});
  emit(two, Op::Ret, Type::Void, {mod.constInt(20)});
  emit(other, Op::Ret, Type::Void, {mod.constInt(30)});

  auto *large = mod.addFunction("large_arithmetic", Type::I32,
      {{Type::I32, false, "n"}}, false);
  auto *le = block(large, "entry");
  chain = large->args[0].get();
  for (int k = 0; k < 80; ++k)
    chain = emit(le, Op::Add, Type::I32, {chain, mod.constInt(1)});
  emit(le, Op::Ret, Type::Void, {chain});

  auto *caller = mod.addFunction("caller", Type::I32, {}, false);
  auto *ce = block(caller, "entry");
  auto *call = emit(ce, Op::Call, Type::I32, {dispatch, mod.constInt(2)});
  auto *kept = emit(ce, Op::Call, Type::I32, {large, mod.constInt(32)});
  emit(ce, Op::Ret, Type::Void, {call});
  require(makeInlineGeneralPass(true)->run(mod), "constant dispatch was not specialized");
  int calls = 0, returns = 0;
  for (auto &bb : caller->blocks)
    for (auto &iu : bb->instrs) {
      if (iu->op == Op::Call) {
        ++calls;
        require(iu.get() == kept, "large arithmetic helper gained specialization budget");
      }
      if (iu->op == Op::Ret) ++returns;
    }
  require(calls == 1 && returns == 1, "constant inline damaged caller control flow");
  // Ret merges the three cloned return flows, each keyed on its cloned block.
  bool merge = false;
  for (auto &bb : caller->blocks)
    for (auto &iu : bb->instrs)
      if (iu->op == Op::Phi && iu->ops.size() == 6) merge = true;
  require(merge, "constant inline lost multi-return merge");
}

static void promotedPhiLifetime() {
  Module mod("promoted-phi-lifetime");
  auto *f = mod.addFunction("test", Type::I32, {{Type::I32, false, "condition"}}, false);
  auto *e = block(f, "entry"), *yes = block(f, "yes");
  auto *no = block(f, "no"), *join = block(f, "join");
  auto *a = emit(e, Op::Alloca, Type::Ptr, {});
  auto *b = emit(e, Op::Alloca, Type::Ptr, {});
  a->n = b->n = 1; a->elem = b->elem = Type::I32;
  emit(e, Op::Store, Type::Void, {mod.constInt(42), a});
  emit(e, Op::CondBr, Type::Void, {f->args[0].get(), yes, no});
  for (auto *arm : {yes, no}) {
    auto *v = emit(arm, Op::Load, Type::I32, {a});
    emit(arm, Op::Store, Type::Void, {v, b});
    emit(arm, Op::Br, Type::Void, {join});
  }
  auto *v = emit(join, Op::Load, Type::I32, {b});
  emit(join, Op::Ret, Type::Void, {v});
  require(makeMem2RegPass()->run(mod), "fixture slots were not promoted");
  require(!makeMem2RegPass()->run(mod), "promotion was not idempotent");
  auto mf = InstructionSelector().select(&mod)[0];
  require(execute(mf, 0) == 42 && execute(mf, 1) == 42,
          "folding a promoted phi lost its replacement value");
}

static void pointerComparisons() {
  for (bool global : {false, true}) {
    Module mod("pointer-comparison");
    auto *f = mod.addFunction("test", Type::I32, {{Type::I32, false, "condition"}}, false);
    auto *e = block(f, "entry"), *yes = block(f, "yes"), *no = block(f, "no");
    Value *base;
    if (global) base = mod.addGlobal("data", Type::I32, 8, false, false, {});
    else { auto *a = emit(e, Op::Alloca, Type::Ptr, {}); a->n = 8; base = a; }
    auto *p = emit(e, Op::Gep, Type::Ptr, {base, mod.constInt(12)});
    emit(e, Op::CondBr, Type::Void, {f->args[0].get(), yes, no});
    for (auto *arm : {yes, no}) {
      auto *q = emit(arm, Op::Gep, Type::Ptr, {base, mod.constInt(12)});
      auto *cmp = emit(arm, Op::ICmp, Type::I32, {p, q}); cmp->cond = Cond::Eq;
      emit(arm, Op::Ret, Type::Void, {cmp});
    }
    auto mf = InstructionSelector().select(&mod)[0];
    require(execute(mf, 0) == 1 && execute(mf, 1) == 1,
            "lazy pointer comparison unbound or cached across sibling blocks");
  }
  Module mod("pointer-call-siblings");
  auto *probe=mod.addFunction("pointer_probe",Type::I32,{{Type::Ptr,true,"p"}},true);
  auto *f=mod.addFunction("test",Type::I32,{{Type::Ptr,true,"base"}},false);
  auto *e=block(f,"entry"), *yes=block(f,"yes"), *no=block(f,"no");
  auto *p=emit(e,Op::Gep,Type::Ptr,{f->args[0].get(),mod.constInt(12)});
  emit(e,Op::CondBr,Type::Void,{f->args[0].get(),yes,no});
  for(auto *arm:{yes,no}) {
    auto *result=emit(arm,Op::Call,Type::I32,{probe,p});
    emit(arm,Op::Ret,Type::Void,{result});
  }
  auto mf=InstructionSelector().select(&mod)[0];
  require(execute(mf,0)==12 && execute(mf,4096)==4108,
          "pointer call cached an address temporary across sibling blocks");
}

int main() {
  try {
    branches(); phiClobber(); swappedGuard(); conditionalPhiEdges();
    constantInlining(); promotedPhiLifetime(); pointerComparisons();
    std::cout << "selector regression tests passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n'; return 1;
  }
}
