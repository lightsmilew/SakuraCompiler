#include "midend/pass/Opt.h"
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

using namespace sakura::ir;
static void require(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}
static Instruction *emit(BasicBlock *b, Op op, Type ty, std::vector<Value *> operands) {
  auto u = std::make_unique<Instruction>(op, ty); u->ops = std::move(operands);
  auto *p = u.get(); b->instrs.push_back(std::move(u)); return p;
}

// Execute memory effects independently of the pass. Both pointer arguments
// can refer to the same object; the test call writes through its argument.
static int32_t execute(Function &f, bool alias) {
  std::unordered_map<Value *, int64_t> values;
  std::unordered_map<int64_t, int32_t> memory{{4096,7},{8192,23}};
  values[f.args[0].get()] = 4096;
  values[f.args[1].get()] = alias ? 4096 : 8192;
  auto value = [&](Value *v) {
    if (auto *c = dynamic_cast<ConstantInt *>(v)) return int64_t(c->v);
    require(values.count(v), "memory interpreter use before definition");
    return values.at(v);
  };
  auto *block = f.entry(); BasicBlock *pred = nullptr;
  for (int fuel=0; fuel<10; ++fuel) {
    BasicBlock *next = nullptr;
    for (auto &u : block->instrs) {
      switch (u->op) {
      case Op::Alloca:
        values[u.get()] = 12288; memory[12288] = 7; break;
      case Op::Phi:
        require(u->ops[0]==pred, "memory interpreter phi predecessor");
        values[u.get()] = value(u->ops[1]); break;
      case Op::Load: values[u.get()] = memory.at(value(u->ops[0])); break;
      case Op::Store: memory[value(u->ops[1])] = int32_t(value(u->ops[0])); break;
      case Op::Add: values[u.get()] = int32_t(value(u->ops[0])+value(u->ops[1])); break;
      case Op::Call: memory[value(u->ops[1])] = 99; break;
      case Op::Br: next = dynamic_cast<BasicBlock *>(u->ops[0]); break;
      case Op::Ret: return int32_t(value(u->ops[0]));
      default: throw std::runtime_error("unsupported memory interpreter op");
      }
    }
    require(next, "memory interpreter missing successor"); pred = block; block = next;
  }
  throw std::runtime_error("memory interpreter did not terminate");
}

static void checkRestore(int mode) {
  Module m("store-back");
  auto *f=m.addFunction("test",Type::I32,
      {{Type::Ptr,true,"p"},{Type::Ptr,true,"q"}},false);
  auto block = [&] {
    f->blocks.push_back(std::make_unique<BasicBlock>("block"));
    return f->blocks.back().get();
  };
  auto *entry=block();
  auto *p=f->args[0].get(); auto *q=f->args[1].get();
  auto *saved=emit(entry,Op::Load,Type::I32,{p});
  BasicBlock *body=entry;
  if (mode==3) {
    body=block();
    emit(entry,Op::Br,Type::Void,{body});
    emit(body,Op::Store,Type::Void,{m.constInt(99),q});
  } else if (mode==1) {
    emit(body,Op::Store,Type::Void,{m.constInt(99),q});
  } else if (mode==2) {
    auto *writer=m.addFunction("writer",Type::Void,{{Type::Ptr,true,"p"}},true);
    emit(body,Op::Call,Type::Void,{writer,q});
  } else {
    emit(body,Op::Add,Type::I32,{saved,m.constInt(1)});
  }
  emit(body,Op::Store,Type::Void,{saved,p});
  auto *result=emit(body,Op::Load,Type::I32,{p});
  emit(body,Op::Ret,Type::Void,{result});
  const int32_t before[]={execute(*f,false),execute(*f,true)};
  auto pass=makeRedundantStorePass(Layer::Cf);
  pass->run(m); pass->run(m);
  require(execute(*f,false)==before[0] && execute(*f,true)==before[1],
          "store-back removal changed aliased memory");
  size_t stores=0;
  for (auto &b:f->blocks) for (auto &u:b->instrs) stores+=u->op==Op::Store;
  require(stores==(mode==0 ? 0u : mode==2 ? 1u : 2u),
          "store-back eligibility ignored a clobber");
}
static void checkPrivateAlias(bool restore) {
  Module m("private-alias");
  auto *f=m.addFunction("test",Type::I32,
      {{Type::Ptr,true,"unused_p"},{Type::Ptr,true,"unused_q"}},false);
  f->blocks.push_back(std::make_unique<BasicBlock>("entry"));
  f->blocks.push_back(std::make_unique<BasicBlock>("body"));
  auto *entry=f->blocks[0].get(), *body=f->blocks[1].get();
  auto *p=emit(entry,Op::Alloca,Type::Ptr,{}); p->n=1;
  emit(entry,Op::Br,Type::Void,{body});
  auto *q=emit(body,Op::Phi,Type::Ptr,{entry,p});
  Value *result=nullptr;
  if (restore) {
    auto *saved=emit(body,Op::Load,Type::I32,{p});
    emit(body,Op::Store,Type::Void,{m.constInt(99),q});
    emit(body,Op::Store,Type::Void,{saved,p});
    result=emit(body,Op::Load,Type::I32,{p});
  } else {
    emit(body,Op::Store,Type::Void,{m.constInt(42),p});
    result=emit(body,Op::Load,Type::I32,{q});
    emit(body,Op::Store,Type::Void,{m.constInt(99),p});
  }
  emit(body,Op::Ret,Type::Void,{result});
  int32_t before=execute(*f,false);
  makeRedundantStorePass(Layer::Cf)->run(m);
  require(execute(*f,false)==before,"pointer phi lost its alloca alias");
}
int main(int argc, char **argv) {
  try {
    bool privateOnly=argc>1 && std::string(argv[1])=="private";
    if (!privateOnly) for (int mode=0;mode<4;++mode) checkRestore(mode);
    if (argc==1 || privateOnly) { checkPrivateAlias(false); checkPrivateAlias(true); }
    std::cout << "memory regression passed\n";
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
