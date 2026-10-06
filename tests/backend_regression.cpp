#include "backend/Machine.h"
#include "backend/MachineOpt.h"
#include "backend/Asm.h"
#include "backend/RA.h"

#include <iostream>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

using namespace sakura::backend;

static void check(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}

static MInst inst(MOp op, int dst = -1, int a = -1, int b = -1, int imm = 0) {
  MInst m;
  m.op = op; m.dst = dst; m.a = a; m.b = b; m.imm = imm;
  return m;
}

static MachineFunc function(std::vector<MInst> code) {
  MachineFunc f;
  f.name = "test";
  f.blocks.push_back({"entry", std::move(code)});
  return f;
}

// Execute the straight-line scheduling fixtures independently of allocation.
static int evaluate(const MachineFunc &f) {
  std::unordered_map<int, int> r, memory{{100, 200}, {200, 37}};
  auto get = [&](int v) { return v < 0 ? 0 : r.at(v); };
  for (const auto &m : f.blocks[0].instrs) {
    switch (m.op) {
    case MOp::Li: r[m.dst] = m.imm; break;
    case MOp::MoveX: r[m.dst] = get(m.a); break;
    case MOp::IAdd: r[m.dst] = get(m.a) + get(m.b); break;
    case MOp::IAddI: r[m.dst] = get(m.a) + m.imm; break;
    case MOp::Lw: r[m.dst] = memory.at(get(m.a)); break;
    case MOp::Ret: return get(m.a);
    default: throw std::runtime_error("unsupported fixture opcode");
    }
  }
  throw std::runtime_error("fixture has no return");
}

static void scheduling() {
  MachineOptimizer opt;
  std::vector<std::vector<MInst>> fixtures{
    // A load reading the result of another load must not pass it.
    {inst(MOp::Li, 0, -1, -1, 100), inst(MOp::Lw, 1, 0),
     inst(MOp::Lw, 2, 1), inst(MOp::Ret, -1, 2)},
    // WAR: the copy must read the old value before the load redefines it.
    {inst(MOp::Li, 0, -1, -1, 100), inst(MOp::Li, 1, -1, -1, 7),
     inst(MOp::MoveX, 2, 1), inst(MOp::Lw, 1, 0), inst(MOp::Ret, -1, 2)},
    // WAW: do not let an earlier definition overwrite the loaded value.
    {inst(MOp::Li, 0, -1, -1, 100), inst(MOp::Li, 1, -1, -1, 7),
     inst(MOp::Lw, 1, 0), inst(MOp::Ret, -1, 1)}
  };
  for (auto &code : fixtures) {
    std::vector<MachineFunc> fs{function(code)};
    int expected = evaluate(fs[0]);
    opt.schedule(fs);
    check(evaluate(fs[0]) == expected, "load scheduling violated a dependency");
  }

  std::vector<MInst> code{inst(MOp::Li, 0, -1, -1, 100),
                          inst(MOp::Li, 1, -1, -1, 0)};
  for (int i = 0; i < 32; ++i) {
    code.push_back(inst(MOp::Lw, i + 2, 0));
    code.push_back(inst(MOp::IAdd, 1, 1, i + 2));
  }
  code.push_back(inst(MOp::Ret, -1, 1));
  std::vector<MachineFunc> fs{function(code)};
  int expected = evaluate(fs[0]);
  opt.schedule(fs);
  check(evaluate(fs[0]) == expected, "scheduling changed unrolled reduction");
  RegisterAllocator().allocate(fs);
  check(fs[0].spillCount == 0, "scheduling made streaming loads spill");
}

static void allocation() {
  std::vector<MachineFunc> leaf{function({inst(MOp::EntryInt, 0, -1, -1, 10),
      inst(MOp::IAddI, 1, 0, -1, 3), inst(MOp::Ret, -1, 1)})};
  std::ostringstream stats;
  RegisterAllocator().allocate(leaf, &stats);
  check(leaf[0].usedCSX.empty(), "leaf unnecessarily saves a register");
  check(stats.str().find("spill-slots=0") != std::string::npos,
        "missing allocation statistics");

  MInst call = inst(MOp::Call);
  call.sym = "clobber";
  std::vector<MachineFunc> across{function({inst(MOp::EntryInt, 0, -1, -1, 10),
      call, inst(MOp::Ret, -1, 0)})};
  RegisterAllocator().allocate(across);
  check(across[0].usedCSX.size() == 1, "call-live value lost callee-saved colour");

  // Every incoming ABI register remains intact until its EntryInt reads it.
  std::vector<MInst> code;
  for (int i = 0; i < 8; ++i) code.push_back(inst(MOp::EntryInt, i, -1, -1, i + 10));
  int sum = 0;
  for (int i = 1; i < 8; ++i) {
    code.push_back(inst(MOp::IAdd, i + 7, sum, i));
    sum = i + 7;
  }
  code.push_back(inst(MOp::Ret, -1, sum));
  std::vector<MachineFunc> args{function(code)};
  RegisterAllocator().allocate(args);
  check(args[0].spillCount == 0, "eight parameters unnecessarily spill");
  int regs[32] = {};
  for (int i = 0; i < 8; ++i) regs[i + 10] = i + 1;
  for (const auto &m : args[0].blocks[0].instrs) {
    if (m.op == MOp::EntryInt) regs[m.dst] = regs[m.imm];
    if (m.op == MOp::IAdd) regs[m.dst] = regs[m.a] + regs[m.b];
    if (m.op == MOp::Ret) check(regs[m.a] == 36, "incoming parameter clobbered");
  }
}

static void globalStorage() {
  sakura::ir::Module mod("global-storage");
  mod.addGlobal("zeros", Type::I32, 10000000, false, false, {});
  mod.addGlobal("positive_zero", Type::F32, 1, false, true,
                {sakura::ConstVal::floatC(0.0f)});
  mod.addGlobal("negative_zero", Type::F32, 1, false, true,
                {sakura::ConstVal::floatC(-0.0f)});
  std::vector<MachineFunc> functions;
  auto asmText = AssemblyWriter().write(&mod, functions);
  check(asmText.find("\t.bss\n\t.p2align 2\n\t.globl zeros") != std::string::npos,
        "zero global occupies file-backed data");
  check(asmText.find("\t.bss\n\t.p2align 2\n\t.globl positive_zero") != std::string::npos,
        "positive zero initializer missed bss");
  check(asmText.find("\t.data\n\t.p2align 2\n\t.globl negative_zero") != std::string::npos &&
        asmText.find("0x80000000") != std::string::npos,
        "negative zero initializer lost its sign bit");
}

// Model RV64 word arithmetic independently, including its sign extension.
static int64_t evaluateStrength(const MachineFunc &f) {
  std::unordered_map<int, int64_t> r;
  auto get = [&](int v) -> int64_t { return v < 0 ? 0 : r.at(v); };
  for (const auto &m : f.blocks[0].instrs) {
    int64_t a = get(m.a), b = get(m.b);
    switch (m.op) {
    case MOp::Li: r[m.dst] = m.imm; break;
    case MOp::LiWide: r[m.dst] = m.cst; break;
    case MOp::IRem: r[m.dst] = int32_t(a) % int32_t(b); break;
    case MOp::IDiv: r[m.dst] = int32_t(a) / int32_t(b); break;
    case MOp::Mul64: r[m.dst] = int64_t(uint64_t(a) * uint64_t(b)); break;
    case MOp::IAdd: r[m.dst] = int32_t(uint32_t(a) + uint32_t(b)); break;
    case MOp::ISub: r[m.dst] = int32_t(uint32_t(a) - uint32_t(b)); break;
    case MOp::IAnd: r[m.dst] = a & b; break;
    case MOp::IAndI: r[m.dst] = a & int64_t(m.imm); break;
    case MOp::IXorI: r[m.dst] = a ^ int64_t(m.imm); break;
    case MOp::Shr64I: r[m.dst] = uint64_t(a) >> m.imm; break;
    case MOp::Sar64I: r[m.dst] = a >> m.imm; break;
    case MOp::SraI: r[m.dst] = int32_t(a) >> m.imm; break;
    case MOp::SrlI: r[m.dst] = int32_t(uint32_t(a) >> m.imm); break;
    case MOp::INeg: r[m.dst] = int32_t(0u - uint32_t(a)); break;
    case MOp::IMul: r[m.dst] = int32_t(uint32_t(a) * uint32_t(b)); break;
    case MOp::SllI: r[m.dst] = int32_t(uint32_t(a) << m.imm); break;
    case MOp::ICmp:
      check(m.imm == int(Cond::Eq), "unsupported strength comparison");
      r[m.dst] = a == b; break;
    case MOp::Ret: return a;
    default: throw std::runtime_error("unsupported strength fixture opcode");
    }
  }
  throw std::runtime_error("strength fixture has no return");
}

static void nonnegativeDivision() {
  MachineOptimizer opt;
  uint32_t random = 0x12345678;
  for (int divisor : {3, 5, 7, 9, 31, 50, 97, 100, 1009, 65535, INT32_MAX}) {
    for (int trial = 0; trial < 200; ++trial) {
      random = random * 1664525u + 1013904223u;
      int x = trial == 0 ? INT32_MAX : trial == 1 ? 0 : int(random & INT32_MAX);
      for (MOp op : {MOp::IDiv, MOp::IRem}) {
        auto f = function({inst(MOp::Li, 0, -1, -1, x),
            inst(MOp::Li, 1, -1, -1, divisor), inst(op, 2, 0, 1),
            inst(MOp::Ret, -1, 2)});
        f.nonNeg.insert(0);
        auto expected = evaluateStrength(f);
        std::vector<MachineFunc> fs{f};
        opt.strengthReduction(fs);
        check(evaluateStrength(fs[0]) == expected, "non-negative magic division changed result");
        for (const auto &m : fs[0].blocks[0].instrs)
          check(m.op != MOp::Mulhu, "non-negative i32 division still uses multiply-high");
      }
    }
  }
}

static void remainderUses() {
  MachineOptimizer opt;
  for (int x : {INT32_MIN, -9, -3, -2, -1, 0, 1, 2, 3, INT32_MAX}) {
    for (bool observeXor : {false, true}) {
      auto cmp = inst(MOp::ICmp, 4, 3, -1, int(Cond::Eq));
      auto f = function({inst(MOp::Li, 0, -1, -1, x),
          inst(MOp::Li, 1, -1, -1, 2), inst(MOp::IRem, 2, 0, 1),
          inst(MOp::IXorI, 3, 2, -1, 1), cmp});
      if (observeXor) f.blocks[0].instrs.push_back(inst(MOp::IAdd, 5, 3, 4));
      f.blocks[0].instrs.push_back(inst(MOp::Ret, -1, observeXor ? 5 : 4));
      auto expected = evaluateStrength(f);
      std::vector<MachineFunc> fs{f};
      opt.strengthReduction(fs);
      check(evaluateStrength(fs[0]) == expected,
            "remainder bit-test changed an observed xor value");
      bool masked = false;
      for (const auto &m : fs[0].blocks[0].instrs)
        if (m.op == MOp::IAnd) masked = true;
      check(masked != observeXor, "bit-test use validation missed its intended case");
    }
  }
}

int main() {
  try {
    scheduling();
    allocation();
    globalStorage();
    remainderUses();
    nonnegativeDivision();
    std::cout << "backend regression tests passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
