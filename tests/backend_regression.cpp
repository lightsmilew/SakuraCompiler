#include "backend/Machine.h"
#include "backend/MachineOpt.h"
#include "backend/Asm.h"
#include "backend/RA.h"

#include <iostream>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <algorithm>
#include <array>
#include <numeric>
#include <random>

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

// Independent MIR interpreter: exercise both paths, non-SSA redefinitions,
// parallel-copy cycles, and call clobbers after allocation. Float copies carry
// raw bits, so signed zero and NaN payloads must survive without arithmetic.
static uint64_t evaluateCopies(const MachineFunc &f, bool physical, int input) {
  std::unordered_map<int, uint64_t> regs[2];
  std::unordered_map<uint64_t, uint64_t> memory{{100, 0}, {200, 0}};
  for (int i = 10; i < 18; ++i) {
    regs[0][i] = input + i - 10;
    regs[1][i] = uint32_t(0x80000000u + i - 10);
  }
  std::unordered_map<std::string, size_t> labels;
  for (size_t i = 0; i < f.blocks.size(); ++i) labels[f.blocks[i].name] = i;
  size_t block = 0, pc = 0;
  auto read = [&](int file, int r) -> uint64_t {
    return r < 0 ? 0 : regs[file].at(r);
  };
  auto word = [](uint64_t v) { return uint64_t(int64_t(int32_t(uint32_t(v)))); };
  for (int steps = 0; steps < 10000; ++steps) {
    if (pc == f.blocks.at(block).instrs.size()) { ++block; pc = 0; continue; }
    const MInst &m = f.blocks.at(block).instrs[pc++];
    auto jump = [&]() { block = labels.at(m.sym); pc = 0; };
    switch (m.op) {
    case MOp::Li: regs[0][m.dst] = uint64_t(int64_t(m.imm)); break;
    case MOp::LiF: regs[1][m.dst] = uint32_t(m.imm); break;
    case MOp::EntryInt:
      regs[0][m.dst] = physical ? read(0, m.imm) : uint64_t(input + m.imm - 10); break;
    case MOp::EntryFlt:
      regs[1][m.dst] = physical ? read(1, m.imm) : uint32_t(0x80000000u + m.imm - 10); break;
    case MOp::MoveX: regs[0][m.dst] = read(0, m.a); break;
    case MOp::MoveF: regs[1][m.dst] = read(1, m.a); break;
    case MOp::FAdd: {
      uint32_t aBits = uint32_t(read(1, m.a)), bBits = uint32_t(read(1, m.b)), bits;
      float a, b;
      std::memcpy(&a, &aBits, 4); std::memcpy(&b, &bBits, 4);
      float sum = a + b; std::memcpy(&bits, &sum, 4);
      regs[1][m.dst] = bits; break;
    }
    case MOp::IAdd: regs[0][m.dst] = word(read(0, m.a) + read(0, m.b)); break;
    case MOp::ISub: regs[0][m.dst] = word(read(0, m.a) - read(0, m.b)); break;
    case MOp::IAddI: regs[0][m.dst] = word(read(0, m.a) + m.imm); break;
    case MOp::Fsw: memory[read(0, m.b) + m.imm] = uint32_t(read(1, m.a)); break;
    case MOp::Lw: regs[0][m.dst] = word(memory.at(read(0, m.a) + m.imm)); break;
    case MOp::Jmp: jump(); break;
    case MOp::BrZ: if (read(0, m.a) == 0) jump(); break;
    case MOp::BrNz: if (read(0, m.a) != 0) jump(); break;
    case MOp::BrCmp: {
      int64_t a = int64_t(read(0, m.a)), b = int64_t(read(0, m.b));
      bool taken = false;
      switch (Cond(m.imm)) {
      case Cond::Eq: taken = a == b; break;
      case Cond::Ne: taken = a != b; break;
      case Cond::Lt: taken = a < b; break;
      case Cond::Le: taken = a <= b; break;
      case Cond::Gt: taken = a > b; break;
      case Cond::Ge: taken = a >= b; break;
      }
      if (taken) jump();
      break;
    }
    case MOp::Call: {
      uint64_t result = 17;
      for (size_t i = 0; i < m.args.size(); ++i)
        result = word(result + (i + 1) * read(m.args[i].ty == Type::F32, m.args[i].vreg));
      if (physical) {
        for (int file = 0; file < 2; ++file)
          for (int r = 0; r < 32; ++r)
            if (!(r == 8 || r == 9 || (r >= 18 && r <= 27))) regs[file][r] = 0xbadc0de;
        regs[m.ty == Type::F32][10] = result;
      }
      if (m.dst >= 0) regs[m.ty == Type::F32][m.dst] = result;
      break;
    }
    case MOp::Ret: return read(m.ty == Type::F32, m.a);
    default: throw std::runtime_error("unsupported copy fixture opcode");
    }
  }
  throw std::runtime_error("copy fixture did not terminate");
}

static void copyAllocation() {
  auto branch = inst(MOp::BrNz, -1, 0); branch.sym = "yes";
  auto join = inst(MOp::Jmp); join.sym = "join";
  auto again = inst(MOp::BrNz, -1, 3); again.sym = "loop";
  MachineFunc diamond;
  diamond.name = "diamond";
  diamond.blocks = {{"entry", {inst(MOp::EntryInt, 0, -1, -1, 10),
                    inst(MOp::MoveX, 1, 0), branch}},
      {"no", {inst(MOp::Li, 0, -1, -1, 11), join}},
      {"yes", {inst(MOp::Li, 0, -1, -1, 19), join}},
      {"join", {inst(MOp::IAdd, 2, 0, 1), inst(MOp::Ret, -1, 2)}}};
  MachineFunc cycle;
  cycle.name = "cycle";
  cycle.blocks = {{"entry", {inst(MOp::Li, 0, -1, -1, 7),
                             inst(MOp::Li, 1, -1, -1, 13),
                             inst(MOp::EntryInt, 3, -1, -1, 10)}},
      {"loop", {inst(MOp::MoveX, 2, 0), inst(MOp::MoveX, 0, 1),
                inst(MOp::MoveX, 1, 2), inst(MOp::IAddI, 3, 3, -1, -1), again}},
      {"exit", {inst(MOp::ISub, 4, 0, 1), inst(MOp::Ret, -1, 4)}}};
  MInst call = inst(MOp::Call, 2); call.sym = "clobber";
  call.args = {{1, Type::I32}, {0, Type::I32}};
  std::vector<MachineFunc> fixtures{
      function({inst(MOp::Li, 0, -1, -1, 7), inst(MOp::MoveX, 1, 0),
                inst(MOp::IAddI, 2, 0, -1, 3), inst(MOp::IAdd, 3, 1, 2), inst(MOp::Ret, -1, 3)}),
      function({inst(MOp::Li, 0, -1, -1, 7), inst(MOp::MoveX, 1, 0),
                inst(MOp::Li, 0, -1, -1, 11), inst(MOp::IAdd, 2, 0, 1), inst(MOp::Ret, -1, 2)}),
      function({inst(MOp::EntryInt, 0, -1, -1, 10), inst(MOp::EntryInt, 1, -1, -1, 11),
                call, inst(MOp::IAdd, 3, 0, 2), inst(MOp::Ret, -1, 3)}), diamond, cycle
  };
  for (const auto &fixture : fixtures)
    for (int input : {1, 2, 3, 4, 5}) {
      auto expected = evaluateCopies(fixture, false, input);
      std::vector<MachineFunc> fs{fixture};
      RegisterAllocator().allocate(fs);
      check(fs[0].spillCount == 0, "copy allocation fixture spilled");
      check(evaluateCopies(fs[0], true, input) == expected, "copy colouring changed a live value");
      MachineOptimizer().removeRedundantMoves(fs);
      check(evaluateCopies(fs[0], true, input) == expected, "post-RA copy cleanup changed a live value");
    }
  check(evaluateCopies(diamond, false, 0) == 11, "zero branch oracle broken");
  std::vector<MachineFunc> fs{diamond}; RegisterAllocator().allocate(fs);
  check(evaluateCopies(fs[0], true, 0) == 11, "copy equality escaped a CFG branch");

  // An incoming parameter and its forwarding call argument should already
  // occupy a1/fa1. Integer and float palettes must stay independent.
  for (bool fp : {false, true}) {
    auto entry = inst(fp ? MOp::EntryFlt : MOp::EntryInt, 0, -1, -1, 11);
    entry.ty = fp ? Type::F32 : Type::I32;
    auto forwarded = inst(MOp::Call, 1); forwarded.sym = "forward";
    forwarded.ty = entry.ty;
    forwarded.args = {{0, entry.ty}, {0, entry.ty}};
    auto ret = inst(MOp::Ret, -1, 1); ret.ty = entry.ty;
    fs = {function({entry, forwarded, ret})};
    auto expected = evaluateCopies(fs[0], false, 7);
    RegisterAllocator().allocate(fs);
    check(evaluateCopies(fs[0], true, 7) == expected, "ABI hint confused register files");
    check(fs[0].blocks[0].instrs.front().dst == 11, "forwarded argument missed ABI colour");
    check(fs[0].blocks[0].instrs.back().a == 10, "call result missed ABI return colour");
  }
  for (uint32_t bits : {0x80000000u, 0x7fc01234u}) {
    auto ret = inst(MOp::Ret, -1, 1); ret.ty = Type::F32;
    fs = {function({inst(MOp::LiF, 0, -1, -1, int32_t(bits)),
                    inst(MOp::MoveF, 1, 0), ret})};
    RegisterAllocator().allocate(fs); MachineOptimizer().removeRedundantMoves(fs);
    check(evaluateCopies(fs[0], true, 1) == bits, "float copy changed raw bits");
  }
  MachineFunc hot;
  hot.name = "hot_phi";
  call = inst(MOp::Call, 4); call.sym = "observe_float";
  call.args = {{0, Type::F32}};
  hot.blocks = {{"entry", {inst(MOp::LiF, 0), inst(MOp::EntryInt, 3, -1, -1, 10)}},
      {"loop", {inst(MOp::MoveF, 1, 0), inst(MOp::LiF, 2, -1, -1, 0x3f800000),
                inst(MOp::FAdd, 1, 1, 2), inst(MOp::MoveF, 0, 1),
                inst(MOp::IAddI, 3, 3, -1, -1), again}},
      {"exit", {call, inst(MOp::Ret, -1, 4)}}};
  fs = {hot}; RegisterAllocator().allocate(fs);
  for (int input : {1, 2, 5})
    check(evaluateCopies(fs[0], true, input) == evaluateCopies(hot, false, input),
          "ABI profitability changed floating reduction");
  for (const auto &m : fs[0].blocks[1].instrs)
    check(m.op != MOp::MoveF || m.dst == m.a,
          "cold ABI hint introduced a copy on every loop iteration");
  sakura::ir::Module mod("copy-origins");
  call = inst(MOp::Call, 18); call.sym = "shuffle";
  call.args = {{11, Type::I32}, {10, Type::I32}, {19, Type::I32}};
  fs = {function({inst(MOp::EntryInt, 18, -1, -1, 10),
        inst(MOp::MoveX, 20, 18), call, inst(MOp::Ret, -1, 20)})};
  std::ostringstream stats;
  auto text = AssemblyWriter().write(&mod, fs, &stats);
  check(stats.str().find("x-mir=1 x-entry=1 x-argument=4 x-result=1 x-return=1") != std::string::npos &&
        stats.str().find("shuffle-stores=0 shuffle-loads=0") != std::string::npos,
        "copy-origin accounting missed ABI shuffle traffic");
  size_t copies = 0, pos = 0;
  while ((pos = text.find("\tmv ", pos)) != std::string::npos) { ++copies; ++pos; }
  check(copies == 8, "copy-origin counts differ from emitted assembly");

  // X10 and F10 are different registers. A float store reading X10 as its
  // address does not make its F10 value a reader of an integer copy to X10.
  fs = {function({inst(MOp::Li, 10, -1, -1, 100), inst(MOp::Li, 11, -1, -1, 200),
      inst(MOp::LiF, 10, -1, -1, 0x3f800000), inst(MOp::LiF, 11, -1, -1, 0x40000000),
      inst(MOp::MoveX, 10, 11), inst(MOp::Fsw, -1, 10, 10),
      inst(MOp::Lw, 12, 11), inst(MOp::Ret, -1, 12)})};
  auto expected = evaluateCopies(fs[0], true, 1);
  MachineOptimizer().removeRedundantMoves(fs);
  check(evaluateCopies(fs[0], true, 1) == expected,
        "copy-to-store fold confused the float value with the integer address");
  for (uint32_t bits : {0x80000000u, 0x7fc01234u}) {
    fs = {function({inst(MOp::Li, 11, -1, -1, 200),
        inst(MOp::LiF, 10, -1, -1, int32_t(bits)), inst(MOp::MoveF, 12, 10),
        inst(MOp::Fsw, -1, 12, 11), inst(MOp::Lw, 13, 11), inst(MOp::Ret, -1, 13)})};
    expected = evaluateCopies(fs[0], true, 1);
    check(MachineOptimizer().removeRedundantMoves(fs) == 1,
          "float store did not consume its copy source directly");
    check(evaluateCopies(fs[0], true, 1) == expected, "float store copy fold changed bits");
  }
}

// Execute the actual assembly copies up to the call, using independent raw
// register bits as the oracle. This exercises the writer, not MIR allocation.
static void argumentShuffles() {
  sakura::ir::Module mod("shuffle");
  auto verify = [&](const std::vector<MArg> &args) {
    std::unordered_map<std::string, uint64_t> regs;
    for (int r = 0; r < 32; ++r) {
      regs[xregName(r)] = 0xfedcba9800000000ULL + uint64_t(r) * 0x10203;
      regs[fregName(r)] = r % 2 ? 0x7fc01000u + r : 0x80000000u + r;
    }
    regs["zero"] = 0;
    regs[fregName(10)] = 0x80000000u; // -0.0 must survive an argument permutation
    regs[fregName(12)] = 0;           // distinguish it from +0.0
    const auto initial = regs;
    auto call = inst(MOp::Call); call.sym = "observe"; call.args = args;
    std::vector<MachineFunc> fs{function({call})};
    auto assembly = AssemblyWriter().write(&mod, fs);
    std::istringstream lines(assembly);
    std::string line;
    bool called = false;
    while (std::getline(lines, line)) {
      if (line.empty() || line[0] != '\t') continue;
      std::istringstream tokens(line);
      std::string op, dst, src;
      tokens >> op;
      if (op == "call") { called = true; break; }
      if (op[0] == '.') continue;
      check(op == "mv" || op == "fmv.s", "argument shuffle emitted memory traffic");
      tokens >> dst >> src;
      check(!dst.empty() && dst.back() == ',', "malformed assembly copy");
      dst.pop_back();
      regs[dst] = regs.at(src);
    }
    check(called, "shuffle fixture did not emit a call");
    ArgCursor cursor;
    for (const auto &arg : args) {
      bool fp = arg.ty == Type::F32;
      auto loc = cursor.next(fp);
      if (loc.reg < 0 || arg.vreg < 0) continue;
      auto name = [&](int r) { return fp ? fregName(r) : xregName(r); };
      check(regs.at(name(loc.reg)) == initial.at(name(arg.vreg)),
            "serialized argument copies changed a value");
    }
  };
  for (bool fp : {false, true}) {
    std::array<int, 7> permutation{{10, 11, 12, 13, 14, 15, 16}};
    do {
      std::vector<MArg> args;
      for (int r : permutation) args.push_back({r, fp ? Type::F32 : Type::Ptr});
      verify(args);
    } while (std::next_permutation(permutation.begin(), permutation.end()));
  }
  std::mt19937 random(20261007);
  const int palette[] = {0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 31};
  for (int trial = 0; trial < 2000; ++trial) {
    std::vector<MArg> args;
    // Includes duplicate sources, self copies, eight-register cycles, both
    // files sharing the same numbers, and already-lowered overflow arguments.
    for (int i = 0; i < 20; ++i) {
      bool fp = (random() & 1) != 0;
      int r = palette[random() % (sizeof(palette) / sizeof(palette[0]))];
      if (fp && r == kScratchF1) r = 2; // reserved scratch is not an allocated value
      args.push_back({r, fp ? Type::F32 : (i % 2 ? Type::I32 : Type::Ptr)});
    }
    verify(args);
  }
  for (auto type : {Type::Ptr, Type::F32})
    verify({{17, type}, {10, type}, {11, type}, {12, type},
            {13, type}, {14, type}, {15, type}, {16, type}});
}

static void assemblyLabels() {
  sakura::ir::Module mod("labels");
  auto jump = inst(MOp::Jmp); jump.sym = "exit";
  auto call = inst(MOp::Call); call.sym = "exit";
  auto address = inst(MOp::LeaGlobal, 10); address.sym = "entry";
  auto ret = inst(MOp::Ret); ret.ty = Type::Void;
  MachineFunc f; f.name = "first";
  f.blocks = {{"entry", {address, call, jump}}, {"exit", {ret}}};
  auto second = f; second.name = "second";
  std::vector<MachineFunc> fs{f, second};
  auto assembly = AssemblyWriter().write(&mod, fs);
  check(assembly.find("# Generated by the Sakura compiler.\n") == 0, "missing generator signature");
  for (int i = 0; i < 4; ++i)
    check(assembly.find(".L" + std::to_string(i) + ":\n") != std::string::npos,
          "local label is missing or duplicated between functions");
  check(assembly.find("\tj .L1\n") != std::string::npos &&
        assembly.find("\tj .L3\n") != std::string::npos, "branch label was not remapped");
  check(assembly.find("\tcall exit\n") != std::string::npos &&
        assembly.find("\tla a0, entry\n") != std::string::npos, "external symbol was renamed");
  check(fs[0].blocks[0].name == "entry" && fs[0].blocks[0].instrs.back().sym == "exit",
        "assembly label mapping mutated MIR diagnostics");
  check(AssemblyWriter().write(&mod, fs) == assembly, "assembly labels are nondeterministic");
}

static void finalLayouts() {
  auto jump = [](const char *target) { auto m = inst(MOp::Jmp); m.sym = target; return m; };
  auto zero = inst(MOp::BrZ, -1, 0); zero.sym = "zero";
  auto bound = inst(MOp::BrCmp, -1, 2, 1, int(Cond::Lt)); bound.sym = "body";
  auto both = inst(MOp::BrNz, -1, 0); both.sym = "empty";
  auto call = inst(MOp::Call, 3); call.sym = "work";
  call.args = {{2, Type::I32}, {0, Type::I32}};
  MachineFunc original;
  original.name = "layout";
  original.blocks = {
    {"entry", {inst(MOp::EntryInt, 0, -1, -1, 10), inst(MOp::Li, 1, -1, -1, 3),
                inst(MOp::Li, 2), inst(MOp::Li, 4), zero, jump("loop")}},
    {"loop", {bound, jump("exit")}},
    {"body", {inst(MOp::IAddI, 2, 2, -1, 1), call, inst(MOp::IAdd, 4, 4, 3),
               both, jump("empty")}},
    {"empty", {}},
    {"step", {jump("loop")}},
    {"zero", {inst(MOp::Li, 4, -1, -1, 99), jump("exit")}},
    {"exit", {inst(MOp::Ret, -1, 4)}}};
  auto verify = [&](const MachineFunc &fixture) {
    std::vector<MachineFunc> fs{fixture};
    MachineOptimizer().finalizeBlockLayout(fs);
    for (int input : {-2, 0, 1, 3})
      check(evaluateCopies(fs[0], false, input) == evaluateCopies(fixture, false, input),
            "branch folding changed a CFG path or call");
    check(fs[0].blocks.front().name == "entry", "layout moved the entry block");
    const size_t blocks = fs[0].blocks.size();
    sakura::ir::Module mod("stable-layout");
    const auto assembly = AssemblyWriter().write(&mod, fs);
    check(MachineOptimizer().finalizeBlockLayout(fs) == 0 && fs[0].blocks.size() == blocks,
          "final layout is not stable");
    check(AssemblyWriter().write(&mod, fs) == assembly, "final layout changed on its second run");
  };
  verify(original); // the empty block falls through in the original layout
  auto explicitFlow = original;
  explicitFlow.blocks[3].instrs.push_back(jump("step"));
  std::array<int, 6> order{{1, 2, 3, 4, 5, 6}};
  do {
    auto permuted = explicitFlow;
    for (size_t i = 0; i < order.size(); ++i) permuted.blocks[i + 1] = explicitFlow.blocks[order[i]];
    verify(permuted);
  } while (std::next_permutation(order.begin(), order.end()));

  // Every signed compare must invert correctly when its taken block is next.
  for (Cond cond : {Cond::Eq, Cond::Ne, Cond::Lt, Cond::Le, Cond::Gt, Cond::Ge}) {
    auto branch = inst(MOp::BrCmp, -1, 0, 1, int(cond)); branch.sym = "yes";
    auto f = function({inst(MOp::EntryInt, 0, -1, -1, 10), inst(MOp::Li, 1),
                       branch, jump("no")});
    f.blocks[0].name = "entry";
    f.blocks.push_back({"yes", {inst(MOp::Li, 2, -1, -1, 5), inst(MOp::Ret, -1, 2)}});
    f.blocks.push_back({"no", {inst(MOp::Li, 2, -1, -1, 9), inst(MOp::Ret, -1, 2)}});
    verify(f);
  }
  // The shared exit prevents linear merging. Placing an arm next to the entry
  // and another next to the exit must still remove two explicit jumps.
  auto choose = inst(MOp::BrNz, -1, 0); choose.sym = "yes";
  MachineFunc separated;
  separated.name = "separated";
  separated.blocks = {
    {"entry", {inst(MOp::EntryInt, 0, -1, -1, 10), choose, jump("no")}},
    {"exit", {inst(MOp::Ret, -1, 2)}},
    {"no", {inst(MOp::Li, 2, -1, -1, 9), jump("exit")}},
    {"yes", {inst(MOp::Li, 2, -1, -1, 5), jump("exit")}}};
  verify(separated);
  std::vector<MachineFunc> placed{separated};
  check(MachineOptimizer().finalizeBlockLayout(placed) == 2,
        "profitable machine block placement did not remove separated jumps");
  auto infinite = function({jump("a")});
  infinite.blocks.push_back({"a", {jump("b")}});
  infinite.blocks.push_back({"b", {jump("a")}});
  std::vector<MachineFunc> fs{infinite};
  MachineOptimizer().finalizeBlockLayout(fs);
  check(fs.size() == 1 && fs[0].blocks.size() == 2 &&
        fs[0].blocks[1].instrs.back().sym == fs[0].blocks[1].name,
        "branch forwarding erased a nonterminating cycle");
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
    case MOp::MoveX: r[m.dst] = a; break;
    case MOp::IAddI: r[m.dst] = m.ty == Type::Ptr ? int64_t(uint64_t(a) + m.imm)
                                                : int64_t(int32_t(uint32_t(a) + m.imm)); break;
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

static void copyAndScale() {
  MachineOptimizer opt;
  for (int x : {INT32_MIN, INT32_MIN+1, -3, 0, 1, 17, INT32_MAX}) {
    std::vector<std::vector<MInst>> fixtures{
      {inst(MOp::Li, 0, -1, -1, x), inst(MOp::MoveX, 1, 0),
       inst(MOp::MoveX, 2, 1), inst(MOp::SllI, 3, 2, -1, 2), inst(MOp::Ret, -1, 3)},
      {inst(MOp::Li, 0, -1, -1, x), inst(MOp::IAddI, 1, 0, -1, -1),
       inst(MOp::SllI, 2, 0, -1, 2), inst(MOp::IAddI, 3, 2, -1, -4),
       inst(MOp::IAdd, 4, 1, 3), inst(MOp::Ret, -1, 4)},
      // Redefining a source invalidates its copy, including non-SSA phis.
      {inst(MOp::Li, 0, -1, -1, x), inst(MOp::MoveX, 1, 0),
       inst(MOp::Li, 0, -1, -1, 7), inst(MOp::IAdd, 2, 0, 1), inst(MOp::Ret, -1, 2)},
      // Cached self updates describe the previous version of the source.
      {inst(MOp::Li, 0, -1, -1, x), inst(MOp::IAddI, 0, 0, -1, -1),
       inst(MOp::SllI, 1, 0, -1, 2), inst(MOp::IAddI, 2, 1, -1, -4), inst(MOp::Ret, -1, 2)}
    };
    for (auto &code : fixtures) {
      std::vector<MachineFunc> fs{function(code)};
      auto expected = evaluateStrength(fs[0]);
      opt.peephole(fs);
      check(evaluateStrength(fs[0]) == expected, "copy/scale peephole changed word arithmetic");
      opt.peephole(fs);
      check(evaluateStrength(fs[0]) == expected, "repeated peephole changed word arithmetic");
    }
  }
  // addiw r, wide, 0 is a sign extension, not a full-width move.
  auto wide = inst(MOp::LiWide, 0); wide.cst = 0x1234567880000001LL;
  std::vector<MachineFunc> fs{function({wide, inst(MOp::IAddI, 1, 0), inst(MOp::Ret, -1, 1)})};
  auto expected = evaluateStrength(fs[0]); opt.peephole(fs);
  check(evaluateStrength(fs[0]) == expected, "addiw zero lost sign extension");
  auto materialize = inst(MOp::LeaGlobal, 18); materialize.sym = "data";
  fs = {function({materialize, inst(MOp::MoveX, 10, 18), inst(MOp::MoveX, 12, 10),
                 inst(MOp::Ret, -1, 12)})};
  opt.peephole(fs); opt.peephole(fs); opt.coalesceMoves(fs);
  check(fs[0].blocks[0].instrs.size() == 2, "global address copy chain retained");
  // Post-RA definition retargeting must count removal and examine the next op.
  fs = {function({inst(MOp::IAddI, 12, 10, -1, 1), inst(MOp::MoveX, 10, 12),
                 inst(MOp::MoveX, 10, 10), inst(MOp::Ret, -1, 10)})};
  check(opt.removeRedundantMoves(fs) == 2, "post-RA retarget/delete progress broken");
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
    copyAllocation();
    argumentShuffles();
    assemblyLabels();
    finalLayouts();
    globalStorage();
    remainderUses();
    nonnegativeDivision();
    copyAndScale();
    std::cout << "backend regression tests passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
