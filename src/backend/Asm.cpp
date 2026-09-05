// RISC-V assembly writer implementation: text-formatting of the machine
// functions (labels, pseudo-moves, fused frame/global addressing, branch
// lowering to label+offset encodings) plus the .data section for globals.
#include "Asm.h"

#include <algorithm>
#include <cstdio>
#include <sstream>
#include <unordered_set>

using namespace sakura::ir;

namespace sakura {
namespace backend {

namespace {

inline bool fits12(int64_t v) { return v >= -2048 && v <= 2047; }
inline int64_t align8(int64_t v) { return (v + 7) & ~7LL; }
inline int64_t align16(int64_t v) { return (v + 15) & ~15LL; }

std::string xrn(int32_t r) { return r <= 0 ? "zero" : xregName(r); }
std::string frn(int32_t r) { return fregName(r); }

// ---------------------------------------------------------------------------
// per-function frame layout (offsets measured from sp after the prologue)
//   0 .. outMax                       outbound stack-argument area
//   outMax .. + shuffleBytes          register-argument shuffle slots
//   .. + spillCount*8                 spill slots
//   .. + localBytes                   ISel-assigned locals
//   .. + usedCSF*8                    saved float callee-saved registers
//   .. + (usedCSX+ra)*8               saved x callee-saved registers / ra
// fs (multiple of 16) ends the frame; incoming stack args live above fs.
// ---------------------------------------------------------------------------
struct Frame {
  int64_t outMax = 0;
  int64_t shuffleBytes = 0;
  int64_t spillBytes = 0;
  int64_t localBytes = 0;
  int64_t spillBase = 0;
  int64_t localBase = 0;
  int64_t fBase = 0;
  int64_t xBase = 0;
  int64_t fs = 0;
  int xN = 0;   // number of saved x slots (includes ra when hasCall)
  int fN = 0;   // number of saved float slots
  bool saveRa = false;
};

// A register-passed call argument that still needs a copy into its ABI slot.
struct ShuffleItem {
  int32_t tgt = -1; // required physical argument register
  int32_t src = -1; // current physical register of the value
  bool isF = false;
  int32_t slot = -1; // assigned shuffle slot when the source must be saved
};

// Determine, for one call, which register args need copying and how many
// shuffle memory slots are required.
int planCall(const MInst &m, std::vector<ShuffleItem> &items) {
  items.clear();
  ArgCursor c;
  for (auto &a : m.args) {
    ArgLoc loc = c.next(a.ty == Type::F32);
    if (a.vreg < 0) continue; // already stored on the stack
    if (loc.reg < 0) continue;
    if (loc.reg != a.vreg)
      items.push_back(ShuffleItem{loc.reg, a.vreg, a.ty == Type::F32, -1});
  }
  // A source that sits in a register which is also a target of another pending
  // argument will be clobbered; those need a memory slot.
  std::unordered_set<int> tset;
  for (auto &it : items) tset.insert((it.isF ? 1000 : 0) + it.tgt);
  int slots = 0;
  for (auto &it : items) {
    if (tset.count((it.isF ? 1000 : 0) + it.src)) it.slot = slots++;
  }
  return slots;
}

int maxShuffleNeeded(const MachineFunc &mf) {
  int need = 0;
  std::vector<ShuffleItem> items;
  for (auto &b : mf.blocks)
    for (auto &m : b.instrs)
      if (m.op == MOp::Call) need = std::max(need, planCall(m, items));
  return need;
}

Frame layoutFrame(const MachineFunc &mf) {
  Frame fr;
  fr.saveRa = mf.hasCall;
  fr.outMax = align8(mf.maxStkArgBytes);
  fr.spillBytes = (int64_t)mf.spillCount * 8;
  fr.localBytes = mf.localBytes;
  fr.xN = (int)mf.usedCSX.size() + (fr.saveRa ? 1 : 0);
  fr.fN = (int)mf.usedCSF.size();
  fr.shuffleBytes = 8LL * maxShuffleNeeded(mf);
  fr.spillBase = fr.outMax + fr.shuffleBytes;
  fr.localBase = fr.spillBase + fr.spillBytes;
  fr.fBase = fr.localBase + fr.localBytes;
  fr.xBase = fr.fBase + (int64_t)fr.fN * 8;
  fr.fs = align16(fr.xBase + (int64_t)fr.xN * 8);
  return fr;
}

// ---------------------------------------------------------------------------
// one instruction line with a 12-bit-imm fallback via t0
// ---------------------------------------------------------------------------
void emitMem(std::ostringstream &os, const char *op, const std::string &reg,
             int64_t off, const std::string &base) {
  if (fits12(off)) {
    os << "\t" << op << " " << reg << ", " << off << "(" << base << ")\n";
  } else {
    os << "\tli t0, " << off << "\n";
    os << "\tadd t0, " << base << ", t0\n";
    os << "\t" << op << " " << reg << ", 0(t0)\n";
  }
}

// load value (i32/f32) addressed by [reg + off]
void emitLoadReg(std::ostringstream &os, bool isF, int32_t dst, int64_t off,
                 int32_t baseReg) {
  const char *op = isF ? "flw" : "lw";
  emitMem(os, op, isF ? frn(dst) : xrn(dst), off, xrn(baseReg));
}
// store value a at [b + off]
void emitStoreReg(std::ostringstream &os, bool isF, int32_t val, int64_t off,
                  int32_t baseReg) {
  const char *op = isF ? "fsw" : "sw";
  emitMem(os, op, isF ? frn(val) : xrn(val), off, xrn(baseReg));
}

// rd = base + off (register-form addi; long offsets go through a scratch)
void emitAddr(std::ostringstream &os, const std::string &dst, int64_t off,
              const std::string &base) {
  if (fits12(off)) {
    os << "\taddi " << dst << ", " << base << ", " << off << "\n";
  } else {
    os << "\tli t0, " << off << "\n";
    os << "\tadd t0, " << base << ", t0\n";
    os << "\tmv " << dst << ", t0\n";
  }
}

// ---------------------------------------------------------------------------
// prologue / epilogue
// ---------------------------------------------------------------------------
void emitPrologue(std::ostringstream &os, const MachineFunc &mf,
                  const Frame &fr) {
  if (fr.fs > 0) {
    if (fits12(fr.fs))
      os << "\taddi sp, sp, -" << fr.fs << "\n";
    else {
      os << "\tli t0, -" << fr.fs << "\n";
      os << "\tadd sp, sp, t0\n";
    }
  }
  int64_t xo = fr.xBase;
  const auto &csX = mf.usedCSX;
  const auto &csF = mf.usedCSF;
  // x saves: ra (if any) at the first slot, then the used callee-saved regs.
  int xi = 0;
  if (fr.saveRa) {
    emitMem(os, "sd", "ra", xo, "sp");
    xo += 8;
    xi = 1;
  }
  for (int r : csX) {
    emitMem(os, "sd", xrn(r), xo, "sp");
    xo += 8;
  }
  (void)xi;
  int64_t fo = fr.fBase;
  for (int r : csF) {
    emitMem(os, "fsw", frn(r), fo, "sp");
    fo += 8;
  }
}

void emitEpilogue(std::ostringstream &os, const MachineFunc &mf,
                  const Frame &fr, const MInst &ret) {
  // Return value first (a0 / fa0 is caller-saved and not restored below).
  if (ret.ty == Type::F32) {
    if (ret.a >= 0) {
      if (ret.a != 10) os << "\tfmv.s fa0, " << frn(ret.a) << "\n";
    } else {
      os << "\tli t0, " << ret.imm << "\n";
      os << "\tfmv.w.x fa0, t0\n";
    }
  } else if (ret.a >= 0) {
    if (ret.a != 10) os << "\tmv a0, " << xrn(ret.a) << "\n";
  } else if (ret.ty != Type::Void) {
    os << "\tli a0, " << ret.imm << "\n";
  }

  // restore callee-saved x registers (reverse order), then ra last of them
  const auto &csX = mf.usedCSX;
  const auto &csF = mf.usedCSF;
  int64_t xo = fr.xBase + (int64_t)(csX.size() + (fr.saveRa ? 1 : 0)) * 8 - 8;
  for (size_t i = csX.size(); i-- > 0;) {
    emitMem(os, "ld", xrn(csX[i]), xo, "sp");
    xo -= 8;
  }
  if (fr.saveRa) {
    emitMem(os, "ld", "ra", xo, "sp");
  }
  int64_t fo = fr.fBase + (int64_t)csF.size() * 8 - 8;
  for (size_t i = csF.size(); i-- > 0;) {
    emitMem(os, "flw", frn(csF[i]), fo, "sp");
    fo -= 8;
  }
  if (fr.fs > 0) {
    if (fits12(fr.fs))
      os << "\taddi sp, sp, " << fr.fs << "\n";
    else {
      os << "\tli t0, " << fr.fs << "\n";
      os << "\tadd sp, sp, t0\n";
    }
  }
  os << "\tret\n";
}

// ---------------------------------------------------------------------------
// register-argument shuffle before a call
// ---------------------------------------------------------------------------
void emitCallShuffle(std::ostringstream &os, const MachineFunc &mf,
                     const Frame &fr, const std::vector<ShuffleItem> &items) {
  if (items.empty()) return;
  // Phase 1: save sources that would be clobbered.
  for (auto &it : items) {
    if (it.slot < 0) continue;
    int64_t off = fr.outMax + (int64_t)it.slot * 8;
    if (it.isF)
      emitMem(os, "fsw", frn(it.src), off, "sp");
    else
      emitMem(os, "sd", xrn(it.src), off, "sp");
  }
  // Phase 2: perform the copies.
  for (auto &it : items) {
    if (it.src == it.tgt) continue;
    if (it.slot >= 0) {
      int64_t off = fr.outMax + (int64_t)it.slot * 8;
      if (it.isF)
        emitMem(os, "flw", frn(it.tgt), off, "sp");
      else
        emitMem(os, "ld", xrn(it.tgt), off, "sp");
    } else if (it.isF) {
      os << "\tfmv.s " << frn(it.tgt) << ", " << frn(it.src) << "\n";
    } else {
      os << "\tmv " << xrn(it.tgt) << ", " << xrn(it.src) << "\n";
    }
  }
}

// ---------------------------------------------------------------------------
// instruction selection -> assembly line
// ---------------------------------------------------------------------------
void emitInst(std::ostringstream &os, const MachineFunc &mf, const Frame &fr,
              const MInst &m) {
  using MOp_ = MOp;
  switch (m.op) {
  case MOp_::Prologue:
    emitPrologue(os, mf, fr);
    return;
  case MOp_::Ret:
    emitEpilogue(os, mf, fr, m);
    return;
  case MOp_::Nop:
    return;
  case MOp_::Jmp:
    os << "\tj " << m.sym << "\n";
    return;
  case MOp_::BrNz:
    os << "\tbnez " << xrn(m.a) << ", " << m.sym << "\n";
    return;
  case MOp_::BrZ:
    os << "\tbeqz " << xrn(m.a) << ", " << m.sym << "\n";
    return;
  case MOp_::BrCmp: {
    const char *op = "beq";
    switch ((Cond)m.imm) {
    case Cond::Ne: op = "bne"; break;
    case Cond::Lt: op = "blt"; break;
    case Cond::Le: op = "bge"; break; // a<=b  <=>  !(a>b)
    case Cond::Gt: op = "blt"; break; // a>b   <=>  b<a
    case Cond::Ge: op = "bge"; break;
    default: op = "beq"; break;
    }
    if ((Cond)m.imm == Cond::Le || (Cond)m.imm == Cond::Gt) {
      os << "\t" << op << " " << xrn(m.b) << ", " << xrn(m.a) << ", " << m.sym
         << "\n";
    } else {
      os << "\t" << op << " " << xrn(m.a) << ", " << xrn(m.b) << ", " << m.sym
         << "\n";
    }
    return;
  }
  case MOp_::LeaFrame:
    emitAddr(os, xrn(m.dst), fr.localBase + m.imm, "sp");
    return;
  case MOp_::LeaGlobal:
    os << "\tla " << xrn(m.dst) << ", " << m.sym << "\n";
    return;
  case MOp_::MoveX:
    os << "\tmv " << xrn(m.dst) << ", " << xrn(m.a) << "\n";
    return;
  case MOp_::MoveF:
    os << "\tfmv.s " << frn(m.dst) << ", " << frn(m.a) << "\n";
    return;
  case MOp_::Li:
    os << "\tli " << xrn(m.dst) << ", " << m.imm << "\n";
    return;
  case MOp_::LiF:
    os << "\tli t0, " << m.imm << "\n";
    os << "\tfmv.w.x " << frn(m.dst) << ", t0\n";
    return;

  // ---- parameter reads
  case MOp_::EntryInt:
    os << "\tmv " << xrn(m.dst) << ", " << xregName(m.imm) << "\n";
    return;
  case MOp_::EntryFlt:
    os << "\tfmv.s " << frn(m.dst) << ", " << fregName(m.imm) << "\n";
    return;
  case MOp_::EntryStkInt:
    emitMem(os, (m.ty == Type::Ptr) ? "ld" : "lw", xrn(m.dst),
            fr.fs + (int64_t)m.imm * 8, "sp");
    return;
  case MOp_::EntryStkFlt:
    emitMem(os, "flw", frn(m.dst), fr.fs + (int64_t)m.imm * 8, "sp");
    return;

  // ---- integer arithmetic
  case MOp_::IAdd:
    os << "\t" << (m.ty == Type::Ptr ? "add" : "addw") << " " << xrn(m.dst)
       << ", " << xrn(m.a) << ", " << xrn(m.b) << "\n";
    return;
  case MOp_::IAddI:
    if (fits12(m.imm))
      os << "\t" << (m.ty == Type::Ptr ? "addi" : "addiw") << " " << xrn(m.dst)
         << ", " << xrn(m.a) << ", " << m.imm << "\n";
    else {
      os << "\tli t0, " << m.imm << "\n";
      os << "\t" << (m.ty == Type::Ptr ? "add" : "addw") << " " << xrn(m.dst)
         << ", " << xrn(m.a) << ", t0\n";
    }
    return;
  case MOp_::ISub:
    os << "\tsubw " << xrn(m.dst) << ", " << xrn(m.a) << ", " << xrn(m.b) << "\n";
    return;
  case MOp_::INeg:
    os << "\tsubw " << xrn(m.dst) << ", zero, " << xrn(m.a) << "\n";
    return;
  case MOp_::IMul:
    os << "\tmulw " << xrn(m.dst) << ", " << xrn(m.a) << ", " << xrn(m.b) << "\n";
    return;
  case MOp_::IDiv:
    os << "\tdivw " << xrn(m.dst) << ", " << xrn(m.a) << ", " << xrn(m.b) << "\n";
    return;
  case MOp_::IRem:
    os << "\tremw " << xrn(m.dst) << ", " << xrn(m.a) << ", " << xrn(m.b) << "\n";
    return;
  case MOp_::IXor:
    os << "\txor " << xrn(m.dst) << ", " << xrn(m.a) << ", " << xrn(m.b) << "\n";
    return;
  case MOp_::IXorI:
    os << "\txori " << xrn(m.dst) << ", " << xrn(m.a) << ", " << m.imm << "\n";
    return;

  case MOp_::ISlt:
    os << "\tslt " << xrn(m.dst) << ", " << xrn(m.a) << ", " << xrn(m.b) << "\n";
    return;
  case MOp_::ISlti:
    os << "\tslti " << xrn(m.dst) << ", " << xrn(m.a) << ", " << m.imm << "\n";
    return;
  case MOp_::ISltu:
    os << "\tsltu " << xrn(m.dst) << ", " << xrn(m.a) << ", " << xrn(m.b) << "\n";
    return;
  case MOp_::ISltiu:
    os << "\tsltiu " << xrn(m.dst) << ", " << xrn(m.a) << ", " << m.imm << "\n";
    return;
  case MOp_::ISltuZ:
    os << "\tsltu " << xrn(m.dst) << ", zero, " << xrn(m.a) << "\n";
    return;

  case MOp_::ICmp: {
    std::string a = xrn(m.a), b = xrn(m.b);
    switch ((Cond)m.imm) {
    case Cond::Eq:
      os << "\txor " << xrn(m.dst) << ", " << a << ", " << b << "\n";
      os << "\tseqz " << xrn(m.dst) << ", " << xrn(m.dst) << "\n";
      break;
    case Cond::Ne:
      os << "\txor " << xrn(m.dst) << ", " << a << ", " << b << "\n";
      os << "\tsnez " << xrn(m.dst) << ", " << xrn(m.dst) << "\n";
      break;
    case Cond::Lt:
      os << "\tslt " << xrn(m.dst) << ", " << a << ", " << b << "\n";
      break;
    case Cond::Gt:
      os << "\tslt " << xrn(m.dst) << ", " << b << ", " << a << "\n";
      break;
    case Cond::Le:
      os << "\tslt " << xrn(m.dst) << ", " << b << ", " << a << "\n";
      os << "\txori " << xrn(m.dst) << ", " << xrn(m.dst) << ", 1\n";
      break;
    default: // Ge
      os << "\tslt " << xrn(m.dst) << ", " << a << ", " << b << "\n";
      os << "\txori " << xrn(m.dst) << ", " << xrn(m.dst) << ", 1\n";
      break;
    }
    return;
  }

  case MOp_::FCmp: {
    switch ((Cond)m.imm) {
    case Cond::Eq:
      os << "\tfeq.s " << xrn(m.dst) << ", " << frn(m.a) << ", " << frn(m.b)
         << "\n";
      break;
    case Cond::Ne:
      os << "\tfeq.s " << xrn(m.dst) << ", " << frn(m.a) << ", " << frn(m.b)
         << "\n";
      os << "\txori " << xrn(m.dst) << ", " << xrn(m.dst) << ", 1\n";
      break;
    case Cond::Lt:
      os << "\tflt.s " << xrn(m.dst) << ", " << frn(m.a) << ", " << frn(m.b)
         << "\n";
      break;
    case Cond::Le:
      os << "\tfle.s " << xrn(m.dst) << ", " << frn(m.a) << ", " << frn(m.b)
         << "\n";
      break;
    case Cond::Gt:
      os << "\tflt.s " << xrn(m.dst) << ", " << frn(m.b) << ", " << frn(m.a)
         << "\n";
      break;
    default:
      os << "\tfle.s " << xrn(m.dst) << ", " << frn(m.b) << ", " << frn(m.a)
         << "\n";
      break;
    }
    return;
  }

  case MOp_::FAdd:
    os << "\tfadd.s " << frn(m.dst) << ", " << frn(m.a) << ", " << frn(m.b)
       << "\n";
    return;
  case MOp_::FSub:
    os << "\tfsub.s " << frn(m.dst) << ", " << frn(m.a) << ", " << frn(m.b)
       << "\n";
    return;
  case MOp_::FMul:
    os << "\tfmul.s " << frn(m.dst) << ", " << frn(m.a) << ", " << frn(m.b)
       << "\n";
    return;
  case MOp_::FDiv:
    os << "\tfdiv.s " << frn(m.dst) << ", " << frn(m.a) << ", " << frn(m.b)
       << "\n";
    return;
  case MOp_::FNeg:
    os << "\tfneg.s " << frn(m.dst) << ", " << frn(m.a) << "\n";
    return;
  case MOp_::I2F:
    os << "\tfcvt.s.w " << frn(m.dst) << ", " << xrn(m.a) << "\n";
    return;
  case MOp_::F2I:
    os << "\tfcvt.w.s " << xrn(m.dst) << ", " << frn(m.a) << ", rtz\n";
    return;
  case MOp_::FMvWX:
    os << "\tfmv.w.x " << frn(m.dst) << ", " << xrn(m.a) << "\n";
    return;

  // ---- register-addressed memory
  case MOp_::Lw:
    emitLoadReg(os, false, m.dst, m.imm, m.a);
    return;
  case MOp_::Flw:
    emitLoadReg(os, true, m.dst, m.imm, m.a);
    return;
  case MOp_::Sw:
    emitStoreReg(os, false, m.a, m.imm, m.b);
    return;
  case MOp_::Fsw:
    emitStoreReg(os, true, m.a, m.imm, m.b);
    return;

  // ---- fused frame addressing
  case MOp_::LwF:
    emitMem(os, "lw", xrn(m.dst), fr.localBase + m.imm, "sp");
    return;
  case MOp_::FlwF:
    emitMem(os, "flw", frn(m.dst), fr.localBase + m.imm, "sp");
    return;
  case MOp_::SwF:
    emitMem(os, "sw", xrn(m.a), fr.localBase + m.imm, "sp");
    return;
  case MOp_::FswF:
    emitMem(os, "fsw", frn(m.a), fr.localBase + m.imm, "sp");
    return;

  // ---- fused global addressing
  case MOp_::LwG:
  case MOp_::FlwG:
  case MOp_::SwG:
  case MOp_::FswG: {
    bool isF = (m.op == MOp_::FlwG || m.op == MOp_::FswG);
    bool store = (m.op == MOp_::SwG || m.op == MOp_::FswG);
    os << "\tla t0, " << m.sym << "\n";
    if (m.imm != 0) {
      if (fits12(m.imm))
        os << "\taddi t0, t0, " << m.imm << "\n";
      else {
        os << "\tli t1, " << m.imm << "\n";
        os << "\tadd t0, t0, t1\n";
      }
    }
    const char *op;
    if (isF)
      op = store ? "fsw" : "flw";
    else
      op = store ? "sw" : "lw";
    os << "\t" << op << " " << (store ? (isF ? frn(m.a) : xrn(m.a))
                                     : (isF ? frn(m.dst) : xrn(m.dst)))
       << ", 0(t0)\n";
    return;
  }

  // ---- spills
  case MOp_::SpillLw:
    emitMem(os, "ld", xrn(m.dst), fr.spillBase + (int64_t)m.imm * 8, "sp");
    return;
  case MOp_::SpillFlw:
    emitMem(os, "flw", frn(m.dst), fr.spillBase + (int64_t)m.imm * 8, "sp");
    return;
  case MOp_::SpillSw:
    emitMem(os, "sd", xrn(m.a), fr.spillBase + (int64_t)m.imm * 8, "sp");
    return;
  case MOp_::SpillFsw:
    emitMem(os, "fsw", frn(m.a), fr.spillBase + (int64_t)m.imm * 8, "sp");
    return;

  // ---- outbound stack args
  case MOp_::StkArgSw:
    emitMem(os, (m.ty == Type::Ptr) ? "sd" : "sw", xrn(m.a), m.imm, "sp");
    return;
  case MOp_::StkArgFsw:
    emitMem(os, "fsw", frn(m.a), m.imm, "sp");
    return;

  // ---- call
  case MOp_::Call: {
    std::vector<ShuffleItem> items;
    planCall(m, items);
    emitCallShuffle(os, mf, fr, items);
    os << "\tcall " << m.sym << "\n";
    if (m.dst >= 0) {
      if (m.ty == Type::F32) {
        if (m.dst != 10) os << "\tfmv.s " << frn(m.dst) << ", fa0\n";
      } else if (m.dst != 10) {
        os << "\tmv " << xrn(m.dst) << ", a0\n";
      }
    }
    return;
  }
  }
}

void emitFunction(std::ostringstream &os, const MachineFunc &mf) {
  Frame fr = layoutFrame(mf);
  os << "\n\t.text\n\t.p2align 2\n\t.globl " << mf.name << "\n";
  os << mf.name << ":\n";
  for (auto &b : mf.blocks) {
    os << b.name << ":\n";
    for (auto &m : b.instrs) emitInst(os, mf, fr, m);
  }
}

// ---------------------------------------------------------------------------
// globals -> .data
// ---------------------------------------------------------------------------
void emitGlobal(std::ostringstream &os, const GlobalVar &g) {
  os << "\n\t.data\n\t.p2align 2\n\t.globl " << g.name << "\n";
  os << g.name << ":\n";
  bool isF = g.elem == Type::F32;
  int64_t n = g.elems;
  int64_t bytes = n * 4;
  if (!g.hasInit) {
    os << "\t.zero " << bytes << "\n";
    return;
  }
  bool allZero = true;
  for (auto &c : g.init) {
    if (isF ? (c.fval != 0.0f) : (c.ival != 0)) {
      allZero = false;
      break;
    }
  }
  if (allZero) {
    os << "\t.zero " << bytes << "\n";
    return;
  }
  int perLine = 8;
  for (int64_t i = 0; i < n; ++i) {
    if (i % perLine == 0) os << "\t.word ";
    else os << ", ";
    const ConstVal &c = g.init[(size_t)i];
    if (isF) {
      char buf[16];
      std::snprintf(buf, sizeof(buf), "0x%08x", c.fbits());
      os << buf;
    } else {
      os << (int32_t)c.ival;
    }
    if (i % perLine == perLine - 1 || i == n - 1) os << "\n";
  }
}

} // namespace

std::string AssemblyWriter::write(Module *mod, std::vector<MachineFunc> &fns) {
  std::ostringstream os;
  os << "\t.text\n";
  for (auto &f : fns) emitFunction(os, f);
  for (auto &g : mod->globals()) emitGlobal(os, *g);
  return os.str();
}

} // namespace backend
} // namespace sakura
