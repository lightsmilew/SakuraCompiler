// Machine-level support implementation: physical register name tables for the
// RISC-V ABI and shared utilities shared by ISel/RA/Asm.
#include "Machine.h"

namespace sakura {
namespace backend {

const char *xregName(int n) {
  static const char *names[32] = {
      "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2",
      "s0",   "s1", "a0", "a1", "a2", "a3", "a4", "a5",
      "a6",   "a7", "s2", "s3", "s4", "s5", "s6", "s7",
      "s8",   "s9", "s10", "s11", "t3", "t4", "t5", "t6"};
  return names[n & 31];
}

const char *fregName(int n) {
  static const char *names[32] = {
      "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7",
      "fs0", "fs1", "fa0", "fa1", "fa2", "fa3", "fa4", "fa5",
      "fa6", "fa7", "fs2", "fs3", "fs4", "fs5", "fs6", "fs7",
      "fs8", "fs9", "fs10", "fs11", "ft8", "ft9", "ft10", "ft11"};
  return names[n & 31];
}

ArgLoc ArgCursor::next(bool isFloat) {
  ArgLoc loc;
  loc.isFloat = isFloat;
  if (isFloat) {
    if (fp_ < 8) {
      loc.reg = 10 + fp_++;          // fa0..fa7
    } else {
      loc.stk = stk_++;
    }
  } else {
    if (gp_ < 8) {
      loc.reg = 10 + gp_++;          // a0..a7
    } else {
      loc.stk = stk_++;
    }
  }
  return loc;
}

// --------------------------------------------------------------------------
// operand classification used by liveness / register allocation
// --------------------------------------------------------------------------
OpSlots slots(const MInst &m) {
  OpSlots s;
  switch (m.op) {
  case MOp::Ret:
    if (m.a >= 0) {
      s.use[0] = m.a;
      s.useFile[0] = m.ty == Type::F32 ? RF_F : RF_X;
    }
    break;
  case MOp::BrNz:
  case MOp::BrZ:
    s.use[0] = m.a;
    s.useFile[0] = RF_X;
    break;
  case MOp::BrCmp:
    s.use[0] = m.a;
    s.useFile[0] = RF_X;
    s.use[1] = m.b;
    s.useFile[1] = RF_X;
    break;
  case MOp::LeaFrame:
  case MOp::LeaGlobal:
  case MOp::Li:
    s.def = m.dst;
    s.defFile = RF_X;
    break;
  case MOp::LiF:
    s.def = m.dst;
    s.defFile = RF_F;
    break;
  case MOp::MoveX:
    s.def = m.dst; s.defFile = RF_X;
    s.use[0] = m.a; s.useFile[0] = RF_X;
    break;
  case MOp::MoveF:
    s.def = m.dst; s.defFile = RF_F;
    s.use[0] = m.a; s.useFile[0] = RF_F;
    break;
  case MOp::EntryInt:
  case MOp::EntryStkInt:
    s.def = m.dst; s.defFile = RF_X;
    break;
  case MOp::EntryFlt:
  case MOp::EntryStkFlt:
    s.def = m.dst; s.defFile = RF_F;
    break;
  case MOp::IAdd:
  case MOp::ISub:
  case MOp::IMul:
  case MOp::IDiv:
  case MOp::IRem:
  case MOp::IXor:
  case MOp::ISlt:
  case MOp::ISltu:
  case MOp::ICmp:
    s.def = m.dst; s.defFile = RF_X;
    s.use[0] = m.a; s.useFile[0] = RF_X;
    s.use[1] = m.b; s.useFile[1] = RF_X;
    break;
  case MOp::IAddI:
  case MOp::INeg:
  case MOp::IXorI:
  case MOp::ISlti:
  case MOp::ISltiu:
  case MOp::ISltuZ:
    s.def = m.dst; s.defFile = RF_X;
    s.use[0] = m.a; s.useFile[0] = RF_X;
    break;
  case MOp::FCmp:
    s.def = m.dst; s.defFile = RF_X;
    s.use[0] = m.a; s.useFile[0] = RF_F;
    s.use[1] = m.b; s.useFile[1] = RF_F;
    break;
  case MOp::FAdd:
  case MOp::FSub:
  case MOp::FMul:
  case MOp::FDiv:
    s.def = m.dst; s.defFile = RF_F;
    s.use[0] = m.a; s.useFile[0] = RF_F;
    s.use[1] = m.b; s.useFile[1] = RF_F;
    break;
  case MOp::FNeg:
    s.def = m.dst; s.defFile = RF_F;
    s.use[0] = m.a; s.useFile[0] = RF_F;
    break;
  case MOp::I2F:
    s.def = m.dst; s.defFile = RF_F;
    s.use[0] = m.a; s.useFile[0] = RF_X;
    break;
  case MOp::F2I:
    s.def = m.dst; s.defFile = RF_X;
    s.use[0] = m.a; s.useFile[0] = RF_F;
    break;
  case MOp::FMvWX:
    s.def = m.dst; s.defFile = RF_F;
    s.use[0] = m.a; s.useFile[0] = RF_X;
    break;
  case MOp::Lw:
    s.def = m.dst; s.defFile = RF_X;
    s.use[0] = m.a; s.useFile[0] = RF_X;
    break;
  case MOp::Flw:
    s.def = m.dst; s.defFile = RF_F;
    s.use[0] = m.a; s.useFile[0] = RF_X;
    break;
  case MOp::Sw:
    s.use[0] = m.a; s.useFile[0] = RF_X;   // value
    s.use[1] = m.b; s.useFile[1] = RF_X;   // address
    break;
  case MOp::StkArgSw:
    s.use[0] = m.a; s.useFile[0] = RF_X;   // value, [sp+imm]
    break;
  case MOp::Fsw:
    s.use[0] = m.a; s.useFile[0] = RF_F;   // value
    s.use[1] = m.b; s.useFile[1] = RF_X;   // address
    break;
  case MOp::StkArgFsw:
    s.use[0] = m.a; s.useFile[0] = RF_F;   // value, [sp+imm]
    break;
  case MOp::LwF:
  case MOp::LwG:
  case MOp::SpillLw:
    s.def = m.dst; s.defFile = RF_X;
    break;
  case MOp::FlwF:
  case MOp::FlwG:
  case MOp::SpillFlw:
    s.def = m.dst; s.defFile = RF_F;
    break;
  case MOp::SwF:
  case MOp::SwG:
  case MOp::SpillSw:
    s.use[0] = m.a; s.useFile[0] = RF_X;
    break;
  case MOp::FswF:
  case MOp::FswG:
  case MOp::SpillFsw:
    s.use[0] = m.a; s.useFile[0] = RF_F;
    break;
  case MOp::Call:
    s.def = m.dst;
    s.defFile = m.ty == Type::F32 ? RF_F : RF_X;
    break;
  default:
    break; // Prologue, Jmp, Nop
  }
  return s;
}

} // namespace backend
} // namespace sakura
