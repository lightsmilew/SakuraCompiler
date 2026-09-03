// RISC-V machine-level IR used by the backend.
// Instructions operate on *virtual registers*; a later graph-colouring pass
// maps them to physical registers (or spill slots).  Frame/global/stack
// addressing is represented with dedicated opcodes so the assembler writer can
// fuse displacements directly into load/store instructions.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "../midend/ir/IR.h"

namespace sakura {
namespace backend {

using ir::Type;
using ir::Cond;

// Register files.
enum RFile { RF_X = 0, RF_F = 1 };

// --------------------------------------------------------------------------
// machine opcodes
// --------------------------------------------------------------------------
enum class MOp : uint8_t {
  // markers / control
  Prologue,      // start of function prologue expansion (no operands)
  Ret,           // a = optional return-value vreg (file = function return)
  Jmp,           // sym = target label
  BrNz,          // if a(X) != 0 jump to sym
  BrZ,           // if a(X) == 0 jump to sym
  BrCmp,         // int compare branch: jump to sym when a rel b holds
                 //   (imm = Cond, signed)

  // address materialisation
  LeaFrame,      // dst(X) = sp + frame-offset imm(local byte offset)
  LeaGlobal,     // dst(X) = address of symbol sym

  // moves
  MoveX, MoveF,  // dst = a  (mv / fmv.s)

  // constant materialisation
  Li,            // dst(X) = imm (i32)
  LiF,           // dst(F) = imm (f32 bit pattern)

  // entry parameter reads (imm = parameter ordinal)
  EntryInt,      // dst(X) = GP argument register for param imm
  EntryFlt,      // dst(F) = FP argument register for param imm
  EntryStkInt,   // dst(X) = incoming stack GP arg slot imm
  EntryStkFlt,   // dst(F) = incoming stack FP arg slot imm

  // integer arithmetic / logic (signed unless noted)
  IAdd, IAddI,   // dst = a + b | a + imm
  ISub,          // dst = a - b
  INeg,          // dst = -a
  IMul, IDiv, IRem,
  IXor, IXorI,   // dst = a ^ b | a ^ imm
  ISlt,          // dst = (a < b) signed
  ISlti,         // dst = (a < imm)
  ISltu,         // dst = (a <u b)
  ISltiu,        // dst = (a <u imm)
  ISltuZ,        // dst = (0 <u a)  == (a != 0)
  ICmp,          // dst(X) = int compare, imm = Cond (lowered by writer)
  FCmp,          // dst(X) = float compare, imm = Cond, uses a,b(F)

  // float arithmetic
  FAdd, FSub, FMul, FDiv, FNeg,  // dst(F) = ...

  // conversions
  I2F,           // dst(F) = (float)a(X)
  F2I,           // dst(X) = (int)a(F), truncation (rtz)
  FMvWX,         // dst(F) = a(X) bit move

  // memory: register-addressed (base vreg in `a`)
  Lw, Flw,       // dst = *(i32|f32) a
  Sw, Fsw,       // *(i32|f32) b = a        (b = address vreg)

  // memory: fused frame / global / spill addressing
  LwF, FlwF, SwF, FswF,   // imm = local frame byte offset (value in `a` for stores)
  LwG, FlwG, SwG, FswG,   // sym = global, imm = byte displacement
  SpillLw, SpillFlw, SpillSw, SpillFsw,  // imm = spill-slot index

  // outbound stack argument store (before a Call)
  StkArgSw, StkArgFsw,   // [sp + imm] = a

  // call (sym = callee, args, dst = result vreg or -1)
  Call,          // imm = outbound stack bytes

  Nop
};

struct MArg {
  int32_t vreg = -1;
  Type ty = Type::I32;
};

struct MInst {
  MOp op = MOp::Nop;
  int32_t dst = -1;   // defined vreg (-1 if none)
  int32_t a = -1;     // first use vreg
  int32_t b = -1;     // second use vreg
  int32_t imm = 0;    // displacement / cond / param ordinal / imm value
  Type ty = Type::I32; // type of the primary def/use value
  std::string sym;  // label / global / callee
  std::vector<MArg> args; // call arguments
  int line = 0;
};

struct MBlock {
  std::string name;
  std::vector<MInst> instrs;
};

struct MachineFunc {
  std::string name;
  Type ret = Type::I32;
  std::vector<Type> params;     // I32/Ptr -> GP, F32 -> FP
  std::vector<MBlock> blocks;
  bool hasCall = false;
  std::vector<int> usedCSX;     // physical callee-saved x registers in use
  std::vector<int> usedCSF;     // physical callee-saved f registers in use
  int localBytes = 0;           // frame bytes for ISel-assigned locals
  int spillCount = 0;           // number of RA spill slots
  int maxStkArgBytes = 0;       // max outbound stack argument bytes
};

// --------------------------------------------------------------------------
// physical register names / conventions
// --------------------------------------------------------------------------
constexpr int kScratchX1 = 5;  // t0
constexpr int kScratchX2 = 6;  // t1
constexpr int kScratchF1 = 0;  // ft0
constexpr int kScratchF2 = 1;  // ft1
constexpr int kX0 = 0, kRA = 1, kSP = 2;

const char *xregName(int n);
const char *fregName(int n);

// Operand classification of an instruction: definition vreg (if any) and the
// up-to-2 explicit use vregs.  `a` of Call-style args and fused-address ops
// are handled by their dedicated opcodes in liveness.
struct OpSlots {
  int def = -1;
  RFile defFile = RF_X;
  int use[3] = {-1, -1, -1};
  RFile useFile[3] = {RF_X, RF_X, RF_X};
};
OpSlots slots(const MInst &m);

// --------------------------------------------------------------------------
// RV64 function-call argument placement (SysY has only i32/f32).
//   GP stream: a0..a7; FP stream: fa0..fa7; once a stream is exhausted its
//   overflow arguments go to the shared stack area in 8-byte slots.  Float
//   arguments never spill into GP registers (matches the reference ABI and
//   keeps caller/callee placement trivially consistent).
// --------------------------------------------------------------------------
struct ArgLoc {
  bool isFloat = false;
  int reg = -1;   // physical GP reg if isFloat==false else physical FP reg
  int stk = -1;   // stack slot index (-1 when passed in a register)
};
class ArgCursor {
public:
  ArgLoc next(bool isFloat);
private:
  int gp_ = 0;  // next GP stream index 0..7
  int fp_ = 0;  // next FP stream index 0..7
  int stk_ = 0; // next stack slot
};

// stack slot byte offset
inline int stkOff(int slot) { return slot * 8; }

} // namespace backend
} // namespace sakura
