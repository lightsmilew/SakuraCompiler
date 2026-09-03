// ---------------------------------------------------------------------------
// SakuraCompiler mid-end IR (MLIR-style).
//
// The mid-end is a self-contained SSA representation modelled on MLIR rather
// than on textual LLVM IR:
//
//   * every opcode is classified into an MLIR-style *dialect* (func / arith /
//     cf / memref) and carries a fully-qualified operation name, so the dump
//     produced by Module::toString() reads like MLIR assembly, e.g.
//
//       module {
//         func.func private @putint(i32)
//         memref.global @a : [4 x i32] = [i32 1, i32 2, i32 3, i32 4]
//         func.func @main() -> i32 {
//           %0 = arith.constant 10 : i32
//           %1 = memref.alloca : i32
//           memref.store %0, %1 : i32
//           %2 = memref.load %1 : i32
//           %3 = arith.addi %2, %0 : i32
//           func.return %3
//         }
//       }
//
//   * SSA values carry MLIR scalar types: i32, f32 and ptr.
//
// Memory model: SysY scalars/arrays are memory-resident objects; only
// expression temporaries flow through SSA values.  memref.alloca allocates a
// stack object (its result is the object's address), memref.gep folds a byte
// offset into an address, and memref.load/memref.store access memory through
// an address.
//
// The flat Op/Type enums are kept as the single dispatch key that the
// back-end instruction selector switches on; the dialect layer on top of them
// is what makes the IR *read* as MLIR.
// ---------------------------------------------------------------------------
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../common/Common.h"

namespace sakura {
namespace ir {

// ---------------------------------------------------------------------------
// Value types.  i32/f32 are the scalar SSA types; ptr is an address (result
// of alloca / gep / a global symbol / an array argument).
// ---------------------------------------------------------------------------
enum class Type : uint8_t { Void, I32, F32, Ptr };

// MLIR-style type spelling used by the dump ("f32", not LLVM's "float").
inline const char *typeStr(Type t) {
  switch (t) {
  case Type::I32: return "i32";
  case Type::F32: return "f32";
  case Type::Ptr: return "ptr";
  default: return "void";
  }
}

// ---------------------------------------------------------------------------
// Opcodes.  The comment on every group names the MLIR dialect the op is
// lowered into; the enum itself stays flat because the back-end switches on
// these single code points during instruction selection.
// ---------------------------------------------------------------------------
enum class Op : uint8_t {
  // func dialect
  Ret,  Call,                     // func.return / func.call

  // cf dialect (structured control flow)
  Br,   CondBr,                   // cf.br / cf.cond_br

  // memref dialect (memory objects; addresses are `ptr` values)
  Alloca, Load, Store, Gep,       // memref.alloca / load / store / gep

  // arith dialect
  Add,  Sub,  Mul,  SDiv, SRem,   // arith.{addi, subi, muli, divsi, remsi}
  FAdd, FSub, FMul, FDiv,         // arith.{addf, subf, mulf, divf}
  ICmp, FCmp,                     // arith.{cmpi, cmpf}; produce i32 0/1
  Sitofp, Fptosi,                 // arith.{sitofp, fptosi}
  Not,                            // arith.xori with i32 1  ('!' on a 0/1 value)

  // tensor dialect: whole-tensor memory operations.  These keep the tensor
  // semantics of the source language intact (a tensor is a first-class value
  // with a statically known shape) so a later vectoriser can act on them; the
  // back-end expands each op into machine loops.
  //
  // Common layout: ops[0] is the *destination buffer* (ptr), ops[1..] are the
  // sources - whole-tensor buffers (ptr) or scalar SSA values (i32/f32, the
  // latter encode "broadcast this scalar to every element").
  //
  //   TAddI/TAddF..TRemI  elementwise arithmetic (dest = a op b)
  //   TNegI/TNegF         elementwise unary negation (dest = -a)
  //   TCopy               whole-tensor copy (dest = src)
  //   TMatmulI/TMatmulF   matrix product dest[M,P] = a[M,N] @ b[N,P]
  //
  // Instruction::elem holds the element scalar type; Instruction::shape the
  // static tensor dims (for matmul it is {M, N, P}).
  TAddI, TAddF, TSubI, TSubF, TMulI, TMulF, TDivI, TDivF, TRemI,
  TNegI, TNegF, TCopy,
  TMatmulI, TMatmulF,
};

// Comparison predicates (shared by arith.cmpi / arith.cmpf).
enum class Cond : uint8_t {
  Eq, Ne, Lt, Le, Gt, Ge
};

// ---------------------------------------------------------------------------
// MLIR dialects recognised by the mid-end.  Each opcode is a member of
// exactly one dialect; opName() yields its fully-qualified spelling, e.g.
// "func.call", "arith.addi", "cf.cond_br".
// ---------------------------------------------------------------------------
enum class Dialect : uint8_t { Func, Arith, Cf, Memref, Tensor };

const char *dialectName(Dialect d);   // "func" | "arith" | "cf" | "memref"
Dialect dialectOf(Op o);              // dialect an opcode belongs to
const char *opMnemonic(Op o);         // e.g. "call", "addi", "cond_br"
inline std::string opName(Op o) {
  return std::string(dialectName(dialectOf(o))) + "." + opMnemonic(o);
}

struct Value {
  Type ty = Type::Void;
  virtual ~Value() = default;
  bool isType(Type t) const { return ty == t; }
};

// ---- constants -----------------------------------------------------------
struct ConstantInt : Value {
  int32_t v = 0;
  explicit ConstantInt(int32_t x) { ty = Type::I32; v = x; }
};
struct ConstantFloat : Value {
  float v = 0.0f;
  explicit ConstantFloat(float x) { ty = Type::F32; v = x; }
  uint32_t bits() const { return f2bits(v); }
};

// A global object.  `elems` is the number of scalar elements (1 for scalars).
struct GlobalVar : Value {
  Type elem = Type::I32;
  int64_t elems = 1;
  bool isConst = false;
  bool hasInit = false;
  std::vector<ConstVal> init;   // flat initial values (elems entries when hasInit)
  std::string name;
  explicit GlobalVar(const std::string &n, Type e, int64_t cnt, bool cnst) {
    name = n; elem = e; elems = cnt; isConst = cnst; ty = Type::Ptr;
  }
};

struct BasicBlock;
struct Instruction;

// Function argument (array arguments have Type::Ptr).
struct Argument : Value {
  int index = 0;
  std::string name;
};

struct Function;

// ---- instructions --------------------------------------------------------
// Every instruction owns its operands as raw pointers to other Values.  It is
// itself a Value when it produces a result (ty != Void).  Instructions are
// owned by their BasicBlock.  Instruction::op is the single dispatch key used
// by the back-end; its MLIR dialect name is available via opName(op).
struct Instruction : Value {
  Op op = Op::Ret;
  Cond cond = Cond::Eq;         // ICmp / FCmp predicate
  std::vector<Value *> ops;     // operands
  std::string name;             // result temp name (e.g. "t3"); dump renumbers
  Type elem = Type::I32;        // Alloca element type / Load-Store element hint
                               // / tensor-op element scalar type
  int64_t n = 0;                // Alloca element count
  std::vector<int64_t> shape;   // tensor-op static shape (dims; {M,N,P} for matmul)
  int line = 0;                 // source line (debug)

  Instruction(Op o, Type t) : op(o) { ty = t; }
  bool hasResult() const { return ty != Type::Void; }
};

// ---- basic block ---------------------------------------------------------
struct BasicBlock : Value {
  std::string name;
  std::vector<std::unique_ptr<Instruction>> instrs;
  bool hasTerminator() const;
  explicit BasicBlock(const std::string &n) { name = n; ty = Type::Void; }
};

// ---- functions -----------------------------------------------------------
struct ParamDesc {
  Type ty = Type::I32;     // I32 / F32 / Ptr (array)
  bool isArray = false;
  std::string name;
};

struct Function : Value {
  std::string name;
  Type ret = Type::I32;
  std::vector<ParamDesc> params;
  std::vector<std::unique_ptr<Argument>> args;      // populated for defs
  std::vector<std::unique_ptr<BasicBlock>> blocks;
  bool isLib = false;        // library function, no body

  BasicBlock *entry() const { return blocks.empty() ? nullptr : blocks.front().get(); }
};

// ---- module --------------------------------------------------------------
class Module {
public:
  explicit Module(const std::string &id) : name_(id) {}

  // constant creation (owned by the module arena)
  ConstantInt *constInt(int32_t v);
  ConstantFloat *constFloat(float v);

  GlobalVar *addGlobal(const std::string &name, Type elem, int64_t elems,
                       bool isConst, bool hasInit,
                       const std::vector<ConstVal> &init);
  Function *addFunction(const std::string &name, Type ret,
                        const std::vector<ParamDesc> &params, bool isLib);

  GlobalVar *getGlobal(const std::string &name) const;
  Function *getFunction(const std::string &name) const;
  void reserveBuiltin(const std::string &name);

  const std::string &id() const { return name_; }
  std::vector<std::unique_ptr<GlobalVar>> &globals() { return globals_; }
  std::vector<std::unique_ptr<Function>> &functions() { return functions_; }

  // Prints the module in MLIR-style assembly (see the file comment above).
  std::string toString();

private:
  std::string name_;
  std::vector<std::unique_ptr<GlobalVar>> globals_;
  std::vector<std::unique_ptr<Function>> functions_;
  std::vector<std::unique_ptr<Value>> arena_;          // constants / misc
  std::unordered_map<std::string, GlobalVar *> globalMap_;
  std::unordered_map<std::string, Function *> funcMap_;
};

} // namespace ir
} // namespace sakura
