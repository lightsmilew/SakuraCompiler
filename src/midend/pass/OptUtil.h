// ---------------------------------------------------------------------------
// OptUtil.h: shared helpers for mid-end optimisation passes.
//
// The SakuraCompiler IR is memory-resident: scalar/array variables are
// addressable objects (memref.alloca / memref.global / array params) accessed
// with memref.load / memref.store through memref.gep addresses.  All passes
// here must therefore be conservative about anything that can write memory:
// a func.call or a store through an address whose base may alias is a
// "kill".  Control flow is structured: a function body is a list of blocks,
// and an scf.while / affine.for region op owns nested block lists.
// ---------------------------------------------------------------------------
#pragma once

#include <functional>
#include <unordered_map>
#include <vector>

#include "../ir/IR.h"

namespace sakura {
namespace ir {

// Recursively visit every block of a function (function body, then each
// region-op's cond/body lists) with a callback that may inspect one block.
using BlockList = std::vector<std::unique_ptr<BasicBlock>>;

// Visit all instructions of one block list recursively (descending into the
// regions of scf.while / affine.for terminator ops).
void walkInstrs(Function *fn,
                const std::function<void(BasicBlock &bb, Instruction &inst)> &f);

// All instructions of the module, in a stable order (function order).
std::vector<Instruction *> collectInstrs(Module &mod);

// Number of operand references to `v` across the whole module.
int countUses(Module &mod, Value *v);

// Redirect every operand equal to `from` to `to` (SSA use replacement).
void replaceAllUses(Module &mod, Value *from, Value *to);

// Side-effect classification (see file comment for the memory model).
bool isPure(Op o);              // no side effects: safe to delete if unused
bool isMemoryRead(Op o);        // memref.load
bool isMemoryWrite(Op o);       // memref.store
bool isCall(Op o);              // func.call (may read/write anything)
bool isTerminator(Op o);        // ends a block

// True when `v` is one of the special structural operands (a BasicBlock, a
// Function or a GlobalVar), i.e. not an SSA value we would ever rewrite.
bool isStructural(Value *v);

// A ConstantInt / ConstantFloat helper (mirrors dynamic_cast).
const ConstantInt *asConstInt(const Value *v);
const ConstantFloat *asConstFloat(const Value *v);

// Integer/float arithmetic semantics shared by constant folding / combine.
// Returns false when the fold is not representable (e.g. div-by-zero).
struct FoldRes {
  bool ok = false;
  int32_t ival = 0;
  float fval = 0.0f;
  bool isFloat = false;
};
// Fold `l op r` (or unary `0 op r` for TNeg-style) with both operands
// constant.  `op` must be one of the arith opcodes (Add..Not, ICmp/FCmp,
// Sitofp/Fptosi); tensor/affine/scf ops are not foldable here.
FoldRes foldArith(Op op, Cond c, const FoldRes &l, const FoldRes &r);

// Classify the "address" of a pointer value: its root object.
enum class RootKind { Alloca, Global, Param, Other };
struct AddrInfo {
  RootKind kind = RootKind::Other;
  Value *root = nullptr;      // the Alloca inst / GlobalVar / Argument
  int64_t off = 0;            // constant byte offset when fully static
  bool constOff = false;      // true when `off` is exact (no dynamic part)
};
// Follows memref.gep chains: address = root + off.
AddrInfo addressOf(Value *ptr);

} // namespace ir
} // namespace sakura
