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
#include <unordered_set>
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

// A block whose terminator was rewritten may have lost an edge: drop every
// cf.phi input (in `list`) keyed on it that no live edge supports, keeping
// phi input lists in sync with the real predecessor edges.
void pruneStalePhiPred(BlockList &list, BasicBlock *p);

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

// ---------------------------------------------------------------------------
// Interprocedural side effects (see OptEffects.cpp).
//
// A per-module summary of what each function can read and write.  Its purpose
// is to give the mid-end the information LLVM gets from inferred function
// attributes (`readnone` / `readonly`): when a call is provably a pure
// function of its arguments, it can be common-subexpression-eliminated and
// it does not invalidate any cached memory value.
// ---------------------------------------------------------------------------
struct FuncEffects {
  bool reads = false;              // may read memory
  bool writes = false;             // may write memory at all
  bool readsParam = false;         // reads memory rooted at an argument
  bool writesParam = false;        // writes memory rooted at an argument
  bool opaque = false;             // calls something opaque (library/indirect)
  // The call provably returns (LLVM's `willreturn`).  This is what makes it
  // legal for LICM to *speculate* an invariant call out of a loop that might
  // not execute it at all: a call that writes nothing and always returns can
  // be run an extra time without changing any observation.
  bool terminates = false;
  std::unordered_set<GlobalVar *> readGlobals;  // globals it may read
  std::unordered_set<GlobalVar *> writeGlobals; // globals it may write
};

class Effects {
public:
  void compute(Module &mod);

  const FuncEffects *of(Function *f) const;

  // The address of this global is handed out somewhere (passed to a call,
  // stored, ...), so code outside the analysed module could write it.
  bool globalEscapes(GlobalVar *g) const;

  // Calling `f` cannot modify any memory the caller can later observe, so
  // every cached load in the caller stays valid across the call.
  bool callWritesNothing(Function *f) const;

  // Calling `f` with some arguments always yields the same result, no matter
  // what the caller did to memory before: `f` writes nothing and everything it
  // reads is provably immutable.  Two such calls with equal arguments are the
  // same value.
  bool callIsValue(Function *f) const;

  // May a call to `f` write the object `a` is an address into?  This is the
  // question loop-invariant code motion has to answer before hoisting a load
  // across a call, and LLVM answers it with BasicAA plus the inferred
  // `readnone`/`readonly` attributes.  Here the attribute half comes from the
  // summary (does the callee write at all, and through what kind of pointer)
  // and the alias half from the escape set: a callee that only writes through
  // its pointer arguments can reach a private object only if that object's
  // address was handed out somewhere in the module.
  //
  // `f == nullptr` (an indirect call) is assumed to write everything.
  bool callMayWrite(Function *f, const AddrInfo &a) const;

  // Does a call to `f` provably return (LLVM's `willreturn`)?  Speculating a
  // call - running it on a path that would not have run it - is only sound
  // when it both has no side effects and always comes back.
  bool terminates(Function *f) const {
    const FuncEffects *e = of(f);
    return e != nullptr && e->terminates;
  }

private:
  std::unordered_map<Function *, FuncEffects> fx_;
  std::unordered_set<GlobalVar *> written_;  // globals written by anyone
  std::unordered_set<Value *> escaped_;      // objects whose address escapes
};

} // namespace ir
} // namespace sakura
