// OptUtil.cpp - see OptUtil.h
#include "OptUtil.h"

#include <cmath>

#include <unordered_set>

namespace sakura {
namespace ir {

// ---------------------------------------------------------------------------
// Traversal / collection helpers
// ---------------------------------------------------------------------------

namespace {

// Append one block and, when its terminator is a region op, descend into the
// region block lists.
void collectBlock(Module &mod, BasicBlock *bb,
                  std::vector<BasicBlock *> *blocks,
                  std::vector<Instruction *> *instrs) {
  if (blocks) blocks->push_back(bb);
  for (auto &iu : bb->instrs) {
    Instruction *inst = iu.get();
    if (instrs) instrs->push_back(inst);
    if (inst->op == Op::ScfWhile) {
      for (auto &b : inst->condRegion) collectBlock(mod, b.get(), blocks, instrs);
      for (auto &b : inst->bodyRegion) collectBlock(mod, b.get(), blocks, instrs);
    } else if (inst->op == Op::AffineFor) {
      for (auto &b : inst->bodyRegion) collectBlock(mod, b.get(), blocks, instrs);
    }
  }
}

} // namespace

std::vector<Instruction *> collectInstrs(Module &mod) {
  std::vector<Instruction *> out;
  for (auto &f : mod.functions()) {
    if (f->isLib) continue;
    for (auto &b : f->blocks) collectBlock(mod, b.get(), nullptr, &out);
  }
  return out;
}

void walkInstrs(Function *fn,
                const std::function<void(BasicBlock &bb, Instruction &inst)> &f) {
  std::function<void(BlockList &)> rec = [&](BlockList &list) {
    for (auto &bb : list) {
      for (auto &iu : bb->instrs) {
        Instruction *inst = iu.get();
        f(*bb, *inst);
        if (inst->op == Op::ScfWhile) {
          rec(inst->condRegion);
          rec(inst->bodyRegion);
        } else if (inst->op == Op::AffineFor) {
          rec(inst->bodyRegion);
        }
      }
    }
  };
  rec(fn->blocks);
}

int countUses(Module &mod, Value *v) {
  int n = 0;
  for (Instruction *inst : collectInstrs(mod))
    for (Value *o : inst->ops)
      if (o == v) ++n;
  return n;
}

void replaceAllUses(Module &mod, Value *from, Value *to) {
  for (Instruction *inst : collectInstrs(mod))
    for (auto &o : inst->ops)
      if (o == from) o = to;
}

// A block `p` whose terminator was just rewritten (e.g. a constant-armed
// cond_br folded to a plain br) may have lost an edge to a successor.  Every
// cf.phi distinguishes its incoming values by predecessor block, so a phi
// input keyed on `p` at a block `p` no longer branches to is now a stale key
// that would corrupt later edge-based rewrites (thunk collapsing re-keys such
// entries onto `p`, merging two values onto one predecessor).  Drop them.
void pruneStalePhiPred(BlockList &list, BasicBlock *p) {
  if (!p) return;
  // blocks p's current terminator still branches to
  std::unordered_set<BasicBlock *> targets;
  if (!p->instrs.empty()) {
    Instruction *last = p->instrs.back().get();
    auto push = [&](Value *v) {
      if (auto *b = dynamic_cast<BasicBlock *>(v)) targets.insert(b);
    };
    switch (last->op) {
    case Op::Br:
      push(last->ops[0]);
      break;
    case Op::CondBr:
      if (last->ops.size() >= 3) {
        push(last->ops[1]);
        push(last->ops[2]);
      }
      break;
    default:
      break; // ret / region terminator: p keeps no outgoing edges
    }
  }
  for (auto &bb : list) {
    if (bb.get() == p) continue;
    for (auto &iu : bb->instrs) {
      Instruction *phi = iu.get();
      if (phi->op != Op::Phi) continue;
      for (size_t s = 0; s + 1 < phi->ops.size();) {
        if (phi->ops[s] == (Value *)p && !targets.count(bb.get()))
          phi->ops.erase(phi->ops.begin() + (long)s,
                         phi->ops.begin() + (long)s + 2);
        else
          s += 2;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

bool isTerminator(Op o) {
  switch (o) {
  case Op::Ret:
  case Op::Br:
  case Op::CondBr:
  case Op::ScfWhile:
  case Op::ScfCondition:
  case Op::ScfYield:
  case Op::AffineFor:
  case Op::AffineYield:
    return true;
  default:
    return false;
  }
}

bool isPure(Op o) {
  switch (o) {
  case Op::Alloca:
  case Op::Gep:
  case Op::Phi: // a merge has no side effects; drop when its result is unused
  case Op::Add:
  case Op::Sub:
  case Op::Mul:
  case Op::SDiv:
  case Op::SRem:
  case Op::FAdd:
  case Op::FSub:
  case Op::FMul:
  case Op::FDiv:
  case Op::ICmp:
  case Op::FCmp:
  case Op::Sitofp:
  case Op::Fptosi:
  case Op::Not:
  case Op::Load: // loads have no side effects; dropping a dead load is safe
    return true;
  default:
    return false;
  }
}

bool isMemoryWrite(Op o) { return o == Op::Store; }
bool isCall(Op o) { return o == Op::Call; }

bool isStructural(Value *v) {
  return dynamic_cast<BasicBlock *>(v) || dynamic_cast<Function *>(v) ||
         dynamic_cast<GlobalVar *>(v);
}

const ConstantInt *asConstInt(const Value *v) {
  return dynamic_cast<const ConstantInt *>(v);
}
const ConstantFloat *asConstFloat(const Value *v) {
  return dynamic_cast<const ConstantFloat *>(v);
}

// ---------------------------------------------------------------------------
// Constant arithmetic
// ---------------------------------------------------------------------------

FoldRes intRes(int32_t v) {
  FoldRes r;
  r.ok = true;
  r.ival = v;
  return r;
}
FoldRes fltRes(float v) {
  FoldRes r;
  r.ok = true;
  r.fval = v;
  r.isFloat = true;
  return r;
}

FoldRes foldArith(Op op, Cond c, const FoldRes &l, const FoldRes &r) {
  // Unary forms first (the second operand is ignored).
  switch (op) {
  case Op::Not: return intRes(l.ival == 0 ? 1 : 0);
  case Op::Sitofp: return fltRes((float)l.ival);
  case Op::Fptosi: return intRes((int32_t)l.fval);
  default: break;
  }
  bool isF = op == Op::FAdd || op == Op::FSub || op == Op::FMul ||
             op == Op::FDiv || op == Op::FCmp;
  if (isF) {
    float a = l.isFloat ? l.fval : (float)l.ival;
    float b = r.isFloat ? r.fval : (float)r.ival;
    switch (op) {
    case Op::FAdd: return fltRes(a + b);
    case Op::FSub: return fltRes(a - b);
    case Op::FMul: return fltRes(a * b);
    case Op::FDiv: return (b == 0.0f) ? FoldRes{} : fltRes(a / b);
    case Op::FCmp: {
      bool t = false;
      switch (c) {
      case Cond::Eq: t = (a == b); break;
      case Cond::Ne: t = (a != b); break;
      case Cond::Lt: t = (a < b); break;
      case Cond::Le: t = (a <= b); break;
      case Cond::Gt: t = (a > b); break;
      case Cond::Ge: t = (a >= b); break;
      }
      return intRes(t ? 1 : 0);
    }
    default: return FoldRes{};
    }
  }
  int32_t a = l.ival;
  int32_t b = r.ival;
  switch (op) {
  case Op::Add: return intRes(a + b);
  case Op::Sub: return intRes(a - b);
  case Op::Mul: return intRes(a * b);
  case Op::SDiv: return (b == 0) ? FoldRes{} : intRes(a / b);
  case Op::SRem: return (b == 0) ? FoldRes{} : intRes(a % b);
  case Op::ICmp: {
    bool t = false;
    switch (c) {
    case Cond::Eq: t = (a == b); break;
    case Cond::Ne: t = (a != b); break;
    case Cond::Lt: t = (a < b); break;
    case Cond::Le: t = (a <= b); break;
    case Cond::Gt: t = (a > b); break;
    case Cond::Ge: t = (a >= b); break;
    }
    return intRes(t ? 1 : 0);
  }
  default: return FoldRes{};
  }
}

// ---------------------------------------------------------------------------
// Address classification (memref.gep chains)
// ---------------------------------------------------------------------------

AddrInfo addressOf(Value *ptr) {
  AddrInfo info;
  if (auto *g = dynamic_cast<GlobalVar *>(ptr)) {
    info.kind = RootKind::Global;
    info.root = g;
    info.constOff = true;
    return info;
  }
  if (auto *a = dynamic_cast<Argument *>(ptr)) {
    if (a->ty == Type::Ptr) {
      info.kind = RootKind::Param;
      info.root = a;
      info.constOff = true;
    }
    return info;
  }
  auto *inst = dynamic_cast<Instruction *>(ptr);
  if (!inst) return info;
  if (inst->op == Op::Alloca) {
    info.kind = RootKind::Alloca;
    info.root = inst;
    info.constOff = true;
    return info;
  }
  if (inst->op != Op::Gep) {
    info.kind = RootKind::Other;
    info.root = inst;
    return info;
  }
  // memref.gep: addr = gep(base, byteOffset).  Fold a chain of constant
  // offsets into one; a dynamic offset anywhere makes the address dynamic.
  AddrInfo base = addressOf(inst->ops[0]);
  info = base;
  if (!base.root) return info;
  if (auto *ci = asConstInt(inst->ops[1])) {
    info.off += ci->v; // keep accumulating along the chain
  } else {
    info.constOff = false;
  }
  return info;
}

} // namespace ir
} // namespace sakura
