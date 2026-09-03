// Instruction selection implementation: walks each IR function's basic blocks
// and maps operations onto machine instructions.  Local variables get frame
// slots (via memref.alloca), arrays get contiguous 4-byte-slot regions, and
// every call gets outbound stack-argument space sized per the RISC-V ABI.
#include "ISel.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>

#include "../common/Common.h"
#include "../midend/pass/Passes.h" // isCfOnly: the back-end only accepts cf

using namespace sakura::ir;
using namespace sakura::backend;

namespace sakura {
namespace backend {

namespace {

inline bool fits12(int64_t v) { return v >= -2048 && v <= 2047; }

Cond invertCond(Cond c) {
  switch (c) {
  case Cond::Eq: return Cond::Ne;
  case Cond::Ne: return Cond::Eq;
  case Cond::Lt: return Cond::Ge;
  case Cond::Le: return Cond::Gt;
  case Cond::Gt: return Cond::Le;
  default: return Cond::Lt; // Ge
  }
}

// ---------------------------------------------------------------------------
// Address expression: one of
//   0 = frame slot   (off = local byte offset from locals area)
//   1 = global       (g + off bytes)
//   2 = register     (reg + off bytes)
// ---------------------------------------------------------------------------
struct PtrVal {
  int kind = 2;
  GlobalVar *g = nullptr;
  int64_t off = 0;
  int32_t reg = -1;
};

// ---------------------------------------------------------------------------
class FnLower {
public:
  MachineFunc mf;
  Function *fn = nullptr;

  std::unordered_map<const Value *, int32_t> vreg;
  std::unordered_map<const Value *, PtrVal> ptrs;
  std::unordered_map<const Instruction *, int64_t> frOff;
  std::unordered_set<const Instruction *> liveAlloca;

  int nextReg = 0;
  int nextFrameByte = 0;
  int maxStk = 0;
  MBlock *cur = nullptr;

  std::string labelFor(BasicBlock *bb) const {
    return ".L" + mf.name + "_" + bb->name;
  }

  int32_t newReg() { return nextReg++; }

  MInst *emit(MInst m) {
    cur->instrs.push_back(std::move(m));
    return &cur->instrs.back();
  }

  // ---- value -> register ----------------------------------------------
  int32_t forceReg(Value *v, int line) {
    auto it = vreg.find(v);
    if (it != vreg.end()) return it->second;
    MInst m;
    if (auto *ci = dynamic_cast<ConstantInt *>(v)) {
      m.op = MOp::Li;
      m.dst = newReg();
      m.imm = ci->v;
      m.line = line;
      emit(m);
      return vreg[v] = m.dst;
    }
    if (auto *cf = dynamic_cast<ConstantFloat *>(v)) {
      m.op = MOp::LiF;
      m.dst = newReg();
      m.imm = (int32_t)cf->bits();
      m.line = line;
      emit(m);
      return vreg[v] = m.dst;
    }
    throw CompileError("internal: unbound value in instruction selection",
                       line);
  }

  int64_t frameSlot(Instruction *a) {
    auto it = frOff.find(a);
    if (it != frOff.end()) return it->second;
    int64_t off = nextFrameByte;
    nextFrameByte += (int64_t)a->n * 4;
    frOff[a] = off;
    return off;
  }

  PtrVal &ptrOf(Value *v) {
    auto it = ptrs.find(v);
    if (it != ptrs.end()) return it->second;
    if (auto *g = dynamic_cast<GlobalVar *>(v)) {
      PtrVal p;
      p.kind = 1;
      p.g = g;
      return ptrs.emplace(v, p).first->second;
    }
    if (auto *ai = dynamic_cast<Instruction *>(v)) {
      if (ai->op == Op::Alloca) {
        PtrVal p;
        p.kind = 0;
        p.off = frameSlot(ai);
        return ptrs.emplace(v, p).first->second;
      }
      throw CompileError("internal: unexpected pointer-producing value",
                         ai->line);
    }
    auto vit = vreg.find(v);
    if (vit != vreg.end()) {
      PtrVal p;
      p.kind = 2;
      p.reg = vit->second;
      return ptrs.emplace(v, p).first->second;
    }
    throw CompileError("internal: unknown address value", 0);
  }

  int32_t materializePtr(PtrVal &p, int line) {
    if (p.kind == 2) {
      if (p.off == 0) return p.reg;
      // register pointer plus a folded constant offset: fold the offset into
      // the register so the value can be used where a bare address is needed
      // (e.g. as a call argument).
      int32_t base = p.reg;
      int64_t off = p.off;
      if (fits12(off)) {
        base = opI(MOp::IAddI, base, (int32_t)off, line, Type::Ptr);
      } else {
        int32_t l = opI(MOp::Li, -1, (int32_t)off, line);
        base = op2(MOp::IAdd, base, l, line, Type::Ptr);
      }
      p.reg = base;
      p.off = 0;
      return base;
    }
    MInst m;
    m.dst = newReg();
    m.ty = Type::Ptr;
    m.line = line;
    if (p.kind == 0) {
      m.op = MOp::LeaFrame;
      m.imm = (int32_t)p.off;
      emit(m);
      // Note: the map entry stays kind0/kind1; loads and stores can always
      // use the fused sp/symbol addressing, so we must not bake a register
      // materialisation into a slot that is shared across basic blocks (the
      // defining instruction could end up not dominating all its uses).
      return m.dst;
    }
    m.op = MOp::LeaGlobal;
    m.sym = p.g->name;
    emit(m);
    int32_t base = m.dst;
    int64_t off = p.off;
    if (off != 0) {
      if (fits12(off)) {
        base = opI(MOp::IAddI, base, (int32_t)off, line, Type::Ptr);
      } else {
        int32_t l = opI(MOp::Li, -1, (int32_t)off, line);
        base = op2(MOp::IAdd, base, l, line, Type::Ptr);
      }
    }
    return base;
  }

  MOp loadOp(Type elem, PtrVal &p) const {
    bool isF = elem == Type::F32;
    switch (p.kind) {
    case 0: return isF ? MOp::FlwF : MOp::LwF;
    case 1: return isF ? MOp::FlwG : MOp::LwG;
    default: return isF ? MOp::Flw : MOp::Lw;
    }
  }
  MOp storeOp(Type elem, PtrVal &p) const {
    bool isF = elem == Type::F32;
    switch (p.kind) {
    case 0: return isF ? MOp::FswF : MOp::SwF;
    case 1: return isF ? MOp::FswG : MOp::SwG;
    default: return isF ? MOp::Fsw : MOp::Sw;
    }
  }
  void memOp(bool isStore, MOp op, PtrVal &p, MInst &m) {
    if (p.kind == 0) {
      m.imm = (int32_t)p.off;
    } else if (p.kind == 1) {
      m.sym = p.g->name;
      m.imm = (int32_t)p.off;
    } else {
      if (isStore)
        m.b = p.reg;
      else
        m.a = p.reg;
      m.imm = (int32_t)p.off;
    }
  }

  // ---- one/two-operand arithmetic helpers ------------------------------
  int32_t op2(MOp op, int32_t a, int32_t b, int line, Type ty = Type::I32) {
    MInst m;
    m.op = op;
    m.dst = newReg();
    m.a = a;
    m.b = b;
    m.ty = ty;
    m.line = line;
    emit(m);
    return m.dst;
  }
  int32_t opI(MOp op, int32_t a, int32_t imm, int line, Type ty = Type::I32) {
    MInst m;
    m.op = op;
    m.dst = newReg();
    m.a = a;
    m.imm = imm;
    m.ty = ty;
    m.line = line;
    emit(m);
    return m.dst;
  }
  int32_t op1(MOp op, int32_t a, int line, Type ty = Type::I32) {
    MInst m;
    m.op = op;
    m.dst = newReg();
    m.a = a;
    m.ty = ty;
    m.line = line;
    emit(m);
    return m.dst;
  }
};

} // namespace

// ==========================================================================
//  per-function lowering
// ==========================================================================
namespace {

void markLiveAllocas(FnLower &L) {
  for (auto &bb : L.fn->blocks) {
    for (auto &iu : bb->instrs) {
      Instruction *i = iu.get();
      for (size_t k = 0; k < i->ops.size(); ++k) {
        Value *op = i->ops[k];
        auto *ai = dynamic_cast<Instruction *>(op);
        if (!ai || ai->op != Op::Alloca) continue;
        if (i->op == Op::Store && i->ops.size() == 2 && i->ops[1] == ai)
          continue; // pure store address -> droppable if never read
        if (i->op == Op::Br || i->op == Op::CondBr) continue;
        L.liveAlloca.insert(ai);
      }
    }
  }
}

void emitParamEntries(FnLower &L) {
  ArgCursor c;
  size_t n = L.fn->params.size();
  for (size_t i = 0; i < n; ++i) {
    bool isF = L.mf.params[i] == Type::F32;
    ArgLoc loc = c.next(isF);
    Argument *arg = L.fn->args[i].get();
    MInst m;
    // reg-passed params: imm = physical register number; stack-passed:
    // imm = stack slot ordinal (writer adds the frame size).
    m.imm = (loc.stk >= 0) ? loc.stk : loc.reg;
    if (isF) {
      m.op = (loc.stk >= 0) ? MOp::EntryStkFlt : MOp::EntryFlt;
      m.ty = Type::F32;
    } else {
      m.op = (loc.stk >= 0) ? MOp::EntryStkInt : MOp::EntryInt;
      m.ty = L.mf.params[i];
    }
    m.dst = L.newReg();
    L.emit(m);
    L.vreg[arg] = m.dst;
    if (L.mf.params[i] == Type::Ptr) {
      PtrVal p;
      p.kind = 2;
      p.reg = m.dst;
      L.ptrs[arg] = p;
    }
  }
}

// split an ICmp operand: returns -1 when the operand is the constant zero.
void cmpOperands(FnLower &L, Instruction *i, int32_t &a, int32_t &b,
                 int line) {
  auto z = [](Value *v) { return dynamic_cast<ConstantInt *>(v) &&
                                 dynamic_cast<ConstantInt *>(v)->v == 0; };
  a = z(i->ops[0]) ? -1 : L.forceReg(i->ops[0], line);
  b = z(i->ops[1]) ? -1 : L.forceReg(i->ops[1], line);
}

// ---------------------------------------------------------------------------
// Whole-tensor ops (TCopy/TAdd*/.../TMatmul*) are expanded into affine.for
// loop nests in the mid-end (midend/pass/TensorLower.cpp), so no tensor op
// should survive to instruction selection.  Reaching here is a pipeline bug.
// ---------------------------------------------------------------------------

void lowerInstruction(FnLower &L, Instruction *i) {
  int line = i->line;
  switch (i->op) {
  case Op::Alloca:
    return; // slot assigned lazily on first use
  case Op::Load: {
    PtrVal &p = L.ptrOf(i->ops[0]);
    MInst m;
    m.op = L.loadOp(i->ty, p);
    m.dst = L.newReg();
    m.ty = i->ty;
    m.line = line;
    L.memOp(false, m.op, p, m);
    L.emit(m);
    L.vreg[i] = m.dst;
    return;
  }
  case Op::Store: {
    // drop stores into slots that are never read
    Value *addr = i->ops[1];
    if (auto *ai = dynamic_cast<Instruction *>(addr))
      if (ai->op == Op::Alloca && !L.liveAlloca.count(ai)) return;
    int32_t v = L.forceReg(i->ops[0], line);
    PtrVal &p = L.ptrOf(addr);
    MInst m;
    m.op = L.storeOp(i->ops[0]->ty, p);
    m.a = v;
    m.ty = i->ops[0]->ty;
    m.line = line;
    L.memOp(true, m.op, p, m);
    L.emit(m);
    return;
  }
  case Op::Gep: {
    PtrVal &base = L.ptrOf(i->ops[0]);
    Value *off = i->ops[1];
    PtrVal out;
    if (auto *ci = dynamic_cast<ConstantInt *>(off)) {
      out = base;
      out.off += ci->v;
    } else {
      out.kind = 2;
      out.reg = L.op2(MOp::IAdd, L.materializePtr(base, line),
                      L.forceReg(off, line), line, Type::Ptr);
    }
    L.ptrs[i] = out;
    return;
  }
  case Op::FAdd:
  case Op::FSub: {
    Value *l = i->ops[0], *r = i->ops[1];
    auto *lfc = dynamic_cast<ConstantFloat *>(l);
    if (i->op == Op::FSub && lfc && lfc->v == 0.0f) {
      L.vreg[i] = L.op1(MOp::FNeg, L.forceReg(r, line), line);
      return;
    }
    L.vreg[i] = L.op2(i->op == Op::FAdd ? MOp::FAdd : MOp::FSub,
                      L.forceReg(l, line), L.forceReg(r, line), line);
    return;
  }
  case Op::Add:
  case Op::Sub: {
    Value *l = i->ops[0], *r = i->ops[1];
    auto lc = dynamic_cast<ConstantInt *>(l);
    auto rc = dynamic_cast<ConstantInt *>(r);
    if (i->op == Op::Sub && lc && lc->v == 0 && !rc) { // unary minus
      L.vreg[i] = L.op1(MOp::INeg, L.forceReg(r, line), line);
      return;
    }
    if (rc) {
      int32_t a = L.forceReg(l, line);
      int64_t C = rc->v;
      if (i->op == Op::Add) {
        if (C == 0) L.vreg[i] = a;
        else if (fits12(C)) L.vreg[i] = L.opI(MOp::IAddI, a, (int32_t)C, line);
        else L.vreg[i] = L.op2(MOp::IAdd, a, L.opI(MOp::Li, -1, (int32_t)C, line), line);
      } else {
        if (C == 0) L.vreg[i] = a;
        else if (fits12(-C)) L.vreg[i] = L.opI(MOp::IAddI, a, (int32_t)(-C), line);
        else L.vreg[i] = L.op2(MOp::ISub, a, L.opI(MOp::Li, -1, (int32_t)C, line), line);
      }
      return;
    }
    if (lc) {
      int32_t b = L.forceReg(r, line);
      L.vreg[i] = L.op2(i->op == Op::Add ? MOp::IAdd : MOp::ISub,
                        L.opI(MOp::Li, -1, (int32_t)lc->v, line), b, line);
      return;
    }
    L.vreg[i] = L.op2(i->op == Op::Add ? MOp::IAdd : MOp::ISub,
                      L.forceReg(l, line), L.forceReg(r, line), line);
    return;
  }
  case Op::FMul:
  case Op::FDiv: {
    L.vreg[i] = L.op2(i->op == Op::FMul ? MOp::FMul : MOp::FDiv,
                      L.forceReg(i->ops[0], line),
                      L.forceReg(i->ops[1], line), line);
    return;
  }
  case Op::Mul:
  case Op::SDiv:
  case Op::SRem: {
    MOp op = i->op == Op::Mul ? MOp::IMul
             : i->op == Op::SDiv ? MOp::IDiv
                                 : MOp::IRem;
    L.vreg[i] = L.op2(op, L.forceReg(i->ops[0], line),
                      L.forceReg(i->ops[1], line), line);
    return;
  }
  case Op::ICmp: {
    // fully-constant comparisons are folded here
    auto lc = dynamic_cast<ConstantInt *>(i->ops[0]);
    auto rc = dynamic_cast<ConstantInt *>(i->ops[1]);
    if (lc && rc) {
      bool b;
      switch (i->cond) {
      case Cond::Eq: b = lc->v == rc->v; break;
      case Cond::Ne: b = lc->v != rc->v; break;
      case Cond::Lt: b = lc->v < rc->v; break;
      case Cond::Le: b = lc->v <= rc->v; break;
      case Cond::Gt: b = lc->v > rc->v; break;
      default: b = lc->v >= rc->v; break;
      }
      MInst m;
      m.op = MOp::Li;
      m.dst = L.newReg();
      m.imm = b ? 1 : 0;
      m.line = line;
      L.emit(m);
      L.vreg[i] = m.dst;
      return;
    }
    MInst m;
    m.op = MOp::ICmp;
    m.dst = L.newReg();
    m.imm = (int32_t)i->cond;
    m.line = line;
    cmpOperands(L, i, m.a, m.b, line); // constant-zero operands -> -1 (x0)
    L.emit(m);
    L.vreg[i] = m.dst;
    return;
  }
  case Op::FCmp: {
    MInst m;
    m.op = MOp::FCmp;
    m.dst = L.newReg();
    m.a = L.forceReg(i->ops[0], line);
    m.b = L.forceReg(i->ops[1], line);
    m.imm = (int32_t)i->cond;
    m.line = line;
    L.emit(m);
    L.vreg[i] = m.dst;
    return;
  }
  case Op::Sitofp:
    L.vreg[i] = L.op1(MOp::I2F, L.forceReg(i->ops[0], line), line);
    return;
  case Op::Fptosi:
    L.vreg[i] = L.op1(MOp::F2I, L.forceReg(i->ops[0], line), line);
    return;
  case Op::Not: {
    Value *b = i->ops[0];
    if (auto *ci = dynamic_cast<ConstantInt *>(b)) {
      MInst m;
      m.op = MOp::Li;
      m.dst = L.newReg();
      m.imm = ci->v ? 0 : 1;
      m.line = line;
      L.emit(m);
      L.vreg[i] = m.dst;
    } else {
      L.vreg[i] = L.opI(MOp::IXorI, L.forceReg(b, line), 1, line);
    }
    return;
  }
  case Op::Call: {
    auto *callee = dynamic_cast<Function *>(i->ops[0]);
    if (!callee) throw CompileError("internal: call to non-function", line);
    MInst m;
    m.op = MOp::Call;
    m.sym = callee->name;
    m.line = line;
    ArgCursor c;
    int stkBytes = 0;
    for (size_t k = 1; k < i->ops.size(); ++k) {
      Value *av = i->ops[k];
      bool isF = av->ty == Type::F32;
      ArgLoc loc = c.next(isF);
      MArg ma;
      ma.ty = av->ty;
      ma.vreg = (av->ty == Type::Ptr) ? L.materializePtr(L.ptrOf(av), line)
                                      : L.forceReg(av, line);
      if (loc.stk >= 0) {
        MInst s;
        s.op = isF ? MOp::StkArgFsw : MOp::StkArgSw;
        s.a = ma.vreg;
        s.ty = av->ty;
        s.imm = backend::stkOff(loc.stk);
        s.line = line;
        L.emit(s);
        stkBytes = std::max(stkBytes, backend::stkOff(loc.stk) + 8);
        ma.vreg = -1; // value already stored; writer classifies by type only
      }
      m.args.push_back(ma);
    }
    m.imm = stkBytes;
    if (callee->ret != Type::Void) {
      m.dst = L.newReg();
      m.ty = callee->ret;
      L.vreg[i] = m.dst;
    }
    L.emit(m);
    L.maxStk = std::max(L.maxStk, stkBytes);
    return;
  }
  case Op::TCopy:
  case Op::TNegI:
  case Op::TNegF:
  case Op::TAddI:
  case Op::TSubI:
  case Op::TMulI:
  case Op::TDivI:
  case Op::TRemI:
  case Op::TAddF:
  case Op::TSubF:
  case Op::TMulF:
  case Op::TDivF:
  case Op::TMatmulI:
  case Op::TMatmulF:
    throw CompileError("internal: tensor op reached ISel; it must be expanded "
                       "in the mid-end (TensorLower)", line);
  default:
    throw CompileError("internal: unhandled opcode in ISel", line);
  }
}

void lowerTerminator(FnLower &L, BasicBlock *bb, size_t bi) {
  Instruction *t = bb->instrs.back().get();
  int line = t->line;
  BasicBlock *nextBB =
      (bi + 1 < L.fn->blocks.size()) ? L.fn->blocks[bi + 1].get() : nullptr;
  std::string nextL = nextBB ? L.labelFor(nextBB) : std::string();

  auto jmp = [&](BasicBlock *target) {
    std::string lab = L.labelFor(target);
    if (lab != nextL) {
      MInst m;
      m.op = MOp::Jmp;
      m.sym = lab;
      m.line = line;
      L.emit(m);
    }
  };

  if (t->op == Op::Br) {
    jmp(dynamic_cast<BasicBlock *>(t->ops[0]));
    return;
  }
  if (t->op == Op::Ret) {
    MInst m;
    m.op = MOp::Ret;
    m.line = line;
    m.ty = L.mf.ret;
    if (!t->ops.empty()) {
      Value *v = t->ops[0];
      if (auto *ci = dynamic_cast<ConstantInt *>(v)) {
        m.imm = ci->v;
      } else if (auto *cf = dynamic_cast<ConstantFloat *>(v)) {
        m.imm = (int32_t)cf->bits();
        m.ty = Type::F32;
      } else {
        m.a = L.forceReg(v, line);
      }
    }
    L.emit(m);
    return;
  }
  // CondBr
  Value *cond = t->ops[0];
  auto *thenB = dynamic_cast<BasicBlock *>(t->ops[1]);
  auto *elseB = dynamic_cast<BasicBlock *>(t->ops[2]);
  std::string thenL = L.labelFor(thenB), elseL = L.labelFor(elseB);

  if (auto *ci = dynamic_cast<ConstantInt *>(cond)) {
    jmp(ci->v ? thenB : elseB);
    return;
  }
  int32_t cv = L.forceReg(cond, line);
  MInst m;
  m.line = line;
  m.a = cv;
  if (thenL == nextL) {
    m.op = MOp::BrZ;
    m.sym = elseL;
    L.emit(m);
  } else if (elseL == nextL) {
    m.op = MOp::BrNz;
    m.sym = thenL;
    L.emit(m);
  } else {
    m.op = MOp::BrNz;
    m.sym = thenL;
    L.emit(m);
    MInst j;
    j.op = MOp::Jmp;
    j.sym = elseL;
    j.line = line;
    L.emit(j);
  }
}

std::vector<int32_t> countUses(FnLower &L) {
  std::vector<int32_t> use((size_t)L.nextReg, 0);
  auto touch = [&](int32_t r) {
    if (r >= 0 && r < (int32_t)use.size()) ++use[(size_t)r];
  };
  for (auto &b : L.mf.blocks)
    for (auto &m : b.instrs) {
      OpSlots sl = slots(m);
      for (int k = 0; k < 3; ++k) touch(sl.use[k]);
      for (auto &a : m.args) touch(a.vreg);
    }
  return use;
}

// compare materialization immediately followed by a test on its result can be
// replaced by a single conditional branch.
void fuseBranchCompares(FnLower &L) {
  auto use = countUses(L);
  for (auto &b : L.mf.blocks) {
    auto &v = b.instrs;
    for (size_t i = 0; i < v.size();) {
      MInst &m = v[i];
      bool branchOnTrue = m.op == MOp::BrNz; // jump sym when cond != 0
      if (branchOnTrue || m.op == MOp::BrZ) {
        if (i >= 1 && v[i - 1].op == MOp::ICmp &&
            v[i - 1].dst == m.a && use[(size_t)m.a] == 1) {
          Cond c = (Cond)v[i - 1].imm;
          if (!branchOnTrue) c = invertCond(c);
          m.op = MOp::BrCmp;
          m.a = v[i - 1].a;
          m.b = v[i - 1].b;
          m.imm = (int32_t)c;
          v.erase(v.begin() + (i - 1));
          --i;
          continue;
        }
        // compare then xor 1 (logical not of a comparison)
        if (i >= 2 && v[i - 1].op == MOp::IXorI && v[i - 1].imm == 1 &&
            v[i - 1].dst == m.a && use[(size_t)m.a] == 1 &&
            v[i - 2].op == MOp::ICmp && v[i - 2].dst == v[i - 1].a &&
            use[(size_t)v[i - 1].a] == 1) {
          Cond c = (Cond)v[i - 2].imm;
          if (branchOnTrue) c = invertCond(c);
          m.op = MOp::BrCmp;
          m.a = v[i - 2].a;
          m.b = v[i - 2].b;
          m.imm = (int32_t)c;
          v.erase(v.begin() + (i - 2), v.begin() + i);
          i -= 2;
          continue;
        }
      }
      ++i;
    }
  }
}

} // namespace

std::vector<MachineFunc> runInstructionSelection(Module *mod) {
  // The mid-end contract: the back-end only ever consumes flat cf IR.  This
  // is enforced by the verify-cf-only pass in runMidEndPipeline(); the check
  // here makes the boundary hold even if a future driver bypasses the
  // pipeline.
  if (!isCfOnly(*mod))
    throw CompileError(
        "internal: instruction selection only accepts cf-only IR (a "
        "structured/whole-tensor op survived the mid-end pipeline)");
  std::vector<MachineFunc> out;
  for (auto &fu : mod->functions()) {
    Function *fn = fu.get();
    if (fn->isLib || fn->blocks.empty()) continue;

    FnLower L;
    L.fn = fn;
    L.mf.name = fn->name;
    L.mf.ret = fn->ret;
    for (auto &p : fn->params) L.mf.params.push_back(p.ty);
    markLiveAllocas(L);

    for (size_t bi = 0; bi < fn->blocks.size(); ++bi) {
      BasicBlock *bb = fn->blocks[bi].get();
      MBlock mb;
      mb.name = L.labelFor(bb);
      L.mf.blocks.push_back(std::move(mb));
      L.cur = &L.mf.blocks.back();
      if (bi == 0) {
        MInst p;
        p.op = MOp::Prologue;
        L.emit(p);
        emitParamEntries(L);
      }
      auto &insts = bb->instrs;
      size_t bodyN = insts.size();
      // all but the trailing terminator
      for (size_t j = 0; j + 1 < bodyN; ++j) {
        Instruction *i = insts[j].get();
        lowerInstruction(L, i);
      }
      lowerTerminator(L, bb, bi);
    }
    L.mf.localBytes = L.nextFrameByte;
    L.mf.maxStkArgBytes = L.maxStk;
    fuseBranchCompares(L);
    out.push_back(std::move(L.mf));
  }
  return out;
}

} // namespace backend
} // namespace sakura
