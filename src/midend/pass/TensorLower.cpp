// TensorLower.cpp - affine-layer pass expanding whole-tensor ops into loops.
// See TensorLower.h.
#include "TensorLower.h"

#include <cstdint>
#include <vector>

namespace sakura {
namespace ir {

namespace {

// True if `o` is a whole-tensor operation (tensor dialect).
bool isTensorDialectOp(Op o) {
  switch (o) {
  case Op::TCopy:
  case Op::TNegI:
  case Op::TNegF:
  case Op::TAddI:
  case Op::TAddF:
  case Op::TSubI:
  case Op::TSubF:
  case Op::TMulI:
  case Op::TMulF:
  case Op::TDivI:
  case Op::TDivF:
  case Op::TRemI:
  case Op::TMatmulI:
  case Op::TMatmulF:
    return true;
  default:
    return false;
  }
}

int64_t shapeTotal(const Instruction *i) {
  int64_t n = 1;
  for (int64_t d : i->shape) n *= d;
  return n;
}

// ---------------------------------------------------------------------------
// IR writer.  `newBlock()` appends to the region list on top of regStack_,
// which mirrors IRBuilder's stack of owning block lists (function body or an
// scf.while/affine.for region), so tensor-op loops nest correctly inside the
// structured control-flow of the affine layer.
// ---------------------------------------------------------------------------
class TensorLowerer {
public:
  explicit TensorLowerer(Module *mod) : mod_(mod) {}

  void run() {
    for (auto &f : mod_->functions()) {
      if (f->isLib || f->blocks.empty()) continue;
      nameSeq_ = 0;
      regStack_.clear();
      regStack_.push_back(&f->blocks);
      lowerBlockList(f->blocks);
      regStack_.pop_back();
    }
  }

private:
  Module *mod_;
  int nameSeq_ = 0;
  std::vector<std::vector<std::unique_ptr<BasicBlock>> *> regStack_;

  BasicBlock *newBlock() {
    auto b = std::make_unique<BasicBlock>("tl" + std::to_string(nameSeq_++));
    BasicBlock *p = b.get();
    regStack_.back()->push_back(std::move(b));
    return p;
  }

  // Append a fresh instruction to a block and return it.
  Instruction *emitTo(BasicBlock *bb, Op op, Type ty, int line) {
    auto i = std::make_unique<Instruction>(op, ty);
    i->line = line;
    Instruction *p = i.get();
    bb->instrs.push_back(std::move(i));
    return p;
  }

  // Create a stack slot holding one i32 (induction variable / accumulator).
  Instruction *emitSlot(BasicBlock *bb, int line) {
    Instruction *s = emitTo(bb, Op::Alloca, Type::Ptr, line);
    s->elem = Type::I32;
    s->n = 1;
    return s;
  }

  // Emit `v = base + byteOff` (memref.gep) into `bb`.
  Instruction *emitGepInto(BasicBlock *bb, Value *base, Value *byteOff,
                           int line) {
    Instruction *g = emitTo(bb, Op::Gep, Type::Ptr, line);
    g->ops = {base, byteOff};
    return g;
  }

  // Recursively expand every tensor op in a block list.  A tensor op inside a
  // block is replaced by an affine.for loop nest (the op becomes the block
  // terminator) and the instructions that followed it move into a fresh exit
  // block; the scan continues there, so a chain of tensor ops in one block
  // lowers to a chain of loops.
  void lowerBlockList(std::vector<std::unique_ptr<BasicBlock>> &list) {
    size_t i = 0;
    while (i < list.size()) {
      BasicBlock *b = list[i].get();

      // Recurse into structured loops owned by this block first: their region
      // bodies may themselves contain whole-tensor ops.
      for (auto &iu : b->instrs) {
        Instruction *in = iu.get();
        if (in->op == Op::ScfWhile) {
          regStack_.push_back(&in->condRegion);
          lowerBlockList(in->condRegion);
          regStack_.pop_back();
          regStack_.push_back(&in->bodyRegion);
          lowerBlockList(in->bodyRegion);
          regStack_.pop_back();
        } else if (in->op == Op::AffineFor) {
          regStack_.push_back(&in->bodyRegion);
          lowerBlockList(in->bodyRegion);
          regStack_.pop_back();
        }
      }

      // First tensor op of this block (there can be several, separated by
      // ordinary code; splitting moves the rest into the exit block).
      size_t k = b->instrs.size();
      for (size_t j = 0; j < b->instrs.size(); ++j) {
        if (isTensorDialectOp(b->instrs[j]->op)) {
          k = j;
          break;
        }
      }
      if (k < b->instrs.size()) {
        expandTensorOp(list, i, k);
        // The block is now terminated by the loop; its trailing instructions
        // (and any further tensor ops) were moved into a fresh exit block
        // inserted right after it, which the next scan iteration handles.
      }
      ++i;
    }
  }

  // Split `list[bi]` at instruction index `k`: instructions after `k` move to
  // a fresh exit block appended right after it (which becomes the loop exit).
  BasicBlock *splitAfter(std::vector<std::unique_ptr<BasicBlock>> &list,
                         size_t bi, size_t k) {
    BasicBlock *b = list[bi].get();
    auto x = std::make_unique<BasicBlock>("tl" + std::to_string(nameSeq_++));
    BasicBlock *xp = x.get();
    list.insert(list.begin() + (bi + 1), std::move(x));
    for (size_t m = k + 1; m < b->instrs.size(); ++m)
      xp->instrs.push_back(std::move(b->instrs[m]));
    b->instrs.resize(k + 1);
    return xp;
  }

  // Return a fresh affine.for for the counting loop `iv = 0 to ub` whose exit
  // is `exit`; appends it to `bb`.
  Instruction *emitFor(BasicBlock *bb, Value *slot, int64_t ub,
                       BasicBlock *exit, int line) {
    Instruction *f = emitTo(bb, Op::AffineFor, Type::Void, line);
    f->ops = {exit, slot, mod_->constInt(0), mod_->constInt((int32_t)ub)};
    f->cond = Cond::Lt;
    f->step = 1;
    return f;
  }

  // ---- dispatcher ------------------------------------------------------
  void expandTensorOp(std::vector<std::unique_ptr<BasicBlock>> &list, size_t bi,
                      size_t k) {
    Instruction *ti = list[bi]->instrs[k].get();
    if (ti->op == Op::TMatmulI || ti->op == Op::TMatmulF)
      expandMatmul(list, bi, k);
    else
      expandElementwise(list, bi, k);
  }

  // ---- elementwise ops --------------------------------------------------
  void expandElementwise(std::vector<std::unique_ptr<BasicBlock>> &list,
                         size_t bi, size_t k) {
    BasicBlock *b = list[bi].get();
    Instruction *ti = b->instrs[k].get();
    const Op op = ti->op; // copy before the instruction is destroyed below
    const Type elem = ti->elem;
    const int line = ti->line;
    const int64_t total = shapeTotal(ti);
    if (total <= 0 || total > INT32_MAX)
      throw CompileError("internal: bad tensor size", line);
    const std::vector<Value *> srcs = ti->ops; // ops[0] dest, ops[1..] sources

    // The loop replaces the op as the block terminator; everything that
    // followed it continues in the new exit block.
    BasicBlock *exit = splitAfter(list, bi, k);
    b->instrs.pop_back();

    Instruction *slot = emitSlot(b, line);
    Instruction *fop = emitFor(b, slot, total, exit, line);

    regStack_.push_back(&fop->bodyRegion);
    BasicBlock *body = newBlock();
    emitElementwiseBody(body, op, elem, line, srcs, slot);
    regStack_.pop_back();
  }

  void emitElementwiseBody(BasicBlock *body, Op op, Type elem, int line,
                           const std::vector<Value *> &srcs, Value *slot) {
    Instruction *iv = emitTo(body, Op::Load, Type::I32, line);
    iv->ops = {slot};
    Instruction *off = emitTo(body, Op::Mul, Type::I32, line);
    off->ops = {iv, mod_->constInt(4)};

    Instruction *dst = emitGepInto(body, srcs[0], off, line);

    std::vector<Value *> vals;
    for (size_t s = 1; s < srcs.size(); ++s) {
      Value *v = srcs[s];
      if (v->ty == Type::Ptr) { // whole-tensor source: load element `off`
        Instruction *a = emitGepInto(body, v, off, line);
        Instruction *ld = emitTo(body, Op::Load, elem, line);
        ld->ops = {a};
        vals.push_back(ld);
      } else {
        vals.push_back(v); // broadcast scalar
      }
    }

    Value *res = nullptr;
    switch (op) {
    case Op::TCopy:
      res = vals[0];
      break;
    case Op::TNegI: {
      Instruction *s = emitTo(body, Op::Sub, Type::I32, line);
      s->ops = {mod_->constInt(0), vals[0]};
      res = s;
      break;
    }
    case Op::TNegF: {
      Instruction *s = emitTo(body, Op::FSub, Type::F32, line);
      s->ops = {mod_->constFloat(0.0f), vals[0]};
      res = s;
      break;
    }
    default: {
      Op ar;
      switch (op) {
      case Op::TAddI: ar = Op::Add; break;
      case Op::TAddF: ar = Op::FAdd; break;
      case Op::TSubI: ar = Op::Sub; break;
      case Op::TSubF: ar = Op::FSub; break;
      case Op::TMulI: ar = Op::Mul; break;
      case Op::TMulF: ar = Op::FMul; break;
      case Op::TDivI: ar = Op::SDiv; break;
      case Op::TDivF: ar = Op::FDiv; break;
      case Op::TRemI: ar = Op::SRem; break;
      default:
        throw CompileError("internal: not an elementwise tensor op (" +
                               opName(op) + ")",
                           line);
      }
      Instruction *a = emitTo(body, ar, elem, line);
      a->ops = {vals[0], vals[1]};
      res = a;
      break;
    }
    }

    Instruction *st = emitTo(body, Op::Store, Type::Void, line);
    st->ops = {res, dst};
    emitTo(body, Op::AffineYield, Type::Void, line);
  }

  // ---- matmul -----------------------------------------------------------
  // dest[M,P] = a[M,N] @ b[N,P], expanded as the canonical kij affine nest:
  //   affine.for i in [0,M):
  //     affine.for j in [0,P):
  //       acc = 0
  //       affine.for k in [0,N):
  //         acc += a[i][k] * b[k][j]
  //       dest[i][j] = acc
  void expandMatmul(std::vector<std::unique_ptr<BasicBlock>> &list, size_t bi,
                    size_t k) {
    BasicBlock *b = list[bi].get();
    Instruction *ti = b->instrs[k].get();
    if (ti->shape.size() != 3)
      throw CompileError("internal: matmul shape must be {M,N,P}", ti->line);
    const int64_t M = ti->shape[0], N = ti->shape[1], P = ti->shape[2];
    if (M <= 0 || N <= 0 || P <= 0 || M > INT32_MAX || N > INT32_MAX ||
        P > INT32_MAX)
      throw CompileError("internal: bad matmul shape", ti->line);
    const Type elem = ti->elem;
    const bool isF = elem == Type::F32;
    const int line = ti->line;
    Value *cBase = ti->ops[0], *aBase = ti->ops[1], *bBase = ti->ops[2];

    BasicBlock *exit = splitAfter(list, bi, k);
    b->instrs.pop_back();

    // Induction slots for i / j / k and the per-(i,j) accumulator.
    Instruction *si = emitSlot(b, line);
    Instruction *sj = emitSlot(b, line);
    Instruction *sk = emitSlot(b, line);
    Instruction *sac = emitSlot(b, line);
    ConstantInt *c4 = mod_->constInt(4);

    // i loop (its exit is `exit`, the block that followed the matmul)
    Instruction *fI = emitFor(b, si, M, exit, line);
    regStack_.push_back(&fI->bodyRegion);
    BasicBlock *bI = newBlock();
    BasicBlock *exitJ = newBlock(); // j-loop exit (sibling of bI in this region)

    // j loop
    Instruction *fJ = emitFor(bI, sj, P, exitJ, line);
    regStack_.push_back(&fJ->bodyRegion);
    BasicBlock *bJ = newBlock();
    BasicBlock *exitK = newBlock(); // k-loop exit (sibling of bJ in this region)

    // acc = 0
    {
      Value *zero = isF ? (Value *)mod_->constFloat(0.0f)
                        : (Value *)mod_->constInt(0);
      Instruction *st = emitTo(bJ, Op::Store, Type::Void, line);
      st->ops = {zero, sac};
    }

    // k loop
    Instruction *fK = emitFor(bJ, sk, N, exitK, line);
    regStack_.push_back(&fK->bodyRegion);
    BasicBlock *bK = newBlock();

    // inner body: acc += a[i][k] * b[k][j]
    {
      Instruction *iv = emitTo(bK, Op::Load, Type::I32, line);
      iv->ops = {si};
      Instruction *jv = emitTo(bK, Op::Load, Type::I32, line);
      jv->ops = {sj};
      Instruction *kv = emitTo(bK, Op::Load, Type::I32, line);
      kv->ops = {sk};

      auto rowMajorOff = [&](Value *r1, int64_t stride, Value *r2) {
        // (r1 * stride + r2) * 4
        Instruction *m = emitTo(bK, Op::Mul, Type::I32, line);
        m->ops = {r1, mod_->constInt((int32_t)stride)};
        Instruction *a = emitTo(bK, Op::Add, Type::I32, line);
        a->ops = {m, r2};
        Instruction *x = emitTo(bK, Op::Mul, Type::I32, line);
        x->ops = {a, c4};
        return x;
      };

      Instruction *aOff = rowMajorOff(iv, N, kv); // a[i][k]
      Instruction *bOff = rowMajorOff(kv, P, jv); // b[k][j]
      Instruction *aAddr = emitGepInto(bK, aBase, aOff, line);
      Instruction *bAddr = emitGepInto(bK, bBase, bOff, line);
      Instruction *av = emitTo(bK, Op::Load, elem, line);
      av->ops = {aAddr};
      Instruction *bv = emitTo(bK, Op::Load, elem, line);
      bv->ops = {bAddr};
      Instruction *acc = emitTo(bK, Op::Load, elem, line);
      acc->ops = {sac};

      Instruction *prod = emitTo(bK, isF ? Op::FMul : Op::Mul, elem, line);
      prod->ops = {av, bv};
      Instruction *sum = emitTo(bK, isF ? Op::FAdd : Op::Add, elem, line);
      sum->ops = {acc, prod};
      Instruction *st = emitTo(bK, Op::Store, Type::Void, line);
      st->ops = {sum, sac};
      emitTo(bK, Op::AffineYield, Type::Void, line);
    }
    regStack_.pop_back(); // end of the k-loop body region

    // j-loop tail (exit of the k loop): dest[i][j] = acc
    {
      Instruction *iv = emitTo(exitK, Op::Load, Type::I32, line);
      iv->ops = {si};
      Instruction *jv = emitTo(exitK, Op::Load, Type::I32, line);
      jv->ops = {sj};
      Instruction *acc = emitTo(exitK, Op::Load, elem, line);
      acc->ops = {sac};

      Instruction *m = emitTo(exitK, Op::Mul, Type::I32, line);
      m->ops = {iv, mod_->constInt((int32_t)P)};
      Instruction *a = emitTo(exitK, Op::Add, Type::I32, line);
      a->ops = {m, jv};
      Instruction *x = emitTo(exitK, Op::Mul, Type::I32, line);
      x->ops = {a, c4};
      Instruction *cAddr = emitGepInto(exitK, cBase, x, line);
      Instruction *st = emitTo(exitK, Op::Store, Type::Void, line);
      st->ops = {acc, cAddr};
      emitTo(exitK, Op::AffineYield, Type::Void, line);
    }
    regStack_.pop_back(); // end of the j-loop body region

    // i-loop tail (exit of the j loop): nothing left to do, close the region
    emitTo(exitJ, Op::AffineYield, Type::Void, line);
    regStack_.pop_back(); // end of the i-loop body region
  }
};

} // namespace

void expandTensorOps(Module *mod) {
  TensorLowerer lowerer(mod);
  lowerer.run();
}

} // namespace ir
} // namespace sakura
