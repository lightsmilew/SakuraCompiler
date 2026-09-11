// ---------------------------------------------------------------------------
// OptMatmul.cpp: rewrite classic ijk matrix-multiply nests into register- and
// cache-friendly forms (LLVM LoopInterchange / register-blocking style).
//
// Matches, at the affine layer:
//
//   affine.for i:
//     affine.for j:
//       acc = 0
//       affine.for k:
//         acc += L[i][k] * R[k][j]
//       D[i][j] = acc
//
// and rewrites the j/k nest to a register-blocked "micro-kernel":
//
//   affine.for i:
//     affine.for jj (step 4):             // column micro-tile
//       acc0..acc3 = 0                     // scalar slots (mem2reg -> regs)
//       affine.for k:
//         lik = L[i][k]
//         acc0 += lik * R[k][jj+0]         // four independent chains
//         acc1 += lik * R[k][jj+1]
//         acc2 += lik * R[k][jj+2]
//         acc3 += lik * R[k][jj+3]
//       D[i][jj+0..3] = acc0..3
//     affine.for j (ubMain..ub):           // remainder columns, scalar acc
//       s = 0
//       affine.for k: s += L[i][k] * R[k][j]
//       D[i][j] = s
//
// Row-major streaming of R and of the destination beats both the column walk
// of R in the original ijk nest and the memory-resident accumulator row of a
// plain ikj rewrite; keeping 4 accumulators in registers also breaks the
// serial add-latency chain of the reduction.  When D aliases the left
// operand L, a full-row scratch is required and the plain ikj form is used.
// ---------------------------------------------------------------------------
#include "Opt.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "OptUtil.h"

namespace sakura {
namespace ir {

namespace {

using BlockList = std::vector<std::unique_ptr<BasicBlock>>;

Instruction *emitTo(BasicBlock *bb, Op op, Type ty, int line) {
  auto i = std::make_unique<Instruction>(op, ty);
  i->line = line;
  Instruction *p = i.get();
  bb->instrs.push_back(std::move(i));
  return p;
}

Instruction *emitSlot(BasicBlock *bb, int line, Type elem = Type::I32,
                      int64_t n = 1) {
  Instruction *s = emitTo(bb, Op::Alloca, Type::Ptr, line);
  s->elem = elem;
  s->n = n;
  return s;
}

Instruction *emitFor(BasicBlock *bb, Value *slot, Value *lb, Value *ub,
                     BasicBlock *exit, int line) {
  Instruction *f = emitTo(bb, Op::AffineFor, Type::Void, line);
  f->ops = {exit, slot, lb, ub};
  f->cond = Cond::Lt;
  f->step = 1;
  return f;
}

BasicBlock *newBlock(BlockList &list, int &seq) {
  auto b = std::make_unique<BasicBlock>("mm" + std::to_string(seq++));
  BasicBlock *p = b.get();
  list.push_back(std::move(b));
  return p;
}

bool isLoadOf(Instruction *ld, Value *slot) {
  return ld && ld->op == Op::Load && ld->ops.size() == 1 && ld->ops[0] == slot;
}

// Match byte offset `(row * rowBytes) + (col * elemBytes)`.
bool matchRowColOff(Value *off, Value *rowSlot, Value *colSlot, int32_t *rowBytes,
                    int32_t *elemBytes) {
  auto *add = dynamic_cast<Instruction *>(off);
  if (!add || add->op != Op::Add || add->ops.size() != 2) return false;

  auto asScaled = [&](Value *v, Value *wantSlot, int32_t *scale) -> bool {
    auto *m = dynamic_cast<Instruction *>(v);
    if (!m || m->op != Op::Mul || m->ops.size() != 2) return false;
    Value *a = m->ops[0], *b = m->ops[1];
    const ConstantInt *ca = asConstInt(a);
    const ConstantInt *cb = asConstInt(b);
    Value *idx = nullptr;
    const ConstantInt *sc = nullptr;
    if (cb && !ca) {
      idx = a;
      sc = cb;
    } else if (ca && !cb) {
      idx = b;
      sc = ca;
    } else
      return false;
    auto *ld = dynamic_cast<Instruction *>(idx);
    if (!isLoadOf(ld, wantSlot)) return false;
    *scale = sc->v;
    return true;
  };

  int32_t s0 = 0, s1 = 0;
  if (asScaled(add->ops[0], rowSlot, &s0) &&
      asScaled(add->ops[1], colSlot, &s1)) {
    *rowBytes = s0;
    *elemBytes = s1;
    return s1 == 4 || s1 == 8;
  }
  if (asScaled(add->ops[0], colSlot, &s0) &&
      asScaled(add->ops[1], rowSlot, &s1)) {
    *elemBytes = s0;
    *rowBytes = s1;
    return s0 == 4 || s0 == 8;
  }
  return false;
}

struct ArrRef {
  Value *base = nullptr;
  int32_t rowBytes = 0;
  int32_t elemBytes = 4;
};

bool matchArrLoad(Instruction *ld, Value *rowSlot, Value *colSlot, ArrRef *out) {
  if (!ld || ld->op != Op::Load || ld->ops.size() != 1) return false;
  auto *gep = dynamic_cast<Instruction *>(ld->ops[0]);
  if (!gep || gep->op != Op::Gep || gep->ops.size() != 2) return false;
  int32_t rb = 0, eb = 0;
  if (!matchRowColOff(gep->ops[1], rowSlot, colSlot, &rb, &eb)) return false;
  out->base = gep->ops[0];
  out->rowBytes = rb;
  out->elemBytes = eb;
  return true;
}

bool matchArrStore(Instruction *st, Value *rowSlot, Value *colSlot, ArrRef *out,
                   Value **val) {
  if (!st || st->op != Op::Store || st->ops.size() != 2) return false;
  auto *gep = dynamic_cast<Instruction *>(st->ops[1]);
  if (!gep || gep->op != Op::Gep || gep->ops.size() != 2) return false;
  int32_t rb = 0, eb = 0;
  if (!matchRowColOff(gep->ops[1], rowSlot, colSlot, &rb, &eb)) return false;
  out->base = gep->ops[0];
  out->rowBytes = rb;
  out->elemBytes = eb;
  *val = st->ops[0];
  return true;
}

Value *rootOf(Value *ptr) {
  AddrInfo a = addressOf(ptr);
  return a.root ? a.root : ptr;
}

bool sameBound(Value *a, Value *b) {
  if (a == b) return true;
  if (auto *ca = asConstInt(a))
    if (auto *cb = asConstInt(b)) return ca->v == cb->v;
  auto *la = dynamic_cast<Instruction *>(a);
  auto *lb = dynamic_cast<Instruction *>(b);
  if (la && lb && la->op == Op::Load && lb->op == Op::Load &&
      la->ops.size() == 1 && lb->ops.size() == 1 && la->ops[0] == lb->ops[0])
    return true;
  return false;
}

struct MatmulNest {
  Instruction *jFor = nullptr;
  Instruction *kFor = nullptr;
  Value *iSlot = nullptr;
  Value *jSlot = nullptr;
  Value *kSlot = nullptr;
  Value *accSlot = nullptr;
  ArrRef left;
  ArrRef right;
  ArrRef dest;
  Value *ub = nullptr;
  Value *lb = nullptr;
  int line = 0;
  Type elem = Type::I32;
  int32_t rowBytes = 0;
  int32_t elemBytes = 4;
};

bool analyseKBody(BasicBlock *body, MatmulNest *m) {
  Instruction *accLd = nullptr, *lLd = nullptr, *rLd = nullptr;
  Instruction *mul = nullptr, *add = nullptr, *accSt = nullptr;
  for (auto &iu : body->instrs) {
    Instruction *in = iu.get();
    if (in->op == Op::AffineYield) continue;
    if (isLoadOf(in, m->accSlot)) {
      accLd = in;
      continue;
    }
    ArrRef ar;
    if (in->op == Op::Load && matchArrLoad(in, m->iSlot, m->kSlot, &ar)) {
      lLd = in;
      m->left = ar;
      continue;
    }
    if (in->op == Op::Load && matchArrLoad(in, m->kSlot, m->jSlot, &ar)) {
      rLd = in;
      m->right = ar;
      continue;
    }
    if ((in->op == Op::Mul || in->op == Op::FMul) && !mul) {
      mul = in;
      continue;
    }
    if ((in->op == Op::Add || in->op == Op::FAdd) && !add) {
      add = in;
      continue;
    }
    if (in->op == Op::Store && in->ops.size() == 2 && in->ops[1] == m->accSlot) {
      accSt = in;
      continue;
    }
    if (in->op == Op::Load || in->op == Op::Mul || in->op == Op::Add ||
        in->op == Op::Gep || in->op == Op::Alloca)
      continue;
    return false;
  }
  if (!accLd || !lLd || !rLd || !mul || !add || !accSt) return false;
  m->elem = mul->ty;
  m->elemBytes = m->left.elemBytes;
  if (m->right.elemBytes != m->elemBytes) return false;
  // Prefer dest-row stride for the accumulator row; fall back to left.
  m->rowBytes = m->left.rowBytes;
  return true;
}

bool analyseJLoop(Instruction *jFor, Value *iSlot, MatmulNest *m) {
  if (jFor->op != Op::AffineFor || jFor->ops.size() != 4) return false;
  if (jFor->cond != Cond::Lt || jFor->step != 1) return false;
  m->jFor = jFor;
  m->iSlot = iSlot;
  m->jSlot = jFor->ops[1];
  m->lb = jFor->ops[2];
  m->ub = jFor->ops[3];
  m->line = jFor->line;

  if (jFor->bodyRegion.empty()) return false;
  BasicBlock *hdr = jFor->bodyRegion[0].get();
  Instruction *kFor = nullptr;
  m->accSlot = nullptr;

  for (auto &iu : hdr->instrs) {
    Instruction *in = iu.get();
    if (in->op == Op::Store && in->ops.size() == 2) {
      if (asConstInt(in->ops[0]) && asConstInt(in->ops[0])->v == 0)
        m->accSlot = in->ops[1];
      else if (asConstFloat(in->ops[0]) && asConstFloat(in->ops[0])->v == 0.0f)
        m->accSlot = in->ops[1];
      continue;
    }
    if (in->op == Op::AffineFor) {
      kFor = in;
      break;
    }
  }
  if (!kFor || !m->accSlot) return false;
  if (kFor->ops.size() != 4 || kFor->cond != Cond::Lt || kFor->step != 1)
    return false;
  if (kFor->ops[3] != m->ub && !sameBound(kFor->ops[3], m->ub)) return false;
  if (kFor->ops[2] != m->lb && !sameBound(kFor->ops[2], m->lb)) return false;
  m->kFor = kFor;
  m->kSlot = kFor->ops[1];

  if (kFor->bodyRegion.size() != 1) return false;
  if (!analyseKBody(kFor->bodyRegion[0].get(), m)) return false;

  auto *exitBlk = dynamic_cast<BasicBlock *>(kFor->ops[0]);
  if (!exitBlk) return false;
  bool sawStore = false;
  for (auto &iu : exitBlk->instrs) {
    Instruction *in = iu.get();
    if (in->op == Op::AffineYield) continue;
    ArrRef d;
    Value *val = nullptr;
    if (matchArrStore(in, m->iSlot, m->jSlot, &d, &val)) {
      auto *ld = dynamic_cast<Instruction *>(val);
      if (!ld || !isLoadOf(ld, m->accSlot)) return false;
      m->dest = d;
      sawStore = true;
      continue;
    }
    if (in->op == Op::Load || in->op == Op::Mul || in->op == Op::Add ||
        in->op == Op::Gep)
      continue;
    return false;
  }
  if (!sawStore) return false;
  if (m->dest.elemBytes != m->elemBytes) return false;
  // Accumulator / dest row length comes from D[i][j]'s row stride.
  m->rowBytes = m->dest.rowBytes;
  return true;
}

void emitRowMajorOff(Module &mod, BasicBlock *bb, Value *row, Value *col,
                     int32_t rowBytes, int32_t elemBytes, int line,
                     Instruction **outOff) {
  Instruction *m1 = emitTo(bb, Op::Mul, Type::I32, line);
  m1->ops = {row, mod.constInt(rowBytes)};
  Instruction *m2 = emitTo(bb, Op::Mul, Type::I32, line);
  m2->ops = {col, mod.constInt(elemBytes)};
  Instruction *ad = emitTo(bb, Op::Add, Type::I32, line);
  ad->ops = {m1, m2};
  *outOff = ad;
}

// Number of j-columns accumulated in independent scalar slots (register
// micro-tile width) inside the micro-kernel rewrite.
constexpr int64_t kBlock = 4;

bool rewriteMicrokernelInList(Module &mod, BlockList &list, BasicBlock &owner,
                              size_t jIdx, MatmulNest &m, int &seq) {
  if (owner.instrs[jIdx].get() != m.jFor) return false;
  auto *exitAfterJ = dynamic_cast<BasicBlock *>(m.jFor->ops[0]);
  if (!exitAfterJ) return false;
  if (m.elemBytes != 4) return false; // micro-kernel handles i32/f32 rows

  const int line = m.line;
  const bool isF = m.elem == Type::F32;
  Value *zero = isF ? (Value *)mod.constFloat(0.0f) : (Value *)mod.constInt(0);
  const int64_t B = kBlock;

  // Drop the old j-for (and its nested k-for / inner allocas).  Allocate
  // fresh induction + accumulator slots on surviving blocks only.
  owner.instrs.erase(owner.instrs.begin() + (long)jIdx);

  Instruction *jjS = emitSlot(&owner, line);
  Instruction *kS = emitSlot(&owner, line);
  {
    Instruction *stj = emitTo(&owner, Op::Store, Type::Void, line);
    stj->ops = {m.lb, jjS};
    Instruction *stk = emitTo(&owner, Op::Store, Type::Void, line);
    stk->ops = {m.lb, kS};
  }

  // ubMain = lb + ((ub-lb) - (ub-lb) % B): the largest multiple-of-B prefix
  // of the column range.  Main blocked loop covers [lb, ubMain), the
  // remainder column loop covers [ubMain, ub).
  Instruction *d = emitTo(&owner, Op::Sub, Type::I32, line);
  d->ops = {m.ub, m.lb};
  Instruction *r = emitTo(&owner, Op::SRem, Type::I32, line);
  r->ops = {d, mod.constInt((int32_t)B)};
  Instruction *a = emitTo(&owner, Op::Sub, Type::I32, line);
  a->ops = {d, r};
  Instruction *ubMain = emitTo(&owner, Op::Add, Type::I32, line);
  ubMain->ops = {m.lb, a};

  BasicBlock *remExit = newBlock(list, seq); // after blocked loop

  // ---- main blocked column loop: jj in [lb, ubMain) step B --------------
  Instruction *fB = emitFor(&owner, jjS, m.lb, ubMain, remExit, line);
  fB->step = B;
  {
    BasicBlock *bbJ = newBlock(fB->bodyRegion, seq);   // block header
    BasicBlock *exitK = newBlock(fB->bodyRegion, seq); // after k-loop, still
                                                       // inside this jj block

    Instruction *acc[kBlock] = {};
    for (int t = 0; t < B; ++t) acc[t] = emitSlot(bbJ, line, m.elem);
    for (int t = 0; t < B; ++t) {
      Instruction *st = emitTo(bbJ, Op::Store, Type::Void, line);
      st->ops = {zero, acc[t]};
    }

    // k loop: the four accumulator chains run in independent registers.
    Instruction *fK = emitFor(bbJ, kS, m.lb, m.ub, exitK, line);
    BasicBlock *bk = newBlock(fK->bodyRegion, seq);
    {
      Instruction *iv = emitTo(bk, Op::Load, Type::I32, line);
      iv->ops = {m.iSlot};
      Instruction *kv = emitTo(bk, Op::Load, Type::I32, line);
      kv->ops = {kS};
      Instruction *jjv = emitTo(bk, Op::Load, Type::I32, line);
      jjv->ops = {jjS};

      Instruction *lOff = nullptr;
      emitRowMajorOff(mod, bk, iv, kv, m.left.rowBytes, m.elemBytes, line,
                      &lOff);
      Instruction *lGep = emitTo(bk, Op::Gep, Type::Ptr, line);
      lGep->ops = {m.left.base, lOff};
      Instruction *c = emitTo(bk, Op::Load, m.elem, line);
      c->ops = {lGep};

      // Base byte offset for R[k][jjv + t]: kv*right.rowBytes + jjv*4.
      Instruction *rOff0 = nullptr;
      emitRowMajorOff(mod, bk, kv, jjv, m.right.rowBytes, m.elemBytes, line,
                      &rOff0);
      for (int t = 0; t < B; ++t) {
        Instruction *roff = rOff0;
        if (t != 0) {
          roff = emitTo(bk, Op::Add, Type::I32, line);
          roff->ops = {rOff0, mod.constInt((int32_t)(t * m.elemBytes))};
        }
        Instruction *rGep = emitTo(bk, Op::Gep, Type::Ptr, line);
        rGep->ops = {m.right.base, roff};
        Instruction *rv = emitTo(bk, Op::Load, m.elem, line);
        rv->ops = {rGep};
        Instruction *prod =
            emitTo(bk, isF ? Op::FMul : Op::Mul, m.elem, line);
        prod->ops = {c, rv};
        Instruction *oa = emitTo(bk, Op::Load, m.elem, line);
        oa->ops = {acc[t]};
        Instruction *sum = emitTo(bk, isF ? Op::FAdd : Op::Add, m.elem, line);
        sum->ops = {oa, prod};
        Instruction *st = emitTo(bk, Op::Store, Type::Void, line);
        st->ops = {sum, acc[t]};
      }
      emitTo(bk, Op::AffineYield, Type::Void, line);
    }

    // After the k loop: flush the four accumulators to D[i][jjv + t].
    BasicBlock *ek = exitK;
    {
      Instruction *iv = emitTo(ek, Op::Load, Type::I32, line);
      iv->ops = {m.iSlot};
      Instruction *jjv = emitTo(ek, Op::Load, Type::I32, line);
      jjv->ops = {jjS};
      Instruction *dOff0 = nullptr;
      emitRowMajorOff(mod, ek, iv, jjv, m.dest.rowBytes, m.elemBytes, line,
                      &dOff0);
      for (int t = 0; t < B; ++t) {
        Instruction *doff = dOff0;
        if (t != 0) {
          doff = emitTo(ek, Op::Add, Type::I32, line);
          doff->ops = {dOff0, mod.constInt((int32_t)(t * m.elemBytes))};
        }
        Instruction *dGep = emitTo(ek, Op::Gep, Type::Ptr, line);
        dGep->ops = {m.dest.base, doff};
        Instruction *v = emitTo(ek, Op::Load, m.elem, line);
        v->ops = {acc[t]};
        Instruction *st = emitTo(ek, Op::Store, Type::Void, line);
        st->ops = {v, dGep};
      }
      emitTo(ek, Op::AffineYield, Type::Void, line);
    }
  }

  // ---- remainder: one scalar accumulator per leftover column -------------
  Instruction *fR = emitFor(remExit, jjS, ubMain, m.ub, exitAfterJ, line);
  {
    BasicBlock *rb = newBlock(fR->bodyRegion, seq);
    BasicBlock *afterR = newBlock(fR->bodyRegion, seq);
    Instruction *s = emitSlot(rb, line, m.elem);
    {
      Instruction *st = emitTo(rb, Op::Store, Type::Void, line);
      st->ops = {zero, s};
    }
    Instruction *fRK = emitFor(rb, kS, m.lb, m.ub, afterR, line);
    BasicBlock *rk = newBlock(fRK->bodyRegion, seq);
    {
      Instruction *iv = emitTo(rk, Op::Load, Type::I32, line);
      iv->ops = {m.iSlot};
      Instruction *kv = emitTo(rk, Op::Load, Type::I32, line);
      kv->ops = {kS};
      Instruction *jv = emitTo(rk, Op::Load, Type::I32, line);
      jv->ops = {jjS};

      Instruction *lOff = nullptr;
      emitRowMajorOff(mod, rk, iv, kv, m.left.rowBytes, m.elemBytes, line,
                      &lOff);
      Instruction *lGep = emitTo(rk, Op::Gep, Type::Ptr, line);
      lGep->ops = {m.left.base, lOff};
      Instruction *c = emitTo(rk, Op::Load, m.elem, line);
      c->ops = {lGep};

      Instruction *rOff = nullptr;
      emitRowMajorOff(mod, rk, kv, jv, m.right.rowBytes, m.elemBytes, line,
                      &rOff);
      Instruction *rGep = emitTo(rk, Op::Gep, Type::Ptr, line);
      rGep->ops = {m.right.base, rOff};
      Instruction *rv = emitTo(rk, Op::Load, m.elem, line);
      rv->ops = {rGep};

      Instruction *prod = emitTo(rk, isF ? Op::FMul : Op::Mul, m.elem, line);
      prod->ops = {c, rv};
      Instruction *oa = emitTo(rk, Op::Load, m.elem, line);
      oa->ops = {s};
      Instruction *sum = emitTo(rk, isF ? Op::FAdd : Op::Add, m.elem, line);
      sum->ops = {oa, prod};
      Instruction *st = emitTo(rk, Op::Store, Type::Void, line);
      st->ops = {sum, s};
      emitTo(rk, Op::AffineYield, Type::Void, line);
    }
    BasicBlock *ar = afterR;
    {
      Instruction *iv = emitTo(ar, Op::Load, Type::I32, line);
      iv->ops = {m.iSlot};
      Instruction *jv = emitTo(ar, Op::Load, Type::I32, line);
      jv->ops = {jjS};
      Instruction *dOff = nullptr;
      emitRowMajorOff(mod, ar, iv, jv, m.dest.rowBytes, m.elemBytes, line,
                      &dOff);
      Instruction *dGep = emitTo(ar, Op::Gep, Type::Ptr, line);
      dGep->ops = {m.dest.base, dOff};
      Instruction *v = emitTo(ar, Op::Load, m.elem, line);
      v->ops = {s};
      Instruction *st = emitTo(ar, Op::Store, Type::Void, line);
      st->ops = {v, dGep};
      emitTo(ar, Op::AffineYield, Type::Void, line);
    }
  }
  return true;
}

bool rewriteNestInList(Module &mod, BlockList &list, BasicBlock &owner,
                       size_t jIdx, MatmulNest &m, int &seq) {
  if (owner.instrs[jIdx].get() != m.jFor) return false;

  auto *exitAfterJ = dynamic_cast<BasicBlock *>(m.jFor->ops[0]);
  if (!exitAfterJ) return false;

  const int line = m.line;
  const bool isF = m.elem == Type::F32;
  Value *zero = isF ? (Value *)mod.constFloat(0.0f) : (Value *)mod.constInt(0);

  const bool destAliases =
      rootOf(m.dest.base) == rootOf(m.left.base) ||
      rootOf(m.dest.base) == rootOf(m.right.base);

  if (m.elemBytes <= 0 || m.rowBytes % m.elemBytes != 0) return false;
  int64_t rowElems = m.rowBytes / m.elemBytes;
  if (rowElems <= 0 || rowElems > 65536) return false;

  // Drop the old j-for (and its nested k-for / inner allocas).  j/k slots
  // lived inside that region, so allocate fresh induction slots on `owner`.
  owner.instrs.erase(owner.instrs.begin() + (long)jIdx);

  Instruction *jSlot = emitSlot(&owner, line);
  Instruction *kSlot = emitSlot(&owner, line);
  // Seed both to lb before the nest (affine.for also stores lb on entry, but
  // keep slots defined for any pre-loop loads).
  {
    Instruction *stj = emitTo(&owner, Op::Store, Type::Void, line);
    stj->ops = {m.lb, jSlot};
    Instruction *stk = emitTo(&owner, Op::Store, Type::Void, line);
    stk->ops = {m.lb, kSlot};
  }

  Instruction *tmpRow = nullptr;
  Value *accBase = m.dest.base;
  if (destAliases) {
    tmpRow = emitSlot(&owner, line, m.elem, rowElems);
    accBase = tmpRow;
  }

  BasicBlock *zeroExit = newBlock(list, seq);
  BasicBlock *afterK = destAliases ? newBlock(list, seq) : exitAfterJ;

  // zero loop
  Instruction *fZ = emitFor(&owner, jSlot, m.lb, m.ub, zeroExit, line);
  {
    BasicBlock *bz = newBlock(fZ->bodyRegion, seq);
    Instruction *jv = emitTo(bz, Op::Load, Type::I32, line);
    jv->ops = {jSlot};
    Instruction *off = nullptr;
    if (destAliases) {
      off = emitTo(bz, Op::Mul, Type::I32, line);
      off->ops = {jv, mod.constInt(m.elemBytes)};
    } else {
      Instruction *iv = emitTo(bz, Op::Load, Type::I32, line);
      iv->ops = {m.iSlot};
      emitRowMajorOff(mod, bz, iv, jv, m.rowBytes, m.elemBytes, line, &off);
    }
    Instruction *gep = emitTo(bz, Op::Gep, Type::Ptr, line);
    gep->ops = {accBase, off};
    Instruction *st = emitTo(bz, Op::Store, Type::Void, line);
    st->ops = {zero, gep};
    emitTo(bz, Op::AffineYield, Type::Void, line);
  }

  // k loop
  Instruction *fK = emitFor(zeroExit, kSlot, m.lb, m.ub, afterK, line);
  BasicBlock *bk = newBlock(fK->bodyRegion, seq);
  BasicBlock *jInnerExit = newBlock(fK->bodyRegion, seq);

  Instruction *likSlot = emitSlot(bk, line, m.elem);
  {
    Instruction *iv = emitTo(bk, Op::Load, Type::I32, line);
    iv->ops = {m.iSlot};
    Instruction *kv = emitTo(bk, Op::Load, Type::I32, line);
    kv->ops = {kSlot};
    Instruction *off = nullptr;
    emitRowMajorOff(mod, bk, iv, kv, m.left.rowBytes, m.elemBytes, line, &off);
    Instruction *gep = emitTo(bk, Op::Gep, Type::Ptr, line);
    gep->ops = {m.left.base, off};
    Instruction *ld = emitTo(bk, Op::Load, m.elem, line);
    ld->ops = {gep};
    Instruction *st = emitTo(bk, Op::Store, Type::Void, line);
    st->ops = {ld, likSlot};
  }

  Instruction *fJ = emitFor(bk, jSlot, m.lb, m.ub, jInnerExit, line);
  {
    BasicBlock *bj = newBlock(fJ->bodyRegion, seq);
    Instruction *iv = emitTo(bj, Op::Load, Type::I32, line);
    iv->ops = {m.iSlot};
    Instruction *jv = emitTo(bj, Op::Load, Type::I32, line);
    jv->ops = {jSlot};
    Instruction *kv = emitTo(bj, Op::Load, Type::I32, line);
    kv->ops = {kSlot};

    Instruction *accOff = nullptr;
    if (destAliases) {
      accOff = emitTo(bj, Op::Mul, Type::I32, line);
      accOff->ops = {jv, mod.constInt(m.elemBytes)};
    } else {
      emitRowMajorOff(mod, bj, iv, jv, m.rowBytes, m.elemBytes, line, &accOff);
    }
    Instruction *accGep = emitTo(bj, Op::Gep, Type::Ptr, line);
    accGep->ops = {accBase, accOff};
    Instruction *acc = emitTo(bj, Op::Load, m.elem, line);
    acc->ops = {accGep};

    Instruction *rOff = nullptr;
    emitRowMajorOff(mod, bj, kv, jv, m.right.rowBytes, m.elemBytes, line,
                    &rOff);
    Instruction *rGep = emitTo(bj, Op::Gep, Type::Ptr, line);
    rGep->ops = {m.right.base, rOff};
    Instruction *rv = emitTo(bj, Op::Load, m.elem, line);
    rv->ops = {rGep};

    Instruction *lik = emitTo(bj, Op::Load, m.elem, line);
    lik->ops = {likSlot};

    Instruction *prod = emitTo(bj, isF ? Op::FMul : Op::Mul, m.elem, line);
    prod->ops = {lik, rv};
    Instruction *sum = emitTo(bj, isF ? Op::FAdd : Op::Add, m.elem, line);
    sum->ops = {acc, prod};
    Instruction *st = emitTo(bj, Op::Store, Type::Void, line);
    st->ops = {sum, accGep};
    emitTo(bj, Op::AffineYield, Type::Void, line);
  }
  emitTo(jInnerExit, Op::AffineYield, Type::Void, line);

  if (destAliases) {
    Instruction *fC = emitFor(afterK, jSlot, m.lb, m.ub, exitAfterJ, line);
    BasicBlock *bc = newBlock(fC->bodyRegion, seq);
    Instruction *iv = emitTo(bc, Op::Load, Type::I32, line);
    iv->ops = {m.iSlot};
    Instruction *jv = emitTo(bc, Op::Load, Type::I32, line);
    jv->ops = {jSlot};
    Instruction *tOff = emitTo(bc, Op::Mul, Type::I32, line);
    tOff->ops = {jv, mod.constInt(m.elemBytes)};
    Instruction *tGep = emitTo(bc, Op::Gep, Type::Ptr, line);
    tGep->ops = {tmpRow, tOff};
    Instruction *tv = emitTo(bc, Op::Load, m.elem, line);
    tv->ops = {tGep};
    Instruction *dOff = nullptr;
    emitRowMajorOff(mod, bc, iv, jv, m.dest.rowBytes, m.elemBytes, line, &dOff);
    Instruction *dGep = emitTo(bc, Op::Gep, Type::Ptr, line);
    dGep->ops = {m.dest.base, dOff};
    Instruction *st = emitTo(bc, Op::Store, Type::Void, line);
    st->ops = {tv, dGep};
    emitTo(bc, Op::AffineYield, Type::Void, line);
  }

  return true;
}

class MatmulIkjPass final : public Pass {
public:
  explicit MatmulIkjPass(Layer l) : layer_(l) {}
  const char *name() const override { return "matmul-ikj"; }
  Layer inLayer() const override { return layer_; }

  bool run(Module &mod) override {
    bool any = false;
    int seq = 0;
    for (auto &f : mod.functions()) {
      if (f->isLib) continue;
      any |= walk(mod, f->blocks, /*iSlot=*/nullptr, seq);
    }
    return any;
  }

private:
  Layer layer_;

  bool walk(Module &mod, BlockList &list, Value *iSlot, int &seq) {
    bool any = false;
    // Index-based: rewriteNestInList may append sibling blocks to `list`.
    for (size_t bi = 0; bi < list.size(); ++bi) {
      BasicBlock *bb = list[bi].get();
      for (size_t i = 0; i < bb->instrs.size(); ++i) {
        Instruction *op = bb->instrs[i].get();
        if (op->op == Op::AffineFor) {
          any |= walk(mod, op->bodyRegion, op->ops[1], seq);
          // Re-fetch: nested walk may have grown bodyRegion only.
          op = bb->instrs[i].get();
          if (iSlot && op->op == Op::AffineFor) {
            MatmulNest m;
            if (analyseJLoop(op, iSlot, &m)) {
              bool ok = false;
              if (rootOf(m.dest.base) == rootOf(m.left.base))
                // Dest aliases the left (row) operand: only the full-row
                // scratch variant is safe (right-hand aliasing is handled by
                // the register-blocked micro-kernel, which defers the write
                // of each column block until its whole k-sum is done).
                ok = rewriteNestInList(mod, list, *bb, i, m, seq);
              else
                ok = rewriteMicrokernelInList(mod, list, *bb, i, m, seq);
              if (ok) {
                any = true;
                // list may have grown; restart this block's instruction scan
                i = (size_t)-1;
                continue;
              }
            }
          }
        } else if (op->op == Op::ScfWhile) {
          any |= walk(mod, op->condRegion, iSlot, seq);
          any |= walk(mod, op->bodyRegion, iSlot, seq);
        }
      }
    }
    return any;
  }
};

} // namespace

std::unique_ptr<Pass> makeMatmulIkjPass(Layer l) {
  return std::make_unique<MatmulIkjPass>(l);
}

} // namespace ir
} // namespace sakura
