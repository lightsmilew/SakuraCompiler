// ---------------------------------------------------------------------------
// OptLoop.cpp: loop-invariant code motion + unrolling of affine.for loops.
//
// Both passes recurse through the region structure so they work at the
// affine and the scf layer:
//
//  * licm hoists invariant pure operations from the loop's condition/body
//    regions to just before the loop op.  Invariant *loads* are hoisted only
//    when no store reachable in the loop may alias their address (and never
//    when the loop contains a call, which can write anything reachable).
//    The affine.for induction slot (and any nested affine.for slot) is
//    treated as a value that changes every iteration, so reads of loop
//    counters are never hoisted.
//
//  * loop-unroll fully unrolls tiny constant-trip affine.for whose body is a
//    single straight-line block without calls or nested region ops, and
//    partially unrolls the remaining step-1 affine.for (factor 4 main loop +
//    step-1 remainder epilogue) when its body is a compact straight-line
//    array/vector kernel: the epilogue loops are never themselves re-partial-
//    unrolled (they carry their own remainder), but a small constant-trip
//    epilogue can still be fully unrolled afterwards.  Nests whose body block
//    stores through a pointer rooted at the induction slot are left alone
//    (each copy would need its own address delta).
// ---------------------------------------------------------------------------
#include "Opt.h"

#include <cstdint>
#include <unordered_map>
#include <unordered_set>

#include "OptUtil.h"

namespace sakura {
namespace ir {

namespace {

// ===========================================================================
// LICM
// ===========================================================================

struct LoopWrites {
  std::vector<AddrInfo> stores;
  // One entry per call in the loop: the callee, or nullptr for an indirect
  // call.  Whether such a call can actually clobber a given load is decided
  // per load by `Effects::callMayWrite` - a `readnone`/`readonly` callee does
  // not stop a load from being hoisted across it, which is what LLVM's
  // inferred function attributes buy it too.
  std::vector<Function *> calls;
  std::vector<Value *> ivRoots;
};

bool storeAliases(const AddrInfo &l, const AddrInfo &s) {
  if (s.constOff && l.constOff && s.root == l.root) return s.off == l.off;
  // Two *distinct* private objects (a global or a stack slot each) can never
  // alias, no matter how the offsets are computed: a GEP chain rooted at one
  // object cannot reach the storage of another.  The offsets being dynamic
  // used to defeat this, which kept the common `c[i][j] = ... a[i][k] ...`
  // matrix pattern from hoisting the invariant `a[i][k]` out of the inner
  // loop - worth ~8x on the 01_mm3/conv2d kernels under QEMU.
  if (s.root && l.root && s.root != l.root) {
    bool lPriv = l.kind == RootKind::Alloca || l.kind == RootKind::Global;
    bool sPriv = s.kind == RootKind::Alloca || s.kind == RootKind::Global;
    if (lPriv && sPriv) return false;
  }
  if (l.constOff && l.kind == RootKind::Alloca) return false;
  return true;
}

class LicmPass final : public Pass {
public:
  explicit LicmPass(Layer l) : layer_(l) {}
  const char *name() const override { return "licm"; }
  Layer inLayer() const override { return layer_; }

  bool run(Module &mod) override {
    bool any = false;
    // Function-attribute summaries (`readnone` / `readonly`) for the alias
    // question "can this call clobber that load?".  Computed once: LICM only
    // moves instructions around, so the call-graph summary stays valid.
    fx_.compute(mod);
    for (auto &f : mod.functions()) {
      if (f->isLib) continue;
      std::function<void(BlockList &)> rec = [&](BlockList &list) {
        for (auto &bb : list) {
          auto &instrs = bb->instrs;
          for (size_t i = 0; i < instrs.size(); ++i) {
            Instruction *op = instrs[i].get();
            if (op->op == Op::ScfWhile) {
              rec(op->condRegion);
              rec(op->bodyRegion);
              if (hoist(mod, *bb, i, op)) any = true;
            } else if (op->op == Op::AffineFor) {
              rec(op->bodyRegion);
              if (hoist(mod, *bb, i, op)) any = true;
            }
          }
        }
      };
      rec(f->blocks);
    }
    return any;
  }

private:
  Layer layer_;
  Effects fx_;

  static void scanWrites(BlockList &list, LoopWrites &w) {
    for (auto &bb : list)
      for (auto &iu : bb->instrs) {
        Instruction *i = iu.get();
        if (i->op == Op::Store)
          w.stores.push_back(addressOf(i->ops[1]));
        else if (i->op == Op::Call)
          w.calls.push_back(i->ops.empty()
                                ? nullptr
                                : dynamic_cast<Function *>(i->ops[0]));
        else if (i->op == Op::AffineFor) {
          if (auto *s = dynamic_cast<Instruction *>(i->ops[1]))
            w.ivRoots.push_back(s);
          scanWrites(i->bodyRegion, w);
        } else if (i->op == Op::ScfWhile) {
          scanWrites(i->condRegion, w);
          scanWrites(i->bodyRegion, w);
        }
      }
  }

  // A call may be hoisted when it is a pure value that always returns: it
  // writes nothing, reads nothing mutable, and provably comes back, so
  // running it one extra time (or out of order) is unobservable.  This is
  // LICM running on an inferred `readnone`/`readonly` + `willreturn` callee,
  // which is how clang lifts fft0's `power(d, mod-2)` out of the scaling loop
  // instead of recomputing ~1350 recursive calls per iteration.
  bool callHoistable(Instruction *i) const {
    if (i->op != Op::Call || i->ops.empty()) return false;
    auto *callee = dynamic_cast<Function *>(i->ops[0]);
    if (!callee) return false; // indirect call
    return fx_.callIsValue(callee) && fx_.terminates(callee);
  }

  bool loadMayChange(Value *addr, LoopWrites &w) const {
    AddrInfo l = addressOf(addr);
    for (Value *r : w.ivRoots)
      if (l.root == r) return true;
    for (Function *c : w.calls)
      if (fx_.callMayWrite(c, l)) return true;
    for (const AddrInfo &s : w.stores)
      if (storeAliases(l, s)) return true;
    return false;
  }

  bool hoist(Module &mod, BasicBlock &owner, size_t idx, Instruction *loop) {
    std::unordered_set<Instruction *> inner;
    std::vector<Instruction *> topLevel;
    std::function<void(BlockList &, bool)> collect =
        [&](BlockList &list, bool top) {
          for (auto &bb : list)
            for (auto &iu : bb->instrs) {
              Instruction *i = iu.get();
              inner.insert(i);
              if (top) topLevel.push_back(i);
              if (i->op == Op::ScfWhile) {
                collect(i->condRegion, false);
                collect(i->bodyRegion, false);
              } else if (i->op == Op::AffineFor) {
                collect(i->bodyRegion, false);
              }
            }
        };
    if (loop->op == Op::ScfWhile) {
      collect(loop->condRegion, true);
      collect(loop->bodyRegion, true);
    } else {
      collect(loop->bodyRegion, true);
    }

    LoopWrites writes;
    if (loop->op == Op::ScfWhile) {
      scanWrites(loop->condRegion, writes);
      scanWrites(loop->bodyRegion, writes);
    } else {
      scanWrites(loop->bodyRegion, writes);
      if (auto *s = dynamic_cast<Instruction *>(loop->ops[1]))
        writes.ivRoots.push_back(s);
    }

    std::unordered_set<Instruction *> inv;
    bool grow = true;
    while (grow) {
      grow = false;
      for (Instruction *i : topLevel) {
        if (inv.count(i)) continue;
        if (isTerminator(i->op)) continue;
        bool hoistable = false;
        if (i->op == Op::Load) {
          hoistable = !loadMayChange(i->ops[0], writes);
        } else if (i->op == Op::Call) {
          hoistable = callHoistable(i);
        } else if (isPure(i->op) && i->op != Op::Alloca &&
                   i->op != Op::Load) {
          hoistable = true;
        }
        if (!hoistable) continue;
        bool all = true;
        for (Value *o : i->ops) {
          if (isStructural(o)) continue;
          auto *d = dynamic_cast<Instruction *>(o);
          if (d && inner.count(d) && !inv.count(d)) {
            all = false;
            break;
          }
        }
        if (all) {
          inv.insert(i);
          grow = true;
        }
      }
    }

    std::vector<Instruction *> toMove;
    for (Instruction *i : topLevel) {
      if (!inv.count(i)) continue;
      if (i->op == Op::ScfWhile || i->op == Op::AffineFor ||
          i->op == Op::ScfCondition || i->op == Op::ScfYield ||
          i->op == Op::AffineYield)
        continue;
      bool usedInLoop = false;
      for (Instruction *u : topLevel) {
        if (u == i) continue;
        for (Value *o : u->ops)
          if (o == (Value *)i) usedInLoop = true;
      }
      if (usedInLoop) toMove.push_back(i);
    }
    if (toMove.empty()) return false;

    std::unordered_set<Instruction *> moveSet(toMove.begin(), toMove.end());
    auto &ownerInstrs = owner.instrs;
    size_t at = idx;
    std::function<void(BlockList &)> take = [&](BlockList &list) {
      for (auto &bb : list) {
        auto &v = bb->instrs;
        for (auto it = v.begin(); it != v.end();) {
          if (moveSet.count(it->get())) {
            ownerInstrs.insert(ownerInstrs.begin() + (long)at,
                               std::move(*it));
            ++at;
            it = v.erase(it);
          } else
            ++it;
        }
      }
    };
    if (loop->op == Op::ScfWhile) {
      take(loop->condRegion);
      take(loop->bodyRegion);
    } else {
      take(loop->bodyRegion);
    }
    return true;
  }
};

// ===========================================================================
// affine.for full + partial unrolling
// ===========================================================================

// Process-unique name suffix for epilogue blocks synthesised by partial
// unrolling: unrolls can occur at the same instruction index in several
// sibling regions, and block names surface as labels, so they must not clash.
static int g_unrollSeq = 0;

static std::unique_ptr<Instruction> cloneInstr(const Instruction *src) {
  auto c = std::make_unique<Instruction>(src->op, src->ty);
  c->cond = src->cond;
  c->elem = src->elem;
  c->n = src->n;
  c->step = src->step;
  c->shape = src->shape;
  c->line = src->line;
  return c;
}

// Coefficient of `v` as an affine function of this loop's induction read.
// The induction variable of an `affine.for` is read from its slot, so inside
// the body the read is a `Load` of `ivSlot`.  A value that equals
// `k*iv + r` with a compile-time `k` and an induction-independent `r` has a
// well-defined coefficient `k`; a copy of the body at `iv + t` then equals the
// `iv` value plus `k*t`.  Only `k` is returned - the additive part `r` never
// has to be modelled, because a rebased copy reuses copy 0's address and adds
// the constant `k*t` to it.
//
// Values defined outside the body are loop-invariant (coefficient 0).  An
// operation is induction-independent when all of its operands are; anything
// else is not affine in the induction variable and bails (returns false), which
// keeps the rewrite conservative.
static bool ivCoeff(
    Value *v, Value *ivSlot, const std::unordered_set<const Instruction *> &body,
    std::unordered_map<const Value *, std::pair<bool, int64_t>> &memo,
    int64_t &k, int depth = 0) {
  if (depth > 24) return false;
  if (dynamic_cast<ConstantInt *>(v)) {
    k = 0;
    return true;
  }
  auto *in = dynamic_cast<Instruction *>(v);
  if (!in || !body.count(in)) { // defined outside the body: loop-invariant
    k = 0;
    return true;
  }
  auto m = memo.find(v);
  if (m != memo.end()) {
    k = m->second.second;
    return m->second.first;
  }
  memo[v] = {false, 0}; // in-progress: break cycles defensively
  bool ok = false;
  int64_t kk = 0;
  auto coeff = [&](Value *x, int64_t &out) {
    return ivCoeff(x, ivSlot, body, memo, out, depth + 1);
  };
  switch (in->op) {
  case Op::Load:
    // Only the induction read itself is affine in the induction variable; any
    // other load has no affine relation to it.  (Induction-independent loads
    // are hoisted out of the body by LICM, which runs before unrolling.)
    if (in->ops.size() == 1 && in->ops[0] == ivSlot) {
      kk = 1;
      ok = true;
    }
    break;
  case Op::Add:
    if (in->ops.size() == 2) {
      int64_t a = 0, b = 0;
      if (coeff(in->ops[0], a) && coeff(in->ops[1], b)) {
        kk = a + b;
        ok = true;
      }
    }
    break;
  case Op::Sub:
    if (in->ops.size() == 2) {
      int64_t a = 0, b = 0;
      if (coeff(in->ops[0], a) && coeff(in->ops[1], b)) {
        kk = a - b;
        ok = true;
      }
    }
    break;
  case Op::Mul:
    if (in->ops.size() == 2) {
      const ConstantInt *c0 = asConstInt(in->ops[0]);
      const ConstantInt *c1 = asConstInt(in->ops[1]);
      int64_t a = 0, b = 0;
      if (c0 && coeff(in->ops[1], b)) {
        kk = c0->v * b;
        ok = true;
      } else if (c1 && coeff(in->ops[0], a)) {
        kk = c1->v * a;
        ok = true;
      }
    }
    break;
  default: {
    // Induction-independent iff every operand is; a use of an induction-
    // dependent operand is non-affine and bails.
    bool allZero = true;
    for (Value *o : in->ops) {
      int64_t a = 0;
      if (!coeff(o, a) || a != 0) {
        allZero = false;
        break;
      }
    }
    if (allZero) {
      kk = 0;
      ok = true;
    }
    break;
  }
  }
  memo[v] = {ok, kk};
  k = kk;
  return ok;
}

// Fill `gepK` with the induction-slope of every body Gep whose byte offset is
// affine in the induction read (slope != 0); other Geps are left out and stay
// recomputed per copy.
static void computeGepCoeffs(
    const std::vector<Instruction *> &body, Value *ivSlot,
    std::unordered_map<const Instruction *, int64_t> &gepK) {
  std::unordered_set<const Instruction *> bodySet(body.begin(), body.end());
  std::unordered_map<const Value *, std::pair<bool, int64_t>> memo;
  for (Instruction *s : body) {
    if (s->op != Op::Gep || s->ops.size() != 2) continue;
    int64_t c = 0;
    if (ivCoeff(s->ops[1], ivSlot, bodySet, memo, c) && c != 0) gepK[s] = c;
  }
}

// Emit one clone of the straight-line loop body for an induction value
// `ivDelta` beyond copy 0 (`ivVal` already carries the value).  `gepK` maps
// each body `Gep` whose byte offset is affine in the induction variable to its
// coefficient `k`.  When `ivDelta != 0` such a Gep is *not* recomputed: the
// copy points at copy 0's address plus the constant `k*ivDelta`, which the
// back end folds straight into the load/store displacement.  The dead per-copy
// offset arithmetic is then removed by the cleanup round that follows
// unrolling, so the group's address arithmetic is computed once.
static void emitUnrolledCopy(
    Module &mod, std::vector<std::unique_ptr<Instruction>> &dst,
    const std::vector<Instruction *> &bodyInstrs, Value *ivSlot, Value *ivVal,
    int64_t ivDelta,
    const std::unordered_map<const Instruction *, int64_t> &gepK,
    std::unordered_map<const Instruction *, Instruction *> &firstGep) {
  std::unordered_map<Instruction *, Instruction *> remap;
  std::vector<Instruction *> copies;
  std::unordered_set<Instruction *> rebased;
  for (Instruction *src : bodyInstrs) {
    auto gk = gepK.find(src);
    if (src->op == Op::Gep && ivDelta != 0 && gk != gepK.end()) {
      int64_t delta = gk->second * ivDelta;
      auto fg = firstGep.find(src);
      if (fg != firstGep.end() && delta >= INT32_MIN && delta <= INT32_MAX) {
        auto c = cloneInstr(src);
        Instruction *cp = c.get();
        cp->ops = {fg->second, mod.constInt((int32_t)delta)};
        remap[src] = cp;
        rebased.insert(cp);
        dst.push_back(std::move(c));
        copies.push_back(cp);
        continue;
      }
    }
    auto c = cloneInstr(src);
    Instruction *cp = c.get();
    if (src->op == Op::Gep && ivDelta == 0 && gk != gepK.end())
      firstGep[src] = cp; // remembered so later copies can rebase on it
    remap[src] = cp;
    dst.push_back(std::move(c));
    copies.push_back(cp);
  }
  for (size_t k = 0; k < copies.size(); ++k) {
    Instruction *src = bodyInstrs[k];
    Instruction *cp = copies[k];
    if (rebased.count(cp)) continue; // operands were set when rebasing
    for (Value *o : src->ops) {
      auto *d = dynamic_cast<Instruction *>(o);
      if (d && remap.count(d))
        cp->ops.push_back(remap[d]);
      else
        cp->ops.push_back(o);
    }
  }
  for (size_t k = 0; k < copies.size(); ++k) {
    Instruction *src = bodyInstrs[k];
    Instruction *cp = copies[k];
    if (src->op == Op::Load && src->ops.size() == 1 && src->ops[0] == ivSlot) {
      replaceAllUses(mod, cp, ivVal);
      for (auto it = dst.begin(); it != dst.end(); ++it)
        if (it->get() == cp) {
          dst.erase(it);
          break;
        }
    }
  }
}

class LoopUnrollPass final : public Pass {
public:
  explicit LoopUnrollPass(Layer l) : layer_(l) {}
  const char *name() const override { return "loop-unroll"; }
  Layer inLayer() const override { return layer_; }

  bool run(Module &mod) override {
    bool any = false;
    for (auto &f : mod.functions()) {
      if (f->isLib) continue;
      // Epilogue loops synthesised by partial unrolling carry their own
      // remainder and must not be partial-unrolled again; recording them here
      // (per function, per run) keeps the pointer stable for the whole sweep.
      std::unordered_set<Instruction *> skip;
      // Index-based: unrolling inserts/clones into a block's instruction list
      // and appends epilogue blocks, so neither the block iteration nor the
      // instruction scan may cache references into vectors that can grow
      // underneath them.
      std::function<void(BlockList &)> rec = [&](BlockList &list) {
        for (size_t bi = 0; bi < list.size(); ++bi) {
          BasicBlock *bb = list[bi].get();
          auto &instrs = bb->instrs;
          size_t i = 0;
          while (i < instrs.size()) {
            Instruction *op = instrs[i].get();
            if (op->op == Op::AffineFor) {
              rec(op->bodyRegion);
              op = instrs[i].get();
              if (!op || op->op != Op::AffineFor) continue;
              // full unroll first (small constant-trip loops), then partial
              if (tryFullUnroll(mod, *bb, i))
                any = true;
              else if (!skip.count(op) &&
                       tryPartialUnroll(mod, list, *bb, i, skip))
                any = true; // af now has step=4; the pre-loop instructions
                            // inserted before it shift it, and the epilogue
                            // loop recorded in `skip` lives in a block
                            // appended to `list`
              else
                ++i;
            } else {
              if (op->op == Op::ScfWhile) {
                rec(op->condRegion);
                rec(op->bodyRegion);
              }
              ++i;
            }
          }
        }
      };
      rec(f->blocks);
    }
    return any;
  }

private:
  Layer layer_;
  static constexpr int64_t kPartialFactor = 4;

  bool collectBody(Instruction *af, std::vector<Instruction *> *bodyInstrs) {
    BlockList &body = af->bodyRegion;
    if (body.size() != 1) return false;
    BasicBlock *B = body[0].get();
    bodyInstrs->clear();
    for (auto &iu : B->instrs) {
      Instruction *i = iu.get();
      if (i->op == Op::AffineYield) continue;
      bodyInstrs->push_back(i);
    }
    if (bodyInstrs->empty()) return false;
    for (Instruction *i : *bodyInstrs) {
      if (i->op == Op::ScfWhile || i->op == Op::Call || i->op == Op::AffineFor)
        return false;
      // Duplicating an explicit terminator would make the copies branch too.
      if (i->op == Op::Ret || i->op == Op::Br || i->op == Op::CondBr ||
          i->op == Op::ScfCondition || i->op == Op::ScfYield)
        return false;
      if (i->op == Op::Store) {
        AddrInfo a = addressOf(i->ops[1]);
        if (a.root == af->ops[1]) return false;
      }
    }
    return true;
  }

  // ------------------------------------------------------------------
  // Partial unrolling: turn  for iv in [lb, ub) step 1  into a step-4 main
  // loop whose single body executes four original iterations (iv+0..iv+3)
  // straight-line, followed by an epilogue loop covering the remainder
  // columns [ubMain, ub).  Both loops reuse the same iv slot.
  // ------------------------------------------------------------------
  bool tryPartialUnroll(Module &mod, BlockList &list, BasicBlock &bb,
                        size_t idx, std::unordered_set<Instruction *> &skip) {
    auto &instrs = bb.instrs;
    Instruction *af = instrs[idx].get();
    if (af->op != Op::AffineFor) return false;
    if (skip.count(af)) return false; // epilogue loops are never re-partialised
    if (af->ops.size() != 4 || af->cond != Cond::Lt) return false;
    if (af->step != 1) return false; // already (partially) unrolled
    Value *exitBB = af->ops[0];
    Value *ivSlot = af->ops[1];
    Value *lb = af->ops[2];
    Value *ub = af->ops[3];

    // Keep dynamic / medium-size constant loops; tiny constant loops are
    // handled by full unrolling.  Static trip count is only an estimate.
    bool constRange = false;
    int64_t trips = -1;
    if (const ConstantInt *l = asConstInt(lb))
      if (const ConstantInt *u = asConstInt(ub)) {
        constRange = true;
        trips = u->v - l->v;
      }
    if (constRange && trips < kPartialFactor) return false;
    if (constRange && trips <= 0) return false;

    std::vector<Instruction *> bodyInstrs;
    if (!collectBody(af, &bodyInstrs)) return false;
    // Only small/medium straight-line bodies benefit (the matmul micro-kernel
    // loop is already register-blocked and must not be expanded further).
    if ((int64_t)bodyInstrs.size() < 4 || (int64_t)bodyInstrs.size() > 14)
      return false;

    // Ownership snapshot of the body so it can be re-emitted freely.
    std::vector<std::unique_ptr<Instruction>> owned;
    std::vector<Instruction *> src;
    owned.reserve(bodyInstrs.size());
    for (Instruction *b : bodyInstrs) {
      owned.push_back(cloneInstr(b));
      src.push_back(owned.back().get());
    }
    for (size_t k = 0; k < src.size(); ++k) {
      for (Value *o : bodyInstrs[k]->ops) {
        auto *d = dynamic_cast<Instruction *>(o);
        Instruction *use = nullptr;
        if (d)
          for (size_t q = 0; q < bodyInstrs.size(); ++q)
            if (bodyInstrs[q] == d) {
              use = src[q];
              break;
            }
        src[k]->ops.push_back(use ? (Value *)use : o);
      }
    }
    const size_t nBody = bodyInstrs.size();
    const int line = af->line;
    BasicBlock *B = af->bodyRegion[0].get();

    // No epilogue when the trip count is statically a multiple of 4.
    bool needRem = !(constRange && trips % kPartialFactor == 0);

    Value *ubMain = ub;
    if (needRem) {
      // ubMain = lb + ((ub-lb) - (ub-lb) % 4), inserted just before the loop.
      size_t at = idx;
      auto emitPre = [&](Op op, Type ty) -> Instruction * {
        auto u = std::make_unique<Instruction>(op, ty);
        u->line = line;
        Instruction *p = u.get();
        instrs.insert(instrs.begin() + (long)at++, std::move(u));
        return p;
      };
      Instruction *dd = emitPre(Op::Sub, Type::I32);
      dd->ops = {ub, lb};
      Instruction *rr = emitPre(Op::SRem, Type::I32);
      rr->ops = {dd, mod.constInt((int32_t)kPartialFactor)};
      Instruction *aa = emitPre(Op::Sub, Type::I32);
      aa->ops = {dd, rr};
      Instruction *um = emitPre(Op::Add, Type::I32);
      um->ops = {lb, aa};
      ubMain = um;
    }

    // Mutate the loop: step 4, new upper bound, optional epilogue target.
    af->step = kPartialFactor;
    af->ops[3] = ubMain;

    // Rebuild the body block: a straight-line run of 4 original iterations.
    B->instrs.clear();
    {
      // Addresses that are affine in the induction variable.  For copies 1..3
      // the address is copy 0's plus `coeff*t`, so it is rebuilt as a nested
      // Gep with a constant offset that the back end folds into the load/store
      // displacement - one address computation per unrolled group instead of
      // four.
      std::unordered_map<const Instruction *, int64_t> gepK;
      computeGepCoeffs(src, ivSlot, gepK);
      std::unordered_map<const Instruction *, Instruction *> firstGep;

      auto iu = std::make_unique<Instruction>(Op::Load, Type::I32);
      iu->line = line;
      iu->ops = {ivSlot};
      Instruction *lv = iu.get();
      B->instrs.push_back(std::move(iu));
      for (int64_t t = 0; t < kPartialFactor; ++t) {
        Value *ivVal = lv;
        if (t != 0) {
          auto au = std::make_unique<Instruction>(Op::Add, Type::I32);
          au->line = line;
          au->ops = {lv, mod.constInt((int32_t)t)};
          ivVal = au.get();
          B->instrs.push_back(std::move(au));
        }
        emitUnrolledCopy(mod, B->instrs, src, ivSlot, ivVal, t, gepK, firstGep);
      }
      auto yu = std::make_unique<Instruction>(Op::AffineYield, Type::Void);
      yu->line = line;
      B->instrs.push_back(std::move(yu));
    }

    // Epilogue loop over [ubMain, ub) running the original (single-step)
    // body; both loops share the same iv slot.
    if (needRem) {
      int seq = ++g_unrollSeq;
      auto br = std::make_unique<BasicBlock>("unr" + std::to_string(seq));
      BasicBlock *rem = br.get();
      list.push_back(std::move(br));
      af->ops[0] = rem;

      auto fu = std::make_unique<Instruction>(Op::AffineFor, Type::Void);
      fu->line = line;
      fu->ops = {exitBB, ivSlot, ubMain, ub};
      fu->cond = Cond::Lt;
      fu->step = 1;
      Instruction *fR = fu.get();
      rem->instrs.push_back(std::move(fu));
      skip.insert(fR);
      auto rb = std::make_unique<BasicBlock>("unrb" + std::to_string(seq));
      BasicBlock *rbody = rb.get();
      fR->bodyRegion.push_back(std::move(rb));
      // Clone the original body into the epilogue and fix up internal refs.
      // cloneInstr copies no operands, so the operands are re-materialised
      // from the source clones (`src`) here; a clone must not iterate its own
      // (still empty) operand vector.
      std::vector<Instruction *> cps;
      cps.reserve(nBody);
      for (size_t k = 0; k < nBody; ++k) {
        auto cu = cloneInstr(src[k]);
        Instruction *cp = cu.get();
        rbody->instrs.push_back(std::move(cu));
        cps.push_back(cp);
      }
      for (size_t k = 0; k < nBody; ++k) {
        Instruction *c = cps[k];
        const Instruction *s = src[k];
        c->ops.reserve(s->ops.size());
        for (Value *o : s->ops) {
          auto *d = dynamic_cast<Instruction *>(o);
          Instruction *use = nullptr;
          if (d)
            for (size_t q = 0; q < nBody; ++q)
              if (src[q] == d) {
                use = cps[q];
                break;
              }
          c->ops.push_back(use ? (Value *)use : o);
        }
      }
      auto yu = std::make_unique<Instruction>(Op::AffineYield, Type::Void);
      yu->line = line;
      rbody->instrs.push_back(std::move(yu));
    }
    return true;
  }

  bool tryFullUnroll(Module &mod, BasicBlock &bb, size_t idx) {
    auto &instrs = bb.instrs;
    Instruction *af = instrs[idx].get();
    if (af->op != Op::AffineFor) return false;
    if (af->ops.size() != 4 || af->cond != Cond::Lt) return false;
    const ConstantInt *lb = asConstInt(af->ops[2]);
    const ConstantInt *ub = asConstInt(af->ops[3]);
    int64_t step = af->step;
    if (!lb || !ub || step <= 0) return false;
    if (lb->v >= ub->v) return false;
    int64_t trips = (ub->v - lb->v + step - 1) / step;

    if (trips > 32) return false;

    std::vector<Instruction *> bodyInstrs;
    if (!collectBody(af, &bodyInstrs)) return false;
    if ((int64_t)bodyInstrs.size() * trips > 64) return false;

    Value *exitBB = af->ops[0];
    Value *ivSlot = af->ops[1];
    // Keep the affine.for (and its body instrs) alive while we clone.
    std::unique_ptr<Instruction> afOwner = std::move(instrs[idx]);
    instrs.erase(instrs.begin() + (long)idx);

    // Rebase each fully-unrolled copy's addresses on copy 0's: copy t uses the
    // induction value lb + t*step, so its address differs from copy 0's by
    // coeff * t * step.
    std::unordered_map<const Instruction *, int64_t> gepK;
    computeGepCoeffs(bodyInstrs, ivSlot, gepK);
    std::unordered_map<const Instruction *, Instruction *> firstGep;
    for (int64_t t = 0; t < trips; ++t) {
      Value *ivConst = mod.constInt((int32_t)(lb->v + t * step));
      emitUnrolledCopy(mod, instrs, bodyInstrs, ivSlot, ivConst, t * step,
                       gepK, firstGep);
    }
    auto br = std::make_unique<Instruction>(Op::Br, Type::Void);
    br->cond = Cond::Eq;
    br->ops = {exitBB};
    instrs.push_back(std::move(br));
    return true;
  }

};

} // namespace

std::unique_ptr<Pass> makeLicmPass(Layer l) {
  return std::make_unique<LicmPass>(l);
}

std::unique_ptr<Pass> makeLoopUnrollPass(Layer l) {
  return std::make_unique<LoopUnrollPass>(l);
}

} // namespace ir
} // namespace sakura
