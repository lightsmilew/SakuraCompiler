// ---------------------------------------------------------------------------
// OptIndVar.cpp: induction-variable strength reduction (cf layer).
//
// A `while` loop over an array computes its element address from scratch every
// iteration - `gep base, add(mul(iv, stride), invariant)` - which puts a
// multiply (or, after the back-end's mul-by-constant lowering, a shift) on the
// address critical path of the innermost loop.  LLVM's LoopStrengthReduce
// replaces such a *derived induction variable* by one that is carried around
// the loop and merely incremented:
//
//     iv_next = add iv, step
//     addr    = gep base, add(mul iv, K), J          (J loop-invariant)
//   =>
//     p_phi   = phi [pre: gep base, (init*K + J)], [latch: p_next]
//     p_next  = gep p_phi, step*K
//     ... every use of addr becomes p_phi
//
// The address then costs one pointer increment per iteration instead of a
// multiply plus offset arithmetic, and - because the back-end folds a constant
// `gep` displacement into the load/store it feeds - the increment often
// disappears into the addressing mode entirely.  The address also stops
// depending on the multiply's latency, which matters most in the strided
// matrix / array kernels this pass targets.
//
// The pass is deliberately narrow: it only rewrites single-latch natural loops
// whose header carries the induction `cf.phi` (the shape mem2reg leaves
// behind), and only offsets that are affine in *one* induction variable with a
// constant coefficient.  Everything else in the offset must be loop-invariant,
// which is checked transitively; an invariant sub-expression that lives inside
// the loop is cloned into the preheader so the new pointer's initial value can
// be computed there.  Every use of an address must be inside the loop, so the
// new pointer's definition always dominates its uses.  The rewrite is then a
// pure reassociation of arithmetic that was already there - no new memory
// access, no change to the trip count or the control flow.
// ---------------------------------------------------------------------------
#include "Opt.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "OptUtil.h"

namespace sakura {
namespace ir {

namespace {

// Cap on how many derived addresses one loop may gain.  Each one costs a
// loop-carried register, so a loop with many address patterns would trade the
// saved multiplies for spills.
constexpr size_t kMaxPerLoop = 4;

// Depth limits for the affine decomposition and the invariance walk
// (defensive; real address expressions are a handful of levels deep).
constexpr int kMaxDepth = 24;
constexpr int kMaxInvDepth = 32;

// One loop's rewrite context: the loop's block set, its preheader, and the
// memo tables for the invariant sub-expressions cloned into that preheader.
struct Rewriter {
  Module &mod;
  Function &f;
  const std::vector<uint8_t> &inLoop;
  std::unordered_map<const Instruction *, size_t> &blockOf;
  BasicBlock *preB = nullptr;
  std::unordered_map<const Instruction *, Value *> preMemo;
  std::unordered_map<std::string, Value *> invMemo;

  Rewriter(Module &m, Function &fn, const std::vector<uint8_t> &il,
           std::unordered_map<const Instruction *, size_t> &bo)
      : mod(m), f(fn), inLoop(il), blockOf(bo) {}

  // Insert just before the preheader's terminator, keeping blockOf in sync so
  // a later (outer) loop sees the clone as living inside its own body and
  // re-clones rather than referencing it from a block that does not dominate.
  void insertPre(std::unique_ptr<Instruction> u, size_t blockIdx) {
    Instruction *p = u.get();
    auto &v = preB->instrs;
    size_t at = v.empty() ? 0 : v.size() - 1;
    v.insert(v.begin() + (long)at, std::move(u));
    blockOf[p] = blockIdx;
  }

  Value *invAdd(Value *a, Value *b, size_t blockIdx);
  Value *invSub(Value *a, Value *b, size_t blockIdx);
  Value *invMulC(Value *a, int64_t c, size_t blockIdx);

  // A value usable from the preheader that is equal to `v` on every iteration
  // of the loop.  Values defined outside the loop are used as they are; an
  // invariant instruction defined *inside* the loop is cloned (its operands
  // must themselves be invariant).  Returns null when `v` varies per
  // iteration.
  Value *preOf(Value *v, size_t blockIdx, int depth) {
    if (depth > kMaxInvDepth) return nullptr;
    auto *in = dynamic_cast<Instruction *>(v);
    if (!in) return v; // constant / global / argument
    auto bit = blockOf.find(in);
    if (bit == blockOf.end() || !inLoop[bit->second]) return v;
    auto mit = preMemo.find(in);
    if (mit != preMemo.end()) return mit->second;
    // Only deterministic, side-effect-free computation can move; a phi, a
    // fresh allocation or a load may yield a different value per iteration.
    if (!isPure(in->op) || in->op == Op::Phi || in->op == Op::Alloca ||
        in->op == Op::Load)
      return nullptr;
    std::vector<Value *> ops;
    for (Value *o : in->ops) {
      if (isStructural(o)) {
        ops.push_back(o);
        continue;
      }
      Value *p = preOf(o, blockIdx, depth + 1);
      if (!p) return nullptr;
      ops.push_back(p);
    }
    auto cl = std::make_unique<Instruction>(in->op, in->ty);
    cl->cond = in->cond;
    cl->elem = in->elem;
    cl->n = in->n;
    cl->step = in->step;
    cl->shape = in->shape;
    cl->line = in->line;
    cl->ops = ops;
    Instruction *cp = cl.get();
    insertPre(std::move(cl), blockIdx);
    preMemo[in] = cp;
    return cp;
  }

  // offset = k * iv + c + inv   (inv an i32 value available in the preheader)
  struct Aff {
    int64_t k = 0;
    int64_t c = 0;
    Value *inv = nullptr;
    bool bad = false; // integral coefficient did not fit an i32
  };

  Aff affAdd(const Aff &a, const Aff &b, size_t bi) {
    Aff r;
    __int128 k = (__int128)a.k + (__int128)b.k;
    __int128 c = (__int128)a.c + (__int128)b.c;
    r.bad = a.bad || b.bad || k < INT32_MIN || k > INT32_MAX ||
            c < INT32_MIN || c > INT32_MAX;
    r.k = (int64_t)k;
    r.c = (int64_t)c;
    r.inv = invAdd(a.inv, b.inv, bi);
    return r;
  }

  Aff affSub(const Aff &a, const Aff &b, size_t bi) {
    Aff r;
    __int128 k = (__int128)a.k - (__int128)b.k;
    __int128 c = (__int128)a.c - (__int128)b.c;
    r.bad = a.bad || b.bad || k < INT32_MIN || k > INT32_MAX ||
            c < INT32_MIN || c > INT32_MAX;
    r.k = (int64_t)k;
    r.c = (int64_t)c;
    r.inv = invSub(a.inv, b.inv, bi);
    return r;
  }

  Aff affMulC(const Aff &a, int64_t c, size_t bi) {
    Aff r;
    __int128 k = (__int128)a.k * (__int128)c;
    __int128 cc = (__int128)a.c * (__int128)c;
    if (k < INT32_MIN || k > INT32_MAX || cc < INT32_MIN || cc > INT32_MAX)
      r.bad = true;
    r.k = (int64_t)k;
    r.c = (int64_t)cc;
    r.inv = invMulC(a.inv, c, bi);
    r.bad = r.bad || a.bad;
    return r;
  }

  bool affine(Value *v, Value *iv, Aff &out, size_t bi, int depth) {
    if (depth > kMaxDepth) return false;
    if (v == iv) {
      out.k = 1;
      return true;
    }
    if (auto *ci = asConstInt(v)) {
      out.c = ci->v;
      return true;
    }
    if (Value *p = preOf(v, bi, 0)) {
      out.inv = p;
      return true;
    }
    auto *in = dynamic_cast<Instruction *>(v);
    if (!in || in->ops.size() != 2) return false;
    Aff a, b;
    if (!affine(in->ops[0], iv, a, bi, depth + 1)) return false;
    if (!affine(in->ops[1], iv, b, bi, depth + 1)) return false;
    switch (in->op) {
    case Op::Add:
      out = affAdd(a, b, bi);
      return true;
    case Op::Sub:
      out = affSub(a, b, bi);
      return true;
    case Op::Mul:
      // One factor must be a bare constant; the other carries the induction
      // variable (or is invariant).  `(iv*a)*(iv*b)` is quadratic: rejected.
      if (a.k == 0 && !a.inv && !a.bad) {
        out = affMulC(b, a.c, bi);
        return true;
      }
      if (b.k == 0 && !b.inv && !b.bad) {
        out = affMulC(a, b.c, bi);
        return true;
      }
      return false;
    default:
      return false;
    }
  }
};

Value *Rewriter::invAdd(Value *a, Value *b, size_t blockIdx) {
  if (!a) return b;
  if (!b) return a;
  char key[64];
  snprintf(key, sizeof key, "a%p|%p", (void *)a, (void *)b);
  auto it = invMemo.find(key);
  if (it != invMemo.end()) return it->second;
  auto u = std::make_unique<Instruction>(Op::Add, Type::I32);
  u->ops = {a, b};
  Value *r = u.get();
  insertPre(std::move(u), blockIdx);
  invMemo[key] = r;
  return r;
}

Value *Rewriter::invSub(Value *a, Value *b, size_t blockIdx) {
  if (!b) return a;
  if (!a) return invMulC(b, -1, blockIdx);
  char key[64];
  snprintf(key, sizeof key, "s%p|%p", (void *)a, (void *)b);
  auto it = invMemo.find(key);
  if (it != invMemo.end()) return it->second;
  auto u = std::make_unique<Instruction>(Op::Sub, Type::I32);
  u->ops = {a, b};
  Value *r = u.get();
  insertPre(std::move(u), blockIdx);
  invMemo[key] = r;
  return r;
}

Value *Rewriter::invMulC(Value *a, int64_t c, size_t blockIdx) {
  if (!a) return nullptr;
  if (c == 1) return a;
  char key[64];
  snprintf(key, sizeof key, "m%p|%lld", (void *)a, (long long)c);
  auto it = invMemo.find(key);
  if (it != invMemo.end()) return it->second;
  auto u = std::make_unique<Instruction>(Op::Mul, Type::I32);
  u->ops = {a, mod.constInt((int32_t)c)};
  Value *r = u.get();
  insertPre(std::move(u), blockIdx);
  invMemo[key] = r;
  return r;
}

class IndVarPass final : public Pass {
public:
  explicit IndVarPass(Layer l) : layer_(l) {}
  const char *name() const override { return "ind-var"; }
  Layer inLayer() const override { return layer_; }

  bool run(Module &mod) override {
    // Only the flat cf layer has cf.phi; earlier layers still carry the
    // structured loop ops, whose addressing is handled by LICM / unrolling.
    if (layer_ != Layer::Cf) return false;
    bool any = false;
    for (auto &f : mod.functions()) {
      if (f->isLib || f->blocks.size() < 3) continue;
      any |= processFunction(mod, *f);
    }
    return any;
  }

private:
  Layer layer_;

  static void succsOf(BasicBlock *bb, std::vector<BasicBlock *> &out) {
    if (bb->instrs.empty()) return;
    Instruction *last = bb->instrs.back().get();
    auto push = [&](Value *v) {
      if (auto *b = dynamic_cast<BasicBlock *>(v)) out.push_back(b);
    };
    if (last->op == Op::Br) {
      if (!last->ops.empty()) push(last->ops[0]);
    } else if (last->op == Op::CondBr && last->ops.size() >= 3) {
      push(last->ops[1]);
      push(last->ops[2]);
    }
  }

  struct IV {
    Instruction *phi = nullptr;
    Value *init = nullptr; // value on the preheader edge
    size_t header = 0, latch = 0, pre = 0;
    int64_t step = 0;
  };

  struct Cand {
    Instruction *gep = nullptr;
    Value *base = nullptr;
    int64_t k = 0;
    Value *inv = nullptr;
    // Constant part of the offset, kept *out* of `inv` on purpose.  The
    // pointer phi is built for `base + k*iv + inv`; a candidate whose offset
    // differs only by this constant is rewritten to `gep phi, c`, which the
    // instruction selector folds into the load/store displacement.  That is
    // what lets the lanes of an unrolled body share one pointer (and one
    // increment) instead of materialising a distinct preheader value each.
    int64_t c = 0;
    const IV *iv = nullptr;
  };

  bool processFunction(Module &mod, Function &f) {
    const size_t n = f.blocks.size();
    std::unordered_map<BasicBlock *, size_t> idx;
    std::unordered_map<const Instruction *, size_t> blockOf;
    for (size_t i = 0; i < n; ++i) {
      idx[f.blocks[i].get()] = i;
      for (auto &iu : f.blocks[i]->instrs) blockOf[iu.get()] = i;
    }

    std::vector<std::vector<size_t>> succ(n), pred(n);
    std::vector<BasicBlock *> tmp;
    for (size_t bi = 0; bi < n; ++bi) {
      tmp.clear();
      succsOf(f.blocks[bi].get(), tmp);
      for (BasicBlock *s : tmp) {
        auto it = idx.find(s);
        if (it == idx.end()) continue;
        succ[bi].push_back(it->second);
        pred[it->second].push_back(bi);
      }
    }

    std::vector<size_t> rpo;
    std::vector<uint8_t> vis(n, 0);
    std::function<void(size_t)> dfs = [&](size_t b) {
      vis[b] = 1;
      for (size_t s : succ[b])
        if (!vis[s]) dfs(s);
      rpo.push_back(b);
    };
    dfs(0);
    std::reverse(rpo.begin(), rpo.end());

    // Dominator sets (same iterative dataflow as dom-cse; these functions are
    // small enough that the O(n^3) bitset form is not worth beating).
    std::vector<std::vector<uint8_t>> dom(n, std::vector<uint8_t>(n, 1));
    for (size_t d = 1; d < n; ++d) dom[0][d] = 0;
    bool changed = true;
    while (changed) {
      changed = false;
      for (size_t bi : rpo) {
        if (bi == 0 || pred[bi].empty()) continue;
        std::vector<uint8_t> next(n, 1);
        for (size_t i = 0; i < n; ++i) next[i] = dom[pred[bi][0]][i];
        for (size_t pi = 1; pi < pred[bi].size(); ++pi)
          for (size_t i = 0; i < n; ++i)
            next[i] = next[i] && dom[pred[bi][pi]][i];
        next[bi] = 1;
        if (next != dom[bi]) {
          dom[bi] = std::move(next);
          changed = true;
        }
      }
    }

    bool any = false;
    for (size_t hh = 0; hh < n; ++hh) {
      // Back edges b -> h with h dominating b: h is a loop header, b a latch.
      std::vector<size_t> latches;
      for (size_t b = 0; b < n; ++b)
        for (size_t s : succ[b])
          if (s == hh && dom[b][hh]) latches.push_back(b);
      // A single latch keeps the header phis' in-loop edge unambiguous.
      if (latches.size() != 1) continue;
      size_t latch = latches[0];
      if (latch == hh) continue;

      // Natural-loop node set: everything reaching the latch without passing
      // through the header, plus the header itself.
      std::vector<uint8_t> inLoop(n, 0);
      inLoop[hh] = 1;
      inLoop[latch] = 1;
      std::vector<size_t> work{latch};
      while (!work.empty()) {
        size_t x = work.back();
        work.pop_back();
        for (size_t p : pred[x]) {
          if (inLoop[p]) continue;
          inLoop[p] = 1;
          work.push_back(p);
        }
      }

      // The unique out-of-loop predecessor is the preheader.
      std::vector<size_t> pre;
      for (size_t p : pred[hh])
        if (!inLoop[p]) pre.push_back(p);
      if (pre.size() != 1) continue;
      size_t ph = pre[0];

      // Induction variables: header phis whose latch edge is `iv +/- step`.
      std::vector<IV> ivs;
      for (auto &iu : f.blocks[hh]->instrs) {
        Instruction *phi = iu.get();
        if (phi->op != Op::Phi) continue;
        Value *init = nullptr;
        Value *nxt = nullptr;
        for (size_t k = 0; k + 1 < phi->ops.size(); k += 2) {
          auto *pb = dynamic_cast<BasicBlock *>(phi->ops[k]);
          if (!pb) continue;
          auto it = idx.find(pb);
          if (it == idx.end()) continue;
          if (it->second == ph)
            init = phi->ops[k + 1];
          else if (it->second == latch)
            nxt = phi->ops[k + 1];
        }
        if (!init || !nxt || init == phi) continue;
        auto *nx = dynamic_cast<Instruction *>(nxt);
        if (!nx || nx->ops.size() != 2) continue;
        int64_t step = 0;
        bool ok = false;
        if (nx->op == Op::Add || nx->op == Op::Sub) {
          const ConstantInt *ca = asConstInt(nx->ops[0]);
          const ConstantInt *cb = asConstInt(nx->ops[1]);
          // `step` is the *signed* increment: `subi iv, c` walks backwards.
          if (nx->ops[0] == (Value *)phi && cb) {
            step = nx->op == Op::Sub ? -cb->v : cb->v;
            ok = true;
          } else if (nx->op == Op::Add && nx->ops[1] == (Value *)phi && ca) {
            step = ca->v;
            ok = true;
          }
        }
        if (!ok || step == 0) continue;
        IV iv;
        iv.phi = phi;
        iv.init = init;
        iv.header = hh;
        iv.latch = latch;
        iv.pre = ph;
        iv.step = step;
        ivs.push_back(iv);
      }
      if (ivs.empty()) continue;

      Rewriter rw(mod, f, inLoop, blockOf);
      rw.preB = f.blocks[ph].get();

      // Derived addresses inside the loop that are affine in one of the IVs.
      std::vector<Cand> cands;
      std::unordered_set<Instruction *> seen;
      for (size_t bi = 0; bi < n; ++bi) {
        if (!inLoop[bi]) continue;
        for (auto &iu : f.blocks[bi]->instrs) {
          Instruction *ins = iu.get();
          if (ins->op != Op::Gep || ins->ops.size() != 2) continue;
          if (seen.count(ins)) continue;
          for (const IV &iv : ivs) {
            Rewriter::Aff aff;
            if (!rw.affine(ins->ops[1], iv.phi, aff, ph, 0)) continue;
            if (aff.bad || aff.k == 0) continue;
            int64_t delta = 0;
            if (!mulFits(aff.k, iv.step, delta)) continue;
            // The base has to be available (and unchanged) in the preheader.
            if (auto *bd = dynamic_cast<Instruction *>(ins->ops[0])) {
              auto bit = blockOf.find(bd);
              if (bit == blockOf.end() || inLoop[bit->second]) continue;
            }
            // Every use must sit inside the loop, so the new pointer's
            // definition in the header always dominates its uses.
            size_t uses = 0, inside = 0;
            for (size_t b2 = 0; b2 < n; ++b2) {
              for (auto &iu2 : f.blocks[b2]->instrs)
                for (Value *o : iu2->ops)
                  if (o == (Value *)ins) {
                    ++uses;
                    if (inLoop[b2]) ++inside;
                  }
            }
            if (uses == 0 || uses != inside) break;
            seen.insert(ins);
            Cand cd;
            cd.gep = ins;
            cd.base = ins->ops[0];
            cd.k = aff.k;
            cd.inv = aff.inv;
            cd.c = aff.c;
            cd.iv = &iv;
            cands.push_back(cd);
            break; // one induction variable per address
          }
        }
      }
      if (cands.empty()) continue;

      // Identical (iv, base, k, invariant) tuples share one pointer phi; the
      // candidates' constants ride along as `gep phi, c` displacements.  The
      // induction variable is part of the key, not just its coefficient: two
      // geps off the same base with the same stride are only interchangeable
      // when they are affine in the *same* loop-carried value.  A merge loop's
      // `buf[0][i]`, `buf[0][j]` and `buf[1][k]` all have base+stride 4, but
      // the three counters drift apart inside the loop, so folding them onto
      // one pointer (even one whose initial value and step happen to match)
      // reads and writes the wrong elements.
      std::unordered_map<std::string, size_t> gidx;
      std::vector<std::vector<const Cand *>> groups;
      for (const Cand &cd : cands) {
        char key[128];
        snprintf(key, sizeof key, "%p|%p|%lld|%p", (void *)cd.iv->phi,
                 (void *)cd.base, (long long)cd.k, (void *)cd.inv);
        auto git = gidx.find(key);
        if (git != gidx.end()) {
          groups[git->second].push_back(&cd);
        } else {
          gidx[key] = groups.size();
          groups.push_back({&cd});
        }
      }
      if (groups.size() > kMaxPerLoop) continue;

      for (const std::vector<const Cand *> &g : groups) {
        // A displacement is only "folded into the load/store" while it fits
        // the 12-bit signed immediate a `lw`/`sw` encodes.  A matrix row
        // stride blows straight through that: `table[k+1][j]` for a 1400-wide
        // `int` array offsets the base by 1400*4 = 5600 bytes, so leaving the
        // 1400 in the displacement makes the accessor re-materialise it every
        // iteration (`li`+`add` in the body - the very address arithmetic this
        // pass exists to delete).  Fold the group's own constant into the
        // pointer's initial value whenever that keeps every *remaining*
        // displacement in range, which is always the case for a lone
        // candidate and for the unrolled lanes (`c` = 0,4,8,12) the shared-
        // pointer form was designed for.
        int64_t c0 = 0;
        if (g.size() == 1) {
          c0 = g[0]->c; // nothing to share with, so bake it in
        } else {
          int64_t lo = g[0]->c, hi = g[0]->c;
          for (const Cand *cd : g) {
            lo = std::min(lo, cd->c);
            hi = std::max(hi, cd->c);
          }
          // Folding the smallest constant makes the first lane displacement-
          // free; only do it when the others stay encodable (and when the
          // group is all-nonnegative, so no lane's displacement grows).
          if (lo >= 0 && hi - lo <= 2047) c0 = lo;
        }
        Instruction *p = buildPointer(rw, mod, f, *g[0], c0);
        if (!p) continue;
        for (const Cand *cd : g) {
          int64_t disp = cd->c - c0;
          if (disp == 0) {
            // The phi already denotes this address: bypass the gep entirely.
            replaceAllUses(mod, cd->gep, p);
          } else {
            // Re-express the address as `phi + disp`; the backend folds the
            // constant into the load/store displacement.
            cd->gep->ops[0] = p;
            cd->gep->ops[1] = mod.constInt((int32_t)disp);
          }
        }
        any = true;
      }
    }
    return any;
  }

  static bool mulFits(int64_t a, int64_t b, int64_t &out) {
    __int128 p = (__int128)a * (__int128)b;
    if (p < INT32_MIN || p > INT32_MAX) return false;
    out = (int64_t)p;
    return true;
  }

  // Materialise the strength-reduced pointer for one derived address:
  //   preheader: p_init = gep base, (init*K + inv + c0)
  //   header:    p      = phi [pre: p_init], [latch: p_next]
  //   latch:     p_next = gep p, step*K
  // `c0` is the part of the offset shared by every candidate of the group; any
  // remainder is re-added per candidate as a gep displacement at the use site
  // (see the caller for why c0 is not always 0).
  Instruction *buildPointer(Rewriter &rw, Module &mod, Function &f,
                            const Cand &cd, int64_t c0 = 0) {
    const IV &iv = *cd.iv;
    int64_t delta = 0;
    if (!mulFits(cd.k, iv.step, delta)) return nullptr;

    BasicBlock *preB = f.blocks[iv.pre].get();
    BasicBlock *headB = f.blocks[iv.header].get();
    BasicBlock *latchB = f.blocks[iv.latch].get();

    // preheader: the offset against the loop's initial induction value, plus
    // the invariant part (already materialised in the preheader by preOf).
    std::vector<std::unique_ptr<Instruction>> preOps;
    Value *off = nullptr;
    if (cd.k == 1) {
      off = iv.init;
    } else {
      auto mu = std::make_unique<Instruction>(Op::Mul, Type::I32);
      mu->ops = {iv.init, mod.constInt((int32_t)cd.k)};
      off = mu.get();
      preOps.push_back(std::move(mu));
    }
    if (cd.inv) {
      auto ad = std::make_unique<Instruction>(Op::Add, Type::I32);
      ad->ops = {off, cd.inv};
      off = ad.get();
      preOps.push_back(std::move(ad));
    }
    if (c0 != 0) {
      auto ad = std::make_unique<Instruction>(Op::Add, Type::I32);
      ad->ops = {off, mod.constInt((int32_t)c0)};
      off = ad.get();
      preOps.push_back(std::move(ad));
    }
    auto pinit = std::make_unique<Instruction>(Op::Gep, Type::Ptr);
    pinit->ops = {cd.base, off};
    Instruction *pInit = pinit.get();
    preOps.push_back(std::move(pinit));

    // header: the pointer phi, kept with the block's other phis
    auto pphi = std::make_unique<Instruction>(Op::Phi, Type::Ptr);
    Instruction *p = pphi.get();

    // latch: the per-iteration increment
    auto pnext = std::make_unique<Instruction>(Op::Gep, Type::Ptr);
    pnext->ops = {p, mod.constInt((int32_t)delta)};
    Instruction *pNext = pnext.get();

    // splice the preheader ops in before its terminator
    auto &pv = preB->instrs;
    size_t at = pv.empty() ? 0 : pv.size() - 1;
    for (auto &u : preOps) {
      Instruction *pp = u.get();
      pv.insert(pv.begin() + (long)at++, std::move(u));
      rw.blockOf[pp] = iv.pre;
    }

    // the phi goes after the last existing phi in the header
    auto &hv = headB->instrs;
    size_t hAt = 0;
    while (hAt < hv.size() && hv[hAt]->op == Op::Phi) ++hAt;
    hv.insert(hv.begin() + (long)hAt, std::move(pphi));
    rw.blockOf[p] = iv.header;

    p->ops = {(Value *)preB, pInit, (Value *)latchB, pNext};

    auto &lv = latchB->instrs;
    size_t lAt = lv.empty() ? 0 : lv.size() - 1;
    lv.insert(lv.begin() + (long)lAt, std::move(pnext));
    rw.blockOf[pNext] = iv.latch;

    return p;
  }
};

} // namespace

std::unique_ptr<Pass> makeIndVarPass(Layer l) {
  return std::make_unique<IndVarPass>(l);
}

} // namespace ir
} // namespace sakura
