#pragma once
#include "IR.h"
#include <algorithm>
#include <functional>
#include <iterator>
#include <unordered_set>
namespace sakura { namespace ir {
// ---- signed range analysis (LLVM computeKnownBits / SCEV, in miniature) ----
// Prove that an i32 value can never be negative, and derive a signed value
// range for it.  The result feeds the signed div/rem strength reduction in
// lowerInstruction: for a non-negative dividend, C's `x % 2^k` and `x / 2^k`
// are exactly the unsigned `and` and logical shift.
//
// The interesting case is the loop induction variable.  A counter
// `i = phi(0, latch: i + c)` is non-negative because it starts at 0 and only
// grows -- but "only grows" is only true while the increment does not wrap
// i32, and that is what the loop guard buys.  Mirroring LLVM's guarded-addrec
// reasoning, the phi rule requires
//
//   * every non-back-edge incoming value to be non-negative, and
//   * the back-edge value to be literally `i + c` (or `i - c`) with a
//     non-zero constant step, and
//   * a comparison against a constant on `i` (or on the back-edge value) whose
//     true edge dominates the back edge -- so the bound holds on *every* path
//     reaching the increment -- with the bound still inside i32 after adding
//     the step (otherwise the increment could have wrapped), and
//   * the remaining non-back-edge incoming bounds.
//
// When the loop guard is not a constant (`i < n` with `n` loaded or passed in,
// which is the norm once the kernels are inlined) there is no bound to read
// off, but a *strict* guard is still enough for the wrap argument: `i < n`
// with `n <= INT32_MAX` implies `i <= INT32_MAX - 1`, so `i + 1` cannot wrap
// and the entry join's non-negativity survives.  Only `+1`/`-1` steps get that
// treatment; a larger step could still step over the extreme.  LLVM's SCEV
// derives the same "no self-wrap" fact from the guard.
//
// Everything else is a plain interval rule (add/sub/mul/select/div/rem),
// dropping to "unknown" the moment an operation could overflow.  The analysis
// is only ever allowed to be wrong in the safe direction.
struct Range {
  long long lo = INT32_MIN; // INT32_MIN/INT32_MAX mean "no bound known"
  long long hi = INT32_MAX;
  bool nonNeg() const { return lo >= 0; }
};

class RangeAnalysis {
public:
  void run(Function *fn) {
    blocks_.clear(); bidx_.clear(); blkOf_.clear(); dom_.clear();
    ranges_.clear(); guards_.clear(); varGuards_.clear();
    fn_ = fn;
    for (auto &bb : fn->blocks) {
      bidx_[bb.get()] = (int)blocks_.size();
      for (auto &iu : bb->instrs) blkOf_[iu.get()] = (int)blocks_.size();
      blocks_.push_back(bb.get());
    }
    if (blocks_.empty()) return;
    computeDom();
    collectGuards();
    // Kildall-style iteration from "everything unknown".  Every rule is
    // monotone and only ever narrows an interval, so the sequence is
    // descending and a generous cap both keeps the cost bounded and lets the
    // long inlined helper chains (whose value flow crosses several nested
    // loops) reach a fixpoint instead of getting wiped out.
    for (int round = 0; round < 256; ++round) {
      bool changed = false;
      for (auto &bb : fn->blocks)
        for (auto &iu : bb->instrs) {
          Instruction *i = iu.get();
          if (i->ty != Type::I32) continue;
          if (update(i, i->op == Op::Phi ? phiRange(i) : evaluate(i)))
            changed = true;
        }
      if (!changed) return;
    }
    ranges_.clear(); // did not converge: claim nothing
  }

  bool nonNeg(Value *v) const {
    if (auto *ci = dynamic_cast<ConstantInt *>(v)) return ci->v >= 0;
    auto it = ranges_.find(v);
    return it != ranges_.end() && it->second.nonNeg();
  }

  // Signed range of `v`; the full i32 range when nothing was proven.  Exposed
  // so instruction selection can ask whether a difference may overflow.
  Range rangeOfValue(Value *v) const { return rangeOf(v); }

private:
  static Cond swapCond(Cond c) {
    switch (c) {
    case Cond::Lt: return Cond::Gt; case Cond::Le: return Cond::Ge;
    case Cond::Gt: return Cond::Lt; case Cond::Ge: return Cond::Le;
    default: return c;
    }
  }
  Function *fn_ = nullptr;
  std::vector<BasicBlock *> blocks_;
  std::unordered_map<const BasicBlock *, int> bidx_;
  std::unordered_map<const Instruction *, int> blkOf_;
  std::vector<std::unordered_set<int>> dom_;
  std::unordered_map<const Value *, Range> ranges_;

  // A branch-implied bound on a value: taken on the *true* edge of the
  // terminator of `blk` (whose true successor is `thenTarget`).
  struct GuardFact {
    int blk;
    int thenTarget;
    bool upper;
    long long bound;
  };
  std::unordered_map<const Value *, std::vector<GuardFact>> guards_;

  // A *strict* branch-implied bound whose boundary is a variable (`v < x`,
  // `v > x`).  The boundary value is unknown, but an i32 never leaves
  // [INT32_MIN, INT32_MAX], so `v < x` still proves `v <= INT32_MAX - 1` (and
  // `v > x` proves `v >= INT32_MIN + 1`).  That is all the induction rule
  // needs to know the `+1`/`-1` step cannot wrap.
  struct VarGuard {
    int blk;
    int thenTarget;
    bool upper; // true: `v < x`, false: `v > x`
  };
  std::unordered_map<const Value *, std::vector<VarGuard>> varGuards_;

  bool dominates(int a, int b) const { return dom_[(size_t)b].count(a) != 0; }

  void computeDom() {
    size_t n = blocks_.size();
    dom_.assign(n, {});
    std::vector<std::vector<int>> preds(n);
    for (size_t k = 0; k < n; ++k) {
      BasicBlock *b = blocks_[k];
      if (b->instrs.empty()) continue;
      Instruction *t = b->instrs.back().get();
      if (t->op == Op::Br && !t->ops.empty()) {
        auto *d = dynamic_cast<BasicBlock *>(t->ops[0]);
        if (d && bidx_.count(d)) preds[(size_t)bidx_[d]].push_back((int)k);
      } else if (t->op == Op::CondBr && t->ops.size() >= 3) {
        for (size_t j = 1; j <= 2; ++j) {
          auto *d = dynamic_cast<BasicBlock *>(t->ops[j]);
          if (d && bidx_.count(d)) preds[(size_t)bidx_[d]].push_back((int)k);
        }
      }
    }
    std::unordered_set<int> all;
    for (size_t k = 0; k < n; ++k) all.insert((int)k);
    dom_[0] = {0};
    for (size_t k = 1; k < n; ++k) dom_[k] = all;
    bool changed = true;
    while (changed) {
      changed = false;
      for (size_t k = 1; k < n; ++k) {
        std::unordered_set<int> d;
        bool first = true;
        for (int p : preds[k]) {
          if (first) {
            d = dom_[(size_t)p];
            first = false;
          } else {
            for (auto it = d.begin(); it != d.end();)
              it = dom_[(size_t)p].count(*it) ? std::next(it) : d.erase(it);
          }
        }
        if (first) d = {0}; // unreachable: keep it conservative but connected
        d.insert((int)k);
        if (d != dom_[k]) {
          dom_[k] = std::move(d);
          changed = true;
        }
      }
    }
  }

  void collectGuards() {
    for (size_t k = 0; k < blocks_.size(); ++k) {
      BasicBlock *b = blocks_[k];
      if (b->instrs.empty()) continue;
      Instruction *t = b->instrs.back().get();
      if (t->op != Op::CondBr || t->ops.size() < 3) continue;
      auto *c = dynamic_cast<Instruction *>(t->ops[0]);
      if (!c || c->op != Op::ICmp || c->ops.size() != 2) continue;
      auto *lc = dynamic_cast<ConstantInt *>(c->ops[0]);
      auto *rc = dynamic_cast<ConstantInt *>(c->ops[1]);
      auto *tb = dynamic_cast<BasicBlock *>(t->ops[1]);
      if (!tb) continue;
      auto tit = bidx_.find(tb);
      if (tit == bidx_.end()) continue;
      if (!lc && !rc) {
        // Both sides are variables: no interval can be read off, but a strict
        // relation still pins the operand away from the i32 extreme, which is
        // what the induction rule uses to rule out a wrapping step.  Record the
        // relation for either operand.
        Cond p0 = c->cond;             // ops[0] <p0> ops[1]
        Cond p1 = swapCond(c->cond); // ops[1] <p1> ops[0]
        if (p0 == Cond::Lt || p0 == Cond::Gt)
          varGuards_[c->ops[0]].push_back({(int)k, tit->second, p0 == Cond::Lt});
        if (p1 == Cond::Lt || p1 == Cond::Gt)
          varGuards_[c->ops[1]].push_back({(int)k, tit->second, p1 == Cond::Lt});
        continue;
      }
      Value *v = nullptr;
      long long K = 0;
      bool vLeft = false;
      if (rc && !lc) {
        v = c->ops[0];
        K = rc->v;
        vLeft = true;
      } else if (lc && !rc) {
        v = c->ops[1];
        K = lc->v;
      } else {
        continue; // both constant (folded away)
      }
      Cond pred = vLeft ? c->cond : swapCond(c->cond);
      bool upper;
      long long bound;
      switch (pred) {
      case Cond::Lt: upper = true;  bound = K - 1; break;
      case Cond::Le: upper = true;  bound = K;     break;
      case Cond::Gt: upper = false; bound = K + 1; break;
      case Cond::Ge: upper = false; bound = K;     break;
      default: continue; // Eq/Ne imply no interval
      }
      guards_[v].push_back({(int)k, tit->second, upper, bound});
    }
  }

  // Is there a guard on `v` that is in force on the edge pi -> succBlk?
  bool boundAt(Value *v, int pi, int succBlk, bool upper, long long &out) {
    auto it = guards_.find(v);
    if (it == guards_.end()) return false;
    bool found = false;
    for (const GuardFact &g : it->second) {
      if (g.upper != upper) continue;
      bool holds;
      if (g.blk == pi) {
        holds = g.thenTarget == succBlk; // pi's own branch: only the true edge
      } else {
        holds = g.thenTarget == pi || dominates(g.thenTarget, pi);
      }
      if (!holds) continue;
      if (!found || (upper ? g.bound < out : g.bound > out)) out = g.bound;
      found = true;
    }
    return found;
  }

  // Is there a strict variable-bounded guard on `v` in force on the edge
  // pi -> succBlk?  See VarGuard above.
  bool varGuardAt(Value *v, int pi, int succBlk, bool upper) {
    auto it = varGuards_.find(v);
    if (it == varGuards_.end()) return false;
    for (const VarGuard &g : it->second) {
      if (g.upper != upper) continue;
      bool holds;
      if (g.blk == pi) {
        holds = g.thenTarget == succBlk; // pi's own branch: only the true edge
      } else {
        holds = g.thenTarget == pi || dominates(g.thenTarget, pi);
      }
      if (holds) return true;
    }
    return false;
  }

  Range rangeOf(Value *v) const {
    if (auto *ci = dynamic_cast<ConstantInt *>(v)) return {ci->v, ci->v};
    auto it = ranges_.find(v);
    return it == ranges_.end() ? Range{} : it->second;
  }

  bool update(Instruction *i, Range r) {
    auto it = ranges_.find(i);
    if (it != ranges_.end() && it->second.lo == r.lo && it->second.hi == r.hi)
      return false;
    ranges_[i] = r;
    return true;
  }

  static Range safeAdd(Range a, Range b) {
    long long lo = a.lo + b.lo, hi = a.hi + b.hi;
    if (lo < INT32_MIN || hi > INT32_MAX) return Range{};
    return {lo, hi};
  }
  static Range safeSub(Range a, Range b) {
    long long lo = a.lo - b.hi, hi = a.hi - b.lo;
    if (lo < INT32_MIN || hi > INT32_MAX) return Range{};
    return {lo, hi};
  }
  static Range safeMul(Range a, Range b) {
    long long products[] = {a.lo*b.lo, a.lo*b.hi, a.hi*b.lo, a.hi*b.hi};
    auto bounds = std::minmax_element(std::begin(products), std::end(products));
    if (*bounds.first < INT32_MIN || *bounds.second > INT32_MAX) return Range{};
    return {*bounds.first, *bounds.second};
  }

  Range evaluate(Instruction *i) {
    return evaluateWith(i, [&](size_t k) { return rangeOf(i->ops[k]); });
  }

  // Range of `i` where its operands' ranges come from `rng`.  Splitting this
  // out of evaluate() lets evalUnder() below re-derive a sub-expression with a
  // hypothetically narrowed operand.
  Range evaluateWith(Instruction *i,
                     const std::function<Range(size_t)> &rng) {
    switch (i->op) {
    case Op::ICmp: // materialises 0/1
    case Op::Not:
      return {0, 1};
    case Op::Add:
      if (i->ops.size() == 2) return safeAdd(rng(0), rng(1));
      break;
    case Op::Sub:
      if (i->ops.size() == 2) return safeSub(rng(0), rng(1));
      break;
    case Op::Mul:
      if (i->ops.size() == 2) return safeMul(rng(0), rng(1));
      break;
    case Op::SDiv: {
      if (i->ops.size() != 2) break;
      auto *c = dynamic_cast<ConstantInt *>(i->ops[1]);
      Range a = rng(0);
      if (!c || c->v <= 0 || a.lo < 0) break; // truncation direction unknown
      return {a.lo / c->v, a.hi / c->v};
    }
    case Op::SRem: {
      if (i->ops.size() != 2) break;
      auto *c = dynamic_cast<ConstantInt *>(i->ops[1]);
      Range a = rng(0);
      // C remainder takes the dividend's sign, so a non-negative dividend by a
      // positive constant lands in [0, c-1].
      if (!c || c->v <= 0 || a.lo < 0) break;
      return {0, std::min<long long>(a.hi, c->v - 1)};
    }
    case Op::Select: {
      if (i->ops.size() != 3) break;
      Range t = rng(1), f = rng(2);
      return {std::min(t.lo, f.lo), std::max(t.hi, f.hi)};
    }
    default:
      break;
    }
    return Range{};
  }

  // Range of `v` recomputed under the assumption that `phi`'s range is `r`.
  // Values that do not depend on `phi` fall back to their already-computed
  // range, which is always a sound over-approximation; `depth` keeps a
  // pathological expression tree from making this blow up.  Used to prove that
  // a loop-carried phi whose back edge is a shrinking function of the phi
  // itself (`x / 2`, `x & mask`, `x >> k`, ...) can never leave the join of
  // its entry values - the shape the inlined bit helpers produce.
  Range evalUnder(Value *v, Instruction *phi, Range r, int depth) {
    if (v == phi) return r;
    if (auto *ci = dynamic_cast<ConstantInt *>(v)) return {ci->v, ci->v};
    auto *in = dynamic_cast<Instruction *>(v);
    if (!in || in->ty != Type::I32 || depth >= 6) return rangeOf(v);
    return evaluateWith(in, [&](size_t k) {
      if (k >= in->ops.size()) return Range{};
      return evalUnder(in->ops[k], phi, r, depth + 1);
    });
  }

  // `v` is the phi incremented by a positive/negative constant step.
  static bool indStep(Instruction *phi, Value *v, bool &up, long long &c) {
    auto *in = dynamic_cast<Instruction *>(v);
    if (!in || in->ty != Type::I32 || in->ops.size() != 2) return false;
    auto *c0 = dynamic_cast<ConstantInt *>(in->ops[0]);
    auto *c1 = dynamic_cast<ConstantInt *>(in->ops[1]);
    if (in->op == Op::Add) {
      if (in->ops[1] == phi && c0)
        c = c0->v;
      else if (in->ops[0] == phi && c1)
        c = c1->v;
      else
        return false;
      up = c >= 0;
      c = c < 0 ? -c : c;
      return c != 0;
    }
    if (in->op == Op::Sub && in->ops[0] == phi && c1) {
      c = c1->v;
      up = c <= 0;
      c = c < 0 ? -c : c;
      return c != 0;
    }
    return false;
  }

  struct BackEdge {
    int pred;     // predecessor block index
    Value *value; // the incoming value (`phi +/- step`)
    bool up;      // step direction
    long long step;
    bool ind;     // `value` really is `phi +/- step`
  };

  Range phiRange(Instruction *phi) {
    int self = blkOf_.count(phi) ? blkOf_.at(phi) : -1;
    if (self < 0) return Range{};
    bool haveNonBack = false;
    Range nb;  // join of the entry values
    // Join of *every* incoming value.  A phi evaluates to one of its incoming
    // values, so this join is a sound over-approximation for any phi - it is
    // the fallback whenever the tight induction-variable reasoning below does
    // not apply.  Without it a plain if/else value merge (which has no back
    // edge at all) would be reported as "unknown", which is what kept the
    // saturating inlined helpers of the bit-twiddling kernels from proving
    // their results non-negative.
    Range all;
    bool haveAll = false;
    std::vector<BackEdge> backs;
    bool backsOk = true;
    for (size_t k = 0; k + 1 < phi->ops.size(); k += 2) {
      auto *pbb = dynamic_cast<BasicBlock *>(phi->ops[k]);
      Value *v = phi->ops[k + 1];
      if (!pbb || bidx_.count(pbb) == 0) continue;
      int pi = bidx_.at(pbb);
      Range r = rangeOf(v);
      if (!haveAll) {
        all = r;
        haveAll = true;
      } else {
        all.lo = std::min(all.lo, r.lo);
        all.hi = std::max(all.hi, r.hi);
      }
      if (dominates(self, pi)) { // the header dominates this predecessor
        bool up = false;
        long long step = 0;
        bool ind = indStep(phi, v, up, step);
        if (!ind) backsOk = false;
        backs.push_back({pi, v, up, step, ind});
      } else {
        // Join (union) of the entry values; the accumulator starts empty, so
        // the first entry value seeds it rather than being met with top.
        if (!haveNonBack) {
          nb = r;
          haveNonBack = true;
        } else {
          nb.lo = std::min(nb.lo, r.lo);
          nb.hi = std::max(nb.hi, r.hi);
        }
      }
    }
    if (!haveAll) return Range{};
    // Tighter form: an induction variable whose trip count is pinned by the
    // loop guards.  The result is intersected with the plain join, so a failed
    // or imprecise induction derivation can only ever lose precision, never
    // soundness.
    if (haveNonBack && backsOk && !backs.empty()) {
      long long lo = nb.lo, hi = nb.hi;
      bool ok = true;
      for (const BackEdge &b : backs) {
        long long bound;
        if (b.up) {
          // An upper bound on the counter where the step is applied: taken on
          // the counter itself (the new value is bound + step), or directly on
          // the newly computed value.
          if (boundAt(phi, b.pred, self, true, bound)) {
            bound += b.step;
          } else if (boundAt(b.value, b.pred, self, true, bound)) {
            // bound already describes the back-edge value
          } else if (b.step == 1 &&
                     varGuardAt(phi, b.pred, self, true)) {
            // No constant in the guard, but `i < n` with n <= INT32_MAX keeps
            // `i` at most INT32_MAX - 1, so `i + 1` cannot wrap and the loop
            // cannot carry the counter negative.  The upper end stays open.
            bound = INT32_MAX;
          } else {
            ok = false;
            break;
          }
          if (bound > INT32_MAX) {
            ok = false;
            break;
          }
          hi = std::max(hi, bound);
        } else {
          if (boundAt(phi, b.pred, self, false, bound)) {
            bound -= b.step;
          } else if (boundAt(b.value, b.pred, self, false, bound)) {
            // bound already describes the back-edge value
          } else if (b.step == 1 && varGuardAt(phi, b.pred, self, false)) {
            bound = INT32_MIN; // `i > n` rules out the `i - 1` underflow
          } else {
            ok = false;
            break;
          }
          if (bound < INT32_MIN) {
            ok = false;
            break;
          }
          lo = std::min(lo, bound);
        }
      }
      if (ok && hi >= lo) {
        all.lo = std::max(all.lo, lo);
        all.hi = std::min(all.hi, hi);
      }
    } else if (haveNonBack && !backs.empty()) {
      // Not a simple +/- induction, but the loop-carried value may still be a
      // *shrinking* function of the phi itself.  If the entry join is already
      // closed under every back edge, the phi can never leave it, so it is a
      // sound (greatest) fixpoint.  `x / 2` and `x / 2^k` in the inlined
      // division helpers are exactly this shape, and proving the phi
      // non-negative is what turns its `% 2^k` into an `andi` and its `/ 2^k`
      // into a logical shift.
      bool stable = true;
      for (const BackEdge &b : backs) {
        Range r = evalUnder(b.value, phi, nb, 0);
        if (r.lo < nb.lo || r.hi > nb.hi) {
          stable = false;
          break;
        }
      }
      if (stable) {
        all.lo = std::max(all.lo, nb.lo);
        all.hi = std::min(all.hi, nb.hi);
      }
    }
    if (all.hi < all.lo) return Range{};
    return all;
  }
};


} }
