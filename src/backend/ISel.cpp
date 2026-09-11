// Instruction selection implementation: walks each IR function's basic blocks
// and maps operations onto machine instructions.  Local variables get frame
// slots (via memref.alloca), arrays get contiguous 4-byte-slot regions, and
// every call gets outbound stack-argument space sized per the RISC-V ABI.
#include "ISel.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
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
        Cond p1 = invertCond(c->cond); // ops[1] <p1> ops[0]
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
      Cond pred = vLeft ? c->cond : invertCond(c->cond);
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
    if (a.lo < 0 || b.lo < 0) return Range{}; // keep the sign analysis simple
    long long hi = a.hi * b.hi; // <= 2^62: fits int64
    if (hi > INT32_MAX) return Range{};
    return {a.lo * b.lo, hi};
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
    if (haveNonBack && backsOk && !backs.empty() && nb.lo >= 0) {
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
          } else if (b.step == 1 && nb.lo >= 0 &&
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

class FnLower {
public:
  MachineFunc mf;
  Function *fn = nullptr;

  std::unordered_map<const Value *, int32_t> vreg;
  std::unordered_map<const Value *, PtrVal> ptrs;
  std::unordered_map<const Instruction *, int64_t> frOff;
  std::unordered_set<const Instruction *> liveAlloca;
  // phi instructions per block (cf.phi), used to place register copies at
  // the end of every predecessor block.
  std::unordered_map<const BasicBlock *, std::vector<Instruction *>> blockPhis;
  std::unordered_map<const Value *, int32_t> phiRegs; // phi -> its vreg

  int nextReg = 0;
  int nextFrameByte = 0;
  int maxStk = 0;
  MBlock *cur = nullptr;
  // Scalar/float constant materialisation cache, scoped to the *current*
  // basic block only.  Reusing a vreg across blocks is unsafe: the `li`
  // lands at the first use, which need not dominate sibling blocks (the same
  // literal used on both arms of a cond_br after a helper was inlined into
  // each arm).  Within one straight-line block reuse is safe and keeps
  // register pressure low on argument-heavy functions.
  const BasicBlock *curBB = nullptr;
  std::unordered_map<const Value *, int32_t> blkConst;

  // Signed ranges of the function's i32 values (see RangeAnalysis).  The
  // signed division / remainder strength reductions below use it: for a
  // non-negative dividend and a power-of-two divisor the C-semantics `%` and
  // `/` are exactly the unsigned `and` and logical `shift`, which is what
  // LLVM's computeKnownBits + DAGCombiner produce.
  RangeAnalysis range;
  bool isNonNeg(Value *v) const { return range.nonNeg(v); }

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
    if (auto *pi = dynamic_cast<Instruction *>(v))
      if (pi->op == Op::Phi) return phiRegs.at(pi); // bound in the pre-scan
    MInst m;
    if (auto *ci = dynamic_cast<ConstantInt *>(v)) {
      auto cIt = blkConst.find(v);
      if (cIt != blkConst.end()) return cIt->second; // same-block reuse
      m.op = MOp::Li;
      m.dst = newReg();
      m.imm = ci->v;
      m.line = line;
      emit(m);
      return blkConst[v] = m.dst; // cached only for the current block
    }
    if (auto *cf = dynamic_cast<ConstantFloat *>(v)) {
      auto cIt = blkConst.find(v);
      if (cIt != blkConst.end()) return cIt->second;
      m.op = MOp::LiF;
      m.dst = newReg();
      m.imm = (int32_t)cf->bits();
      m.line = line;
      emit(m);
      return blkConst[v] = m.dst; // same-block reuse only
    }
    std::string desc = "?";
    if (auto *inst = dynamic_cast<Instruction *>(v))
      desc = "inst op=" + std::to_string((int)inst->op) +
             " line=" + std::to_string(inst->line);
    else if (dynamic_cast<Argument *>(v)) desc = "argument";
    else if (dynamic_cast<GlobalVar *>(v)) desc = "global";
    throw CompileError("internal: unbound value in instruction selection [" +
                           desc + "]",
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
      if (ai->op == Op::Phi) {
        // A pointer phi (induction-variable strength reduction): its value is
        // the virtual register the predecessor copies write.
        auto it = phiRegs.find(ai);
        if (it != phiRegs.end()) {
          PtrVal p;
          p.kind = 2;
          p.reg = it->second;
          return ptrs.emplace(v, p).first->second;
        }
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
      int32_t base = p.reg;
      int64_t off = p.off;
      // `lw`/`sw` encode at most a 12-bit signed displacement.  A wider folded
      // constant - which is exactly what the unroller produces (`a[i*N+k]`
      // becomes base, base+4096, base+8192, ... inside one body) - cannot be
      // encoded, and the writer would otherwise silently re-expand it *after*
      // the optimisation pipeline as `li t0, C; add t0, base, t0; lw 0(t0)`:
      // three instructions per access, re-materialised on every iteration.
      // Emitting the `li` here instead keeps the constant visible to machine
      // LICM (it is loop-invariant, so it moves into the preheader) and to the
      // constant CSE, leaving just one `add` per access in the loop.  The
      // pointer cache is deliberately not updated: this register only dominates
      // the access being lowered, not the gep's other uses.
      if (off != 0 && !fits12(off)) {
        int32_t l = opI(MOp::Li, -1, (int32_t)off, m.line);
        base = op2(MOp::IAdd, base, l, m.line, Type::Ptr);
        off = 0;
      }
      if (isStore)
        m.b = base;
      else
        m.a = base;
      m.imm = (int32_t)off;
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

  int32_t liImm(int32_t v, int line) {
    MInst m;
    m.op = MOp::Li;
    m.dst = newReg();
    m.imm = v;
    m.line = line;
    emit(m);
    return m.dst;
  }

  // `x <cond> C` / `C <cond> x` with a 12-bit C, folded into `slti` / `xori`
  // instead of materialising C in a register first.
  //
  // Two wins, both of which LLVM's RISC-V ISel gets from the `(slti ...)`
  // patterns in RISCVInstrInfo.td: the `li` disappears, and - the part that
  // actually shows up on the loop benchmarks - C stops being a live value
  // across the loop body, which is one fewer register for the register
  // allocator to keep alive.  On `x == C` the fold additionally lets the
  // compare stay a `xori; seqz` pair rather than needing `C` in a second
  // register (`xor dst, a, b; seqz`) the way the generic form does.
  //
  // Returns false when the shape does not apply (C too large, or the
  // `C + 1` that `<=`/`>` need wrapping); the caller then falls back to the
  // register form.
  bool cmpImm(Instruction *i, int64_t C, bool constOnRight, int line,
              int32_t &out) {
    if (!fits12(C)) return false;
    Cond c = i->cond;
    if (!constOnRight) { // `C <cond> x` == `x <flip(cond)> C`
      switch (c) {
      case Cond::Lt: c = Cond::Gt; break;
      case Cond::Le: c = Cond::Ge; break;
      case Cond::Gt: c = Cond::Lt; break;
      case Cond::Ge: c = Cond::Le; break;
      default: break; // Eq / Ne are symmetric
      }
    }
    // `<=`/`>` are rewritten through `C + 1`; that step must stay 12-bit too
    // (and not wrap, which `fits12` already rules out for INT32_MAX).
    if ((c == Cond::Le || c == Cond::Gt) && !fits12(C + 1)) return false;
    // `x == 0` / `x != 0` are already a bare `seqz`/`snez` through the
    // generic path; folding them here would only add a `xori`.
    if ((c == Cond::Eq || c == Cond::Ne) && C == 0) return false;
    int32_t x = forceReg(i->ops[constOnRight ? 0 : 1], line);
    switch (c) {
    case Cond::Eq:
    case Cond::Ne: {
      // (x ^ C) == 0 / != 0, i.e. `xori; seqz|snez`
      int32_t t = opI(MOp::IXorI, x, (int32_t)C, line);
      MInst m;
      m.op = MOp::ICmp;
      m.dst = newReg();
      m.a = t;
      m.b = -1; // x0
      m.imm = (int32_t)c;
      m.line = line;
      emit(m);
      out = m.dst;
      return true;
    }
    case Cond::Lt:
      out = opI(MOp::ISlti, x, (int32_t)C, line);
      return true;
    case Cond::Le:
      out = opI(MOp::ISlti, x, (int32_t)(C + 1), line);
      return true;
    case Cond::Gt:
      out = opI(MOp::ISlti, x, (int32_t)(C + 1), line);
      out = opI(MOp::IXorI, out, 1, line);
      return true;
    default: // Ge
      out = opI(MOp::ISlti, x, (int32_t)C, line);
      out = opI(MOp::IXorI, out, 1, line);
      return true;
    }
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

// True when `v` is provably 0 or 1, i.e. a boolean living in an i32.  All
// conditions in this IR are already normalised to that range (arith.cmpi and
// `!` both produce 0/1), but a *select* only is one when both of its arms are,
// so the check recurses there - a select of, say, 10 and 20 must not be
// mistaken for a flag, or the mask sequences below would compute `-(10)`.
bool isBool01(Value *v, int depth = 0) {
  if (depth > 8) return false;
  if (auto *cc = dynamic_cast<ConstantInt *>(v)) return cc->v == 0 || cc->v == 1;
  auto *ci = dynamic_cast<Instruction *>(v);
  if (!ci) return false;
  switch (ci->op) {
  case Op::ICmp:
  case Op::FCmp:
  case Op::Not:
    return true;
  case Op::Select:
    return ci->ops.size() == 3 && isBool01(ci->ops[1], depth + 1) &&
           isBool01(ci->ops[2], depth + 1);
  default:
    return false;
  }
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
  case Op::Phi:
    return; // a pure merge: the vreg was bound in the pre-scan and the
            // incoming copies are emitted at the predecessors' terminators
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
    if (i->op == Op::Mul) {
      auto lc = dynamic_cast<ConstantInt *>(i->ops[0]);
      auto rc = dynamic_cast<ConstantInt *>(i->ops[1]);
      Value *var = nullptr;
      int32_t C = 0;
      if (rc && !lc) {
        var = i->ops[0];
        C = rc->v;
      } else if (lc && !rc) {
        var = i->ops[1];
        C = lc->v;
      }
      if (var && C > 0 && (C & (C - 1)) == 0) {
        int sh = 0;
        int32_t t = C;
        while (t > 1) {
          t >>= 1;
          ++sh;
        }
        if (sh >= 1 && sh <= 31) {
          L.vreg[i] = L.opI(MOp::SllI, L.forceReg(var, line), sh, line);
          return;
        }
      }
    } else {
      // Signed div/rem of a value that can never be negative by a positive
      // power of two: C's `x % 2^k` is then exactly `x & (2^k-1)` and `x / 2^k`
      // exactly `x >>u k`, so the whole sign-bias correction sequence the
      // back-end would otherwise emit (srai/srli/add/and/sub) disappears.
      // This is what LLVM's computeKnownBits + DAGCombiner do for the same
      // shape, and it is by far the most common index/parity pattern in the
      // array kernels (`i % 16` in the crypto/hash cases, `i & 1`-style
      // parity tests everywhere else).
      auto rc = dynamic_cast<ConstantInt *>(i->ops[1]);
      if (rc && rc->v > 1 && (rc->v & (rc->v - 1)) == 0 &&
          i->ops.size() == 2 && L.isNonNeg(i->ops[0])) {
        int sh = 0;
        for (int32_t t = rc->v; t > 1; t >>= 1) ++sh;
        if (i->op == Op::SRem) {
          // `andi` only encodes a 12-bit signed immediate, so wider masks
          // (e.g. `i % 65536`) have to materialise the constant first.
          int32_t mask = rc->v - 1;
          int32_t a = L.forceReg(i->ops[0], line);
          L.vreg[i] = mask <= 2047
                          ? L.opI(MOp::IAndI, a, mask, line)
                          : L.op2(MOp::IAnd, a, L.opI(MOp::Li, -1, mask, line),
                                  line);
          return;
        }
        if (sh <= 31) {
          L.vreg[i] = L.opI(MOp::SrlI, L.forceReg(i->ops[0], line), sh, line);
          return;
        }
      }
    }
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
    // If one side is a small constant, fold it into an immediate compare
    // (`slti`/`xori`) before falling back to `li` + register compare.
    static const bool noCmpImm = std::getenv("SAKU_NO_CMPIMM") != nullptr;
    int32_t immRes = 0;
    if (!noCmpImm && rc &&
        L.cmpImm(i, rc->v, /*constOnRight=*/true, line, immRes)) {
      L.vreg[i] = immRes;
      return;
    }
    if (!noCmpImm && lc &&
        L.cmpImm(i, lc->v, /*constOnRight=*/false, line, immRes)) {
      L.vreg[i] = immRes;
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
  case Op::Select: {
    // RISC-V has no conditional move, so a select becomes a mask sequence:
    //   mask = -c          (c is 0/1 -> 0x0 or 0xFFFF...FFFF)
    //   res  = b ^ ((a ^ b) & mask)
    // Two operand shapes collapse it further, and both are the ones the
    // if-converter actually produces from `if (c) x = k;` / `if (c) x += v;`:
    //   b == 0  ->  res = a & mask
    //   a == 0  ->  res = b & (c - 1)
    Value *cv = i->ops[0], *av = i->ops[1], *bv = i->ops[2];
    if (i->ty != Type::I32)
      throw CompileError("internal: non-integer select reached ISel", line);
    // The mask trick needs the condition *normalised* to 0/1: `-c` is an
    // all-ones mask only for c == 1.  A comparison already yields 0/1;
    // anything else (a phi merging branch conditions, an `&&` result) is
    // normalised with `sltu c, x0, c` == (c != 0), which is what the source
    // `cond_br` meant anyway.
    int32_t c;
    bool known01 = isBool01(cv);
    if (known01) {
      c = L.forceReg(cv, line);
    } else {
      c = L.op1(MOp::ISltuZ, L.forceReg(cv, line), line);
    }
    auto *ac = dynamic_cast<ConstantInt *>(av);
    auto *bc = dynamic_cast<ConstantInt *>(bv);
    // `select(c, y, 0)` and `select(c, 0, y)` with y itself 0/1 need no mask:
    // `y & c` (resp. `y & !c`) is exact, because for a flag the sign-extending
    // negation `-c` is just `c` again.  This is the shape the front end leaves
    // behind for every short-circuited `A && B` (it lowers to
    // `select(A, B, 0)`), which sits in the innermost loop of the
    // bit-at-a-time kernels in huffman / crc / crypto; the generic form spends
    // an `INeg` on each one.
    if (bc && bc->v == 0 && isBool01(av)) {
      L.vreg[i] = L.op2(MOp::IAnd, L.forceReg(av, line), c, line);
      return;
    }
    if (ac && ac->v == 0 && isBool01(bv)) {
      int32_t nc = L.opI(MOp::IXorI, c, 1, line);
      L.vreg[i] = L.op2(MOp::IAnd, L.forceReg(bv, line), nc, line);
      return;
    }
    if (bc && bc->v == 0) {
      int32_t mask = L.op1(MOp::INeg, c, line);
      L.vreg[i] = L.op2(MOp::IAnd, L.forceReg(av, line), mask, line);
      return;
    }
    if (ac && ac->v == 0) {
      int32_t nmask = L.opI(MOp::IAddI, c, -1, line); // c-1 == 0 or -1
      L.vreg[i] = L.op2(MOp::IAnd, L.forceReg(bv, line), nmask, line);
      return;
    }
    // Something the mask tricks above cannot see: a value that is *known* to
    // be 0/1 but is not a constant.  Comparisons, `!` and other selects all
    // have that range, and that is exactly what the if-converter leaves
    // behind when it merges a boolean phi.
    auto is01 = [](Value *v) -> bool { return isBool01(v); };
    // `select(c, 1, y)` with both c and y 0/1 is just `c | y`, and its
    // mirror is `!c | y`.  The front end renders every short-circuited
    // `A || B` as this select (one arm constant-true), and the generic mask
    // sequence would spend four instructions building what one `or` does.
    // This is the same fold InstCombine performs on `select i1 %c, true, %y`,
    // and it is the shape at the top of huffman's bit-at-a-time loops.
    if (ac && ac->v == 1 && !bc && is01(bv)) {
      L.vreg[i] = L.op2(MOp::IOr, c, L.forceReg(bv, line), line);
      return;
    }
    if (bc && bc->v == 1 && !ac && is01(av)) {
      int32_t nc = L.opI(MOp::IXorI, c, 1, line);
      L.vreg[i] = L.op2(MOp::IOr, nc, L.forceReg(av, line), line);
      return;
    }
    // `select(c, y + z, y)` (and `y - z`) is the other canonical
    // if-conversion shape: `if (c) y += z;`.  Folding the add into the mask -
    // `y + (z & -c)` - drops an instruction *and* keeps the accumulation on
    // the critical path instead of round-tripping through xor.  Arithmetic
    // wraps in i32, so lifting the select inside the add is exact.
    if (auto *ai = dynamic_cast<Instruction *>(av)) {
      if ((ai->op == Op::Add || ai->op == Op::Sub) && ai->ops.size() == 2) {
        Value *y = ai->ops[0], *z = ai->ops[1];
        if (ai->op == Op::Add && y != bv) std::swap(y, z);
        if (y == bv) {
          int32_t mask = L.op1(MOp::INeg, c, line);
          int32_t m = L.op2(MOp::IAnd, L.forceReg(z, line), mask, line);
          L.vreg[i] = L.op2(ai->op == Op::Add ? MOp::IAdd : MOp::ISub,
                            L.forceReg(y, line), m, line);
          return;
        }
      }
    }
    // `select(lo < hi, hi, lo)` and its mirrors are smax/smin.  RISC-V has no
    // min/max instruction, but the sign of the difference *is* the mask:
    //   d = lo - hi;  m = d >> 31
    //   smax = lo - (d & m)      smin = hi + (d & m)
    // One instruction cheaper than the generic `b ^ ((a ^ b) & -c)` (the `-c`
    // and the trailing `xor` fold into the subtraction) and, more importantly,
    // it shortens the chain from compare -> xor -> and -> xor to
    // sub -> shift -> and -> sub.  This is InstCombine's select-to-min/max
    // followed by the RISC-V smax/smin expansion, and it is the body of the
    // `max`/`min` chains in the h-4 / h-9 relaxation kernels.  Skipped when
    // either arm is a flag: the two `isBool01` paths above are shorter still.
    if (auto *ci = dynamic_cast<Instruction *>(cv)) {
      if (ci->op == Op::ICmp && ci->ops.size() == 2 && !is01(av) && !is01(bv)) {
        Value *u = ci->ops[0], *v = ci->ops[1];
        Value *lo = nullptr, *hi = nullptr;
        switch (ci->cond) {
        case Cond::Lt: // u < v
        case Cond::Le: // u <= v (equality returns either, so the mask is exact)
          lo = u;
          hi = v;
          break;
        case Cond::Gt: // u > v
        case Cond::Ge: // u >= v
          lo = v;
          hi = u;
          break;
        default:
          break;
        }
        const bool maxSide = lo && av == hi && bv == lo;
        const bool minSide = lo && av == lo && bv == hi;
        // `d = lo - hi` below has to reproduce the *true* difference: the mask
        // is read off its sign, and a wrapped i32 difference can flip that
        // sign (`INT32_MIN - 1` comes out positive).  Only take the short
        // form when the ranges prove the subtraction cannot overflow; a value
        // with no bound known is [INT32_MIN, INT32_MAX] and fails this test.
        Range lr = L.range.rangeOfValue(lo), hr = L.range.rangeOfValue(hi);
        const bool noOverflow = lr.lo - hr.hi >= (long long)INT32_MIN &&
                                lr.hi - hr.lo <= (long long)INT32_MAX;
        if ((maxSide || minSide) && noOverflow) {
          int32_t d = L.op2(MOp::ISub, L.forceReg(lo, line),
                            L.forceReg(hi, line), line);
          int32_t m = L.opI(MOp::SraI, d, 31, line);
          int32_t mand = L.op2(MOp::IAnd, d, m, line);
          L.vreg[i] = maxSide
                          ? L.op2(MOp::ISub, L.forceReg(lo, line), mand, line)
                          : L.op2(MOp::IAdd, L.forceReg(hi, line), mand, line);
          return;
        }
      }
    }
    int32_t a = L.forceReg(av, line);
    int32_t b = L.forceReg(bv, line);
    int32_t x = L.op2(MOp::IXor, a, b, line);
    // Merging two flags needs no mask either: `b ^ ((a^b) & c)` already picks
    // `a` when c == 1, and the `-c` the general form builds is `c` again when
    // c is a flag.
    int32_t m = isBool01(av) && isBool01(bv)
                    ? L.op2(MOp::IAnd, x, c, line)
                    : L.op2(MOp::IAnd, x, L.op1(MOp::INeg, c, line), line);
    L.vreg[i] = L.op2(MOp::IXor, m, b, line);
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
    throw CompileError("internal: unhandled opcode in ISel: '" +
                           opName(i->op) + "'",
                       line);
  }
}

void lowerTerminator(FnLower &L, BasicBlock *bb, size_t bi) {
  Instruction *t = bb->instrs.back().get();
  int line = t->line;

  // cf.phi lowering: a successor block may need the value this block feeds
  // into each of its phis.  The copy lands at the end of *this* block, so on
  // a loop back edge it is exactly the "phi copy" a register-allocated loop
  // counter needs.  Copies are emitted for every branch target; a copy whose
  // destination is never reached on the taken path only writes a dead vreg
  // (its phi block is only ever read through this or another predecessor's
  // copy), so this is safe even without edge splitting.
  //
  // The phis of one target execute SIMULTANEOUSLY (SSA merge semantics): when
  // they feed each other across the edge - the A=D;D=C;C=B;temp rotation of a
  // register-rotation loop - emitting them in declaration order reads a vreg
  // that an earlier copy has already overwritten.  They are therefore
  // scheduled like parallel copies: a copy is emitted only once its
  // destination is no longer read by a still-pending copy, and a remaining
  // pure cycle is broken through a fresh scratch vreg that captures the
  // pre-copy value its readers expect.
  auto emitPhiCopies = [&](BasicBlock *target) {
    auto it = L.blockPhis.find(target);
    if (it == L.blockPhis.end()) return;
    struct Copy {
      int32_t dst = -1, src = -1;
      bool isConst = false, isF = false;
      int32_t imm = 0;
    };
    std::vector<Copy> P;
    P.reserve(it->second.size());
    for (Instruction *ph : it->second) {
      Value *val = nullptr;
      for (size_t k = 0; k + 1 < ph->ops.size(); k += 2)
        if (ph->ops[k] == (Value *)bb) {
          val = ph->ops[k + 1];
          break;
        }
      if (!val) continue;
      int32_t dst = L.phiRegs.at(ph);
      bool isF = ph->ty == Type::F32;
      Copy c;
      c.dst = dst;
      c.isF = isF;
      if (ph->ty == Type::Ptr) {
        // A pointer phi - created by induction-variable strength reduction,
        // which carries an address around the loop.  Addresses live in the
        // pointer map (a `gep` never defines a plain value register), so the
        // incoming value is materialised into a register here.
        c.src = L.materializePtr(L.ptrOf(val), line);
        if (c.src == dst) continue;
      } else if (auto *ci = dynamic_cast<ConstantInt *>(val)) {
        c.isConst = true;
        c.imm = ci->v;
      } else if (auto *cf = dynamic_cast<ConstantFloat *>(val)) {
        c.isConst = true;
        c.imm = (int32_t)cf->bits();
        c.isF = true;
      } else {
        c.src = L.forceReg(val, line);
        if (c.src == dst) continue; // self copy: nothing to do
      }
      P.push_back(c);
    }
    auto emitCopy = [&](const Copy &c) {
      MInst m;
      m.line = line;
      if (c.isConst) {
        m.op = c.isF ? MOp::LiF : MOp::Li;
        m.dst = c.dst;
        m.imm = c.imm;
        m.ty = c.isF ? Type::F32 : Type::I32;
      } else {
        m.op = c.isF ? MOp::MoveF : MOp::MoveX;
        m.dst = c.dst;
        m.a = c.src;
        m.ty = c.isF ? Type::F32 : Type::I32;
      }
      L.emit(m);
    };
    std::vector<char> done(P.size(), 0);
    size_t remaining = P.size();
    auto dstReadLater = [&](int32_t d) {
      for (size_t j = 0; j < P.size(); ++j)
        if (!done[j] && !P[j].isConst && P[j].src == d) return true;
      return false;
    };
    while (remaining > 0) {
      size_t pick = P.size();
      for (size_t i = 0; i < P.size(); ++i) {
        if (done[i]) continue;
        if (!dstReadLater(P[i].dst)) {
          pick = i;
          break;
        }
      }
      if (pick != P.size()) { // leaf: safe to emit now
        emitCopy(P[pick]);
        done[pick] = 1;
        --remaining;
        continue;
      }
      // every remaining destination is read by a remaining copy: pure cycles
      // (e.g. an A<->B swap).  Preserve one destination's pre-copy value in a
      // fresh scratch, redirect its readers, then overwrite it.
      size_t i = 0;
      while (done[i]) ++i;
      Copy &c = P[i];
      int32_t tmp = L.newReg();
      {
        MInst m;
        m.op = c.isF ? MOp::MoveF : MOp::MoveX;
        m.dst = tmp;
        m.a = c.dst; // the vreg's current (pre-copy) value
        m.ty = c.isF ? Type::F32 : Type::I32;
        m.line = line;
        L.emit(m);
      }
      for (size_t j = 0; j < P.size(); ++j)
        if (!done[j] && j != i && P[j].src == c.dst) P[j].src = tmp;
      emitCopy(c);
      done[i] = 1;
      --remaining;
    }
  };

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
    auto *target = dynamic_cast<BasicBlock *>(t->ops[0]);
    emitPhiCopies(target);
    jmp(target);
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
  // incoming phi values for both arms (see the note above: emitted before the
  // conditional branch, which is safe without edge splitting)
  emitPhiCopies(thenB);
  if (elseB != thenB) emitPhiCopies(elseB);
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
// replaced by a single conditional branch.  Phi copies emitted between the
// compare and the branch (mem2reg) are transparent: we scan back across pure
// move/load-immediate instructions that do not feed the branch operand.
//
// The fusion rebinds the branch to read the compare's *operands* at the
// branch's own position, which is after those skipped copies.  A copy that
// writes one of them would therefore change the value the branch tests, so
// the fold is only taken when no skipped instruction defines `cmp.a`/`cmp.b`.
// Loop rotation produces exactly that shape: the rotated body ends with the
// loop test *and* the phi's back-edge copy (`mv rd, rs`), and if the test
// reads the phi register `rd` the pair must stay `cmp rd, ..; mv rd, rs; BrNz`
// - the compare reads the old value, the branch consumes its result.
void fuseBranchCompares(FnLower &L) {
  auto use = countUses(L);
  auto transparent = [](const MInst &m) {
    return m.op == MOp::MoveX || m.op == MOp::MoveF || m.op == MOp::Li ||
           m.op == MOp::LiF;
  };
  for (auto &b : L.mf.blocks) {
    auto &v = b.instrs;
    // True when a transparent instruction in [from, to) writes `ra` or `rb`.
    auto clobbersBetween = [&](size_t from, size_t to, int32_t ra, int32_t rb) {
      for (size_t k = from; k < to; ++k) {
        const MInst &t = v[k];
        if (!transparent(t)) continue;
        if (t.dst >= 0 && (t.dst == ra || t.dst == rb)) return true;
      }
      return false;
    };
    for (size_t i = 0; i < v.size();) {
      MInst &m = v[i];
      bool branchOnTrue = m.op == MOp::BrNz; // jump sym when cond != 0
      if (branchOnTrue || m.op == MOp::BrZ) {
        // scan back over transparent instructions to find the compare
        long long j = (long long)i - 1;
        while (j >= 0 && transparent(v[(size_t)j]) &&
               v[(size_t)j].dst != m.a && v[(size_t)j].a != m.a)
          --j;
        if (j >= 0 && v[(size_t)j].op == MOp::ICmp &&
            v[(size_t)j].dst == m.a && use[(size_t)m.a] == 1 &&
            !clobbersBetween((size_t)j + 1, i, v[(size_t)j].a,
                             v[(size_t)j].b)) {
          Cond c = (Cond)v[(size_t)j].imm;
          if (!branchOnTrue) c = invertCond(c);
          m.op = MOp::BrCmp;
          m.a = v[(size_t)j].a;
          m.b = v[(size_t)j].b;
          m.imm = (int32_t)c;
          v.erase(v.begin() + j);
          --i;
          continue;
        }
        // compare then xor 1 (logical not of a comparison)
        if (j >= 1 && v[(size_t)j].op == MOp::IXorI &&
            v[(size_t)j].imm == 1 && v[(size_t)j].dst == m.a &&
            use[(size_t)m.a] == 1 && v[(size_t)j - 1].op == MOp::ICmp &&
            v[(size_t)j - 1].dst == v[(size_t)j].a &&
            use[(size_t)v[(size_t)j].a] == 1 &&
            !clobbersBetween((size_t)j + 1, i, v[(size_t)j - 1].a,
                             v[(size_t)j - 1].b)) {
          Cond c = (Cond)v[(size_t)j - 1].imm;
          if (branchOnTrue) c = invertCond(c);
          m.op = MOp::BrCmp;
          m.a = v[(size_t)j - 1].a;
          m.b = v[(size_t)j - 1].b;
          m.imm = (int32_t)c;
          // remove the xor and the compare but keep any transparent phi
          // copies that sit between them and the branch.
          v.erase(v.begin() + j);      // xor
          v.erase(v.begin() + (j - 1)); // compare
          i -= 2;
          continue;
        }
      }
      ++i;
    }
  }
}

} // namespace

std::vector<MachineFunc> InstructionSelector::select(Module *mod) {
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
    L.range.run(fn);

    // Bind every cf.phi to a fresh virtual register up front: a predecessor
    // block can be emitted before the block that owns the phi (loop back
    // edges), so the copy source/destination vregs must exist by then.  The
    // phi itself produces no machine instruction - predecessors write its
    // vreg and the phi block reads it.
    for (auto &bb : fn->blocks)
      for (auto &iu : bb->instrs)
        if (iu->op == Op::Phi) {
          L.phiRegs[iu.get()] = L.newReg();
          L.blockPhis[bb.get()].push_back(iu.get());
        }

    for (size_t bi = 0; bi < fn->blocks.size(); ++bi) {
      BasicBlock *bb = fn->blocks[bi].get();
      MBlock mb;
      mb.name = L.labelFor(bb);
      L.mf.blocks.push_back(std::move(mb));
      L.cur = &L.mf.blocks.back();
      L.curBB = bb;
      L.blkConst.clear(); // constants are only reusable inside one block
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

    // Export the range analysis' non-negativity facts to the machine-level
    // strength reducer: for a non-negative dividend a constant division is an
    // *unsigned* division, and RISC-V's `mulhu` computes it in two instructions
    // where the signed magic needs six (see MachineOpt's magicu).
    //
    // A register with several definitions is a phi: ISel writes it from a copy
    // in every predecessor, so `nonNeg` may only be attached when the phi
    // itself is proven non-negative - the same condition the analysis used to
    // accept every incoming value, which is exactly what those copies carry.
    {
      std::unordered_map<int32_t, int> defs;
      for (auto &b : L.mf.blocks)
        for (auto &m : b.instrs)
          if (m.dst >= 0) ++defs[m.dst];
      auto regOf = [&](Value *v) -> int32_t {
        auto it = L.vreg.find(v);
        if (it != L.vreg.end()) return it->second;
        auto p = L.phiRegs.find(v);
        return p != L.phiRegs.end() ? p->second : -1;
      };
      auto mark = [&](Value *v) {
        if (!L.isNonNeg(v)) return;
        int32_t r = regOf(v);
        if (r < 0) return;
        bool isPhi = L.phiRegs.find(v) != L.phiRegs.end();
        if (defs[r] == 1 || isPhi) L.mf.nonNeg.insert(r);
      };
      for (auto &bb : fn->blocks)
        for (auto &iu : bb->instrs) mark(iu.get());
      for (auto &a : fn->args) mark(a.get());
    }

    out.push_back(std::move(L.mf));
  }
  return out;
}

} // namespace backend
} // namespace sakura
