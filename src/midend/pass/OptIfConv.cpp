// ---------------------------------------------------------------------------
// OptIfConv.cpp: branchless if-conversion (the port of SimplifyCFG's
// two-entry-PHI select formation plus LLVM's branchless `select` lowering
// strategy).
//
// The bit-twiddling kernels (crc*, crypto*, huffman*, gzip*) are written as
// dense if-chains over one or two bits, and after mem2reg each `if` is a clean
// two-entry diamond:
//
//       S:  %c = arith.cmpi ...        ; the condition
//           cf.cond_br %c, ^A, ^B
//       ^A: (pure ops ending in) cf.br ^M
//       ^B: (pure ops ending in) cf.br ^M
//       ^M: %p = cf.phi [^A, %va], [^B, %vb]     (the merge value)
//           ...
//
// LLVM turns that into `%p = select %c, %va, %vb` in S, deletes the two arms
// and lets the back-end emit a mask sequence instead of a branch.  RISC-V has
// no conditional move, so the selector expands to a few ALU ops, which is far
// cheaper than the mispredicting branch + the register copies the phi would
// otherwise need.  This is what closes the remaining gap on crc1 / crypto-1 /
// huffman-01.
//
// Only *speculatable* integer ops are hoisted out of the arms: arithmetic
// without division (x/0 and INT_MIN/-1 trap), comparisons, `select` and `gep`.
// Loads (may fault), stores/calls (side effects), floats (no FPR<->XPR
// move-back) and phis in the arms are all rejected, as is any diamond whose
// merged values are not available at the condition block.
//
// The transform is repeated to a fixed point: removing one diamond merges the
// two blocks (`cfg-simplify` afterwards), which usually exposes the next `if`
// of the chain as another two-entry diamond.
// ---------------------------------------------------------------------------
#include "Opt.h"

#include <algorithm>
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "OptUtil.h"

namespace sakura {
namespace ir {
namespace {

// Ops that are safe to execute unconditionally.  Division/remainder are
// deliberately absent (a speculated x/0 or INT_MIN/-1 traps), as are loads.
bool speculatable(Op o) {
  switch (o) {
  case Op::Add: case Op::Sub: case Op::Mul:
  case Op::ICmp: case Op::Not: case Op::Select: case Op::Gep:
    return true;
  default:
    return false;
  }
}

constexpr int kMaxSpeculated = 8; // instructions hoisted out of both arms
constexpr int kMaxRounds = 64;    // one diamond per round, per function

// A merge phi and the two values it selected between.
struct Plan {
  Instruction *phi;
  Value *vt;
  Value *vf;
};

struct Cfg {
  std::unordered_map<BasicBlock *, int> idx;
  std::vector<BasicBlock *> blk;
  std::vector<std::vector<int>> preds, succs;
  int n = 0;
};

class IfConvPass final : public Pass {
public:
  const char *name() const override { return "if-convert"; }
  Layer inLayer() const override { return Layer::Cf; }

  bool run(Module &mod) override {
    // A/B switch: the win is measured separately per benchmark.
    if (std::getenv("SAKU_NO_IFCONV")) return false;
    bool any = false;
    for (auto &f : mod.functions()) {
      if (f->isLib || f->blocks.empty()) continue;
      for (int r = 0; r < kMaxRounds; ++r)
        if (!processFunction(mod, *f)) break;
        else any = true;
    }
    return any;
  }

private:
  // ---------------------------------------------------------------- CFG ----
  static void addEdge(Cfg &c, int a, int b) {
    for (int s : c.succs[(size_t)a])
      if (s == b) return;
    c.succs[(size_t)a].push_back(b);
    c.preds[(size_t)b].push_back(a);
  }

  static void buildCfg(Function &f, Cfg &c) {
    c.n = (int)f.blocks.size();
    c.blk.clear();
    c.idx.clear();
    for (int i = 0; i < c.n; ++i) {
      c.idx[f.blocks[(size_t)i].get()] = i;
      c.blk.push_back(f.blocks[(size_t)i].get());
    }
    c.preds.assign((size_t)c.n, {});
    c.succs.assign((size_t)c.n, {});
    for (int i = 0; i < c.n; ++i) {
      auto &v = c.blk[(size_t)i]->instrs;
      if (v.empty()) continue;
      Instruction *t = v.back().get();
      if (t->op == Op::Br && !t->ops.empty()) {
        addEdge(c, i, c.idx.at((BasicBlock *)t->ops[0]));
      } else if (t->op == Op::CondBr && t->ops.size() >= 3) {
        addEdge(c, i, c.idx.at((BasicBlock *)t->ops[1]));
        addEdge(c, i, c.idx.at((BasicBlock *)t->ops[2]));
      }
    }
  }

  static int singleSucc(const Cfg &c, int b) {
    if (c.succs[(size_t)b].size() != 1) return -1;
    return c.succs[(size_t)b][0];
  }

  // Does block `x` dominate block `t`?  Answered by deleting `x` and asking
  // whether `t` is still reachable from the entry: obviously correct, and only
  // asked for the handful of values that feed a diamond being rewritten.
  static bool dominates(const Cfg &c, int x, int t) {
    if (x == t) return true;
    if (t == 0) return x == 0; // nothing but the entry dominates the entry
    if (x == 0) return true;   // the entry dominates everything reachable
    std::vector<char> seen((size_t)c.n, 0);
    std::vector<int> stk{0};
    seen[0] = 1;
    while (!stk.empty()) {
      int b = stk.back();
      stk.pop_back();
      for (int s : c.succs[(size_t)b]) {
        if (s == x || seen[(size_t)s]) continue;
        if (s == t) return false; // reached without x
        seen[(size_t)s] = 1;
        stk.push_back(s);
      }
    }
    return true;
  }

  // A value may be read at the condition block S when it is a constant / a
  // named object, is defined in S, is defined in one of the hoisted arm blocks
  // (it will move along), or is defined in a block that dominates S.
  //
  // Values defined in the merge block M are rejected: M's body is spliced in
  // *after* the selects, so reading one would both be a same-block use before
  // def and, for the phis, a read of an instruction about to be deleted.
  static bool availableAt(
      Value *v, const Cfg &c, int sIdx, int mIdx,
      const std::unordered_map<Instruction *, int> &owner,
      const std::unordered_set<int> &arms) {
    if (isStructural(v) || asConstInt(v)) return true;
    if (asConstFloat(v)) return false; // no float speculation here
    auto *i = dynamic_cast<Instruction *>(v);
    if (!i) return false;
    auto it = owner.find(i);
    if (it == owner.end()) return false;
    int d = it->second;
    if (d == mIdx) return false;
    if (d == sIdx || arms.count(d)) return true;
    return dominates(c, d, sIdx);
  }

  // Every non-terminator instruction of an arm must be safe to hoist into S.
  static bool armHoistable(const Cfg &c, int b,
                           const std::unordered_map<Instruction *, int> &owner,
                           int sIdx, int mIdx,
                           const std::unordered_set<int> &arms, int &cost) {
    auto &v = c.blk[(size_t)b]->instrs;
    for (size_t k = 0; k + 1 < v.size(); ++k) {
      Instruction *i = v[k].get();
      if (!speculatable(i->op)) return false;
      if (i->ty != Type::I32 && i->ty != Type::Ptr) return false;
      if (i->op == Op::Phi) return false;
      ++cost;
      if (cost > kMaxSpeculated) return false;
      for (Value *o : i->ops)
        if (!availableAt(o, c, sIdx, mIdx, owner, arms)) return false;
    }
    return true;
  }

  // ------------------------------------------------------------ transform --
  bool processFunction(Module &mod, Function &f) {
    Cfg c;
    buildCfg(f, c);
    if (c.n < 3) return false;

    std::unordered_map<Instruction *, int> owner;
    for (int b = 0; b < c.n; ++b)
      for (auto &iu : c.blk[(size_t)b]->instrs) owner[iu.get()] = b;

    for (int si = 0; si < c.n; ++si) {
      BasicBlock *S = c.blk[(size_t)si];
      if (S->instrs.empty()) continue;
      Instruction *term = S->instrs.back().get();
      if (term->op != Op::CondBr || term->ops.size() < 3) continue;
      Value *cond = term->ops[0];
      int ai = c.idx.at((BasicBlock *)term->ops[1]);
      int bi = c.idx.at((BasicBlock *)term->ops[2]);
      if (ai == bi) continue;

      // Locate the merge M.  One side may *be* M (the block the other arm
      // falls/joins into); that is the shape `if (c) { ... } rest;` produces.
      int mi = -1;
      if (singleSucc(c, ai) >= 0 && singleSucc(c, ai) == singleSucc(c, bi) &&
          singleSucc(c, ai) != si) {
        mi = singleSucc(c, ai);
      } else if (singleSucc(c, bi) == ai && ai != si) {
        mi = ai;
      } else if (singleSucc(c, ai) == bi && bi != si) {
        mi = bi;
      } else {
        continue;
      }
      if (mi == si) continue;

      // M is entered exactly from the two sides of this branch.
      if (c.preds[(size_t)mi].size() != 2) continue;
      const int predT = (mi == ai) ? si : ai; // predecessor carrying the true side
      const int predF = (mi == bi) ? si : bi;

      std::unordered_set<int> arms;
      if (mi != ai) {
        if (c.preds[(size_t)ai].size() != 1 || c.preds[(size_t)ai][0] != si)
          continue;
        arms.insert(ai);
      }
      if (mi != bi) {
        if (c.preds[(size_t)bi].size() != 1 || c.preds[(size_t)bi][0] != si)
          continue;
        arms.insert(bi);
      }

      int cost = 0;
      bool hoistable = true;
      for (int a : arms)
        if (!armHoistable(c, a, owner, si, mi, arms, cost)) hoistable = false;
      if (!hoistable) continue;

      // Collect the merge phis and check their two incoming values.
      BasicBlock *M = c.blk[(size_t)mi];
      std::vector<Plan> plans;
      bool ok = true;
      int phis = 0;
      for (auto &iu : M->instrs) {
        Instruction *ph = iu.get();
        if (ph->op != Op::Phi) continue;
        ++phis;
        if (ph->ty != Type::I32 || ph->ops.size() != 4) { ok = false; break; }
        Value *vt = nullptr, *vf = nullptr;
        for (size_t k = 0; k + 1 < ph->ops.size(); k += 2) {
          auto *pb = dynamic_cast<BasicBlock *>(ph->ops[k]);
          if (!pb) { ok = false; break; }
          int p = c.idx.at(pb);
          if (p == predT) vt = ph->ops[k + 1];
          else if (p == predF) vf = ph->ops[k + 1];
          else { ok = false; break; }
        }
        if (!ok || !vt || !vf) { ok = false; break; }
        // The phi consumes both values itself, so drop their phi-only uses
        // before testing availability (a phi is not hoistable).
        if (!availableAt(vt, c, si, mi, owner, arms) ||
            !availableAt(vf, c, si, mi, owner, arms)) {
          ok = false;
          break;
        }
        plans.push_back({ph, vt, vf});
      }
      if (!ok || phis == 0 || plans.empty()) continue;

      apply(mod, f, c, si, mi, cond, arms, plans);
      return true; // CFG changed: rebuild and look again
    }
    return false;
  }

  void apply(Module &mod, Function &f, Cfg &c, int si, int mi, Value *cond,
             const std::unordered_set<int> &arms,
             const std::vector<Plan> &plans) {
    BasicBlock *S = c.blk[(size_t)si];
    BasicBlock *M = c.blk[(size_t)mi];

    // 1. create the selects (they reference the hoisted arm values).
    std::vector<std::unique_ptr<Instruction>> sels;
    for (const Plan &p : plans) {
      auto sel = std::make_unique<Instruction>(Op::Select, p.phi->ty);
      sel->line = p.phi->line;
      sel->ops = {cond, p.vt, p.vf};
      Instruction *raw = sel.get();
      sels.push_back(std::move(sel));
      replaceAllUses(mod, p.phi, raw);
    }

    // 2. move the arm bodies into S, right before its terminator.
    size_t at = S->instrs.size() - 1;
    std::vector<BasicBlock *> toErase;
    for (int a : arms) {
      BasicBlock *A = c.blk[(size_t)a];
      auto &v = A->instrs;
      for (size_t k = 0; k + 1 < v.size(); ++k)
        S->instrs.insert(S->instrs.begin() + (long)at++, std::move(v[k]));
      toErase.push_back(A);
    }
    for (auto &sel : sels)
      S->instrs.insert(S->instrs.begin() + (long)at++, std::move(sel));

    // 3. S now branches straight to the merge.
    auto br = std::make_unique<Instruction>(Op::Br, Type::Void);
    br->line = S->instrs.back()->line;
    br->ops = {M};
    S->instrs.back() = std::move(br);

    // 4. the merge phis have become selects; drop them.
    {
      auto &v = M->instrs;
      for (size_t k = 0; k < v.size();) {
        if (v[k]->op == Op::Phi) v.erase(v.begin() + (long)k);
        else ++k;
      }
    }

    // 5. delete the now-unreachable arm blocks.
    for (BasicBlock *A : toErase) {
      auto it = std::find_if(
          f.blocks.begin(), f.blocks.end(),
          [&](const std::unique_ptr<BasicBlock> &p) { return p.get() == A; });
      if (it != f.blocks.end()) f.blocks.erase(it);
    }

    // 6. M is now entered only from S, so S safely falls through into it.
    //    Merging the two (which is what cfg-simplify does) is what lets the
    //    *next* `if` of the chain be recognised as a fresh two-entry diamond
    //    on the following round: without it the pass would stop after one
    //    diamond per linear chain.
    {
      auto &sv = S->instrs;
      if (!sv.empty() && sv.back()->op == Op::Br && sv.back()->ops.size() == 1 &&
          sv.back()->ops[0] == (Value *)M && M != S) {
        // Any phi in a successor of M that is keyed on M must be re-keyed on S
        // (M is about to be destroyed), otherwise it would hold a dangling
        // BasicBlock*.
        for (int y : c.succs[(size_t)mi])
          for (auto &iu : c.blk[(size_t)y]->instrs)
            if (iu->op == Op::Phi)
              for (size_t k = 0; k + 1 < iu->ops.size(); k += 2)
                if (iu->ops[k] == (Value *)M) iu->ops[k] = S;
        // Drop the `br M` placeholder and splice M's body (terminator
        // included) in its place: S keeps its own identity, so the branch
        // target that used to name M now names the merged block.
        sv.pop_back();
        auto &mv = M->instrs;
        while (!mv.empty()) {
          sv.push_back(std::move(mv.front()));
          mv.erase(mv.begin());
        }
        auto it = std::find_if(
            f.blocks.begin(), f.blocks.end(),
            [&](const std::unique_ptr<BasicBlock> &p) { return p.get() == M; });
        if (it != f.blocks.end()) f.blocks.erase(it);
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> makeIfConvPass() { return std::make_unique<IfConvPass>(); }

} // namespace ir
} // namespace sakura
