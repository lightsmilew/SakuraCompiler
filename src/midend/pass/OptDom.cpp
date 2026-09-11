// ---------------------------------------------------------------------------
// OptDom.cpp: whole-CFG dominance-based CSE of pure expressions (dom-cse).
//
// mem-cse and the machine-level BlockLocalCSE only fold along straight-line
// / extended-basic-block chains: their local tables are reset at every join
// (loop header, if/else merge, ...).  An expression that is recomputed in a
// block dominated by an earlier identical computation therefore survives
// both passes -- e.g. the latch and the loop-header of a lowered loop both
// re-deriving the same address arithmetic from the same phi/SSA operands, or
// a dominated tail block repeating an expression its dominator already
// produced.
//
// This pass rebuilds the flat-cf dominator tree of each function and folds
// any pure expression (arith / cast / gep -- the same set mem-cse local CSE
// treats as expression ops) whose operands are dominated by an earlier
// identical instruction.  In SSA every value is defined exactly once, so a
// dominating computation of the same op/operands can never be invalidated;
// only the availability of a def across block boundaries has to be
// dominance-checked, which the dominator walk below does exactly.
//
// Register it at the cf layer after mem2reg, where scalar memory traffic has
// been promoted to phi values and the remaining redundancy is pure SSA.
// ---------------------------------------------------------------------------
#include "Opt.h"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <unordered_map>
#include <vector>

#include "OptUtil.h"

namespace sakura {
namespace ir {

namespace {

class DomCsePass final : public Pass {
public:
  explicit DomCsePass(Layer l) : layer_(l) {}
  const char *name() const override { return "dom-cse"; }
  Layer inLayer() const override { return layer_; }

  bool run(Module &mod) override {
    bool any = false;
    fx_.compute(mod);
    for (auto &f : mod.functions()) {
      if (f->isLib || f->blocks.size() < 2) continue;
      any |= processFunction(mod, *f);
    }
    return any;
  }

private:
  Layer layer_;
  Effects fx_;
  bool noCallCse_ = false;

  // Key for a value-like call: the callee plus the argument values.  Prefixed
  // so it can never collide with an `exprKey` (which starts with the opcode).
  static std::string callKey(Instruction *call) {
    char b[32];
    snprintf(b, sizeof b, "C:%p:", (void *)call->ops[0]);
    std::string k = b;
    for (size_t i = 1; i < call->ops.size(); ++i)
      k += valueKey(call->ops[i]) + ";";
    return k;
  }

  // Pure, CSE-able expression opcodes (mirrors mem-cse's expression set;
  // alloca allocates fresh storage and load may change between executions, so
  // both are excluded).
  static bool isCseOp(Op op) {
    switch (op) {
    case Op::Add: case Op::Sub: case Op::Mul: case Op::SDiv: case Op::SRem:
    case Op::FAdd: case Op::FSub: case Op::FMul: case Op::FDiv:
    case Op::ICmp: case Op::FCmp: case Op::Sitofp: case Op::Fptosi:
    case Op::Not: case Op::Gep: case Op::Select:
      return true;
    default:
      return false;
    }
  }

  static std::string valueKey(Value *v) {
    if (auto *ci = asConstInt(v)) {
      char b[24];
      snprintf(b, sizeof b, "i%08x", (uint32_t)ci->v);
      return b;
    }
    if (auto *cf = asConstFloat(v)) {
      char b[32];
      snprintf(b, sizeof b, "f%08x", cf->bits());
      return b;
    }
    char b[32];
    snprintf(b, sizeof b, "p%p", (void *)v);
    return b;
  }

  // Canonical identity key for an expression.  Commutative ops sort their two
  // operands so `a+b` and `b+a` fold together.
  static std::string exprKey(Op op, Cond cond,
                             const std::vector<Value *> &os) {
    std::string k =
        std::to_string((int)op) + ":" + std::to_string((int)cond) + ":";
    std::vector<std::string> parts;
    for (Value *o : os) {
      if (isStructural(o)) return std::string(); // never canonical
      parts.push_back(valueKey(o));
    }
    bool comm = op == Op::Add || op == Op::Mul || op == Op::FAdd ||
                op == Op::FMul ||
                (op == Op::FCmp && (cond == Cond::Eq || cond == Cond::Ne)) ||
                (op == Op::ICmp && (cond == Cond::Eq || cond == Cond::Ne));
    if (comm && parts.size() == 2 && parts[0] > parts[1])
      std::swap(parts[0], parts[1]);
    for (auto &p : parts) k += p + ";";
    return k;
  }

  bool processFunction(Module &mod, Function &f) {
    noCallCse_ = std::getenv("SAKU_NO_CALLCSE") != nullptr;
    const size_t n = f.blocks.size();
    std::unordered_map<BasicBlock *, size_t> idx;
    for (size_t i = 0; i < n; ++i) idx[f.blocks[i].get()] = i;

    // Successor / predecessor adjacency over the flat cf.
    std::vector<std::vector<size_t>> succ(n), pred(n);
    for (auto &bb : f.blocks) {
      size_t bi = idx[bb.get()];
      Instruction *last =
          bb->instrs.empty() ? nullptr : bb->instrs.back().get();
      if (!last) continue;
      auto bump = [&](Value *t) {
        if (auto *tb = dynamic_cast<BasicBlock *>(t))
          if (idx.count(tb)) {
            succ[bi].push_back(idx[tb]);
            pred[idx[tb]].push_back(bi);
          }
      };
      if (last->op == Op::Br) bump(last->ops[0]);
      if (last->op == Op::CondBr && last->ops.size() >= 3) {
        bump(last->ops[1]);
        bump(last->ops[2]);
      }
    }

    // Reverse post-order from the entry.
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

    // Dominator sets by iterative dataflow (small functions: good enough).
    // dom[b] starts as the universe (all nodes may dominate) for every block
    // except the entry; sweeping in RPO shrinks each set to {b} u intersect
    // of its predecessors' current sets until fixpoint.
    std::vector<std::vector<uint8_t>> dom(n, std::vector<uint8_t>(n, 1));
    for (size_t d = 1; d < n; ++d) dom[0][d] = 0; // entry: {entry} only
    bool changed = true;
    while (changed) {
      changed = false;
      for (size_t bi : rpo) {
        if (bi == 0) continue;
        if (pred[bi].empty()) continue; // unreachable: keep {bi}
        // dom(b) = {b} u intersect dom(p)
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

    // Immediate dominator: the member of dom(b)\{b} with the largest dom set
    // (the "closest" dominator).
    std::vector<size_t> idom(n, 0);
    for (size_t b = 0; b < n; ++b) {
      if (b == 0) continue;
      size_t best = n;
      size_t bestSize = 0;
      for (size_t d = 0; d < n; ++d)
        if (d != b && dom[b][d]) {
          size_t cnt = 0;
          for (size_t i = 0; i < n; ++i) cnt += dom[d][i] ? 1 : 0;
          if (cnt > bestSize) {
            bestSize = cnt;
            best = d;
          }
        }
      idom[b] = best;
    }

    // Availability of each expression at block entry: the map of the block's
    // immediate dominator (dominators are processed before their dominated
    // blocks in RPO, so the inherited map is final).  A def recorded while
    // walking a block overwrites any deeper entry for the same key, so the
    // map handed to dominated children holds the *closest* dominating def.
    //
    // availBlk tracks the block that owns each entry.  Folding a use onto a
    // dominating def is only valid here when the def is *also* earlier in
    // layout order: the back-end binds virtual registers by walking the block
    // list, so a def that dominates a use but sits later in the list (which
    // loop rotation can produce - the rotation guard keeps a branch to the
    // loop exit, so the exit's dominators are no longer all before it) would
    // have no register bound yet when the use is lowered.
    bool any = false;
    std::vector<std::unordered_map<std::string, Value *>> avail(n);
    std::vector<std::unordered_map<std::string, size_t>> availBlk(n);
    for (size_t bi : rpo) {
      if (bi != 0 && idom[bi] < n) {
        avail[bi] = avail[idom[bi]];
        availBlk[bi] = availBlk[idom[bi]];
      }
      BasicBlock &bb = *f.blocks[bi];
      auto &instrs = bb.instrs;
      size_t i = 0;
      while (i < instrs.size()) {
        Instruction *inst = instrs[i].get();
        if (isTerminator(inst->op)) break;
        // A call to a function that is a pure function of its arguments is as
        // CSE-able as an `add`: equal arguments cannot give a different
        // result, because everything it reads is immutable module data.  This
        // catches helpers called with the same arguments from sibling blocks,
        // which mem-cse's straight-line table cannot see.  Dominance is enough
        // of a guard here because the globals a value-like callee reads are
        // never written by any function in the module.
        std::string k;
        bool cseable = false;
        if (isCseOp(inst->op)) {
          k = exprKey(inst->op, inst->cond, inst->ops);
          cseable = !k.empty();
        } else if (inst->op == Op::Call && inst->ty != Type::Void &&
                   !inst->ops.empty() && !noCallCse_) {
          if (auto *callee = dynamic_cast<Function *>(inst->ops[0]))
            if (fx_.callIsValue(callee)) {
              k = callKey(inst);
              cseable = true;
            }
        }
        if (cseable) {
          auto it = avail[bi].find(k);
          if (it != avail[bi].end() && it->second != (Value *)inst &&
              availBlk[bi][k] <= bi) {
            replaceAllUses(mod, inst, it->second);
            instrs.erase(instrs.begin() + i);
            any = true;
            continue; // do not advance: next instruction shifted in
          }
          avail[bi][k] = inst;
          availBlk[bi][k] = bi;
        }
        ++i;
      }
    }
    return any;
  }
};

} // namespace

std::unique_ptr<Pass> makeDomCsePass(Layer l) {
  return std::make_unique<DomCsePass>(l);
}

} // namespace ir
} // namespace sakura
