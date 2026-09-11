// ---------------------------------------------------------------------------
// OptInline.cpp: general multi-block function inlining (inline-general).
//
// inline-small only splices single-block straight-line leaves.  This pass
// inlines the *general* case at the flat cf layer: a non-recursive function
// with branches, several return blocks and (before mem2reg) a memory-resident
// body is cloned into the caller at the call site:
//
//   * the caller block is split at the call: the prefix keeps running into
//     the cloned entry, and a fresh `after` block receives everything that
//     followed the call;
//   * every clone of a callee return block ends with a jump to `after`;
//   * when the callee returns a value through more than one return block the
//     several flows are merged by a cf.phi at the head of `after` (the same
//     op mem2reg introduces, and the same way the reference project merges
//     multi-return inlining);
//   * the call's uses are redirected to the merge value and the call is
//     erased.
//
// This is the reference FunctionInliningPass's multi-block path, restricted
// to what the SakuraCompiler mid-end needs: it runs right after tail-rec
// elimination so a recursion reduced to a flat loop can still be inlined into
// its callers, and before mem2reg so the cloned scalar slots are promoted in
// the caller like any other local.  Whole-function growth is bounded per
// caller (call sites and cloned-instruction budgets).
// ---------------------------------------------------------------------------
#include "Opt.h"

#include <unordered_map>
#include <unordered_set>

#include "OptUtil.h"

namespace sakura {
namespace ir {

namespace {

constexpr int kMaxCalleeInstrs = 64; // cloned instructions per call
constexpr int kMaxCallerGrowth = 400; // total cloned instrs / caller, lifetime
// A callee reached from at most this many call sites is almost always worth
// inlining regardless of size: the clone removes the call overhead on a hot
// path, and - more importantly for array kernels - it replaces a pointer
// *parameter* with whatever the caller actually passed.  When the argument is
// a distinct global array that turns a `may-alias` into a `no-alias`, which is
// what lets the invariant loads inside the loop be hoisted.  LLVM's inliner
// has the same effect through its LastCallToStaticBonus.  Measured on
// 01_mm3: inlining `mm` into `main` is worth ~8x there, and the same shape
// (small helper called a handful of times) recurs across the suite.
constexpr int kRareCallSites = 2;
constexpr int kMaxCalleeInstrsRare = 320;
constexpr int kMaxCallerGrowthRare = 1400;

class InlineGeneralPass final : public Pass {
public:
  const char *name() const override { return "inline-general"; }
  Layer inLayer() const override { return Layer::Cf; }

  bool run(Module &mod) override {
    computeCalleeClosure(mod);
    countCallSites(mod);
    bigInline_ = std::getenv("SAKU_NO_BIGINLINE") == nullptr;
    bool any = false;
    bool grow = true;
    while (grow) { // an inline can expose call sites worth another round
      grow = false;
      for (auto &f : mod.functions()) {
        if (f->isLib || f->blocks.empty()) continue;
        if (process(mod, *f)) {
          any = true;
          grow = true;
        }
      }
    }
    return any;
  }

private:
  // Functions a callee can reach through zero or more calls (only flat cf
  // calls matter; region op bodies were refused above).  If the caller is
  // reachable from the callee the two sit on a call cycle: inlining the
  // callee's clone keeps one live call to it (the recursion inside), so the
  // site would stay eligible forever and the caller would grow without bound
  // across the fixpoint rounds.  Real recursion plus mutual recursion both
  // die here.
  std::unordered_map<Function *, std::unordered_set<Function *>> reach_;

  void computeCalleeClosure(Module &mod) {
    std::unordered_map<Function *, std::vector<Function *>> callees;
    for (auto &f : mod.functions()) {
      if (f->isLib) continue;
      for (auto &bb : f->blocks)
        for (auto &iu : bb->instrs) {
          if (iu->op != Op::Call) continue;
          auto *t = dynamic_cast<Function *>(iu->ops[0]);
          if (t && !t->isLib) callees[f.get()].push_back(t);
        }
    }
    for (auto &kv : callees) {
      // seed: direct callees
      reach_[kv.first].insert(kv.second.begin(), kv.second.end());
    }
    // transitive fixpoint (graphs here are tiny)
    bool ch = true;
    while (ch) {
      ch = false;
      for (auto &f : mod.functions()) {
        if (f->isLib) continue;
        auto &r = reach_[f.get()];
        std::vector<Function *> extra;
        for (Function *t : r) {
          auto it2 = reach_.find(t);
          if (it2 == reach_.end()) continue; // leaf callee: no outgoing calls
          for (Function *u : it2->second)
            if (!r.count(u)) extra.push_back(u);
        }
        if (!extra.empty()) {
          r.insert(extra.begin(), extra.end());
          ch = true;
        }
      }
    }
  }

  // How many call sites (across the whole module) target each callee.  Used
  // for the "rarely called => inline regardless of size" budget below.
  std::unordered_map<Function *, int> callSites_;
  bool bigInline_ = true;

  void countCallSites(Module &mod) {
    callSites_.clear();
    for (auto &f : mod.functions()) {
      if (f->isLib) continue;
      for (auto &bb : f->blocks)
        for (auto &iu : bb->instrs) {
          if (iu->op != Op::Call) continue;
          if (auto *t = dynamic_cast<Function *>(iu->ops[0]))
            if (!t->isLib) ++callSites_[t];
        }
    }
  }

  bool hasRegion(Function &f) const {
    for (auto &bb : f.blocks)
      for (auto &iu : bb->instrs)
        if (iu->op == Op::ScfWhile || iu->op == Op::AffineFor) return true;
    return false;
  }

  // Inlined instructions accumulated into one caller over the whole pass; a
  // lifetime cap (not a per-process one) is what keeps the outer fixpoint
  // from growing a caller forever when a clone keeps offering the same
  // call site back.
  mutable std::unordered_map<Function *, int> grownBy_;
  int calleeCost(Function *callee) const {
    int n = 0;
    for (auto &bb : callee->blocks) n += (int)bb->instrs.size();
    return n;
  }

  bool eligible(Module &mod, Function *callee, Function *caller,
                Instruction *call) const {
    if (!callee || callee->isLib) return false;
    if (callee == caller) return false; // direct recursion
    if (callee->name == "main") return false;
    if (callee->blocks.empty()) return false;
    // A callee that can reach itself (direct or mutual recursion that
    // survived tail-rec elimination) must stay out of the clone pool: the
    // clone would keep a live recursive call inside it, so the site stays
    // eligible across the fixpoint rounds, the caller keeps growing, and the
    // resulting huge function strains the back-end (register pressure /
    // spill correctness).  The reference inliner refuses recursive callees
    // the same way.
    {
      auto it = reach_.find(callee);
      if (it != reach_.end() && it->second.count(callee)) return false;
    }
    // The clone is a straight splice of flat blocks; refuse anything that
    // would need region fix-ups (none should survive at the cf layer, kept
    // defensive).  Phis in the *caller* are fine: splitting a block only
    // re-keys the split block's phi entries to the new `after` block (see
    // doInline), so a caller that already carries an inline-merge phi can be
    // inlined into again.
    if (hasRegion(*callee) || hasRegion(*caller)) return false;
    // A callee with only one or two call sites gets a much larger budget: the
    // clone is not duplicated anywhere, so the size cost is paid once and the
    // argument-specialisation win applies at every one of those sites.
    auto csIt = callSites_.find(callee);
    const int nSites = (csIt == callSites_.end()) ? 0 : csIt->second;
    const bool rare = bigInline_ && nSites > 0 && nSites <= kRareCallSites;
    const int calleeCap = rare ? kMaxCalleeInstrsRare : kMaxCalleeInstrs;
    const int callerCap = rare ? kMaxCallerGrowthRare : kMaxCallerGrowth;
    if (calleeCost(callee) > calleeCap) return false;
    if (call->ops.size() - 1 != callee->args.size()) return false;
    if (callee->ret != call->ty) return false; // incl. void matching
    // A non-void callee must return on some path; void callees must have a
    // return so the inlined body can reach the `after` block.
    bool anyRet = false;
    for (auto &bb : callee->blocks)
      for (auto &iu : bb->instrs)
        if (iu->op == Op::Ret) anyRet = true;
    if (!anyRet) return false;
    // call-cycle guard + lifetime budget (see comments above)
    auto it = reach_.find(callee);
    if (it != reach_.end() && it->second.count(caller)) return false;
    if (grownBy_[caller] + calleeCost(callee) > callerCap) return false;
    return true;
  }

  bool process(Module &mod, Function &caller) {
    bool any = false;
    for (;;) {
      bool found = false;
      for (auto &bb : caller.blocks) {
        for (size_t i = 0; i < bb->instrs.size(); ++i) {
          Instruction *inst = bb->instrs[i].get();
          if (inst->op != Op::Call) continue;
          auto *callee = dynamic_cast<Function *>(inst->ops[0]);
          if (!callee || !eligible(mod, callee, &caller, inst)) continue;
          doInline(mod, caller, bb.get(), i, callee);
          grownBy_[&caller] += calleeCost(callee);
          any = true;
          found = true;
          break; // block list changed; rescan the function
        }
        if (found) break;
      }
      if (!found) break;
    }
    return any;
  }

  // Per-function suffix counters.  Block names become assembly labels, so
  // they must stay unique inside a function across every inlining round (the
  // outer run() loop revisits functions several times).
  std::unordered_map<Function *, int> freshByFn_;
  int nextFresh(Function &caller) { return freshByFn_[&caller]++; }

  // resolve a callee SSA value to its caller-context value
  struct CloneCtx {
    std::unordered_map<BasicBlock *, BasicBlock *> bb;
    std::unordered_map<Instruction *, Instruction *> inst;
    std::vector<Value *> callArgs; // caller values for callee args
  };

  Value *resolve(Value *v, const CloneCtx &cx) const {
    if (auto *arg = dynamic_cast<Argument *>(v)) {
      if (arg->index < (int)cx.callArgs.size())
        return cx.callArgs[(size_t)arg->index];
      return v;
    }
    if (auto *i = dynamic_cast<Instruction *>(v)) {
      auto it = cx.inst.find(i);
      return it != cx.inst.end() ? (Value *)it->second : v;
    }
    if (auto *b = dynamic_cast<BasicBlock *>(v)) {
      auto it = cx.bb.find(b);
      return it != cx.bb.end() ? (Value *)it->second : v;
    }
    return v;
  }

  void doInline(Module &mod, Function &caller, BasicBlock *bb, size_t ci,
                Function *callee) {
    std::string suffix = "_inl" + std::to_string(nextFresh(caller));
    Instruction *call = bb->instrs[ci].get();
    CloneCtx cx;
    cx.callArgs.assign(call->ops.begin() + 1, call->ops.end());

    // 1) clone every callee block; 2) clone every non-Ret instruction (SSA
    // defs and branch targets resolved in a second pass once the whole map
    // exists: clones may refer forward across blocks).
    std::vector<std::unique_ptr<BasicBlock>> clones;
    for (auto &oldBB : callee->blocks) {
      auto nb = std::make_unique<BasicBlock>(oldBB->name + suffix);
      BasicBlock *nbP = nb.get();
      cx.bb[oldBB.get()] = nbP;
      clones.push_back(std::move(nb));
      for (auto &iu : oldBB->instrs) {
        Instruction *src = iu.get();
        if (src->op == Op::Ret) continue; // return sites rewritten below
        auto c = std::make_unique<Instruction>(src->op, src->ty);
        c->cond = src->cond;
        c->elem = src->elem;
        c->n = src->n;
        c->step = src->step;
        c->shape = src->shape;
        c->line = src->line;
        c->ops = src->ops; // raw operand copies; remapped by the pass below
        Instruction *cp = c.get();
        cx.inst[src] = cp;
        nbP->instrs.push_back(std::move(c));
      }
    }
    // resolve operands
    for (auto &nb : clones)
      for (auto &iu : nb->instrs)
        for (Value *&op : iu->ops) op = resolve(op, cx);

    // 3) the `after` block: receives every return jump and the caller's tail
    auto after = std::make_unique<BasicBlock>(bb->name + suffix + "_after");
    BasicBlock *afterP = after.get();

    // 4) turn cloned return blocks into jumps to `after`, collecting the
    //    returned values (mapped) for the merge.
    struct RetSite {
      BasicBlock *from = nullptr;
      Value *val = nullptr;
    };
    std::vector<RetSite> retVals;
    {
      size_t bi = 0;
      for (auto &oldBB : callee->blocks) {
        BasicBlock *clone = clones[bi++].get();
        // find the cloned Ret in this block (if any)
        for (auto &iu : oldBB->instrs) {
          Instruction *src = iu.get();
          if (src->op != Op::Ret) continue;
          auto br = std::make_unique<Instruction>(Op::Br, Type::Void);
          br->cond = Cond::Eq;
          br->ops = {(Value *)afterP};
          clone->instrs.push_back(std::move(br));
          if (src->ops.empty() || call->ty == Type::Void) continue;
          retVals.push_back({clone, resolve(src->ops[0], cx)});
        }
      }
    }

    // 5) merge value: 0 / 1 / N return flows
    Value *mergeVal = nullptr;
    if (call->hasResult()) {
      if (retVals.size() == 1) {
        mergeVal = retVals[0].val;
      } else if (retVals.size() > 1) {
        auto ph = std::make_unique<Instruction>(Op::Phi, call->ty);
        for (const RetSite &rs : retVals) {
          ph->ops.push_back((Value *)rs.from);
          ph->ops.push_back(rs.val);
        }
        mergeVal = ph.get();
        afterP->instrs.insert(afterP->instrs.begin(), std::move(ph));
      }
      // retVals.empty(): callee never returns on the inlined path; callers
      // that read the (poisoned) result were refused by `anyRet` above only
      // for non-void, so guard: nothing to replace
    }
    if (mergeVal) replaceAllUses(mod, call, mergeVal);

    // 6) split the caller block at the call:
    //      prefix ... br entryClone        (bb)
    //      entryClone ... returns -> after
    //      after:  [phi] <tail instructions>
    bb->instrs.erase(bb->instrs.begin() + (long)ci); // drop the call
    while (bb->instrs.size() > ci) {
      afterP->instrs.push_back(std::move(bb->instrs[ci]));
      bb->instrs.erase(bb->instrs.begin() + (long)ci);
    }
    auto entryBr = std::make_unique<Instruction>(Op::Br, Type::Void);
    entryBr->cond = Cond::Eq;
    entryBr->ops = {(Value *)cx.bb[callee->entry()]};
    bb->instrs.push_back(std::move(entryBr));

    // 6b) bb's tail (with its terminator) now lives in `after`, so any phi
    // whose incoming edge was keyed on bb must be re-keyed to `after`: the
    // value that used to flow out of bb's end now flows out of after's end.
    // (This is what lets a caller that already carries an inline-merge phi be
    // split again for a later call.)
    for (auto &x : caller.blocks)
      for (auto &iu : x->instrs) {
        Instruction *phi = iu.get();
        if (phi->op != Op::Phi) continue;
        for (size_t s = 0; s + 1 < phi->ops.size(); s += 2)
          if (phi->ops[s] == (Value *)bb) phi->ops[s] = (Value *)afterP;
      }

    // 7) splice the clone blocks + `after` right after the caller block.
    size_t pos = 0;
    while (pos < caller.blocks.size() && caller.blocks[pos].get() != bb) ++pos;
    ++pos; // after bb
    for (auto &c : clones) {
      caller.blocks.insert(caller.blocks.begin() + (long)pos++,
                           std::move(c));
    }
    caller.blocks.insert(caller.blocks.begin() + (long)pos, std::move(after));
  }
};

} // namespace

std::unique_ptr<Pass> makeInlineGeneralPass() {
  return std::make_unique<InlineGeneralPass>();
}

} // namespace ir
} // namespace sakura
