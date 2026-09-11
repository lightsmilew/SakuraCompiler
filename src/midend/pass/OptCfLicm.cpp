// ---------------------------------------------------------------------------
// OptCfLicm.cpp: loop-invariant code motion on the *flat* cf layer (licm-cf).
//
// The `licm` pass in OptLoop.cpp works on the structured region ops
// (affine.for / scf.while), which only exist above the cf layer.  By the time
// the pipeline has flattened to basic blocks and run the inliner, a second
// class of invariance becomes visible that the structured pass could never
// see: after `inline-general` a callee's array *parameters* are replaced by
// the caller's actual arguments.  When those are distinct global arrays the
// memory-dependence question "may this store clobber that load?" flips from
// `may-alias` to `no-alias`, and an invariant load that had to stay inside the
// loop (because a pointer parameter might have pointed anywhere) can finally
// be hoisted.
//
// That case is not academic: in 01_mm3 the inner loop
//
//     C[i][j] = C[i][j] * A[i][k] + B[k][j];
//
// re-reads the loop-invariant A[i][k] on every iteration.  With `mm` inlined
// into `main` the A/C roots are the globals @A/@C, so the reload can move to
// the j-loop preheader.  LLVM's LICM does exactly this (its flat-CFG form,
// running after inlining, with BasicAA); measured on the QEMU guest it is the
// difference between 940ms and 120ms for that kernel.
//
// The analysis mirrors the structured licm: natural loops from the
// dominance-verified back edges, writes summarised once per loop, and the same
// conservative `storeAliases` test (distinct private objects never alias, even
// when both offsets are dynamic).  Loops are processed innermost-first so an
// inner hoist can enable an outer one, and the whole per-function walk repeats
// until it reaches a fixpoint.
// ---------------------------------------------------------------------------
#include "Opt.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <unordered_set>

#include "OptUtil.h"

namespace sakura {
namespace ir {

namespace {

// ---------------------------------------------------------------------------
// memory-dependence summary, shared shape with OptLoop.cpp's licm
// ---------------------------------------------------------------------------
struct LoopWrites {
  // One entry per call in the loop: the callee, or nullptr for an indirect
  // call.  Whether a call actually clobbers a given load is decided per load
  // by `Effects::callMayWrite`: a `readnone`/`readonly` callee (inferred from
  // the whole call graph) does not stop a load from being hoisted across it.
  // That matters directly for fft0, where the loop counter bound `d` is a
  // global reloaded after every call to the recursive `multiply`, which
  // provably writes nothing.
  std::vector<Function *> calls;
  std::vector<AddrInfo> stores;
};

bool storeAliases(const AddrInfo &l, const AddrInfo &s) {
  if (s.constOff && l.constOff && s.root == l.root) return s.off == l.off;
  // Two distinct private objects (a global or a stack slot each) can never
  // alias regardless of how their offsets are computed: a gep chain rooted at
  // one object cannot reach the storage of the other.  This is what makes the
  // `C[i][j] = ... A[i][k] ...` pattern hoistable.
  if (s.root && l.root && s.root != l.root) {
    bool lPriv = l.kind == RootKind::Alloca || l.kind == RootKind::Global;
    bool sPriv = s.kind == RootKind::Alloca || s.kind == RootKind::Global;
    if (lPriv && sPriv) return false;
  }
  if (l.constOff && l.kind == RootKind::Alloca) return false;
  return true;
}

bool loadMayChange(Value *addr, const LoopWrites &w, const Effects &fx) {
  AddrInfo l = addressOf(addr);
  for (Function *c : w.calls)
    if (fx.callMayWrite(c, l)) return true;
  for (const AddrInfo &s : w.stores)
    if (storeAliases(l, s)) return true;
  return false;
}

// A call may be speculated into the preheader when it is a pure value that
// always returns: it writes nothing, reads nothing mutable, and provably comes
// back, so running it on a path that would not have run it is unobservable.
// That is LLVM's LICM acting on inferred `readnone`/`readonly` +
// `willreturn`, and it is what lifts fft0's `power(d, mod-2)` out of the
// scaling loop - clang runs it once, we used to run it per iteration.
bool callHoistable(Instruction *i, const Effects &fx) {
  if (i->op != Op::Call || i->ops.empty()) return false;
  auto *callee = dynamic_cast<Function *>(i->ops[0]);
  if (!callee) return false; // indirect call
  return fx.callIsValue(callee) && fx.terminates(callee);
}

class CfLicmPass final : public Pass {
public:
  const char *name() const override { return "licm-cf"; }
  Layer inLayer() const override { return Layer::Cf; }

  bool run(Module &mod) override {
    bool any = false;
    // Function-attribute summaries (`readnone` / `readonly`): the pass only
    // moves instructions, so one computation covers every round.
    fx_.compute(mod);
    for (auto &f : mod.functions()) {
      if (f->isLib || f->blocks.empty()) continue;
      if (splitPreheaders(*f)) any = true;
      // An outer-loop hoist can expose an inner-loop one and vice versa (a
      // fresh preheader value is invariant for the enclosing loop), so iterate
      // until a round changes nothing.
      for (int round = 0; round < 8; ++round)
        if (!processFunction(*f)) break;
        else any = true;
    }
    return any;
  }

  // LLVM's LoopSimplify gives every natural loop a dedicated preheader block
  // whose only successor is the header.  Without one, a loop whose entry edge
  // is a `cond_br` (the shape every loop that sits behind a test has - the
  // branch both enters the loop and skips it) offers `hoistOutOf` nowhere to
  // put a hoisted value: the entry block also runs on the paths that never
  // reach the loop, and speculating a load onto those paths is not always
  // safe, so the pass has to give up.
  //
  // Splitting the entry edge costs one unconditional jump that is executed
  // once per loop *entry* (outside the loop, so it never shows up in a
  // profile) and buys the hoist for every invariant load in the loop.  That
  // is what lets the second `while` of `row_reduce` in conv2d-1 lose its
  // per-iteration `r*N_eff` multiply: the global reload that used to pin the
  // address expression inside the loop moves to the new preheader, which in
  // turn lets ind-var hand the loop a running pointer.
  bool splitPreheaders(Function &f) {
    const size_t n = f.blocks.size();
    if (n < 3) return false;

    std::unordered_map<BasicBlock *, size_t> idx;
    for (size_t i = 0; i < n; ++i) idx[f.blocks[i].get()] = i;

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
            next[i] = (uint8_t)(next[i] && dom[pred[bi][pi]][i]);
        next[bi] = 1;
        if (next != dom[bi]) {
          dom[bi] = std::move(next);
          changed = true;
        }
      }
    }

    // Headers that need a fresh entry block: their current entry edges are
    // either shared (the terminator is a cond_br) or several.
    struct Split {
      size_t h;
      std::vector<size_t> outside;
    };
    std::vector<Split> todo;
    for (size_t h = 1; h < n; ++h) {
      bool isHeader = false;
      std::vector<size_t> latches;
      for (size_t b = 0; b < n; ++b)
        for (size_t s : succ[b])
          if (s == h && dom[b][h]) {
            isHeader = true;
            latches.push_back(b);
          }
      if (!isHeader) continue;

      std::vector<uint8_t> inLoop(n, 0);
      inLoop[h] = 1;
      std::vector<size_t> work(latches.begin(), latches.end());
      while (!work.empty()) {
        size_t x = work.back();
        work.pop_back();
        if (inLoop[x]) continue;
        inLoop[x] = 1;
        for (size_t p : pred[x])
          if (p != h && !inLoop[p]) work.push_back(p);
      }

      std::vector<size_t> outside;
      for (size_t p : pred[h])
        if (!inLoop[p]) outside.push_back(p);
      if (outside.empty()) continue;

      // Already dedicated?  Then leave the loop exactly as it is.
      if (outside.size() == 1) {
        const std::vector<std::unique_ptr<Instruction>> &pv =
            f.blocks[outside[0]]->instrs;
        if (!pv.empty() && pv.back()->op == Op::Br && pv.back()->ops.size() == 1 &&
            dynamic_cast<BasicBlock *>(pv.back()->ops[0]) == f.blocks[h].get())
          continue;
      }
      todo.push_back({h, outside});
    }
    if (todo.empty()) return false;

    // Build one new block per loop and splice it in directly after the
    // primary entry edge, so the fall-through path through the split is the
    // straight-line one and the block layout keeps the loop contiguous.
    std::vector<std::pair<size_t, std::unique_ptr<BasicBlock>>> inserts;
    for (const Split &s : todo) {
      BasicBlock *hB = f.blocks[s.h].get();
      auto np = std::make_unique<BasicBlock>(hB->name + "_pre");
      BasicBlock *npB = np.get();
      // Turn every out-of-loop edge p -> h into p -> np.
      for (size_t p : s.outside) {
        if (f.blocks[p]->instrs.empty()) continue;
        Instruction *t = f.blocks[p]->instrs.back().get();
        for (Value *&o : t->ops)
          if (o == (Value *)hB) o = npB;
      }
      // ... and re-key the header phis on the edge that moved.
      for (auto &iu : hB->instrs) {
        Instruction *phi = iu.get();
        if (phi->op != Op::Phi) continue;
        for (size_t k = 0; k + 1 < phi->ops.size(); k += 2)
          for (size_t p : s.outside)
            if (phi->ops[k] == (Value *)f.blocks[p].get()) phi->ops[k] = npB;
      }
      auto br = std::make_unique<Instruction>(Op::Br, Type::Void);
      br->ops = {hB};
      br->line = hB->instrs.empty() ? 0 : hB->instrs.front()->line;
      np->instrs.push_back(std::move(br));

      size_t anchor = s.outside[0];
      for (size_t p : s.outside) anchor = std::min(anchor, p);
      inserts.push_back({anchor, std::move(np)});
    }

    std::unordered_map<size_t, std::vector<std::unique_ptr<BasicBlock>>> after;
    for (auto &e : inserts) after[e.first].push_back(std::move(e.second));
    BlockList out;
    out.reserve(n + inserts.size());
    for (size_t i = 0; i < n; ++i) {
      out.push_back(std::move(f.blocks[i]));
      auto it = after.find(i);
      if (it != after.end())
        for (auto &b : it->second) out.push_back(std::move(b));
    }
    f.blocks = std::move(out);
    return true;
  }

private:
  Effects fx_;

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

  // Is `v` defined outside every block of the loop?  Structural operands
  // (blocks / functions / globals) and non-instructions (constants,
  // arguments) are always "outside".
  static bool definedOutside(Value *v, const std::unordered_set<Instruction *> &loopDefs) {
    if (auto *i = dynamic_cast<Instruction *>(v)) return !loopDefs.count(i);
    return true;
  }

  bool processFunction(Function &f) {
    const size_t n = f.blocks.size();
    if (n < 2) return false;

    std::unordered_map<BasicBlock *, size_t> idx;
    for (size_t i = 0; i < n; ++i) idx[f.blocks[i].get()] = i;

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

    // Dominator sets: iterative dataflow over the reverse postorder (the
    // functions here are small, so the O(n^3) bitset form is fine).
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
            next[i] = (uint8_t)(next[i] && dom[pred[bi][pi]][i]);
        next[bi] = 1;
        if (next != dom[bi]) {
          dom[bi] = std::move(next);
          changed = true;
        }
      }
    }

    // Collect every natural loop (one per distinct header) with its unique
    // preheader, then sort innermost-first.
    struct Loop {
      size_t header;
      size_t pre;
      std::vector<uint8_t> inLoop;
      std::vector<size_t> nodes; // block indices in the loop, header first
    };
    std::vector<Loop> loops;
    for (size_t h = 1; h < n; ++h) {
      bool isHeader = false;
      for (size_t b = 0; b < n; ++b)
        for (size_t s : succ[b])
          if (s == h && dom[b][h]) isHeader = true;
      if (!isHeader) continue;

      std::vector<uint8_t> inLoop(n, 0);
      inLoop[h] = 1;
      std::vector<size_t> work;
      for (size_t b = 0; b < n; ++b)
        for (size_t s : succ[b])
          if (s == h && dom[b][h]) work.push_back(b);
      while (!work.empty()) {
        size_t x = work.back();
        work.pop_back();
        if (inLoop[x]) continue;
        inLoop[x] = 1;
        for (size_t p : pred[x])
          if (p != h && !inLoop[p]) work.push_back(p);
      }

      // Preheader: the unique predecessor outside the loop that reaches the
      // header through an unconditional branch.  With several such edges
      // there is no single place that dominates the loop, so skip the loop.
      size_t pre = n;
      bool multi = false;
      for (size_t p : pred[h]) {
        if (inLoop[p]) continue;
        const std::vector<std::unique_ptr<Instruction>> &pv = f.blocks[p]->instrs;
        bool uncond = !pv.empty() && pv.back()->op == Op::Br &&
                      pv.back()->ops.size() == 1 &&
                      dynamic_cast<BasicBlock *>(pv.back()->ops[0]) == f.blocks[h].get();
        if (!uncond) continue;
        if (pre != n) multi = true;
        pre = p;
      }
      if (pre == n || multi) continue;
      if (!dom[h][pre]) continue; // the preheader must dominate the header

      Loop lp;
      lp.header = h;
      lp.pre = pre;
      lp.inLoop = std::move(inLoop);
      for (size_t b = 0; b < n; ++b)
        if (lp.inLoop[b]) lp.nodes.push_back(b);
      loops.push_back(std::move(lp));
    }
    if (loops.empty()) return false;
    std::sort(loops.begin(), loops.end(),
              [](const Loop &a, const Loop &b) { return a.nodes.size() < b.nodes.size(); });

    bool any = false;
    for (Loop &lp : loops)
      if (hoistOutOf(f, lp.header, lp.pre, lp.inLoop, lp.nodes)) any = true;
    return any;
  }

  bool hoistOutOf(Function &f, size_t header, size_t pre,
                  const std::vector<uint8_t> &inLoop,
                  const std::vector<size_t> &nodes) {
    (void)header;

    // Sum up what the loop can write, plus the set of instructions it defines.
    LoopWrites writes;
    std::unordered_set<Instruction *> loopDefs;
    for (size_t b : nodes)
      for (auto &iu : f.blocks[b]->instrs) {
        Instruction *i = iu.get();
        loopDefs.insert(i);
        if (i->op == Op::Store && i->ops.size() >= 2)
          writes.stores.push_back(addressOf(i->ops[1]));
        else if (i->op == Op::Call)
          writes.calls.push_back(i->ops.empty()
                                     ? nullptr
                                     : dynamic_cast<Function *>(i->ops[0]));
      }

    // Candidate selection.  Iterated to a fixpoint so that hoisting a value
    // makes its single-use consumers invariant in turn (they are then picked
    // up in the next round below, since they become loop-external operands).
    std::unordered_set<Instruction *> hoisted;
    bool grew = true;
    while (grew) {
      grew = false;
      for (size_t b : nodes) {
        std::vector<std::unique_ptr<Instruction>> &v = f.blocks[b]->instrs;
        for (auto &iu : v) {
          Instruction *i = iu.get();
          if (hoisted.count(i)) continue;
          if (isTerminator(i->op)) continue;
          if (i->op == Op::Phi) continue; // phis belong to their block
          bool inv = false;
          if (i->op == Op::Load && i->ops.size() == 1) {
            inv = !loadMayChange(i->ops[0], writes, fx_);
          } else if (i->op == Op::Call) {
            inv = callHoistable(i, fx_);
          } else if (isPure(i->op) && i->op != Op::Alloca && i->op != Op::Load) {
            inv = true;
          }
          if (!inv) continue;
          bool allOpsOutside = true;
          for (Value *o : i->ops) {
            if (isStructural(o)) continue;
            if (!definedOutside(o, loopDefs) && !hoisted.count(dynamic_cast<Instruction *>(o))) {
              allOpsOutside = false;
              break;
            }
          }
          if (!allOpsOutside) continue;
          hoisted.insert(i);
          grew = true;
        }
      }
    }
    if (hoisted.empty()) return false;

    // Keep only defs actually used *within* the loop: hoisting a value whose
    // only uses are outside the loop would be pointless.
    std::unordered_set<Instruction *> useInLoop;
    for (size_t b = 0; b < f.blocks.size(); ++b) {
      if (!inLoop[b]) continue;
      for (auto &iu : f.blocks[b]->instrs)
        for (Value *o : iu->ops) {
          auto *d = dynamic_cast<Instruction *>(o);
          if (d && hoisted.count(d)) useInLoop.insert(d);
        }
    }
    std::vector<Instruction *> toMove;
    for (Instruction *i : hoisted)
      if (useInLoop.count(i)) toMove.push_back(i);
    if (toMove.empty()) return false;

    // Drain the candidates out of the loop blocks, preserving each block's
    // relative order, so the splicing below cannot reorder a def after its use
    // in the same block.
    std::vector<std::unique_ptr<Instruction>> moved;
    std::unordered_set<Instruction *> moveSet(toMove.begin(), toMove.end());
    for (size_t b : nodes) {
      std::vector<std::unique_ptr<Instruction>> &v = f.blocks[b]->instrs;
      for (auto it = v.begin(); it != v.end();) {
        if (moveSet.count(it->get())) {
          moved.push_back(std::move(*it));
          it = v.erase(it);
        } else {
          ++it;
        }
      }
    }
    if (moved.empty()) return false;

    // Splice them into the preheader just before its terminator.  The order
    // they were drained in respects all loop-internal dependences (a def was
    // visited before its single-use consumer within a block), and anything
    // they read from outside the loop already dominates the preheader.
    std::vector<std::unique_ptr<Instruction>> &pv = f.blocks[pre]->instrs;
    size_t at = pv.size();
    for (size_t q = 0; q < pv.size(); ++q)
      if (isTerminator(pv[q]->op)) {
        at = q;
        break;
      }
    for (auto &m : moved)
      pv.insert(pv.begin() + (long)at++, std::move(m));
    return true;
  }
};

} // namespace

std::unique_ptr<Pass> makeCfLicmPass() { return std::make_unique<CfLicmPass>(); }

} // namespace ir
} // namespace sakura
