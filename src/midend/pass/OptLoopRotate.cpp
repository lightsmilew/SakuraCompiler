// ---------------------------------------------------------------------------
// OptLoopRotate.cpp: loop rotation on the flat cf layer (LLVM LoopRotate).
//
// A `while (cond) { body }` loop as it reaches the cf layer is a two-branch
// loop: the header tests the condition (`cond_br ^body, ^exit`) and the latch
// closes the cycle with an *unconditional* jump back to the header, so each
// iteration pays for a test and a jump:
//
//     ^pre:    cf.br ^header
//     ^header: %c = <cond using phis>; cf.cond_br %c, ^body, ^exit
//     ^body:   ... ; cf.br ^latch
//     ^latch:  ... ; cf.br ^header
//
// Loop rotation turns it inside out: the condition is evaluated once before
// the loop (the "guard", which stays in the old header block) and then
// re-tested at the bottom, so the unconditional back edge is replaced by the
// conditional one:
//
//     ^pre:    cf.br ^header
//     ^header: %c0 = <cond with phi->preheader-edge values>
//              cf.cond_br %c0, ^body, ^exit
//     ^body:   phis; ... ; cf.br ^latch
//     ^latch:  ... ; %c1 = <cond with phi->latch-edge values>
//              cf.cond_br %c1, ^body, ^exit
//
// The header phis move to the body entry: the value from the preheader is
// still the value on the first entry (its phi entry is retargeted from the
// preheader block to the guard, which is now the block that branches into the
// body), and the latch's entry already names a real predecessor of the
// rotated body.
//
// Rotation is applied only where the machine-code layout can realise the win:
// the body entry must immediately follow the header and the exit must
// immediately follow the latch.  The back-end emits no jump when a branch's
// target is the physically next block, so in exactly those cases the loop
// loses one taken branch per iteration with no compensating jump anywhere.
//
// The analysis (CFG, dominators, natural loop) is re-derived after every
// rotation, so each rewrite sees the real, updated control flow.
// ---------------------------------------------------------------------------
#include "Opt.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "OptUtil.h"

namespace sakura {
namespace ir {

namespace {

class LoopRotatePass final : public Pass {
public:
  explicit LoopRotatePass(Layer l) : layer_(l) {}
  const char *name() const override { return "loop-rotate"; }
  Layer inLayer() const override { return layer_; }

  bool run(Module &mod) override {
    // Only the flat cf layer has cf.phi / cf.cond_br; the structured layers
    // keep their loops in region ops, which the canonicalizer owns.
    if (layer_ != Layer::Cf) return false;
    bool any = false;
    for (auto &f : mod.functions()) {
      if (f->isLib || f->blocks.size() < 3) continue;
      // One rotation per round; the CFG is re-derived in between.
      for (int round = 0; round < 64; ++round) {
        if (!rotateOne(*f)) break;
        any = true;
      }
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
    } else if (last->op == Op::CondBr) {
      if (last->ops.size() >= 3) {
        push(last->ops[1]);
        push(last->ops[2]);
      }
    }
  }

  // The value a phi carries on the edge from `pred`, or null when the phi has
  // no entry for it.
  static Value *phiEdge(Instruction *phi, BasicBlock *pred) {
    for (size_t k = 0; k + 1 < phi->ops.size(); k += 2)
      if (phi->ops[k] == (Value *)pred) return phi->ops[k + 1];
    return nullptr;
  }

  // Rewrite `ops` so that every operand present in `sub` is replaced by its
  // mapped value (a no-op for operands that are not keys).
  static void substitute(std::vector<Value *> &ops,
                         const std::unordered_map<Value *, Value *> &sub) {
    for (Value *&o : ops) {
      auto it = sub.find(o);
      if (it != sub.end()) o = it->second;
    }
  }

  // Rotate the first eligible loop of `f`; true when one was rotated.
  bool rotateOne(Function &f) {
    const size_t n = f.blocks.size();
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

    // Dominator sets (iterative dataflow; same form as dom-cse / ind-var).
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

    for (size_t hh = 0; hh < n; ++hh) {
      // Back edges b -> hh with hh dominating b make hh a loop header; a
      // single latch keeps the header phis' in-loop edge unambiguous.
      std::vector<size_t> latches;
      for (size_t b = 0; b < n; ++b)
        for (size_t s : succ[b])
          if (s == hh && dom[b][hh]) latches.push_back(b);
      if (latches.size() != 1) continue;
      const size_t latch = latches[0];
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
          if (p == hh || inLoop[p]) continue;
          inLoop[p] = 1;
          work.push_back(p);
        }
      }

      // Exactly one preheader: the guard evaluates the condition on the
      // preheader edge, and every header phi must have exactly one
      // out-of-loop value.
      std::vector<size_t> pre;
      for (size_t p : pred[hh])
        if (!inLoop[p]) pre.push_back(p);
      if (pre.size() != 1) continue;
      const size_t ph = pre[0];
      if (pred[hh].size() != 2 || pred[hh][0] == pred[hh][1]) continue;

      BasicBlock *H = f.blocks[hh].get();
      if (H->instrs.empty()) continue;
      Instruction *term = H->instrs.back().get();
      if (term->op != Op::CondBr || term->ops.size() != 3) continue;

      // One successor is the in-loop body entry, the other is the header's
      // (single) way out of the loop.
      auto *t1 = dynamic_cast<BasicBlock *>(term->ops[1]);
      auto *t2 = dynamic_cast<BasicBlock *>(term->ops[2]);
      if (!t1 || !t2 || t1 == t2) continue;
      auto i1 = idx.find(t1), i2 = idx.find(t2);
      if (i1 == idx.end() || i2 == idx.end()) continue;
      size_t B, E;
      if (inLoop[i1->second] && !inLoop[i2->second]) {
        B = i1->second;
        E = i2->second;
      } else if (inLoop[i2->second] && !inLoop[i1->second]) {
        B = i2->second;
        E = i1->second;
      } else {
        continue; // nested header, or a branch that leaves the loop twice
      }
      if (B == hh || E == hh) continue;
      // The body entry is entered only from the header, so it carries no phis
      // of its own to reconcile: the header's phis move there verbatim.
      if (pred[B].size() != 1 || pred[B][0] != hh) continue;
      bool bodyHasPhi = false;
      for (auto &iu : f.blocks[B]->instrs)
        if (iu->op == Op::Phi) bodyHasPhi = true;
      if (bodyHasPhi) continue;

      // The latch must be a plain back edge; its other contents are kept.
      BasicBlock *L = f.blocks[latch].get();
      if (L->instrs.empty() || L->instrs.back()->op != Op::Br) continue;
      if (L->instrs.back()->ops.empty() || L->instrs.back()->ops[0] != (Value *)H)
        continue;

      // Layout gate: this is where the win comes from, so require it.
      if (idx[f.blocks[B].get()] != hh + 1) continue;
      if (latch + 1 >= n || idx[f.blocks[E].get()] != latch + 1) continue;

      // The latch's new edge into the exit block must not invalidate the exit's
      // phis.  The exit gains the latch as a predecessor, which would need a new
      // incoming value for every phi there - the value the loop carried on the
      // back edge - and that is not always expressible (the phi's header entry
      // may name a header phi that the rotated loop no longer defines on the
      // guard path).  Only rotate when the exit has no phis to reconcile.
      bool exitOk = true;
      for (auto &iu : f.blocks[E]->instrs)
        if (iu->op == Op::Phi) exitOk = false;
      if (!exitOk) continue;


      // Split the header (non-destructively) into the phis and the condition
      // body; nothing is moved until every below check has passed, because a
      // rejected candidate must leave the function untouched.
      std::vector<Instruction *> phis, body;
      bool hasRegion = false;
      for (auto &iu : H->instrs) {
        if (iu->op == Op::Phi)
          phis.push_back(iu.get());
        else if (iu.get() != term) {
          // The condition is cloned by copying the instruction's fields, which
          // cannot carry a region op's nested blocks; such an op cannot be part
          // of a loop test anyway.
          if (iu->op == Op::ScfWhile || iu->op == Op::AffineFor)
            hasRegion = true;
          body.push_back(iu.get());
        }
      }
      if (hasRegion) continue;
      if (phis.empty()) continue; // no loop-carried value: nothing to rotate

      // Every header phi must be used only inside the loop (after rotation the
      // phis live in the body entry, which no longer dominates the exit); every
      // condition instruction must be used only in the header, because the
      // guard rewrites its operands in place.
      bool ok = true;
      for (Instruction *p : phis) {
        for (size_t b = 0; b < n && ok; ++b) {
          if (inLoop[b]) continue;
          for (auto &iu : f.blocks[b]->instrs)
            for (Value *o : iu->ops)
              if (o == (Value *)p) ok = false;
        }
      }
      for (Instruction *b : body) {
        for (size_t b2 = 0; b2 < n && ok; ++b2) {
          if (b2 == hh) continue;
          for (auto &iu : f.blocks[b2]->instrs)
            for (Value *o : iu->ops)
              if (o == (Value *)b) ok = false;
        }
      }
      if (!ok) continue;

      // Per-edge values of every header phi; a header phi must have an entry
      // for exactly the preheader and the latch.
      std::unordered_map<Value *, Value *> onPre, onLatch;
      for (Instruction *p : phis) {
        if (p->ops.size() != 4) {
          ok = false;
          break;
        }
        Value *pv = phiEdge(p, f.blocks[ph].get());
        Value *lv = phiEdge(p, f.blocks[latch].get());
        if (!pv || !lv) {
          ok = false;
          break;
        }
        onPre[p] = pv;
        onLatch[p] = lv;
      }
      if (!ok) continue;

      // ---- rewrite (no bail-out past this point) --------------------------
      // 1) clone the condition into the latch with the latch-edge values.  The
      //    condition may itself be a phi (or a value with no instruction of its
      //    own in the header), so the terminator operand is substituted too.
      std::vector<std::unique_ptr<Instruction>> clones;
      std::unordered_map<Value *, Value *> cloned;
      for (Instruction *b : body) {
        auto cl = std::make_unique<Instruction>(b->op, b->ty);
        cl->cond = b->cond;
        cl->elem = b->elem;
        cl->n = b->n;
        cl->step = b->step;
        cl->shape = b->shape;
        cl->line = b->line;
        cl->ops = b->ops;
        substitute(cl->ops, onLatch);
        substitute(cl->ops, cloned);
        cloned[b] = cl.get();
        clones.push_back(std::move(cl));
      }
      Value *latchCond = term->ops[0];
      if (auto cit = cloned.find(latchCond); cit != cloned.end())
        latchCond = cit->second;
      if (auto lit = onLatch.find(latchCond); lit != onLatch.end())
        latchCond = lit->second;

      // 2) the header becomes the guard: its own copy of the condition is
      //    retargeted at the preheader-edge values
      for (Instruction *b : body) substitute(b->ops, onPre);
      if (auto pit = onPre.find(term->ops[0]); pit != onPre.end())
        term->ops[0] = pit->second;

      // 3) move the phis to the body entry and retarget the preheader edge at
      //    the guard (the block that now branches into the body)
      {
        std::vector<std::unique_ptr<Instruction>> moved;
        for (auto &iu : H->instrs)
          if (iu && iu->op == Op::Phi) moved.push_back(std::move(iu));
        auto &bv = f.blocks[B]->instrs;
        bv.insert(bv.begin(), std::make_move_iterator(moved.begin()),
                  std::make_move_iterator(moved.end()));
        auto &hv = H->instrs;
        hv.erase(std::remove_if(hv.begin(), hv.end(),
                                [](const std::unique_ptr<Instruction> &p) {
                                  return p == nullptr;
                                }),
                 hv.end());
        for (size_t k = 0; k < phis.size(); ++k) {
          Instruction *p = bv[k].get();
          for (size_t j = 0; j + 1 < p->ops.size(); j += 2)
            if (p->ops[j] == (Value *)f.blocks[ph].get()) p->ops[j] = (Value *)H;
        }
      }

      // 4) the latch's back edge becomes the rotated test
      auto &lv = L->instrs;
      size_t at = lv.size() - 1;
      for (auto &c : clones)
        lv.insert(lv.begin() + (long)at++, std::move(c));
      auto rot = std::make_unique<Instruction>(Op::CondBr, Type::I32);
      rot->ops = {latchCond, f.blocks[B].get(), f.blocks[E].get()};
      lv.back() = std::move(rot);

      return true;
    }
    return false;
  }
};

} // namespace

std::unique_ptr<Pass> makeLoopRotatePass(Layer l) {
  return std::make_unique<LoopRotatePass>(l);
}

} // namespace ir
} // namespace sakura
