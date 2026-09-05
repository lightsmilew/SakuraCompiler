// ---------------------------------------------------------------------------
// OptLoop.cpp: loop-invariant code motion + full unrolling of small constant
// affine.for loops.
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
//  * loop-unroll fully unrolls affine.for whose bounds are constants and
//    whose body is a single straight-line block without calls or nested
//    region ops.  The induction-slot loads inside each copy are replaced by
//    the iteration constant.
// ---------------------------------------------------------------------------
#include "Opt.h"

#include <unordered_map>
#include <unordered_set>

#include "OptUtil.h"

namespace sakura {
namespace ir {

namespace {

// ===========================================================================
// LICM
// ===========================================================================

// Everything in a loop that can change the value of a memory location.
struct LoopWrites {
  bool hasCall = false;
  std::vector<AddrInfo> stores;
  std::vector<Value *> ivRoots; // alloca slots of nested affine.for
};

bool storeAliases(const AddrInfo &l, const AddrInfo &s) {
  if (s.constOff && l.constOff && s.root == l.root) return s.off == l.off;
  if (l.constOff && s.constOff && s.root && l.root && s.root != l.root) {
    // two distinct private objects (alloca/global) never alias
    bool lPriv = l.kind == RootKind::Alloca || l.kind == RootKind::Global;
    bool sPriv = s.kind == RootKind::Alloca || s.kind == RootKind::Global;
    if (lPriv && sPriv) return false;
  }
  // a store to a private alloca can never touch a global / another alloca,
  // but a global can be reached through a param, and a param can alias
  // anything non-private -> be conservative
  if (l.constOff && l.kind == RootKind::Alloca) return false; // s != same obj
  return true;
}

class LicmPass final : public Pass {
public:
  explicit LicmPass(Layer l) : layer_(l) {}
  const char *name() const override { return "licm"; }
  Layer inLayer() const override { return layer_; }

  bool run(Module &mod) override {
    bool any = false;
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

  static void scanWrites(BlockList &list, LoopWrites &w) {
    for (auto &bb : list)
      for (auto &iu : bb->instrs) {
        Instruction *i = iu.get();
        if (i->op == Op::Store)
          w.stores.push_back(addressOf(i->ops[1]));
        else if (i->op == Op::Call)
          w.hasCall = true;
        else if (i->op == Op::AffineFor) {
          if (auto *s = dynamic_cast<Instruction *>(i->ops[1]))
            w.ivRoots.push_back(s); // implicit per-iteration write
          scanWrites(i->bodyRegion, w);
        } else if (i->op == Op::ScfWhile) {
          scanWrites(i->condRegion, w);
          scanWrites(i->bodyRegion, w);
        }
      }
  }

  bool loadMayChange(Value *addr, LoopWrites &w) const {
    AddrInfo l = addressOf(addr);
    for (Value *r : w.ivRoots)
      if (l.root == r) return true;
    if (w.hasCall) return true;
    for (const AddrInfo &s : w.stores)
      if (storeAliases(l, s)) return true;
    return false;
  }

  // Hoist invariant instructions out of `loop`, which lives at `idx` of
  // `owner`.  Returns true when something moved.
  bool hoist(Module &mod, BasicBlock &owner, size_t idx, Instruction *loop) {
    // every instruction defined inside the loop's own regions
    std::unordered_set<Instruction *> inner;
    std::vector<Instruction *> topLevel; // directly in cond/body top blocks
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
      // the loop's own induction slot changes every iteration
      if (auto *s = dynamic_cast<Instruction *>(loop->ops[1]))
        writes.ivRoots.push_back(s);
    }

    // fixpoint over the direct region instructions to find invariants
    std::unordered_set<Instruction *> inv; // pure + all operands invariant
    bool grow = true;
    while (grow) {
      grow = false;
      for (Instruction *i : topLevel) {
        if (inv.count(i)) continue;
        if (isTerminator(i->op)) continue;
        bool hoistable = false;
        if (i->op == Op::Load) {
          hoistable = !loadMayChange(i->ops[0], writes);
        } else if (isPure(i->op) && i->op != Op::Alloca &&
                   i->op != Op::Load) {
          hoistable = true;
        }
        if (!hoistable) continue;
        bool all = true;
        for (Value *o : i->ops) {
          if (isStructural(o)) continue;
          auto *d = dynamic_cast<Instruction *>(o);
          if (d && inner.count(d) && !inv.count(d)) { all = false; break; }
        }
        if (all) { inv.insert(i); grow = true; }
      }
    }

    std::vector<Instruction *> toMove; // dependency-safe order
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
// affine.for full unrolling
// ===========================================================================
class LoopUnrollPass final : public Pass {
public:
  explicit LoopUnrollPass(Layer l) : layer_(l) {}
  const char *name() const override { return "loop-unroll"; }
  Layer inLayer() const override { return layer_; }

  bool run(Module &mod) override {
    bool any = false;
    for (auto &f : mod.functions()) {
      if (f->isLib) continue;
      std::function<void(BlockList &)> rec = [&](BlockList &list) {
        for (auto &bb : list) {
          auto &instrs = bb->instrs;
          size_t i = 0;
          while (i < instrs.size()) {
            Instruction *op = instrs[i].get();
            if (op->op == Op::AffineFor) {
              rec(op->bodyRegion);
              if (tryUnroll(mod, *bb, i))
                any = true; // instrs changed; do not advance
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

  bool tryUnroll(Module &mod, BasicBlock &bb, size_t idx) {
    auto &instrs = bb.instrs;
    Instruction *af = instrs[idx].get();
    if (af->op != Op::AffineFor) return false;
    if (af->ops.size() != 4 || af->cond != Cond::Lt) return false;
    const ConstantInt *lb = asConstInt(af->ops[2]);
    const ConstantInt *ub = asConstInt(af->ops[3]);
    int64_t step = af->step;
    if (!lb || !ub || step <= 0) return false;
    if (lb->v >= ub->v) return false; // zero-trip handled by const-fold
    int64_t trips = (ub->v - lb->v + step - 1) / step;

    const int64_t kMaxTrips = 32;
    if (trips > kMaxTrips) return false;

    BlockList &body = af->bodyRegion;
    if (body.size() != 1) return false;
    BasicBlock *B = body[0].get();
    std::vector<Instruction *> bodyInstrs;
    for (auto &iu : B->instrs) {
      Instruction *i = iu.get();
      if (i->op == Op::AffineYield) continue;
      bodyInstrs.push_back(i);
    }
    if (bodyInstrs.empty()) return false;
    for (Instruction *i : bodyInstrs) {
      if (i->op == Op::ScfWhile || i->op == Op::AffineFor ||
          i->op == Op::Call)
        return false;
      if (i->op == Op::Store) {
        AddrInfo a = addressOf(i->ops[1]);
        if (a.root == af->ops[1]) return false; // writes to the IV slot
      }
    }
    const int64_t kMaxInstrs = 64;
    if ((int64_t)bodyInstrs.size() * trips > kMaxInstrs) return false;

    Value *exitBB = af->ops[0];
    Value *ivSlot = af->ops[1];

    std::unique_ptr<Instruction> afOwner = std::move(instrs[idx]);
    instrs.erase(instrs.begin() + idx);
    af = nullptr; // owned by afOwner now (unused further)

    for (int64_t t = 0; t < trips; ++t) {
      Value *ivConst = mod.constInt((int32_t)(lb->v + t * step));
      std::unordered_map<Instruction *, Instruction *> remap;
      std::vector<Instruction *> copies;
      for (Instruction *src : bodyInstrs) {
        auto c = std::make_unique<Instruction>(src->op, src->ty);
        c->cond = src->cond;
        c->elem = src->elem;
        c->n = src->n;
        c->step = src->step;
        c->shape = src->shape;
        c->line = src->line;
        Instruction *cp = c.get();
        remap[src] = cp;
        instrs.push_back(std::move(c));
        copies.push_back(cp);
      }
      // fill operands after the map is complete (handles forward refs that
      // point to clones already created this round)
      for (size_t k = 0; k < copies.size(); ++k) {
        Instruction *src = bodyInstrs[k];
        Instruction *cp = copies[k];
        for (Value *o : src->ops) {
          auto *d = dynamic_cast<Instruction *>(o);
          if (d && remap.count(d)) cp->ops.push_back(remap[d]);
          else cp->ops.push_back(o);
        }
      }
      // replace this copy's IV-slot loads with the iteration constant
      for (size_t k = 0; k < copies.size(); ++k) {
        Instruction *src = bodyInstrs[k];
        Instruction *cp = copies[k];
        if (src->op == Op::Load && src->ops.size() == 1 &&
            src->ops[0] == ivSlot) {
          replaceAllUses(mod, cp, ivConst);
          for (auto it = instrs.begin(); it != instrs.end(); ++it)
            if (it->get() == cp) { instrs.erase(it); break; }
        }
      }
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
