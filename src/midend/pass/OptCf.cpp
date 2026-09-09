// ---------------------------------------------------------------------------
// OptCf.cpp: flat-cf cleanup (cfg-simplify) and simple self-tail-recursion
// elimination (tail-rec-elim).
//
//  * cfg-simplify folds constant-armed branches and identical-arm cond_brs,
//    merges a block that ends in an unconditional jump into its
//    single-predecessor-free successor, removes an empty "goto" block reached
//    from a single predecessor (re-keying any cf.phi at its target), and
//    drops blocks unreachable from the function entry.  Only valid at the cf
//    layer (after canonicalize-control-flow).
//
//  * tail-rec-elim converts `...; %r = func.call @f(args); func.return %r`
//    inside @f into a loop over parameter slots.  Restricted to functions
//    whose parameters are all scalars; array parameters are left untouched.
// ---------------------------------------------------------------------------
#include "Opt.h"

#include <unordered_map>
#include <unordered_set>

#include "OptUtil.h"

namespace sakura {
namespace ir {

namespace {

// ===========================================================================
// cfg-simplify
// ===========================================================================
class CfgSimplifyPass final : public Pass {
public:
  const char *name() const override { return "cfg-simplify"; }
  Layer inLayer() const override { return Layer::Cf; }

  bool run(Module &mod) override {
    bool any = false;
    for (auto &f : mod.functions()) {
      if (f->isLib || f->blocks.size() < 2) continue;
      bool ch = true;
      while (ch) {
        ch = false;
        ch |= foldConstBranches(f->blocks);
        ch |= foldSameTargets(f->blocks);
        ch |= mergeJumps(f->blocks);
        ch |= dropEmptyThunks(f->blocks);
        ch |= dropUnreachable(f->blocks);
        if (ch) any = true;
      }
    }
    return any;
  }

private:
  static bool foldConstBranches(BlockList &list) {
    bool ch = false;
    for (auto &bb : list)
      for (auto &iu : bb->instrs) {
        Instruction *i = iu.get();
        if (i->op == Op::CondBr && i->ops.size() == 3) {
          if (auto *ci = asConstInt(i->ops[0])) {
            i->op = Op::Br;
            i->ops = {ci->v ? i->ops[1] : i->ops[2]};
            i->cond = Cond::Eq;
            // the folded-away arm's edge is gone; drop its phi inputs so
            // later thunk collapsing never re-keys two values onto one block
            pruneStalePhiPred(list, bb.get());
            ch = true;
          }
        }
      }
    return ch;
  }

  // Successor blocks of a block's terminator (cf layer: br / cond_br).
  static std::vector<BasicBlock *> termSucc(BasicBlock *b) {
    std::vector<BasicBlock *> r;
    Instruction *last = b->instrs.empty() ? nullptr : b->instrs.back().get();
    if (!last) return r;
    auto push = [&](Value *t) {
      if (auto *tb = dynamic_cast<BasicBlock *>(t)) r.push_back(tb);
    };
    if (last->op == Op::Br) push(last->ops[0]);
    if (last->op == Op::CondBr && last->ops.size() >= 3) {
      push(last->ops[1]);
      push(last->ops[2]);
    }
    return r;
  }

  // A cond_br whose two arms already reach the same block is unconditional.
  static bool foldSameTargets(BlockList &list) {
    bool ch = false;
    for (auto &bb : list)
      for (auto &iu : bb->instrs) {
        Instruction *i = iu.get();
        if (i->op == Op::CondBr && i->ops.size() == 3 &&
            i->ops[1] == i->ops[2]) {
          i->op = Op::Br;
          i->ops = {i->ops[1]};
          i->cond = Cond::Eq;
          ch = true;
        }
      }
    return ch;
  }

  // Values compare as the same phi input when they are the same SSA value or
  // two identical constants.
  static bool samePhiInput(Value *a, Value *b) {
    if (a == b) return true;
    if (auto *ca = asConstInt(a))
      if (auto *cb = asConstInt(b)) return ca->v == cb->v;
    if (auto *fa = asConstFloat(a))
      if (auto *fb = asConstFloat(b)) return fa->bits() == fb->bits();
    return false;
  }

  // Remove one empty "goto" block e (single instruction `br t`) when it has
  // exactly one predecessor p.  The edge p -> e -> t collapses to p -> t and
  // every cf.phi at t whose incoming edge was keyed on e is re-keyed to p.
  //
  // The rewrite is refused when t already receives a *different* value from
  // p on a direct edge: a phi distinguishes incoming edges by predecessor
  // block only, so after collapsing e the two distinct value flows would be
  // indistinguishable (this is what keeps, e.g., the disambiguating empty
  // latch of a lowered if/else in place).  Only one thunk is dropped per call
  // because predecessor counts shift as blocks disappear.
  static bool dropEmptyThunks(BlockList &list) {
    if (list.size() < 2) return false;
    std::unordered_map<BasicBlock *, std::vector<BasicBlock *>> preds;
    for (auto &bb : list) preds[bb.get()] = {};
    for (auto &bb : list)
      for (BasicBlock *s : termSucc(bb.get()))
        if (preds.count(s)) preds[s].push_back(bb.get());

    for (size_t ei = 0; ei < list.size(); ++ei) {
      BasicBlock *e = list[ei].get();
      if (list.front().get() == e) continue; // never drop the entry
      auto &ins = e->instrs;
      if (ins.size() != 1) continue; // "empty": only a terminator
      Instruction *br = ins[0].get();
      if (br->op != Op::Br) continue;
      auto *t = dynamic_cast<BasicBlock *>(br->ops[0]);
      if (!t || t == e) continue;
      auto &pe = preds[e];
      if (pe.size() != 1) continue;
      BasicBlock *p = pe[0];
      if (p == e || !preds.count(t)) continue;
      // Collapsing e when p == t would turn p's own terminator edge into a
      // self-loop (p --cond--> e --br--> p becomes p --cond--> p).  A
      // self-loop header whose exit successor reads its cf.phis cannot be
      // lowered safely by the backend (phi copies for the self edge would run
      // unconditionally in front of the exit branch and clobber the values
      // the taken arm expects), so keep the empty latch block in place.
      if (p == t) continue;

      // 1) phi re-keying / conflict check across every block that names e as
      //    an incoming predecessor (in practice only t, e's sole successor).
      struct ReKey {
        Instruction *phi;
        size_t slot; // index of the (pred, value) pair whose pred == e
        Value *v;
      };
      std::vector<ReKey> rekeys;
      bool conflict = false;
      auto checkBlock = [&](BasicBlock &dst) {
        for (auto &iu : dst.instrs) {
          Instruction *phi = iu.get();
          if (phi->op != Op::Phi) continue;
          for (size_t s = 0; s + 1 < phi->ops.size(); s += 2) {
            if (phi->ops[s] != (Value *)e) continue;
            Value *v = phi->ops[s + 1];
            // A direct p -> dst edge would already carry a phi input for p;
            // collapsing e is only sound when both agree on the value.
            for (size_t q = 0; q + 1 < phi->ops.size(); q += 2)
              if (q != s && phi->ops[q] == (Value *)p &&
                  !samePhiInput(phi->ops[q + 1], v)) {
                conflict = true;
                return;
              }
            rekeys.push_back({phi, s / 2, v});
          }
        }
      };
      checkBlock(*t);
      // defensive: e should not appear as a phi predecessor anywhere else
      if (!conflict)
        for (auto &bb : list)
          if (bb.get() != t) {
            checkBlock(*bb);
            if (conflict) break;
          }
      if (conflict) continue;

      // 2) rewrite p's terminator edge e -> t.
      Instruction *pt = p->instrs.empty() ? nullptr : p->instrs.back().get();
      if (!pt) continue;
      bool found = false;
      auto reroute = [&](Value *&op) {
        if (op == (Value *)e) {
          op = t;
          found = true;
        }
      };
      if (pt->op == Op::Br) reroute(pt->ops[0]);
      if (pt->op == Op::CondBr && pt->ops.size() >= 3) {
        reroute(pt->ops[1]);
        reroute(pt->ops[2]);
      }
      if (!found) continue;

      // 3) re-key / drop the phi inputs named after e.
      for (ReKey &rk : rekeys) {
        Instruction *phi = rk.phi;
        size_t pair = rk.slot; // ops[2*pair] == e
        // already a p input?
        bool hasP = false;
        for (size_t s = 0; s + 1 < phi->ops.size(); s += 2)
          if (phi->ops[s] == (Value *)p) hasP = true;
        if (hasP) {
          // duplicate input carrying the same value: erase the e pair
          phi->ops.erase(phi->ops.begin() + (long)(2 * pair),
                         phi->ops.begin() + (long)(2 * pair + 2));
        } else {
          phi->ops[2 * pair] = p; // re-key incoming edge to p
        }
      }

      // 4) erase the thunk itself.
      for (auto it = list.begin(); it != list.end(); ++it)
        if (it->get() == e) {
          list.erase(it);
          break;
        }
      return true;
    }
    return false;
  }

  static bool mergeJumps(BlockList &list) {
    // successor block ids per block
    auto succ = [&](BasicBlock *b) -> std::vector<BasicBlock *> {
      Instruction *last = b->instrs.empty() ? nullptr : b->instrs.back().get();
      std::vector<BasicBlock *> r;
      if (!last) return r;
      if (last->op == Op::Br)
        if (auto *t = dynamic_cast<BasicBlock *>(last->ops[0])) r.push_back(t);
      if (last->op == Op::CondBr) {
        if (last->ops.size() >= 3) {
          if (auto *t = dynamic_cast<BasicBlock *>(last->ops[1])) r.push_back(t);
          if (auto *t = dynamic_cast<BasicBlock *>(last->ops[2])) r.push_back(t);
        }
      }
      return r;
    };
    std::unordered_map<BasicBlock *, int> pred;
    for (auto &bb : list) pred[bb.get()] = 0;
    for (auto &bb : list)
      for (BasicBlock *s : succ(bb.get()))
        if (pred.count(s)) pred[s]++;

    bool ch = false;
    for (size_t ai = 0; ai < list.size(); ++ai) {
      BasicBlock *a = list[ai].get();
      Instruction *last = a->instrs.empty() ? nullptr : a->instrs.back().get();
      if (!last || last->op != Op::Br) continue;
      BasicBlock *b = dynamic_cast<BasicBlock *>(last->ops[0]);
      if (!b || b == a) continue;
      if (pred[b] != 1) continue;
      // b is only entered from a: splice b's body into a, drop the jump.
      // guard: b must not be the function entry / a region target (cf layer
      // has no regions; the entry is list.front()).
      if (list[0].get() == b) continue;
      // When this pass runs again after mem2reg, b may hold cf.phi headers.
      // A phi must stay at the head of its block and describe every incoming
      // edge, so a plain splice is not a valid rewrite there -- skip.
      bool hasPhi = false;
      for (auto &iu : b->instrs)
        if (iu->op == Op::Phi) { hasPhi = true; break; }
      if (hasPhi) continue;
      // Splicing moves every definition of b to a's position.  That keeps
      // the linear def-before-use layout intact only when no block *before*
      // a uses a value defined in b (b may dominate such blocks although it
      // is textually later - e.g. a return block whose load is shared by an
      // earlier branch).
      std::unordered_set<Instruction *> bDefs;
      for (auto &iu : b->instrs) bDefs.insert(iu.get());
      bool safe = true;
      for (size_t ui = 0; ui < ai && safe; ++ui)
        for (auto &iu : list[ui]->instrs)
          for (Value *o : iu->ops)
            if (auto *d = dynamic_cast<Instruction *>(o))
              if (bDefs.count(d)) { safe = false; break; }
      if (!safe) continue;
      // A cf.phi at one of b's successors keys its incoming value on b; that
      // edge must be re-keyed to a after the splice.  a can normally not yet
      // feed such a phi (its only outgoing edge was the jump into b that we
      // are about to remove) - except when an earlier fold left a *stale*
      // a-keyed input behind (a no longer branches there, but the phi still
      // lists it).  Re-keying would then merge two different values onto the
      // same predecessor, which a phi cannot express.  Detect that case and
      // refuse to merge; a stale a-keyed input carries a value for an edge
      // that no longer exists, so it can never equal the live b-edge value.
      for (auto &x : list) {
        if (x.get() == b) continue;
        for (auto &iu : x->instrs) {
          Instruction *phi = iu.get();
          if (phi->op != Op::Phi) continue;
          for (size_t s = 0; s + 1 < phi->ops.size(); s += 2) {
            if (phi->ops[s] != (Value *)b) continue;
            for (size_t q = 0; q + 1 < phi->ops.size(); q += 2) {
              if (q != s && phi->ops[q] == (Value *)a) {
                safe = false;
                break;
              }
            }
            if (!safe) break;
          }
          if (!safe) break;
        }
        if (!safe) break;
      }
      if (!safe) continue;
      a->instrs.pop_back(); // drop the br
      for (auto &iu : b->instrs) a->instrs.push_back(std::move(iu));
      // Every block that carried b's instruction now carries a's instead, and
      // a's only successor used to be b.  A cf.phi that keyed an incoming
      // value on b (b's successors, e.g. the merge block an inliner or mem2reg
      // introduced) therefore has to re-key that edge to a -- the value flow
      // is unchanged, and the pre-check above ruled out any conflicting
      // a-keyed input on the same phi.
      for (auto &x : list) {
        if (x.get() == b) continue;
        for (auto &iu : x->instrs) {
          Instruction *phi = iu.get();
          if (phi->op != Op::Phi) continue;
          for (size_t s = 0; s + 1 < phi->ops.size(); s += 2)
            if (phi->ops[s] == (Value *)b) phi->ops[s] = (Value *)a;
        }
      }
      for (auto it = list.begin(); it != list.end(); ++it)
        if (it->get() == b) { list.erase(it); break; }
      ch = true;
      break;
    }
    return ch;
  }

  static bool dropUnreachable(BlockList &list) {
    if (list.empty()) return false;
    std::unordered_set<BasicBlock *> reach;
    std::vector<BasicBlock *> work = {list.front().get()};
    while (!work.empty()) {
      BasicBlock *b = work.back();
      work.pop_back();
      if (!reach.insert(b).second) continue;
      Instruction *last = b->instrs.empty() ? nullptr : b->instrs.back().get();
      if (!last) continue;
      auto push = [&](Value *t) {
        if (auto *tb = dynamic_cast<BasicBlock *>(t))
          if (!reach.count(tb)) work.push_back(tb);
      };
      if (last->op == Op::Br) push(last->ops[0]);
      if (last->op == Op::CondBr && last->ops.size() >= 3) {
        push(last->ops[1]);
        push(last->ops[2]);
      }
    }
    bool ch = false;
    std::unordered_set<BasicBlock *> dead;
    for (auto &bb : list)
      if (!reach.count(bb.get())) dead.insert(bb.get());
    if (!dead.empty()) {
      // A cf.phi distinguishes its incoming values by predecessor block.  A
      // block erased above no longer has any edge into a survivor, so every
      // phi entry keyed on a dead predecessor must go too -- otherwise the
      // phi carries a dangling block pointer into the next pass.  (The
      // inliner / mem2reg can leave such phis in place when a later fold
      // turns a predecessor unreachable, so this is not just defensive.)
      for (auto &bb : list) {
        if (dead.count(bb.get())) continue;
        for (auto &iu : bb->instrs) {
          Instruction *phi = iu.get();
          if (phi->op != Op::Phi) continue;
          for (size_t s = 0; s + 1 < phi->ops.size();) {
            if (dead.count(dynamic_cast<BasicBlock *>(phi->ops[s]))) {
              phi->ops.erase(phi->ops.begin() + (long)s,
                             phi->ops.begin() + (long)s + 2);
            } else {
              s += 2;
            }
          }
        }
      }
      for (auto it = list.begin(); it != list.end();) {
        if (dead.count(it->get())) {
          it = list.erase(it);
          ch = true;
        } else
          ++it;
      }
    }
    return ch;
  }
};

// ===========================================================================
// tail-recursion elimination (simple form)
// ===========================================================================
class TailRecElimPass final : public Pass {
public:
  const char *name() const override { return "tail-rec-elim"; }
  Layer inLayer() const override { return Layer::Cf; }

  bool run(Module &mod) override {
    bool any = false;
    for (auto &f : mod.functions()) {
      if (f->isLib) continue;
      any |= process(mod, *f);
    }
    return any;
  }

private:
  // locate a self call that is directly returned: block pattern
  //   ...; %c = func.call @f(...); func.return %c
  struct Site {
    BasicBlock *bb = nullptr;
    Instruction *call = nullptr; // may be in the same block
    Instruction *ret = nullptr;
  };

  // Unique block label for the preheader (block names are emitted as part of
  // the .L{fn}_{block} assembly labels, so they only need to be unique inside
  // this function).
  static std::string freshBlockName(Function &f) {
    std::unordered_set<std::string> used;
    for (auto &b : f.blocks) used.insert(b->name);
    std::string n = "tre.pre";
    for (int i = 0; used.count(n); ++i)
      n = "tre.pre" + std::to_string(i);
    return n;
  }

  bool process(Module &mod, Function &f) {
    if (f.params.empty()) return false;
    for (const ParamDesc &p : f.params)
      if (p.ty == Type::Ptr) return false; // array params unsupported

    std::vector<Site> sites;
    for (auto &bb : f.blocks) {
      auto &ins = bb->instrs;
      if (ins.size() < 2) continue;
      Instruction *last = ins.back().get();
      if (last->op != Op::Ret) continue;
      Instruction *prev = ins[ins.size() - 2].get();
      if (prev->op != Op::Call) continue;
      auto *callee = dynamic_cast<Function *>(prev->ops[0]);
      if (callee != &f) continue;
      // the ret must return the call result
      bool tail = true;
      if (f.ret == Type::Void) {
        if (!last->ops.empty()) tail = false;
      } else {
        if (last->ops.empty() || last->ops[0] != (Value *)prev) tail = false;
      }
      if (!tail) continue;
      sites.push_back({bb.get(), prev, last});
    }
    if (sites.empty()) return false;

    // A self tail call turns into "store the new arguments into the parameter
    // slots and jump back to the loop header".  The loop header re-reads the
    // slots, so the seed stores that initialise them may only run once.  The
    // original entry therefore splits into a once-executed preheader
    //
    //     pre:   %slot_i = alloca     store %arg_i -> %slot_i   br body
    //
    // and a body (the old entry) that starts by reloading every slot:
    //
    //     body:  %a_i = load %slot_i   ...                     <- loop header
    //     ...
    //     <tail site>  store <new args> -> %slot_i   br body
    //
    // Re-entering the *old* entry would reset the parameters to their initial
    // values on every iteration and re-run the allocas (growing the stack).
    BasicBlock *body = f.blocks.front().get();
    auto pre = std::make_unique<BasicBlock>(freshBlockName(f));
    BasicBlock *preP = pre.get();

    std::vector<Instruction *> slots; // per param alloca
    std::vector<Instruction *> seeds; // per param initial store (keeps the raw arg)
    int k = 0;
    for (size_t pi = 0; pi < f.params.size(); ++pi) {
      Type pt = f.params[pi].ty == Type::F32 ? Type::F32 : Type::I32;
      auto al = std::make_unique<Instruction>(Op::Alloca, Type::Ptr);
      al->elem = pt;
      al->n = 1;
      Instruction *alP = al.get();
      slots.push_back(alP);
      preP->instrs.insert(preP->instrs.begin() + k++, std::move(al));

      auto st = std::make_unique<Instruction>(Op::Store, Type::Void);
      st->ops = {f.args[pi].get(), alP};
      Instruction *stP = st.get();
      seeds.push_back(stP);
      preP->instrs.insert(preP->instrs.begin() + k++, std::move(st));
    }
    auto preBr = std::make_unique<Instruction>(Op::Br, Type::Void);
    preBr->cond = Cond::Eq;
    preBr->ops = {body};
    preP->instrs.push_back(std::move(preBr));
    f.blocks.insert(f.blocks.begin(), std::move(pre));

    // Loop-header reloads at the top of the body (each iteration picks up the
    // arguments written by the previous tail call).
    std::vector<Instruction *> loaders;
    k = 0;
    for (size_t pi = 0; pi < f.params.size(); ++pi) {
      Type pt = f.params[pi].ty == Type::F32 ? Type::F32 : Type::I32;
      auto ld = std::make_unique<Instruction>(Op::Load, pt);
      ld->elem = pt;
      ld->ops = {slots[pi]};
      Instruction *ldP = ld.get();
      loaders.push_back(ldP);
      body->instrs.insert(body->instrs.begin() + k++, std::move(ld));
    }

    // Any use of a scalar Argument now reads the slot -- except inside the
    // parameter's own seed store in the preheader (that store must write the
    // raw entry value into the slot, not read the load that follows it).
    for (size_t pi = 0; pi < f.args.size(); ++pi) {
      Value *arg = f.args[pi].get();
      for (auto &bb : f.blocks)
        for (auto &in : bb->instrs)
          if (in.get() != seeds[pi])
            for (Value *&op : in->ops)
              if (op == arg) op = loaders[pi];
    }

    // rewrite each tail site: store new args into the slots and jump back to
    // the loop header
    bool any = false;
    for (Site &s : sites) {
      if (s.call->ops.size() - 1 != f.params.size()) continue;
      auto &ins = s.bb->instrs;
      size_t ci = 0;
      while (ins[ci].get() != s.call) ++ci;
      // store each call argument into its slot
      std::vector<std::unique_ptr<Instruction>> stores;
      for (size_t pi = 0; pi < f.params.size(); ++pi) {
        auto st = std::make_unique<Instruction>(Op::Store, Type::Void);
        st->ops = {s.call->ops[1 + pi], slots[pi]};
        stores.push_back(std::move(st));
      }
      // drop the call and the ret, append stores + branch to the body
      ins.erase(ins.begin() + (long)ci); // call
      ins.pop_back();                    // ret
      for (auto &st : stores) ins.push_back(std::move(st));
      auto br = std::make_unique<Instruction>(Op::Br, Type::Void);
      br->cond = Cond::Eq;
      br->ops = {body};
      ins.push_back(std::move(br));
      any = true;
    }
    return any;
  }
};

} // namespace

std::unique_ptr<Pass> makeCfgSimplifyPass() {
  return std::make_unique<CfgSimplifyPass>();
}

std::unique_ptr<Pass> makeTailRecElimPass() {
  return std::make_unique<TailRecElimPass>();
}

} // namespace ir
} // namespace sakura
