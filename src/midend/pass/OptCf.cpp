// ---------------------------------------------------------------------------
// OptCf.cpp: flat-cf cleanup (cfg-simplify) and simple self-tail-recursion
// elimination (tail-rec-elim).
//
//  * cfg-simplify folds constant-armed branches, merges a block that ends in
//    an unconditional jump into its single-predecessor-free successor, and
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
        ch |= mergeJumps(f->blocks);
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
            ch = true;
          }
        }
      }
    return ch;
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
      a->instrs.pop_back(); // drop the br
      for (auto &iu : b->instrs) a->instrs.push_back(std::move(iu));
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
    for (auto it = list.begin(); it != list.end();) {
      if (!reach.count(it->get())) {
        it = list.erase(it);
        ch = true;
      } else
        ++it;
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
