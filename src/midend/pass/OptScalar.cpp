// ---------------------------------------------------------------------------
// OptScalar.cpp: constant folding / propagation, algebraic simplification,
// dead-code elimination and unused global/function removal.
//
// Ported (in spirit) from the reference project's ConstantFoldingPass /
// InstructionCombinePass / DCEPass / RemoveUnusedGlobalAndFunctionPass, but
// implemented on SakuraCompiler's flat-op, memory-resident IR.  All passes
// recurse through the structured regions (affine.for / scf.while bodies) so
// they work at every layer of the pipeline.
// ---------------------------------------------------------------------------
#include "Opt.h"

#include <unordered_set>

#include "OptUtil.h"

namespace sakura {
namespace ir {

namespace {

// ---------------------------------------------------------------------------
// Helpers shared by the scalar passes
// ---------------------------------------------------------------------------

// Try to resolve `v` to a compile-time value.
FoldRes valueOf(const Value *v) {
  FoldRes r;
  if (auto *ci = asConstInt(v)) { r.ok = true; r.ival = ci->v; }
  else if (auto *cf = asConstFloat(v)) { r.ok = true; r.fval = cf->v; r.isFloat = true; }
  return r;
}

// True when every operand except a structural target (block / function /
// global) is a compile-time constant.
bool allOperandsConst(const Instruction *inst, int from = 0) {
  for (size_t i = (size_t)from; i < inst->ops.size(); ++i) {
    const Value *v = inst->ops[i];
    if (isStructural(const_cast<Value *>(v))) continue;
    if (!valueOf(v).ok) return false;
  }
  return true;
}

bool isIntArith(Op o) {
  switch (o) {
  case Op::Add: case Op::Sub: case Op::Mul: case Op::SDiv: case Op::SRem:
  case Op::ICmp: case Op::Not:
    return true;
  default: return false;
  }
}

bool isFpArith(Op o) {
  switch (o) {
  case Op::FAdd: case Op::FSub: case Op::FMul: case Op::FDiv: case Op::FCmp:
    return true;
  default: return false;
  }
}

// Recursion over every block list of a function, exposing the owning vector
// so a pass can splice/erase whole blocks later.
struct ListCtx {
  BlockList *list = nullptr; // null when invoked on the function body root
};
using ListFun =
    std::function<void(BlockList &list, bool isRoot)>;

void forEachBlockList(Function *fn, const ListFun &f) {
  std::function<void(BlockList &, bool)> rec = [&](BlockList &list, bool root) {
    f(list, root);
    for (auto &bb : list)
      for (auto &iu : bb->instrs) {
        Instruction *inst = iu.get();
        if (inst->op == Op::ScfWhile) {
          rec(inst->condRegion, false);
          rec(inst->bodyRegion, false);
        } else if (inst->op == Op::AffineFor) {
          rec(inst->bodyRegion, false);
        }
      }
  };
  rec(fn->blocks, true);
}

// Count how many times `v` appears as an operand (module-wide).
class UseCounter {
public:
  explicit UseCounter(Module &mod) {
    for (Instruction *i : collectInstrs(mod))
      for (Value *o : i->ops) use_[o]++;
  }
  int of(Value *v) const {
    auto it = use_.find(v);
    return it == use_.end() ? 0 : it->second;
  }
  // remove uses of every value in `defs` (erased instructions)
  void subtract(const std::unordered_set<Instruction *> &defs) {
    for (Instruction *i : defs)
      for (Value *o : i->ops) use_[o]--;
  }

private:
  std::unordered_map<Value *, int> use_;
};

} // namespace

// ===========================================================================
// constant-folding
// ===========================================================================
namespace {

class ConstFoldPass final : public Pass {
public:
  explicit ConstFoldPass(Layer l) : layer_(l) {}
  const char *name() const override { return "const-fold"; }
  Layer inLayer() const override { return layer_; }

  bool run(Module &mod) override {
    bool any = false;
    for (auto &f : mod.functions()) {
      if (f->isLib) continue;
      if (foldList(mod, f->blocks)) any = true;
    }
    return any;
  }

private:
  Layer layer_;

  // foldOne outcome: 0 = untouched, 1 = mutated in place (kept),
  // 2 = fold applies but the definition is *dead* and must be erased
  // (uses already rewritten to a constant by the caller).
  enum Fold { FoldNone = 0, FoldKept = 1, FoldErased = 2 };

  bool foldList(Module &mod, BlockList &list) {
    bool any = false;
    // Two-phase per sweep: phase 1 walks every instruction and either
    // rewrites it in place or (for a constant-folded definition) replaces all
    // of its uses and records it for removal; phase 2 erases recorded
    // definitions by identity after the walk, so an erase never disturbs an
    // iterator/pointer that the current scan still holds.
    for (;;) {
      std::unordered_set<Instruction *> eraseSet;
      bool sweep = false;
      for (auto &bb : list) {
        for (auto &iu : bb->instrs) {
          Instruction *inst = iu.get();
          if (inst->op == Op::ScfWhile) {
            if (foldList(mod, inst->condRegion)) sweep = true;
            if (foldList(mod, inst->bodyRegion)) sweep = true;
            continue;
          }
          if (inst->op == Op::AffineFor) {
            if (foldList(mod, inst->bodyRegion)) sweep = true;
            continue;
          }
          switch (foldOne(mod, inst, eraseSet)) {
          case FoldErased:
            any = true;
            sweep = true;
            break;
          case FoldKept:
            any = true;
            sweep = true;
            break;
          default:
            break;
          }
        }
      }
      if (!sweep) break;
      if (!eraseSet.empty()) {
        for (auto &bb : list) {
          for (auto it = bb->instrs.begin(); it != bb->instrs.end();) {
            if (eraseSet.count(it->get())) {
              it = bb->instrs.erase(it);
            } else {
              if ((*it)->op == Op::ScfWhile) {
                // nested regions may also hold recorded defs; erase them too
                eraseIn(eraseSet, (*it)->condRegion);
                eraseIn(eraseSet, (*it)->bodyRegion);
              } else if ((*it)->op == Op::AffineFor) {
                eraseIn(eraseSet, (*it)->bodyRegion);
              }
              ++it;
            }
          }
        }
      }
    }
    return any;
  }

  static void eraseIn(std::unordered_set<Instruction *> &eraseSet,
                      BlockList &list) {
    for (auto &bb : list)
      for (auto it = bb->instrs.begin(); it != bb->instrs.end();) {
        if (eraseSet.count(it->get())) {
          it = bb->instrs.erase(it);
        } else {
          if ((*it)->op == Op::ScfWhile) {
            eraseIn(eraseSet, (*it)->condRegion);
            eraseIn(eraseSet, (*it)->bodyRegion);
          } else if ((*it)->op == Op::AffineFor) {
            eraseIn(eraseSet, (*it)->bodyRegion);
          }
          ++it;
        }
      }
  }

  // Returns what foldList must do with `inst`.  Erasures are *deferred*:
  // when the fold makes `inst` dead, its uses are rewritten immediately and
  // the pointer is recorded in `eraseSet` for the phase-2 sweep.
  Fold foldOne(Module &mod, Instruction *inst,
               std::unordered_set<Instruction *> &eraseSet) {
    Op op = inst->op;

    // --- branch on a constant condition -------------------------------
    if (op == Op::CondBr && inst->ops.size() == 3) {
      FoldRes c = valueOf(inst->ops[0]);
      if (c.ok) {
        Value *target = c.ival ? inst->ops[1] : inst->ops[2];
        inst->op = Op::Br;
        inst->ops = {target};
        inst->cond = Cond::Eq;
        return FoldKept;
      }
      return FoldNone;
    }
    // scf.condition on statically false: the loop never runs; make the cond
    // region fall through to the exit.  (A statically true condition would
    // mean an infinite loop, so it is left alone.)
    if (op == Op::ScfCondition && inst->ops.size() == 3) {
      FoldRes c = valueOf(inst->ops[0]);
      if (c.ok && !c.ival) {
        inst->op = Op::Br;
        inst->ops = {inst->ops[2]}; // -> exit
        inst->cond = Cond::Eq;
        return FoldKept;
      }
      return FoldNone;
    }
    // affine.for that provably runs zero times (const lb >= const ub).
    if (op == Op::AffineFor && inst->ops.size() == 4 && inst->cond == Cond::Lt &&
        inst->step > 0) {
      FoldRes lb = valueOf(inst->ops[2]);
      FoldRes ub = valueOf(inst->ops[3]);
      if (lb.ok && ub.ok && lb.ival >= ub.ival) {
        // drop the loop: br to its exit block
        inst->op = Op::Br;
        inst->ops = {inst->ops[0]};
        inst->cond = Cond::Eq;
        inst->step = 0;
        return FoldKept;
      }
      return FoldNone;
    }
    // --- gep(base, 0) -> base -----------------------------------------
    if (op == Op::Gep) {
      const ConstantInt *ci = asConstInt(inst->ops[1]);
      if (ci && ci->v == 0) {
        Value *base = inst->ops[0];
        replaceAllUses(mod, inst, base);
        eraseSet.insert(inst); // deferred erase
        return FoldErased;
      }
      return FoldNone;
    }
    // --- constant arithmetic / casts / comparisons ----------------------
    if (isIntArith(op) || isFpArith(op) || op == Op::Sitofp ||
        op == Op::Fptosi) {
      if (!allOperandsConst(inst)) return FoldNone;
      FoldRes l = valueOf(inst->ops[0]);
      FoldRes r = inst->ops.size() > 1 ? valueOf(inst->ops[1]) : FoldRes{};
      FoldRes out = foldArith(op, inst->cond, l, r);
      if (!out.ok) return FoldNone;
      Value *c = out.isFloat ? (Value *)mod.constFloat(out.fval)
                             : (Value *)mod.constInt(out.ival);
      replaceAllUses(mod, inst, c);
      eraseSet.insert(inst); // deferred erase
      return FoldErased;
    }
    return FoldNone;
  }
};

} // namespace

std::unique_ptr<Pass> makeConstFoldPass(Layer l) {
  return std::make_unique<ConstFoldPass>(l);
}

// ===========================================================================
// algebraic simplification
// ===========================================================================
namespace {

class AlgebraicPass final : public Pass {
public:
  explicit AlgebraicPass(Layer l) : layer_(l) {}
  const char *name() const override { return "algebraic"; }
  Layer inLayer() const override { return layer_; }

  bool run(Module &mod) override {
    bool any = false;
    // a few fixpoint rounds: each fold can unlock the next
    for (int round = 0; round < 4; ++round) {
      bool changed = false;
      for (auto &f : mod.functions()) {
        if (f->isLib) continue;
        if (rewrite(mod, f->blocks)) changed = true;
      }
      any |= changed;
      if (!changed) break;
    }
    return any;
  }

private:
  Layer layer_;

  bool rewrite(Module &mod, BlockList &list) {
    bool any = false;
    for (auto &bb : list) {
      size_t i = 0;
      while (i < bb->instrs.size()) {
        Instruction *inst = bb->instrs[i].get();
        Op op = inst->op;
        if (op == Op::ScfWhile) {
          any |= rewrite(mod, inst->condRegion);
          any |= rewrite(mod, inst->bodyRegion);
          ++i;
          continue;
        }
        if (op == Op::AffineFor) {
          any |= rewrite(mod, inst->bodyRegion);
          ++i;
          continue;
        }
        Value *repl = trySimplify(mod, inst);
        if (repl && repl != (Value *)inst) {
          replaceAllUses(mod, inst, repl);
          bb->instrs.erase(bb->instrs.begin() + i);
          any = true;
          continue; // re-look at index i (next instruction shifted in)
        }
        ++i;
      }
    }
    return any;
  }

  // Returns a replacement SSA value when the instruction simplifies.
  Value *trySimplify(Module &mod, Instruction *inst) {
    Op op = inst->op;
    if (inst->ops.empty()) return nullptr;

    Value *a = inst->ops[0];
    Value *b = inst->ops.size() > 1 ? inst->ops[1] : nullptr;
    const ConstantInt *ca = asConstInt(a);
    const ConstantInt *cb = b ? asConstInt(b) : nullptr;
    const ConstantFloat *fa = asConstFloat(a);
    const ConstantFloat *fb = b ? asConstFloat(b) : nullptr;

    switch (op) {
    case Op::Add:
      if (ca && ca->v == 0 && !asConstInt(b)) return b; // 0 + x
      if (cb && cb->v == 0) return a;                   // x + 0
      return nullptr;
    case Op::Sub:
      if (cb && cb->v == 0) return a;                   // x - 0
      if (a == b) return mod.constInt(0);               // x - x
      return nullptr;
    case Op::Mul:
      if (ca && ca->v == 0) return mod.constInt(0);     // 0 * x
      if (cb && cb->v == 0) return mod.constInt(0);     // x * 0
      if (ca && ca->v == 1 && !asConstInt(b)) return b; // 1 * x
      if (cb && cb->v == 1) return a;                   // x * 1
      return nullptr;
    case Op::FAdd:
      if (fa && fa->v == 0.0f) return b;
      if (fb && fb->v == 0.0f) return a;
      return nullptr;
    case Op::FSub:
      if (fb && fb->v == 0.0f) return a;
      return nullptr;
    case Op::FMul:
      if (fa && fa->v == 1.0f) return b;
      if (fb && fb->v == 1.0f) return a;
      return nullptr;
    case Op::ICmp: {
      if (a == b && inst->ty == Type::I32) {
        bool t = (inst->cond == Cond::Eq || inst->cond == Cond::Le ||
                  inst->cond == Cond::Ge);
        return mod.constInt(t ? 1 : 0);
      }
      return nullptr;
    }
    case Op::Not: {
      // !(!x) -> x ; !(cmp c) -> cmp !c  (x ^ 1 on a 0/1 compare result)
      if (auto *u = dynamic_cast<Instruction *>(a))
        if (u->op == Op::Not) return u->ops[0];
      if (auto *u = dynamic_cast<Instruction *>(a))
        if (u->op == Op::ICmp) {
          // rewrite the Not itself into the negated compare: same SSA result
          inst->op = Op::ICmp;
          inst->cond = invertCond(u->cond);
          inst->ops = u->ops; // reuse the compare's operands
          return nullptr;     // mutated in place; not a "replacement"
        }
      return nullptr;
    }
    case Op::Gep: {
      const ConstantInt *c = asConstInt(b);
      if (c && c->v == 0) return a; // gep(base, 0)
      return nullptr;
    }
    default:
      return nullptr;
    }
  }

  static Cond invertCond(Cond c) {
    switch (c) {
    case Cond::Eq: return Cond::Ne;
    case Cond::Ne: return Cond::Eq;
    case Cond::Lt: return Cond::Ge;
    case Cond::Le: return Cond::Gt;
    case Cond::Gt: return Cond::Le;
    case Cond::Ge: return Cond::Lt;
    default: return Cond::Eq;
    }
  }
};

} // namespace

std::unique_ptr<Pass> makeAlgebraicPass(Layer l) {
  return std::make_unique<AlgebraicPass>(l);
}

// ===========================================================================
// dead-code elimination (+ write-only alloca removal)
// ===========================================================================
namespace {

class DeadCodePass final : public Pass {
public:
  explicit DeadCodePass(Layer l) : layer_(l) {}
  const char *name() const override { return "dead-code"; }
  Layer inLayer() const override { return layer_; }

  bool run(Module &mod) override {
    bool any = false;
    // round 1: unused pure / load / gep instructions
    for (;;) {
      bool ch = dropDeadInstructions(mod);
      any |= ch;
      if (!ch) break;
    }
    // round 2: write-only allocas (drop their stores + the alloca); then
    // clean up the geps that only fed those stores.
    any |= dropWriteOnlyAllocas(mod);
    for (;;) {
      bool ch = dropDeadInstructions(mod);
      any |= ch;
      if (!ch) break;
    }
    return any;
  }

private:
  Layer layer_;

  bool dropDeadInstructions(Module &mod) {
    bool any = false;
    for (auto &f : mod.functions()) {
      if (f->isLib) continue;
      // Function-wide reference counts, built once.  (countUses() rescans the
      // whole module per instruction and turned a straight-line 100k-inst
      // function into an O(n^2) crawl; SSA defs are function-local so a single
      // walk over the function body gives the exact same answer.)
      std::unordered_map<Value *, int> uses;
      std::function<void(BlockList &)> countRec = [&](BlockList &list) {
        for (auto &bb : list)
          for (auto &iu : bb->instrs) {
            Instruction *inst = iu.get();
            for (Value *o : inst->ops)
              if (auto *oi = dynamic_cast<Instruction *>(o)) uses[oi]++;
            if (inst->op == Op::ScfWhile) {
              countRec(inst->condRegion);
              countRec(inst->bodyRegion);
            } else if (inst->op == Op::AffineFor) {
              countRec(inst->bodyRegion);
            }
          }
      };
      countRec(f->blocks);
      // Erase every pure/load instruction that nobody reads.  Erasing drops
      // the def's own operand counts, so a later instruction in the same walk
      // sees up-to-date liveness; residual chains (def textually before its
      // dead consumer) are caught by the caller's fixpoint loop.
      std::function<void(BlockList &)> eraseRec = [&](BlockList &list) {
        for (auto &bb : list) {
          size_t i = 0;
          while (i < bb->instrs.size()) {
            Instruction *inst = bb->instrs[i].get();
            Op op = inst->op;
            if (op == Op::ScfWhile) {
              eraseRec(inst->condRegion);
              eraseRec(inst->bodyRegion);
              ++i;
              continue;
            }
            if (op == Op::AffineFor) {
              eraseRec(inst->bodyRegion);
              ++i;
              continue;
            }
            if (isTerminator(op)) { ++i; continue; }
            // pure and result unused => dead
            if (isPure(op) && uses[inst] == 0) {
              for (Value *o : inst->ops)
                if (auto *oi = dynamic_cast<Instruction *>(o)) uses[oi]--;
              bb->instrs.erase(bb->instrs.begin() + (long)i);
              any = true;
              continue;
            }
            ++i;
          }
        }
      };
      eraseRec(f->blocks);
    }
    return any;
  }

  bool dropWriteOnlyAllocas(Module &mod) {
    bool any = false;
    for (auto &f : mod.functions()) {
      if (f->isLib) continue;
      bool fnChanged = true;
      while (fnChanged) {
        fnChanged = false;
        // fresh per-iteration snapshots: erasing makes stale pointers unsafe
        auto instrsOf = [](Function *fn) {
          std::vector<Instruction *> out;
          std::function<void(BlockList &)> gather = [&](BlockList &list) {
            for (auto &bb : list)
              for (auto &iu : bb->instrs) {
                out.push_back(iu.get());
                Instruction *inst = iu.get();
                if (inst->op == Op::ScfWhile) {
                  gather(inst->condRegion);
                  gather(inst->bodyRegion);
                } else if (inst->op == Op::AffineFor) {
                  gather(inst->bodyRegion);
                }
              }
          };
          gather(fn->blocks);
          return out;
        };
        auto instrs = instrsOf(f.get());

        // Find *all* removable write-only allocas on the intact snapshot and
        // drop them together.  Candidate sets are independent (two write-only
        // slots can never share a store/gep/load), so removing them one by one
        // with a full rescan per candidate -- the old shape -- was quadratic.
        std::unordered_set<Instruction *> kill;
        std::vector<Instruction *> killAllocas;
        for (Instruction *a : instrs) {
          if (a->op != Op::Alloca) continue;
          if (killAllocas.size() && std::find(killAllocas.begin(),
                                              killAllocas.end(),
                                              a) != killAllocas.end())
            continue; // already slated (unreachable, defensive)
          // every memory op whose address is derived from this slot
          std::unordered_set<Instruction *> rooted;
          bool loadPresent = false;
          for (Instruction *u : instrs) {
            if (u->op == Op::Load || u->op == Op::Store || u->op == Op::Gep) {
              Value *addr = (u->op == Op::Store) ? u->ops[1] : u->ops[0];
              if (addressOf(addr).root == a) {
                rooted.insert(u);
                if (u->op == Op::Load) loadPresent = true;
              }
            }
          }
          if (loadPresent) continue; // the slot is read somewhere: keep it
          // Nothing may consume this slot's address or any value derived from
          // it outside the rooted memory ops (a call/return/arith consumer
          // would observe the address after we drop it).
          bool bad = false;
          for (Instruction *x : instrs) {
            if (bad) break;
            for (Value *o : x->ops) {
              auto *oi = dynamic_cast<Instruction *>(o);
              if (!oi) continue;
              if (oi == a) {
                // direct use of the slot pointer must be as a rooted
                // load/store/gep base (that op's address already roots it)
                if (x->op == Op::Load || x->op == Op::Store || x->op == Op::Gep)
                  continue;
                bad = true; // escaped into a call / terminator / arith
                break;
              }
              if (rooted.count(oi) && !rooted.count(x)) {
                bad = true; // result of a rooted op consumed by non-rooted op
                break;
              }
            }
          }
          if (bad) continue;
          kill.insert(a);
          killAllocas.push_back(a);
          for (Instruction *u : rooted) kill.insert(u);
        }
        if (killAllocas.empty()) break;

        // Two phases so we never re-run address analysis after an erase: an
        // address operand can point at an instruction erased earlier in the
        // same sweep (its consumer has not been erased yet), which would
        // dereference freed memory.  Phase 1 recorded the kill set while the
        // function was still intact; phase 2 erases by identity only.
        std::function<void(BlockList &)> eraseKill = [&](BlockList &list) {
          for (auto &bb : list) {
            for (auto it = bb->instrs.begin(); it != bb->instrs.end();) {
              if (kill.count(it->get())) {
                it = bb->instrs.erase(it);
              } else {
                if ((*it)->op == Op::ScfWhile) {
                  eraseKill((*it)->condRegion);
                  eraseKill((*it)->bodyRegion);
                } else if ((*it)->op == Op::AffineFor) {
                  eraseKill((*it)->bodyRegion);
                }
                ++it;
              }
            }
          }
        };
        eraseKill(f->blocks);
        any = fnChanged = true;
      }
    }
    return any;
  }
};

} // namespace

std::unique_ptr<Pass> makeDeadCodePass(Layer l) {
  return std::make_unique<DeadCodePass>(l);
}

// ===========================================================================
// remove-unused: uncalled user functions and unreferenced globals
// ===========================================================================
namespace {

class RemoveUnusedPass final : public Pass {
public:
  explicit RemoveUnusedPass(Layer l) : layer_(l) {}
  const char *name() const override { return "remove-unused"; }
  Layer inLayer() const override { return layer_; }

  bool run(Module &mod) override {
    bool changed = false;
    // reachable user functions (transitively) from main
    std::unordered_set<Function *> reach;
    Function *entry = mod.getFunction("main");
    if (entry) reach.insert(entry);
    bool grow = true;
    while (grow) {
      grow = false;
      for (auto &f : mod.functions()) {
        if (!reach.count(f.get())) continue;
        for (Instruction *i : collectInstrsOf(mod, f.get()))
          if (i->op == Op::Call && i->ops.size() >= 1)
            if (auto *callee = dynamic_cast<Function *>(i->ops[0]))
              if (!callee->isLib && !reach.count(callee)) {
                reach.insert(callee);
                grow = true;
              }
      }
    }
    // erase unreachable user functions
    std::vector<Function *> deadFuncs;
    for (auto &f : mod.functions())
      if (!f->isLib && !reach.count(f.get())) deadFuncs.push_back(f.get());
    if (!deadFuncs.empty()) {
      for (Function *d : deadFuncs) {
        auto it = std::find_if(
            mod.functions().begin(), mod.functions().end(),
            [&](const auto &up) { return up.get() == d; });
        if (it != mod.functions().end()) mod.functions().erase(it);
      }
      changed = true;
    }
    // globals referenced by the remaining (reachable) functions
    std::unordered_set<GlobalVar *> usedGlobals;
    for (auto &f : mod.functions()) {
      if (f->isLib) continue;
      for (Instruction *i : collectInstrsOf(mod, f.get()))
        for (Value *o : i->ops)
          if (auto *g = dynamic_cast<GlobalVar *>(o)) usedGlobals.insert(g);
    }
    for (auto it = mod.globals().begin(); it != mod.globals().end();) {
      if (!usedGlobals.count(it->get())) {
        it = mod.globals().erase(it);
        changed = true;
      } else
        ++it;
    }
    return changed;
  }

private:
  Layer layer_;

  static std::vector<Instruction *> collectInstrsOf(Module &mod,
                                                    Function *fn) {
    std::vector<Instruction *> out;
    std::function<void(BlockList &)> rec = [&](BlockList &list) {
      for (auto &bb : list)
        for (auto &iu : bb->instrs) {
          out.push_back(iu.get());
          Instruction *inst = iu.get();
          if (inst->op == Op::ScfWhile) {
            rec(inst->condRegion);
            rec(inst->bodyRegion);
          } else if (inst->op == Op::AffineFor) {
            rec(inst->bodyRegion);
          }
        }
    };
    rec(fn->blocks);
    return out;
  }
};

} // namespace

std::unique_ptr<Pass> makeRemoveUnusedPass(Layer l) {
  return std::make_unique<RemoveUnusedPass>(l);
}

} // namespace ir
} // namespace sakura
