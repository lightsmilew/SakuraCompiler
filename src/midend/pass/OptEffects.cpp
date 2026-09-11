// ---------------------------------------------------------------------------
// OptEffects.cpp: interprocedural side-effect summary (see OptUtil.h).
//
// LLVM's mid-end is built on inferred function attributes (`readnone`,
// `readonly`, `nounwind`, `willreturn`).  Two of its CSE wins that we lacked
// show up directly in this suite:
//
//   * `fft`'s butterfly body evaluates `x + multiply(wn, y)` and
//     `x - multiply(wn, y)` in a row.  clang emits *one* call to `multiply`;
//     we emitted two, paying for the whole recursive modular-multiply twice.
//     clang is allowed to do that only because it proved `multiply` writes
//     nothing and reads nothing whose value can change between the two calls.
//   * a call to a function that cannot write memory does not invalidate the
//     store->load forwarding table, so more loads stay forwardable.
//
// The model below is deliberately simple but sound:
//
//   * A function's effect is the union of its own accesses and those of every
//     callee, iterated to a fixed point over the call graph.  Recursion needs
//     no special case: the iteration starts optimistic (no effects) and only
//     ever grows, so `multiply` calling itself converges to "reads nothing,
//     writes nothing" instead of being poisoned into "unknown".
//   * An object whose address never leaves a direct load/store comes back as
//     `non-escaping`.  No pointer to it can exist outside the module, so the
//     only writes it can ever see are the direct module stores we enumerated;
//     in particular a call to an opaque library function cannot reach it.
//     That is what lets `mod` stay provably constant even though `main` calls
//     getarray/putarray.
// ---------------------------------------------------------------------------
#include "OptUtil.h"

#include <unordered_map>
#include <unordered_set>

namespace sakura {
namespace ir {

namespace {

// Address chains are short; the cap only stops a pathological gep cycle.
constexpr int kMaxAddrDepth = 12;

// A callee with no body (a library function) could do anything at all, which
// includes writing caller memory.
FuncEffects opaqueEffects() {
  FuncEffects e;
  e.reads = e.writes = e.readsParam = e.writesParam = e.opaque = true;
  e.terminates = false;
  return e;
}

// Does the function's own control flow contain a cycle (a loop), or a
// structured loop region?  A function with no loops at all always falls off
// the end, so - once every callee also returns - it provably terminates.
//
// This is the cheap, syntactic half of LLVM's `willreturn` inference, and it
// is enough for the recursion that matters here: `multiply` / `power` in
// fft0 are branch-only recursions (`if (b == 0) return 0; ... cur =
// multiply(a, b/2); ...`), so they come back, while the optimistic start of
// the call-graph fixed point lets the self-recursion stay "terminating"
// instead of being poisoned to "unknown".
bool bodyHasLoop(Function *f) {
  bool loop = false;
  std::function<void(BlockList &)> rec = [&](BlockList &list) {
    std::unordered_set<BasicBlock *> inList;
    for (auto &bb : list) inList.insert(bb.get());
    std::unordered_map<BasicBlock *, int> color; // 0 unseen, 1 on stack, 2 done
    std::function<void(BasicBlock *)> dfs = [&](BasicBlock *b) {
      color[b] = 1;
      if (!b->instrs.empty()) {
        Instruction *t = b->instrs.back().get();
        std::vector<BasicBlock *> succ;
        auto push = [&](Value *v) {
          if (auto *s = dynamic_cast<BasicBlock *>(v)) succ.push_back(s);
        };
        if (t->op == Op::Br && !t->ops.empty()) {
          push(t->ops[0]);
        } else if (t->op == Op::CondBr && t->ops.size() >= 3) {
          push(t->ops[1]);
          push(t->ops[2]);
        }
        for (BasicBlock *s : succ) {
          if (!inList.count(s)) continue; // leaves this list: no cycle here
          if (color[s] == 1) loop = true;
          else if (color[s] == 0) dfs(s);
        }
      }
      color[b] = 2;
    };
    for (auto &bb : list)
      if (!color[bb.get()]) dfs(bb.get());
    // Nested regions: any loop region is a potential non-terminating cycle.
    for (auto &bb : list)
      for (auto &iu : bb->instrs) {
        Instruction *i = iu.get();
        if (i->op == Op::ScfWhile) {
          loop = true;
          rec(i->condRegion);
          rec(i->bodyRegion);
        } else if (i->op == Op::AffineFor) {
          loop = true;
          rec(i->bodyRegion);
        }
      }
  };
  rec(f->blocks);
  return loop;
}

} // namespace

void Effects::compute(Module &mod) {
  fx_.clear();
  written_.clear();
  escaped_.clear();

  std::vector<Instruction *> all = collectInstrs(mod);

  // ---- which objects' addresses could be handed to someone else ----------
  // `v` is used only as (part of) a memory address when every use is a load,
  // a store *address*, or a gep that itself satisfies this.  A store *value*,
  // a call argument or a return hands the pointer out, which makes the object
  // reachable by code we cannot analyse.
  std::unordered_map<Value *, std::vector<Instruction *>> users;
  for (Instruction *i : all)
    for (Value *o : i->ops) {
      if (dynamic_cast<BasicBlock *>(o) || dynamic_cast<Function *>(o))
        continue;
      users[o].push_back(i);
    }
  std::function<bool(Value *, int)> addrOnly = [&](Value *v, int d) -> bool {
    if (d > kMaxAddrDepth) return false;
    auto it = users.find(v);
    if (it == users.end()) return true;
    for (Instruction *u : it->second) {
      if (u->op == Op::Gep) {
        if (!addrOnly(u, d + 1)) return false;
      } else if (u->op == Op::Load) {
        continue; // v is the address being read
      } else if (u->op == Op::Store && u->ops.size() >= 2 && u->ops[1] == v) {
        continue; // v is the address being written
      } else {
        return false;
      }
    }
    return true;
  };
  for (auto &g : mod.globals())
    if (!addrOnly(g.get(), 0)) escaped_.insert(g.get());
  for (Instruction *i : all)
    if (i->op == Op::Alloca && !addrOnly(i, 0)) escaped_.insert(i);

  // ---- per-function local summary + call-graph edges ---------------------
  std::unordered_map<Function *, std::vector<Function *>> callees;
  std::unordered_map<Function *, bool> ownDag; // no loop of its own
  for (auto &up : mod.functions()) {
    Function *f = up.get();
    if (f->isLib) {
      fx_[f] = opaqueEffects();
      continue;
    }
    FuncEffects e;
    // Optimistic start for `terminates`: a function with no loop of its own
    // falls through unless one of its callees refuses to come back.  Starting
    // true and only ever clearing it makes the fixed point below prove
    // self-recursive functions (`multiply`) terminate rather than giving up.
    ownDag[f] = !bodyHasLoop(f);
    e.terminates = ownDag[f];
    auto &cal = callees[f];
    walkInstrs(f, [&](BasicBlock &, Instruction &inst) {
      if (inst.op == Op::Load && !inst.ops.empty()) {
        e.reads = true;
        AddrInfo a = addressOf(inst.ops[0]);
        if (a.kind == RootKind::Global && a.root)
          e.readGlobals.insert(static_cast<GlobalVar *>(a.root));
        else if (a.kind != RootKind::Alloca)
          e.readsParam = true; // param-rooted, or an address we cannot name
      } else if (inst.op == Op::Store && inst.ops.size() >= 2) {
        AddrInfo a = addressOf(inst.ops[1]);
        if (a.kind == RootKind::Global && a.root) {
          e.writes = true;
          e.writeGlobals.insert(static_cast<GlobalVar *>(a.root));
        } else if (a.kind == RootKind::Alloca && a.root) {
          // A private slot: invisible to every other function, so the store
          // cannot be observed.  Only a slot whose address leaked can be read
          // after the callee returns.
          if (escaped_.count(a.root)) {
            e.writes = true;
            e.writesParam = true;
          }
        } else {
          e.writes = true;
          e.writesParam = true;
        }
      } else if (inst.op == Op::Call && !inst.ops.empty()) {
        if (auto *c = dynamic_cast<Function *>(inst.ops[0])) {
          if (c->isLib)
            e.opaque = true;
          else
            cal.push_back(c);
        } else {
          e.opaque = true; // indirect call
        }
      }
    });
    fx_[f] = e;
  }

  // ---- fixed point over the call graph -----------------------------------
  bool changed = true;
  for (int guard = 0; changed && guard < 128; ++guard) {
    changed = false;
    for (auto &kv : callees) {
      FuncEffects &e = fx_[kv.first];
      for (Function *c : kv.second) {
        auto it = fx_.find(c);
        if (it == fx_.end()) {
          if (!e.opaque) {
            e.opaque = true;
            changed = true;
          }
          continue;
        }
        const FuncEffects &ce = it->second;
        auto up1 = [&](bool &dst, bool src) {
          if (src && !dst) {
            dst = true;
            changed = true;
          }
        };
        up1(e.reads, ce.reads);
        up1(e.writes, ce.writes);
        up1(e.readsParam, ce.readsParam);
        up1(e.writesParam, ce.writesParam);
        up1(e.opaque, ce.opaque);
        for (GlobalVar *g : ce.readGlobals)
          if (e.readGlobals.insert(g).second) changed = true;
        for (GlobalVar *g : ce.writeGlobals)
          if (e.writeGlobals.insert(g).second) changed = true;
      }
      // `terminates` is not a monotone *growth* like the flags above: it is
      // recomputed from scratch each round, starting optimistic and only ever
      // going false as callees are found not to return.
      bool t = ownDag[kv.first] && !e.opaque;
      if (t)
        for (Function *c : kv.second) {
          auto it = fx_.find(c);
          if (it == fx_.end() || !it->second.terminates) {
            t = false;
            break;
          }
        }
      if (t != e.terminates) {
        e.terminates = t;
        changed = true;
      }
    }
  }

  // ---- module-wide write set ---------------------------------------------
  for (auto &kv : fx_)
    for (GlobalVar *g : kv.second.writeGlobals) written_.insert(g);
}

const FuncEffects *Effects::of(Function *f) const {
  if (!f) return nullptr;
  auto it = fx_.find(f);
  return it == fx_.end() ? nullptr : &it->second;
}

bool Effects::globalEscapes(GlobalVar *g) const { return escaped_.count(g) != 0; }

bool Effects::callWritesNothing(Function *f) const {
  const FuncEffects *e = of(f);
  return e && !e->writes && !e->writesParam && !e->opaque;
}

bool Effects::callIsValue(Function *f) const {
  const FuncEffects *e = of(f);
  if (!e || e->opaque) return false;
  // Anything that can write, or that reads memory owned by the caller, makes
  // the result depend on more than the arguments.
  if (e->writes || e->writesParam || e->readsParam) return false;
  // Every global it reads must be provably immutable: either no module
  // function ever writes it, or it escapes (in which case an opaque callee
  // might, and we give up).
  for (GlobalVar *g : e->readGlobals)
    if (written_.count(g) || escaped_.count(g)) return false;
  return true;
}

bool Effects::callMayWrite(Function *f, const AddrInfo &a) const {
  const FuncEffects *e = of(f);
  if (!e) return true;      // a callee we have no summary for
  if (e->opaque) return true;
  // Reads-only: the call leaves every memory location exactly as it was.
  if (!e->writes && !e->writesParam) return false;
  switch (a.kind) {
  case RootKind::Global: {
    auto *g = static_cast<GlobalVar *>(a.root);
    if (!g) return true;
    if (e->writeGlobals.count(g)) return true;
    // The callee only names globals it can see, so any *other* global is
    // reachable from it solely through a pointer argument - which needs the
    // address to have escaped to someone.
    if (!e->writesParam) return false;
    return escaped_.count(g) != 0;
  }
  case RootKind::Alloca:
    // A private slot is just a stack address.  Another function can only
    // write it if a pointer to it was passed out.
    return a.root && escaped_.count(a.root) != 0;
  default:
    // Parameter-rooted, a pointer loaded from memory, or something we could
    // not classify: assume it can be reached.
    return true;
  }
}

} // namespace ir
} // namespace sakura
