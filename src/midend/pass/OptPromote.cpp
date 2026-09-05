// ---------------------------------------------------------------------------
// OptPromote.cpp: mem2reg - promote scalar memory slots to SSA registers.
//
// The mid-end IR is memory-resident: a local scalar is a memref.alloca slot
// (i32/f32, n==1) read with memref.load and written with memref.store.  A
// loop counter therefore reloads and re-stores its slot on every iteration,
// and the back-end cannot keep it in a register.  This pass lifts every
// *non-escaping* scalar slot into SSA form:
//
//   * slot addresses must be used only as the direct address of load/store
//     (no gep, no call/return/terminator use) - the slot never escapes,
//   * every load must be guaranteed to see a store first along all paths
//     (slots that can be read before any write keep their memory semantics;
//     SysY leaves those reads undefined so we must not invent a value),
//   * definitions that reach a block along several paths are merged by a
//     cf.phi placed at the block's iterated dominance frontier,
//   * loads are rewritten to read the reaching definition directly and the
//     slot (alloca + all its loads/stores) is erased.
//
// The pass runs at the flat cf layer where control flow is explicit (a
// block's predecessors are known from cf.br / cf.cond_br), so phi placement
// and the SSA rename are the classic Cytron / Cooper-Harvey-Kennedy
// algorithms.  Immediate dominators come from the Lengauer-Tarjan algorithm.
// Phis survive into instruction selection, which lowers them to register
// copies at the end of every predecessor block.
// ---------------------------------------------------------------------------
#include "Opt.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "OptUtil.h"

namespace sakura {
namespace ir {
namespace {

// ---- per-slot events ------------------------------------------------------
struct Ev {
  Instruction *inst = nullptr; // the load or the store
  int pos = 0;                 // position inside its block (event ordering)
  bool isStore = false;
};

struct Slot {
  Instruction *alloca = nullptr; // Op::Alloca with elem i32/f32 and n == 1
  Type ty = Type::I32;           // slot scalar type (alloca->elem)
  std::vector<Ev> events;        // all loads/stores, function-wide
  bool escaped = false;          // address used outside a direct load/store
};

// ---- CFG helpers over the flat block list --------------------------------
class Cfg {
public:
  int n = 0;
  std::vector<BasicBlock *> blk;             // index -> block
  std::unordered_map<BasicBlock *, int> idx; // block -> index
  std::vector<std::vector<int>> preds, succs;

  std::vector<int> idom;      // immediate dominator (idom[0] == 0)
  std::vector<int> tin, tout; // dom-tree DFS numbers: dom(a,b) via intervals
  std::vector<std::vector<int>> domKids; // dominator-tree children
  std::vector<std::vector<int>> df;      // dominance frontier per block
  int dfsClock = 0;

  explicit Cfg(Function &f) {
    for (auto &b : f.blocks) {
      idx[b.get()] = n;
      blk.push_back(b.get());
      ++n;
    }
    preds.assign(n, {});
    succs.assign(n, {});
    for (int b = 0; b < n; ++b) {
      BasicBlock *bb = blk[b];
      if (bb->instrs.empty()) continue;
      Instruction *t = bb->instrs.back().get();
      if (t->op == Op::Br) {
        auto *target = dynamic_cast<BasicBlock *>(t->ops[0]);
        addEdge(b, idx.at(target));
      } else if (t->op == Op::CondBr) {
        auto *ta = dynamic_cast<BasicBlock *>(t->ops[1]);
        auto *tb = dynamic_cast<BasicBlock *>(t->ops[2]);
        addEdge(b, idx.at(ta));
        addEdge(b, idx.at(tb));
      }
    }
  }

  void addEdge(int b, int y) {
    for (int s : succs[b])
      if (s == y) return; // no duplicate edges
    succs[b].push_back(y);
    preds[y].push_back(b);
  }

  // Lengauer-Tarjan immediate dominators (classic algorithm, near-linear).
  // The CFG can contain long chains, so both the preorder DFS and the
  // union-find path compression are written iteratively.
  void computeDominators() {
    idom.assign(n, -1);
    if (n == 0) return;

    // ---- 1) iterative DFS preorder -----------------------------------
    // num[b]  : preorder number of block b (0 = unreachable from entry)
    // vertex[i]: block holding preorder number i
    // par[i]  : preorder number of the DFS parent of vertex[i]
    std::vector<int> num(n, 0), vertex(n + 1, -1), par(n + 1, 0);
    int ndfs = 0;
    {
      num[0] = ++ndfs;
      vertex[ndfs] = 0;
      std::vector<std::pair<int, size_t>> stk; // (block, next succ index)
      stk.push_back({0, 0});
      while (!stk.empty()) {
        int b = stk.back().first;
        size_t &ci = stk.back().second;
        if (ci < succs[b].size()) {
          int y = succs[b][ci++];
          if (num[y] != 0) continue;
          num[y] = ++ndfs;
          vertex[ndfs] = y;
          par[ndfs] = num[b];
          stk.push_back({y, 0});
        } else {
          stk.pop_back();
        }
      }
    }
    const int N = ndfs;
    if (N == 1) { // single-block function: only the entry is reachable
      idom[0] = 0;
      finishDominatorTree();
      return;
    }

    // ---- 2) semidominators, reverse preorder --------------------------
    // The union-find forest is keyed by preorder number; lab[v] tracks the
    // vertex on v's ancestor path with the smallest semidominator.
    std::vector<int> semi(N + 1), idomT(N + 1, 0);
    std::vector<int> anc(N + 1, 0), lab(N + 1, 0);
    std::vector<std::vector<int>> bucket(N + 1);
    for (int i = 1; i <= N; ++i) {
      semi[i] = i;
      lab[i] = i;
    }
    // eval(v): vertex with minimum semi[] on the path from v to the root of
    // its current forest component; compresses the path in the process.
    auto eval = [&](int v) -> int {
      if (anc[v] == 0) return lab[v];
      std::vector<int> chain; // v, anc[v], ... up to the child of the root
      for (int c = v; anc[c] != 0; c = anc[c]) chain.push_back(c);
      for (size_t i = chain.size() - 1; i >= 1; --i) {
        int below = chain[i - 1], above = chain[i]; // top-down compression
        if (semi[lab[above]] < semi[lab[below]]) lab[below] = lab[above];
        anc[below] = anc[above];
      }
      return lab[v];
    };

    for (int i = N; i >= 2; --i) {
      int w = vertex[i];
      int s = i;
      for (int p : preds[w]) { // p is a block index
        int d = num[p];
        if (d == 0) continue; // edge from unreachable code
        int u = (d < i) ? d : semi[eval(d)];
        if (u < s) s = u;
      }
      semi[i] = s;
      bucket[s].push_back(i);
      anc[i] = par[i]; // link i under its DFS parent
      for (int v : bucket[par[i]]) { // resolve the parent's bucket now
        int u = eval(v);
        idomT[v] = (semi[u] < semi[v]) ? u : par[i];
      }
      bucket[par[i]].clear();
    }
    // ---- 3) refine into true immediate dominators ---------------------
    for (int i = 2; i <= N; ++i)
      if (idomT[i] != semi[i]) idomT[i] = idomT[idomT[i]];
    idom[0] = 0; // the entry dominates itself (sentinel for walkers)
    for (int i = 2; i <= N; ++i)
      idom[vertex[i]] = vertex[idomT[i]];

    finishDominatorTree();
  }

  // Dominator-tree DFS numbers (dominates via intervals) and the dominance
  // frontier.  Shared by every dominator backend.
  void finishDominatorTree() {
    tin.assign(n, 0);
    tout.assign(n, 0);
    domKids.assign(n, {});
    dfsClock = 0;
    for (int c = 1; c < n; ++c)
      if (idom[c] >= 0) domKids[(size_t)idom[c]].push_back(c);
    // iterative DFS of the dominator tree (dom trees can be deep chains)
    std::vector<std::pair<int, size_t>> stk;
    stk.push_back({0, 0});
    while (!stk.empty()) {
      int b = stk.back().first;
      size_t &ci = stk.back().second;
      if (ci == 0) tin[b] = ++dfsClock; // first visit
      if (ci < domKids[b].size()) {
        int c = domKids[b][ci++];
        stk.push_back({c, 0});
      } else {
        tout[b] = dfsClock;
        stk.pop_back();
      }
    }

    // dominance frontier: y in df(b) when b dominates a predecessor of y but
    // not y itself.  Walk every (pred, y) edge up the dominator chain.
    df.assign(n, {});
    for (int y = 0; y < n; ++y) {
      for (int p : preds[(size_t)y]) {
        int runner = p;
        while (runner >= 0 && runner != idom[y]) {
          df[(size_t)runner].push_back(y);
          runner = (runner == 0) ? -1 : idom[runner]; // idom[0] == 0
        }
      }
    }
  }

  bool reachable(int b) const { return b < n && idom[b] >= 0; }

  // true when block a dominates block b (a == b included)
  bool dominates(int a, int b) const {
    if (a < 0 || b < 0 || a >= n || b >= n) return false;
    return tin[a] <= tin[b] && tout[b] <= tout[a];
  }
};

class Mem2RegPass final : public Pass {
public:
  const char *name() const override { return "mem2reg"; }
  Layer inLayer() const override { return Layer::Cf; }
  bool run(Module &mod) override {
    bool any = false;
    for (auto &f : mod.functions()) {
      if (f->isLib || f->blocks.empty()) continue;
      if (hasStructuredOps(*f)) continue; // defensive: not the flat cf yet
      if (promote(mod, *f)) any = true;
    }
    return any;
  }

private:
  static bool hasStructuredOps(Function &f) {
    for (auto &bb : f.blocks)
      for (auto &iu : bb->instrs)
        if (iu->op == Op::ScfWhile || iu->op == Op::AffineFor) return true;
    return false;
  }

  // ---------------------------------------------------------------- helpers
  std::unordered_map<Instruction *, BasicBlock *> owner;
  BasicBlock *findBlock(Instruction *inst) const {
    auto it = owner.find(inst);
    return it == owner.end() ? nullptr : it->second;
  }

  // ---- identify candidate slots ----------------------------------------
  std::vector<Slot> collectSlots(Function &f) {
    std::unordered_map<Instruction *, int> byAlloca; // alloca -> slot index
    std::unordered_map<Instruction *, int> posOf;
    for (auto &bb : f.blocks)
      for (size_t j = 0; j < bb->instrs.size(); ++j)
        posOf[bb->instrs[j].get()] = (int)j;

    std::vector<Slot> slots;
    for (auto &bb : f.blocks)
      for (auto &iu : bb->instrs)
        if (iu->op == Op::Alloca && (iu->elem == Type::I32 ||
                                     iu->elem == Type::F32)) {
          Slot s;
          s.alloca = iu.get();
          s.ty = iu->elem;
          slots.push_back(s);
        }
    for (size_t i = 0; i < slots.size(); ++i) byAlloca[slots[i].alloca] = (int)i;

    // classify the uses of each scalar alloca across the function
    for (auto &bb : f.blocks) {
      for (auto &iu : bb->instrs) {
        Instruction *inst = iu.get();
        for (size_t k = 0; k < inst->ops.size(); ++k) {
          Value *o = inst->ops[k];
          auto *asI = dynamic_cast<Instruction *>(o);
          if (!asI || asI->op != Op::Alloca) continue;
          auto it = byAlloca.find(asI);
          if (it == byAlloca.end()) continue; // not a scalar slot
          Slot &s = slots[it->second];
          if (inst->op == Op::Load && k == 0) {
            s.events.push_back(Ev{inst, posOf.at(inst), false});
          } else if (inst->op == Op::Store && k == 1) {
            s.events.push_back(Ev{inst, posOf.at(inst), true});
          } else {
            s.escaped = true; // gep / call / ret / cond / store-as-value ...
          }
        }
      }
    }
    // Trust the values that actually flow through the slot rather than the
    // alloca's own elem stamp: some producers label an f32 accumulator alloca
    // as i32, which would make every phi below type i32 while merging f32
    // incoming values (a miscompile once the backend lowers the phi copies).
    // A slot is always touched by loads/stores of one uniform type.
    for (auto &s : slots) {
      for (const Ev &e : s.events) {
        Type t = e.isStore ? e.inst->ops[0]->ty : e.inst->ty;
        if (t == Type::I32 || t == Type::F32) {
          s.ty = t;
          break;
        }
      }
    }
    return slots;
  }

  // A slot may only be promoted when no load can observe an unwritten slot:
  // every load must sit strictly below a store, or textually after one in its
  // own block.  Anything else keeps its memory semantics (left untouched).
  bool allLoadsInitialized(const Slot &s, const Cfg &cfg) const {
    for (const Ev &load : s.events) {
      if (load.isStore) continue;
      BasicBlock *lb = findBlock(load.inst);
      int L = lb ? cfg.idx.at(lb) : -1;
      if (L < 0) return false;
      bool ok = false;
      for (const Ev &st : s.events) {
        if (!st.isStore) continue;
        BasicBlock *db = findBlock(st.inst);
        int D = db ? cfg.idx.at(db) : -1;
        if (D < 0) continue;
        if (D == L) {
          if (st.pos < load.pos) { ok = true; break; }
        } else if (cfg.dominates(D, L)) {
          ok = true;
          break;
        }
      }
      if (!ok) return false;
    }
    return true;
  }

  // ---- promote one slot ---------------------------------------------------
  // All mutations are staged: on success the caller gets the loads to rewrite
  // and the instructions to erase; on failure the slot's phis are removed and
  // nothing else happens (all-or-nothing).
  bool promoteSlot(Module &mod, Function &f, const Cfg &cfg, const Slot &s,
                   std::unordered_map<Instruction *, Value *> &rep,
                   std::unordered_set<Instruction *> &eraseSet) {
    int n = cfg.n;
    // per-block events for this slot, in program order
    std::vector<std::vector<Ev>> blockEvents((size_t)n);
    std::vector<int> defBlocks;
    for (const Ev &e : s.events) {
      int b = cfg.idx.at(findBlock(e.inst));
      blockEvents[(size_t)b].push_back(e);
    }
    for (int b = 0; b < n; ++b)
      std::sort(blockEvents[(size_t)b].begin(), blockEvents[(size_t)b].end(),
                [](const Ev &x, const Ev &y) { return x.pos < y.pos; });
    for (int b = 0; b < n; ++b)
      for (const Ev &e : blockEvents[(size_t)b])
        if (e.isStore) {
          defBlocks.push_back(b);
          break;
        }
    if (defBlocks.empty()) return false;

    // phi placement: iterated dominance frontier of the store blocks
    std::vector<bool> hasPhi((size_t)n, false);
    std::vector<Instruction *> blockPhi((size_t)n, nullptr);
    {
      std::vector<int> work = defBlocks;
      std::vector<char> inWork((size_t)n, 0);
      for (int d : defBlocks) inWork[(size_t)d] = 1;
      while (!work.empty()) {
        int x = work.back();
        work.pop_back();
        inWork[(size_t)x] = 0;
        for (int y : cfg.df[(size_t)x]) {
          if (hasPhi[(size_t)y]) continue;
          hasPhi[(size_t)y] = true;
          auto ph = std::make_unique<Instruction>(Op::Phi, s.ty);
          Instruction *phP = ph.get();
          BasicBlock *yb = cfg.blk[(size_t)y];
          auto it = yb->instrs.begin();
          while (it != yb->instrs.end() && (*it)->op == Op::Phi) ++it;
          yb->instrs.insert(it, std::move(ph));
          blockPhi[(size_t)y] = phP;
          if (!inWork[(size_t)y]) {
            work.push_back(y);
            inWork[(size_t)y] = 1;
          }
        }
      }
    }

    // ---- SSA rename over the dominator tree ----------------------------
    // stack: the current reaching definition for the slot on this path.
    // A store pushes a fresh definition; a load reads the top; successor phi
    // entries record the top at the end of the block.
    //
    // Iterative (explicit enter/exit frames): a block's pushes stay on the
    // stack while its dominator-tree children are processed and are popped by
    // its exit frame, exactly like the recursive formulation.
    std::vector<Value *> stack;
    bool failed = false;
    std::unordered_map<Instruction *, Value *> localRep;
    struct Frame {
      int b;
      bool exit;
      size_t mark; // stack size at entry (used by the exit frame)
    };
    std::vector<Frame> work;
    work.push_back({0, false, 0});
    while (!work.empty() && !failed) {
      Frame fr = work.back();
      work.pop_back();
      int b = fr.b;
      if (fr.exit) {
        stack.resize(fr.mark);
        continue;
      }
      size_t mark = stack.size();
      if (hasPhi[(size_t)b]) stack.push_back(blockPhi[(size_t)b]);
      for (const Ev &e : blockEvents[(size_t)b]) {
        if (e.isStore) {
          stack.push_back(e.inst->ops[0]);
        } else {
          // a load of the slot: read the current reaching definition
          if (stack.empty()) {
            failed = true; // unreachable given allLoadsInitialized
            break;
          }
          localRep[e.inst] = stack.back();
        }
      }
      if (failed) break;
      for (int y : cfg.succs[(size_t)b]) {
        if (!hasPhi[(size_t)y]) continue;
        Value *inc = stack.empty() ? nullptr : stack.back();
        if (!inc) {
          // no reaching definition on this edge.  The filter guarantees every
          // read of the merge goes down a path that carries a definition, so
          // this placeholder is unobservable; use 0 to stay deterministic.
          inc = (s.ty == Type::F32) ? (Value *)mod.constFloat(0.0f)
                                    : (Value *)mod.constInt(0);
        }
        Instruction *ph = blockPhi[(size_t)y];
        ph->ops.push_back((Value *)cfg.blk[(size_t)b]);
        ph->ops.push_back(inc);
      }
      // schedule the exit (below the children) then the children
      work.push_back({b, true, mark});
      for (int c : cfg.domKids[(size_t)b]) work.push_back({c, false, 0});
    }

    if (!failed) {
      // every phi must carry one entry per (reachable) predecessor; the
      // function entry can never host a phi (no copy could define it before
      // the first use).
      for (int b = 0; b < n && !failed; ++b) {
        if (!hasPhi[(size_t)b]) continue;
        if (b == 0) {
          failed = true;
          break;
        }
        int want = 0;
        for (int p : cfg.preds[(size_t)b])
          if (cfg.reachable(p)) ++want;
        if ((int)blockPhi[(size_t)b]->ops.size() / 2 != want) failed = true;
      }
    }
    if (failed) {
      // roll back this slot's phis; the memory ops stay untouched
      for (int b = 0; b < n; ++b) {
        if (!hasPhi[(size_t)b]) continue;
        BasicBlock *yb = cfg.blk[(size_t)b];
        for (auto it = yb->instrs.begin(); it != yb->instrs.end();) {
          if (it->get() == blockPhi[(size_t)b]) it = yb->instrs.erase(it);
          else ++it;
        }
      }
      return false;
    }

    for (auto &kv : localRep) rep[kv.first] = kv.second;
    for (const Ev &e : s.events) eraseSet.insert(e.inst);
    eraseSet.insert(s.alloca);
    return true;
  }

  // ---- fold trivial phis --------------------------------------------------
  // A phi whose incoming values are all the same collapses to that value.
  // Runs to a fixpoint (one fold can expose another).  A phi that only
  // receives itself is left alone.
  void foldPhis(Module &mod, Function &f,
                std::unordered_set<Instruction *> &eraseSet) {
    bool again = true;
    while (again) {
      again = false;
      for (auto &bb : f.blocks) {
        for (auto it = bb->instrs.begin(); it != bb->instrs.end(); ++it) {
          Instruction *ph = it->get();
          if (ph->op != Op::Phi) continue;
          Value *uniq = nullptr;
          bool same = true;
          for (size_t k = 1; k < ph->ops.size() && same; k += 2) {
            Value *v = ph->ops[k];
            if (!uniq) {
              uniq = v;
            } else if (v != uniq) {
              const ConstantInt *a = asConstInt(v), *b = asConstInt(uniq);
              const ConstantFloat *fa = asConstFloat(v), *fb = asConstFloat(uniq);
              if (!(a && b && a->v == b->v) &&
                  !(fa && fb && fa->bits() == fb->bits()))
                same = false;
            }
          }
          if (!same || !uniq || uniq == (Value *)ph) continue;
          replaceAllUses(mod, ph, uniq);
          it = bb->instrs.erase(it); // iterator valid: points at next
          eraseSet.erase(ph);
          again = true;
          break; // block changed; rescan from the top
        }
        if (again) break;
      }
    }
  }

  // ---- per function ------------------------------------------------------
  bool promote(Module &mod, Function &f) {
    owner.clear();
    for (auto &bb : f.blocks)
      for (auto &iu : bb->instrs) owner[iu.get()] = bb.get();

    Cfg cfg(f);
    cfg.computeDominators();
    for (int b = 0; b < cfg.n; ++b)
      if (b != 0 && !cfg.reachable(b)) return false; // leftover dead blocks

    std::vector<Slot> slots = collectSlots(f);
    if (slots.empty()) return false;

    bool any = false;
    std::unordered_map<Instruction *, Value *> rep;
    std::unordered_set<Instruction *> eraseSet;
    for (Slot &s : slots) {
      if (s.escaped || s.alloca->n != 1) continue;
      if (!allLoadsInitialized(s, cfg)) continue;
      bool hasLoad = false, hasStore = false;
      for (const Ev &e : s.events)
        (e.isStore ? hasStore : hasLoad) = true;
      if (!hasLoad || !hasStore) continue; // write-only / read-only slots
      if (promoteSlot(mod, f, cfg, s, rep, eraseSet)) any = true;
    }
    if (!any) return false;

    // redirect every load to its reaching definition, collapse trivial phis,
    // then erase the promoted slots (allocas + loads + stores).
    //
    // Order matters: folding a phi can rewrite *other* phis' incoming values
    // to a load that is about to be erased, so the load->def rewrite must run
    // again after foldPhis.  Only then is it safe to drop the slots.
    for (auto &kv : rep) replaceAllUses(mod, kv.first, kv.second);
    foldPhis(mod, f, eraseSet);
    for (auto &kv : rep) replaceAllUses(mod, kv.first, kv.second);
    for (auto &bb : f.blocks)
      for (auto it = bb->instrs.begin(); it != bb->instrs.end();)
        if (eraseSet.count(it->get())) it = bb->instrs.erase(it);
        else ++it;
    return true;
  }
};

} // namespace

std::unique_ptr<Pass> makeMem2RegPass() {
  return std::make_unique<Mem2RegPass>();
}

} // namespace ir
} // namespace sakura
