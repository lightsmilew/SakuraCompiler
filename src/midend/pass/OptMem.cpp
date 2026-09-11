// ---------------------------------------------------------------------------
// OptMem.cpp: memory forwarding + local CSE (mem-cse).
//
// The IR is memory-resident, so the biggest source of redundant work is
// scalar/array traffic through memref slots.  This pass walks each block list
// in layout order and, along "extended basic block" chains (blocks with a
// single in-list predecessor that appears earlier), maintains:
//
//   * the value currently held in each memory location (store->load
//     forwarding, load->load CSE, dead-store removal), and
//   * a table of pure arithmetic results (local CSE for add/mul/cmp/...).
//
// State is reset whenever control can arrive from more than one place (joins,
// loop headers, region entry) so all rewrites are dominance-safe.  The memory
// model mirrors OptUtil: private alloca slots are tracked exactly; a store
// whose target may alias anything outside (param-rooted / dynamic addresses,
// globals) or a call invalidates the affected entries.
// ---------------------------------------------------------------------------
#include "Opt.h"

#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>

#include "OptUtil.h"

namespace sakura {
namespace ir {

namespace {

class MemCsePass final : public Pass {
public:
  explicit MemCsePass(Layer l) : layer_(l) {}
  const char *name() const override { return "mem-cse"; }
  Layer inLayer() const override { return layer_; }

  bool run(Module &mod) override {
    bool any = false;
    fx_.compute(mod);
    for (auto &f : mod.functions()) {
      if (f->isLib) continue;
      process(mod, f->blocks, &any);
    }
    return any;
  }

private:
  Layer layer_;
  Effects fx_;

  // -----------------------------------------------------------------------
  // key materialisation
  // -----------------------------------------------------------------------

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

  // canonical key for an arithmetic expression (commutative ops sorted)
  static std::string exprKey(Op op, Cond cond, const std::vector<Value *> &os) {
    std::string k =
        std::to_string((int)op) + ":" + std::to_string((int)cond) + ":";
    std::vector<std::string> parts;
    for (Value *o : os) {
      if (isStructural(o)) return std::string(); // never canonical
      parts.push_back(valueKey(o));
    }
    bool comm = op == Op::Add || op == Op::Mul || op == Op::FAdd ||
                op == Op::FMul ||
                (op == Op::FCmp && (cond == Cond::Eq || cond == Cond::Ne));
    if (comm && parts.size() == 2 && parts[0] > parts[1])
      std::swap(parts[0], parts[1]);
    for (auto &p : parts) k += p + ";";
    return k;
  }

  // Key for a call to a value-like function: the callee plus the argument
  // values.  `exprKey` cannot be used because the callee is a structural
  // operand and would make the key empty.
  static std::string callKey(Instruction *call) {
    char b[32];
    snprintf(b, sizeof b, "C:%p:", (void *)call->ops[0]);
    std::string k = b;
    for (size_t i = 1; i < call->ops.size(); ++i)
      k += valueKey(call->ops[i]) + ";";
    return k;
  }

  bool isPureExprOp(Op op) const {
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

  // -----------------------------------------------------------------------
  // main traversal
  // -----------------------------------------------------------------------

  void process(Module &mod, BlockList &list, bool *any) {
    // predecessor counts within this list
    std::unordered_map<BasicBlock *, int> preds;
    std::unordered_map<BasicBlock *, int> index;
    for (size_t i = 0; i < list.size(); ++i) {
      index[list[i].get()] = (int)i;
      preds[list[i].get()] = 0;
    }
    for (auto &bb : list) {
      Instruction *last =
          bb->instrs.empty() ? nullptr : bb->instrs.back().get();
      if (!last) continue;
      auto bump = [&](Value *t) {
        if (auto *b = dynamic_cast<BasicBlock *>(t))
          if (preds.count(b)) preds[b]++;
      };
      switch (last->op) {
      case Op::Br: bump(last->ops[0]); break;
      case Op::CondBr:
        if (last->ops.size() >= 3) {
          bump(last->ops[1]);
          bump(last->ops[2]);
        }
        break;
      case Op::ScfCondition:
        if (last->ops.size() >= 3) {
          bump(last->ops[1]);
          bump(last->ops[2]);
        }
        break;
      default: break;
      }
    }

    // forward propagation state
    std::unordered_map<std::string, Value *> exprs; // pure arithmetic results
    std::unordered_map<std::string, Value *> mem;   // memory contents

    // A loop's exit block (ops[0] of an scf.while / affine.for anywhere in
    // this list) is reached from inside the loop as well, an implicit edge
    // that is invisible to the predecessor count above (the terminator lives
    // in a nested region list).  The loop body may write anything, so such
    // blocks are always treated as joins and state is reset before them.
    std::unordered_set<BasicBlock *> loopExit;
    for (auto &bb : list)
      for (auto &iu : bb->instrs) {
        Instruction *i = iu.get();
        if ((i->op == Op::ScfWhile || i->op == Op::AffineFor) &&
            i->ops.size() >= 1)
          if (auto *ex = dynamic_cast<BasicBlock *>(i->ops[0]))
            if (index.count(ex)) loopExit.insert(ex);
      }

    for (auto &bb : list) {
      // EBB continuation test: exactly one in-list pred placed earlier.
      //
      // Memory state may only cross a block boundary along a *straight
      // line*: the unique predecessor must end in an unconditional br.  When
      // the predecessor ends in a conditional branch the other arm may reach
      // a common join, and forwarding the predecessor's end-state into this
      // block would then be dominance-unsafe (the join would inherit stale
      // slot contents).  Conservative reset keeps every rewrite sound.
      bool singlePred = preds[bb.get()] == 1;
      BasicBlock *onlyPred = nullptr;
      if (singlePred) {
        for (auto &pb : list) {
          Instruction *last =
              pb->instrs.empty() ? nullptr : pb->instrs.back().get();
          if (!last) continue;
          auto isTgt = [&](Value *t) { return t == (Value *)bb.get(); };
          if ((last->op == Op::Br && isTgt(last->ops[0])) ||
              (last->op == Op::CondBr && last->ops.size() >= 3 &&
               (isTgt(last->ops[1]) || isTgt(last->ops[2]))) ||
              (last->op == Op::ScfCondition && last->ops.size() >= 3 &&
               (isTgt(last->ops[1]) || isTgt(last->ops[2])))) {
            onlyPred = pb.get();
            break;
          }
        }
        if (!onlyPred || index[onlyPred] >= index[bb.get()]) singlePred = false;
        if (onlyPred) {
          Instruction *pt = onlyPred->instrs.empty()
                                ? nullptr
                                : onlyPred->instrs.back().get();
          if (!pt || pt->op != Op::Br) singlePred = false; // straight-line only
          // State is a single forward map threaded through *layout order*.
          // It therefore holds the end-state of the previous block in the
          // list, which is the predecessor's end-state only when the
          // predecessor is laid out immediately before this block.  Any block
          // in between (e.g. an unreachable leftover sitting between the
          // entry and a branch target, whose pred-count is 0 yet which still
          // repopulates the map as it is walked) would otherwise leak its own
          // slot contents into `bb`.  Requiring adjacency keeps the inherited
          // map exactly equal to the unique predecessor's end-state.
          if (onlyPred && index[onlyPred] + 1 != index[bb.get()])
            singlePred = false;
        }
      }
      if (!singlePred || loopExit.count(bb.get())) {
        exprs.clear();
        mem.clear();
      }
      processBlock(mod, *bb, exprs, mem, any);
    }
  }

  // Straight-line processing of one block.  `exprs` / `mem` carry the state
  // inherited from the block's unique predecessor (or empty).
  void processBlock(Module &mod, BasicBlock &bb,
                    std::unordered_map<std::string, Value *> &exprs,
                    std::unordered_map<std::string, Value *> &mem,
                    bool *any) {
    auto &instrs = bb.instrs;
    size_t i = 0;
    while (i < instrs.size()) {
      Instruction *inst = instrs[i].get();
      Op op = inst->op;

      // region ops: process nested lists independently, then the block ends
      if (op == Op::ScfWhile) {
        process(mod, inst->condRegion, any);
        process(mod, inst->bodyRegion, any);
        return; // terminator
      }
      if (op == Op::AffineFor) {
        process(mod, inst->bodyRegion, any);
        return; // terminator
      }
      if (isTerminator(op)) return;

      if (op == Op::Load) {
        Value *addr = inst->ops[0];
        std::string key = memKey(addr);
        auto it = mem.find(key);
        if (it != mem.end() && it->second != (Value *)inst) {
          replaceAllUses(mod, inst, it->second);
          instrs.erase(instrs.begin() + i);
          *any = true;
        } else {
          mem[key] = inst; // a later identical load re-reads the same content
          ++i;
        }
      } else if (op == Op::Store) {
        Value *val = inst->ops[0];
        Value *addr = inst->ops[1];
        std::string key = memKey(addr);
        auto it = mem.find(key);
        // redundant store: memory already holds the same value
        bool sameVal = it != mem.end() && sameValue(it->second, val);
        storeKill(addr, mem);
        if (sameVal) {
          instrs.erase(instrs.begin() + i);
          *any = true;
        } else {
          mem[key] = val;
          ++i;
        }
      } else if (op == Op::Call) {
        auto *callee = (!inst->ops.empty())
                           ? dynamic_cast<Function *>(inst->ops[0])
                           : nullptr;
        // A call to a function that cannot write memory leaves every cached
        // load valid, so the forwarding table survives it.  Without the
        // inferred "writes nothing" attribute (LLVM's readonly) this is a
        // full barrier and every tracked slot is dropped.
        const bool noWrite = callee && fx_.callWritesNothing(callee);
        if (!noWrite) mem.clear();
        // Two calls of a function whose result depends only on its arguments
        // are the same value, so the second is redundant -- this is what
        // collapses fft's `x + multiply(wn, y)` / `x - multiply(wn, y)` pair
        // into one call, as clang does.
        static const bool noCallCse = std::getenv("SAKU_NO_CALLCSE") != nullptr;
        if (!noCallCse && noWrite && inst->ty != Type::Void &&
            fx_.callIsValue(callee)) {
          std::string k = callKey(inst);
          auto it = exprs.find(k);
          if (it != exprs.end() && it->second != (Value *)inst) {
            replaceAllUses(mod, inst, it->second);
            instrs.erase(instrs.begin() + i);
            *any = true;
            continue;
          }
          exprs[k] = inst;
        }
        ++i;
      } else if (isPureExprOp(op)) {
        std::string k = exprKey(op, inst->cond, inst->ops);
        if (!k.empty()) {
          auto it = exprs.find(k);
          if (it != exprs.end() && it->second != (Value *)inst) {
            replaceAllUses(mod, inst, it->second);
            instrs.erase(instrs.begin() + i);
            *any = true;
          } else {
            exprs[k] = inst;
            ++i;
          }
        } else
          ++i;
      } else {
        ++i;
      }
    }
  }

  // memory-location key for a load/store address
  static std::string memKey(Value *addr) {
    AddrInfo info = addressOf(addr);
    if ((info.kind == RootKind::Alloca || info.kind == RootKind::Global) &&
        info.constOff) {
      char b[64];
      snprintf(b, sizeof b, "o%p@%lld", (void *)info.root,
               (long long)info.off);
      return b;
    }
    char b[32];
    snprintf(b, sizeof b, "d%p", (void *)addr);
    return b;
  }

  // Key prefix identifying every tracked location rooted at one object.
  static std::string rootPrefix(const void *root) {
    char b[48];
    snprintf(b, sizeof b, "o%p@", root);
    return b;
  }

  // Invalidate the entries a store at `addr` may clobber.
  //
  // When the address is rooted at a known object - even with a *dynamic*
  // offset, as in `nextvalue[cnt]` - the only tracked locations it can reach
  // are that same object and the dynamic (param/unknown-rooted) ones: two
  // distinct allocas/globals are disjoint objects, and indexing off the end of
  // one into another is undefined (the same assumption maySameLocation and
  // LLVM's BasicAA make).  Global scalars therefore stay in the table across a
  // store to a global array - which is what lets `cnt`, `pos`, `bits` and
  // `buf` be read once per iteration of a hash-table or bit-stream loop
  // instead of once per array store.
  static void storeKill(Value *addr,
                        std::unordered_map<std::string, Value *> &mem) {
    AddrInfo info = addressOf(addr);
    if (info.kind == RootKind::Alloca || info.kind == RootKind::Global) {
      if (info.constOff) {
        // exact object + offset: only that one location is overwritten
        char b[64];
        snprintf(b, sizeof b, "o%p@%lld", (void *)info.root,
                 (long long)info.off);
        mem.erase(b);
      }
      // Everything else rooted at the same object is unknown-offset and must
      // go too; the dynamic entries may point into it as well.
      std::string pfx = rootPrefix(info.root);
      for (auto it = mem.begin(); it != mem.end();) {
        bool sameRoot = it->first.compare(0, pfx.size(), pfx) == 0;
        bool dynamic = !it->first.empty() && it->first[0] == 'd';
        if (sameRoot || dynamic)
          it = mem.erase(it);
        else
          ++it;
      }
      return;
    }
    // param-rooted / unknown address: may alias anything non-private
    mem.clear();
  }

  static bool sameValue(Value *a, Value *b) {
    if (a == b) return true;
    if (auto *ca = asConstInt(a))
      if (auto *cb = asConstInt(b)) return ca->v == cb->v;
    if (auto *fa = asConstFloat(a))
      if (auto *fb = asConstFloat(b)) return fa->bits() == fb->bits();
    return false;
  }
};

} // namespace

std::unique_ptr<Pass> makeMemCsePass(Layer l) {
  return std::make_unique<MemCsePass>(l);
}

// ===========================================================================
// redundant-store: block-local dead stores + store-back of identical loads
// ===========================================================================
namespace {

class RedundantStorePass final : public Pass {
public:
  explicit RedundantStorePass(Layer l) : layer_(l) {}
  const char *name() const override { return "redundant-store"; }
  Layer inLayer() const override { return layer_; }

  bool run(Module &mod) override {
    bool any = false;
    for (auto &f : mod.functions()) {
      if (f->isLib) continue;
      std::function<void(BlockList &)> rec = [&](BlockList &list) {
        for (auto &bb : list) {
          any |= cleanBlock(*bb);
          for (auto &iu : bb->instrs) {
            Instruction *in = iu.get();
            if (in->op == Op::AffineFor)
              rec(in->bodyRegion);
            else if (in->op == Op::ScfWhile) {
              rec(in->condRegion);
              rec(in->bodyRegion);
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

  static std::string exactKey(Value *addr) {
    AddrInfo info = addressOf(addr);
    if ((info.kind == RootKind::Alloca || info.kind == RootKind::Global) &&
        info.constOff) {
      char b[64];
      snprintf(b, sizeof b, "o%p@%lld", (void *)info.root,
               (long long)info.off);
      return b;
    }
    char b[32];
    snprintf(b, sizeof b, "d%p", (void *)addr);
    return b;
  }

  // Conservative "may these two addresses denote the same memory location?"
  // Used when a load is seen: any pending store that may alias the load's
  // address must be kept alive (dropped from the pending set) because the
  // load could read the value that store wrote.  Returning false here lets
  // the pass delete a store, so false is only allowed when the two objects
  // are provably disjoint (distinct private allocas/globals, or exact,
  // different constant offsets into the same object).  Everything doubtful
  // returns true.
  static bool maySameLocation(Value *a, Value *b) {
    AddrInfo A = addressOf(a), B = addressOf(b);
    if (!A.root || !B.root) return true; // unresolved / unknown root
    if (A.root == B.root) {
      if (A.constOff && B.constOff) return A.off == B.off;
      return true; // same object, offset not statically exact
    }
    // Different root objects.  Two distinct allocas, an alloca vs anything
    // else, and two distinct globals are all disjoint objects; anything that
    // touches a param/unknown root may alias a global (params receive arrays
    // from callers whose addresses are not tracked here).
    if (A.kind == RootKind::Alloca || B.kind == RootKind::Alloca) return false;
    bool aObj = A.kind == RootKind::Global;
    bool bObj = B.kind == RootKind::Global;
    if (aObj && bObj) return false; // two distinct global objects
    return true;
  }

  bool cleanBlock(BasicBlock &bb) {
    bool any = false;
    auto &instrs = bb.instrs;

    for (size_t i = 0; i < instrs.size();) {
      Instruction *in = instrs[i].get();
      if (in->op == Op::Store && in->ops.size() == 2) {
        auto *ld = dynamic_cast<Instruction *>(in->ops[0]);
        if (ld && ld->op == Op::Load && ld->ops.size() == 1 &&
            ld->ops[0] == in->ops[1]) {
          instrs.erase(instrs.begin() + (long)i);
          any = true;
          continue;
        }
      }
      ++i;
    }

    std::unordered_map<std::string, size_t> lastStore;
    // A pending store can only be marked dead by a later store to the *same
    // location*; any intervening load (or call / region op) that may observe
    // it must keep it alive.  The SSA pointer values %p and %q below are
    // distinct values for the same address (e.g. two separate geps for
    // res[i][j]); keying loads by pointer identity would therefore miss that
    // a load of %q reads what a pending store to %p wrote, and the store
    // would be wrongly eliminated.  Track the store's address operand (not
    // only a pointer-identity key) so each load can invalidate every pending
    // store whose address may alias it.  `lastStore` is the O(1) index of a
    // pending entry by key; both containers are updated together.
    std::vector<std::pair<Value *, size_t>> pending;
    std::vector<char> dead(instrs.size(), 0);
    for (size_t i = 0; i < instrs.size(); ++i) {
      Instruction *in = instrs[i].get();
      if (in->op == Op::Store && in->ops.size() == 2) {
        Value *addr = in->ops[1];
        std::string k = exactKey(addr);
        auto it = lastStore.find(k);
        if (it != lastStore.end()) dead[it->second] = 1;
        lastStore[k] = i;
        // drop the superseded pending entry (same key) and record the new
        // store; entries a load already invalidated are no longer in either
        // container
        for (auto p = pending.begin(); p != pending.end();)
          if (exactKey(p->first) == k) p = pending.erase(p);
          else ++p;
        pending.push_back({addr, i});
      } else if (in->op == Op::Load && in->ops.size() == 1) {
        // A load may read any pending store whose location overlaps this
        // address: drop those so a later store cannot mark them dead.
        Value *ld = in->ops[0];
        for (auto p = pending.begin(); p != pending.end();)
          if (maySameLocation(ld, p->first)) {
            std::string pk = exactKey(p->first);
            auto mit = lastStore.find(pk);
            if (mit != lastStore.end() && mit->second == p->second)
              lastStore.erase(mit);
            p = pending.erase(p);
          } else ++p;
      } else if (in->op == Op::Call || in->op == Op::AffineFor ||
                 in->op == Op::ScfWhile) {
        lastStore.clear();
        pending.clear();
      }
    }
    for (size_t i = instrs.size(); i-- > 0;) {
      if (dead[i]) {
        instrs.erase(instrs.begin() + (long)i);
        any = true;
      }
    }
    return any;
  }
};

} // namespace

std::unique_ptr<Pass> makeRedundantStorePass(Layer l) {
  return std::make_unique<RedundantStorePass>(l);
}

} // namespace ir
} // namespace sakura
