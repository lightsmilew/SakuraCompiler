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
    for (auto &f : mod.functions()) {
      if (f->isLib) continue;
      process(mod, f->blocks, &any);
    }
    return any;
  }

private:
  Layer layer_;

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

  bool isPureExprOp(Op op) const {
    switch (op) {
    case Op::Add: case Op::Sub: case Op::Mul: case Op::SDiv: case Op::SRem:
    case Op::FAdd: case Op::FSub: case Op::FMul: case Op::FDiv:
    case Op::ICmp: case Op::FCmp: case Op::Sitofp: case Op::Fptosi:
    case Op::Not: case Op::Gep:
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
        mem.clear(); // a call may write any global / param / escaped slot
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

  // Invalidate the entries a store at `addr` may clobber.
  static void storeKill(Value *addr,
                        std::unordered_map<std::string, Value *> &mem) {
    AddrInfo info = addressOf(addr);
    if ((info.kind == RootKind::Alloca || info.kind == RootKind::Global) &&
        info.constOff) {
      // exact object + offset: only that location is overwritten
      char b[64];
      snprintf(b, sizeof b, "o%p@%lld", (void *)info.root,
               (long long)info.off);
      mem.erase(b);
      if (info.kind == RootKind::Global) {
        // a store to a global can be observed through an aliasing param
        for (auto it = mem.begin(); it != mem.end();) {
          if (it->first[0] == 'd') it = mem.erase(it);
          else ++it;
        }
      }
      return;
    }
    // dynamic / param-rooted address: may alias anything non-private
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

} // namespace ir
} // namespace sakura
