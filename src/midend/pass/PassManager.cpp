// PassManager.cpp: layer-name helpers and the pass-sequence executor.
#include "PassManager.h"
#include <chrono>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <unordered_set>

#include "OptUtil.h"

namespace sakura {
namespace ir {

const char *layerName(Layer l) {
  switch (l) {
  case Layer::Affine: return "affine";
  case Layer::Scf:    return "scf";
  default:            return "cf";
  }
}

bool PassManager::run(Module &mod) {
  // IRBuilder output is the top (affine) layer: print it before any pass so
  // --dump-affine shows the IR exactly as the builder produced it.
  Layer cur = Layer::Affine;
  if (printer_) printer_(layerName(cur), mod);

  // Debug hook: SAKURA_SKIP=<comma separated names or substrings> disables
  // matching passes so a bad transformation can be bisected without a
  // rebuild.  Skips are recorded in results_ as "skipped".
  std::vector<std::string> skip;
  if (const char *s = getenv("SAKURA_SKIP")) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
      if (!item.empty()) skip.push_back(item);
    }
  }
  auto skipped = [&](const char *name) {
    for (auto &sk : skip)
      if (strstr(name, sk.c_str())) return true;
    return false;
  };

  bool changed = false;
  results_.clear();
  auto t_total = std::chrono::steady_clock::now();
  const bool doTime = getenv("SAKURA_TIME") != nullptr;
  std::vector<std::pair<std::string, double>> times;
  for (auto &pass : passes_) {
    if (pass->inLayer() != cur)
      throw CompileError("internal: pass '" + std::string(pass->name()) +
                         "' expects " + layerName(pass->inLayer()) +
                         "-layer IR but the module is at the " +
                         layerName(cur) + " layer");
    if (skipped(pass->name())) {
      results_.emplace_back(std::string(pass->name()) + " [skipped]", false);
      continue;
    }
    auto t0 = std::chrono::steady_clock::now();
    bool ch = pass->run(mod);
    auto t1 = std::chrono::steady_clock::now();
    if (doTime)
      times.emplace_back(pass->name(),
                         std::chrono::duration<double, std::milli>(t1 - t0)
                             .count());
    results_.emplace_back(pass->name(), ch);
    changed |= ch;
    if (getenv("SAKURA_DUMP")) {
      static int seq = 0;
      std::ofstream of("/tmp/pd_" + std::to_string(seq++) + "_" +
                       pass->name() + ".txt");
      of << mod.toString();
    }
    if (getenv("SAKURA_CHECK")) {
      // after each pass, every operand must still point at a live value;
      // a dangling operand here is the pass that dropped a definition
      std::unordered_set<Value *> live;
      std::unordered_set<BasicBlock *> liveBlocks;
      for (auto &f : mod.functions()) {
        if (f->isLib) continue;
        for (auto &a : f->args) live.insert(a.get());
        std::function<void(BlockList &)> rec = [&](BlockList &list) {
          for (auto &bb : list) {
            liveBlocks.insert(bb.get());
            for (auto &iu : bb->instrs) {
              live.insert(iu.get());
              Instruction *i = iu.get();
              if (i->op == Op::ScfWhile) {
                rec(i->condRegion);
                rec(i->bodyRegion);
              } else if (i->op == Op::AffineFor) {
                rec(i->bodyRegion);
              }
            }
          }
        };
        rec(f->blocks);
      }
      // structural integrity: no null / duplicate entries in any block list
      bool dup = false;
      std::function<void(BlockList &)> dupRec = [&](BlockList &list) {
        std::unordered_set<BasicBlock *> seenB;
        for (auto &bb : list) {
          if (!bb) { std::cerr << "[SAKURA_CHECK] " << pass->name()
                               << ": null BasicBlock in a list\n"; dup = true; }
          if (bb && !seenB.insert(bb.get()).second) {
            std::cerr << "[SAKURA_CHECK] " << pass->name()
                      << ": duplicate BasicBlock in a list\n";
            dup = true;
          }
          if (bb) {
            std::unordered_set<Instruction *> seenI;
            for (auto &iu : bb->instrs) {
              if (!iu) { std::cerr << "[SAKURA_CHECK] " << pass->name()
                                   << ": null Instruction in block\n";
                        dup = true; continue; }
              if (!seenI.insert(iu.get()).second) {
                std::cerr << "[SAKURA_CHECK] " << pass->name()
                          << ": duplicate Instruction in a block\n";
                dup = true;
              }
              if (iu->op == Op::ScfWhile) {
                dupRec(iu->condRegion);
                dupRec(iu->bodyRegion);
              } else if (iu->op == Op::AffineFor) {
                dupRec(iu->bodyRegion);
              }
            }
          }
        }
      };
      for (auto &f : mod.functions())
        if (!f->isLib) dupRec(f->blocks);
      if (dup) throw CompileError("SAKURA_CHECK: structural corruption", 0);
      bool bad = false;
      for (auto &f : mod.functions()) {
        if (f->isLib) continue;
        std::function<void(BlockList &)> rec = [&](BlockList &list) {
          for (auto &bb : list)
            for (auto &iu : bb->instrs) {
              Instruction *i = iu.get();
              for (Value *o : i->ops) {
                if (auto *tb = dynamic_cast<BasicBlock *>(o)) {
                  // structural check: an erased block must not be targeted
                  if (!liveBlocks.count(tb)) {
                    std::cerr << "[SAKURA_CHECK] " << pass->name()
                              << " targets erased block: fn=" << f->name
                              << " op=" << opName(i->op) << " line=" << i->line
                              << "\n";
                    bad = true;
                  }
                  continue;
                }
                if (dynamic_cast<Function *>(o) ||
                    dynamic_cast<GlobalVar *>(o))
                  continue; // cross-function refs are legitimate
                if (asConstInt(o) || asConstFloat(o)) continue;
                auto *oi = dynamic_cast<Instruction *>(o);
                if (oi && !live.count(oi)) {
                  std::cerr << "[SAKURA_CHECK] " << pass->name()
                            << " leaves dangling operand: fn=" << f->name
                            << " op=" << opName(i->op) << " line=" << i->line
                            << " dangle-op=" << opName(oi->op)
                            << " dangle-line=" << oi->line << "\n";
                  bad = true;
                }
              }
              if (i->op == Op::ScfWhile) {
                rec(i->condRegion);
                rec(i->bodyRegion);
              } else if (i->op == Op::AffineFor) {
                rec(i->bodyRegion);
              }
            }
        };
        rec(f->blocks);
      }
      if (bad) throw CompileError("SAKURA_CHECK: dangling operand detected", 0);
    }
    if (pass->outLayer() != cur) {
      cur = pass->outLayer();
      if (printer_) printer_(layerName(cur), mod);
    }
  }
  if (doTime) {
    double tot = 0;
    for (auto &t : times) tot += t.second;
    std::cerr << "[SAKURA_TIME] total " << (long)tot << " ms\n";
    for (auto &t : times)
      if (t.second > 10)
        std::cerr << "[SAKURA_TIME] " << t.first << " " << (long)t.second
                  << " ms\n";
  }
  (void)t_total;
  return changed;
}

} // namespace ir
} // namespace sakura
