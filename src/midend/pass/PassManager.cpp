// PassManager.cpp: layer-name helpers and the pass-sequence executor.
#include "PassManager.h"

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

  bool changed = false;
  results_.clear();
  for (auto &pass : passes_) {
    if (pass->inLayer() != cur)
      throw CompileError("internal: pass '" + std::string(pass->name()) +
                         "' expects " + layerName(pass->inLayer()) +
                         "-layer IR but the module is at the " +
                         layerName(cur) + " layer");
    bool ch = pass->run(mod);
    results_.emplace_back(pass->name(), ch);
    changed |= ch;
    if (pass->outLayer() != cur) {
      cur = pass->outLayer();
      if (printer_) printer_(layerName(cur), mod);
    }
  }
  return changed;
}

} // namespace ir
} // namespace sakura
