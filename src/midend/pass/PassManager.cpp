// PassManager.cpp: layer-name helpers and the pass-sequence executor.
#include "PassManager.h"

#include <cstdio>
#include <cstdlib>
#include <string>

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

  // Debug aid: SAKURA_PM_DUMP=<dir> writes the module after every pass as
  // <n>_<pass-name>.txt, the equivalent of LLVM's -print-after-all.  Purely a
  // developer switch: unset in normal compiles, so it costs nothing.
  const char *dumpDir = getenv("SAKURA_PM_DUMP");
  int dumpSeq = 0;
  auto dumpAfter = [&](const char *pname) {
    if (!dumpDir) return;
    char buf[512];
    std::snprintf(buf, sizeof buf, "%s/%04d_%s.txt", dumpDir, dumpSeq++, pname);
    if (FILE *fp = std::fopen(buf, "w")) {
      std::string s = mod.toString();
      std::fwrite(s.data(), 1, s.size(), fp);
      std::fclose(fp);
    }
  };

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
    dumpAfter(pass->name());
    if (pass->outLayer() != cur) {
      cur = pass->outLayer();
      if (printer_) printer_(layerName(cur), mod);
      dumpAfter(("layer-" + std::string(layerName(cur))).c_str());
    }
  }
  return changed;
}

} // namespace ir
} // namespace sakura
