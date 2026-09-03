// ---------------------------------------------------------------------------
// PassManager.h: a minimal MLIR-style pass manager for the layered mid-end.
//
// The mid-end IR is organised in three control-flow layers, each of which can
// be optimised independently:
//
//     affine (top)   IRBuilder output.  Canonical counting loops are
//                    affine.for region ops; whole-tensor ops are still
//                    whole-tensor until expand-tensor-ops runs.
//          |        lower-affine-to-scf        (affine.for -> scf.while)
//          v
//     scf (middle)   every structured loop is the generic scf.while
//          |        canonicalize-control-flow  (scf.while -> flat cf)
//          v
//     cf (bottom)    the flat cf.br / cf.cond_br CFG -- the only form the
//                    back-end ever consumes
//
// A pass declares the layer it expects (inLayer) and the layer it leaves the
// module in (outLayer).  Optimisation passes keep the layer unchanged; a
// lowering pass steps the IR down exactly one layer.  PassManager::run()
// checks the annotations while executing the sequence, so a wrongly ordered
// pipeline fails fast instead of feeding non-cf IR to the back-end.
//
// An optional "layer printer" hook is invoked every time the pipeline enters
// a layer (the starting affine layer plus each lowering boundary); the driver
// wires --dump-affine / --dump-scf / --dump-cf to it.
// ---------------------------------------------------------------------------
#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "../ir/IR.h"

namespace sakura {
namespace ir {

// The three control-flow layers of the mid-end.
enum class Layer : uint8_t { Affine, Scf, Cf };

// "affine" | "scf" | "cf"
const char *layerName(Layer l);

// One whole-module IR transformation.
class Pass {
public:
  virtual ~Pass() = default;

  // Stable identifier, e.g. "lower-affine-to-scf" (used in diagnostics).
  virtual const char *name() const = 0;

  // Control-flow layer the IR must be in before this pass runs.
  virtual Layer inLayer() const = 0;

  // Layer the pass leaves the IR in.  Defaults to inLayer(), which is what
  // optimisation passes want; lowering passes override it.
  virtual Layer outLayer() const { return inLayer(); }

  // Transform `mod`.  Returns true when the IR was modified.
  virtual bool run(Module &mod) = 0;
};

// Executes an ordered list of passes.  Passes are owned by the manager.
class PassManager {
public:
  // Invoked with the module every time the pipeline enters a layer.
  using LayerPrinter = std::function<void(const char *layer, Module &mod)>;

  void setLayerPrinter(LayerPrinter printer) { printer_ = std::move(printer); }
  void addPass(std::unique_ptr<Pass> pass) {
    passes_.push_back(std::move(pass));
  }

  // Execute all registered passes in order.  The module must start in the
  // affine layer (i.e. straight out of IRBuilder).  Returns true if any pass
  // modified the IR.
  bool run(Module &mod);

private:
  std::vector<std::unique_ptr<Pass>> passes_;
  LayerPrinter printer_;
};

} // namespace ir
} // namespace sakura
