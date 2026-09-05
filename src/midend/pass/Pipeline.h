// ---------------------------------------------------------------------------
// Pipeline.h: the single high-level mid-end entry point.
//
// main (and any other driver) never assembles passes: it calls
// runMidEndPipeline() on the affine-layer module produced by IRBuilder and
// gets back a pure-cf module ready for the back-end.  All IR lowering and
// per-layer optimisation live inside this one function, which builds the
// PassManager and registers the standard pass sequence:
//
//     affine (top)
//       expand-tensor-ops                 (whole-tensor -> affine.for)
//       [ affine-level optimisation passes ]
//       lower-affine-to-scf               (affine.for -> scf.while)
//     scf (middle)
//       [ scf-level optimisation passes ]
//       canonicalize-control-flow         (scf.while -> flat cf)
//     cf (bottom)
//       [ cf-level optimisation passes ]
//       verify-cf-only                    (back-end gate)
//
// New optimisation passes are added by editing this sequence, never by
// touching the driver.
// ---------------------------------------------------------------------------
#pragma once

#include <functional>
#include <iosfwd>

#include "../ir/IR.h"

namespace sakura {
namespace ir {

// Optimisation budget.  O0 keeps the historical conversion-only pipeline
// (useful as a correctness baseline); O1/O2 enable the ported optimisation
// passes (O2 = default, strongest).
enum class OptLevel : uint8_t { O0, O1, O2 };

// Optional per-layer view hook.  Invoked with the layer name ("affine" /
// "scf" / "cf") and the module every time the pipeline enters a layer, i.e.
// on IRBuilder output and at each lowering boundary.  The driver wires the
// --dump-affine / --dump-scf / --dump-cf flags to this.
using LayerView = std::function<void(const char *layer, Module &mod)>;

// Lower `mod` from the affine layer (IRBuilder output) to pure cf, running
// every conversion and optimisation pass through an internal PassManager.
// After the call returns, the module contains only cf / func / arith /
// memref ops (guaranteed by the trailing verify-cf-only pass).
//
// When `stats` is non-null, a one-line "<pass>: changed/unchanged" report is
// written after the pipeline has run (driven by --pass-stats).
void runMidEndPipeline(Module &mod, const LayerView &view = nullptr,
                       OptLevel level = OptLevel::O2,
                       std::ostream *stats = nullptr);

} // namespace ir
} // namespace sakura
