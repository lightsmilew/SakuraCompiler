// ---------------------------------------------------------------------------
// Pipeline.cpp: RISCVBackend - the standard back-end pass sequence.
//
// run() hands the pure-cf module to the five stages and returns the emitted
// assembly; the driver never calls the individual stages itself.
// ---------------------------------------------------------------------------
#include "Pipeline.h"

#include <cstdio>
#include <cstdlib>

namespace sakura {
namespace backend {

static const char *mopName(MOp op) {
  switch (op) {
#define CS(x) case MOp::x: return #x
  CS(Prologue); CS(Ret); CS(Jmp); CS(BrNz); CS(BrZ); CS(BrCmp);
  CS(LeaFrame); CS(LeaGlobal); CS(MoveX); CS(MoveF);
  CS(Li); CS(LiWide); CS(LiF);
  CS(EntryInt); CS(EntryFlt); CS(EntryStkInt); CS(EntryStkFlt);
  CS(IAdd); CS(IAddI); CS(ISub); CS(INeg); CS(IMul); CS(IDiv); CS(IRem); CS(Mulh);
  CS(Mulhu); CS(Mul64);
  CS(SllI); CS(SrlI); CS(SraI); CS(Shl64I); CS(Shr64I); CS(Sar64I);
  CS(IXor); CS(IXorI); CS(IAnd); CS(IAndI);
  CS(ISlt); CS(ISlti); CS(ISltu); CS(ISltiu); CS(ISltuZ);
  CS(ICmp); CS(FCmp); CS(FAdd); CS(FSub); CS(FMul); CS(FDiv); CS(FNeg);
  CS(I2F); CS(F2I); CS(FMvWX);
  CS(Lw); CS(Flw); CS(Sw); CS(Fsw);
  CS(LwF); CS(FlwF); CS(SwF); CS(FswF); CS(SdF);
  CS(LwG); CS(FlwG); CS(SwG); CS(FswG);
  CS(SpillLw); CS(SpillFlw); CS(SpillSw); CS(SpillFsw);
  CS(StkArgSw); CS(StkArgFsw); CS(Call); CS(Nop);
#undef CS
  default: return "?";
  }
}


// Debug aid: SAKU_MIR_<STAGE>=1 dumps the machine IR at that stage on stderr.
// Stages: ISEL / POST_MACHOPT / POST_RA.  Unset in normal compiles.
static void dumpMir(const char *tag, const std::vector<MachineFunc> &fns) {
  if (!std::getenv(tag)) return;
  for (const auto &f : fns) {
    fprintf(stderr, "func %s:\n", f.name.c_str());
    for (const auto &b : f.blocks) {
      fprintf(stderr, "  %s:\n", b.name.c_str());
      for (const auto &m : b.instrs)
        fprintf(stderr, "    %-12s d=%-3d a=%-3d b=%-3d imm=%d sym=%s\n",
                mopName(m.op), m.dst, m.a, m.b, m.imm, m.sym.c_str());
    }
  }
}

std::string RISCVBackend::run(ir::Module *mod, const BackendOptions &opts) {
  // ---- 1) instruction selection: pure-cf IR -> machine IR --------------
  // Throws if a structured op survived the mid-end (defensive re-check of
  // the verify-cf-only contract, in case a future driver bypasses it).
  std::vector<MachineFunc> fns = isel_.select(mod);

  if (std::getenv("SAKU_MIRDUMP")) dumpMir("SAKU_MIRDUMP", fns);
  dumpMir("SAKU_MIR_ISEL", fns);

  // ---- 2) machine-level optimisation (peephole / schedule / CSE / LICM) --
  // Runs on virtual registers before allocation; off for the -O0 baseline.
  // Each stage can be disabled individually (SAKU_NO_SR / _PEEPHOLE / _SCHED /
  // _BLOCKCSE / _MLICM / _COALESCE) to bisect an optimisation-induced bug.
  size_t sr = 0, peephole = 0, scheduled = 0, blockcse = 0, hoisted = 0,
         coalesced = 0, postra = 0, rotated = 0, globalcse = 0, blockcse2 = 0,
         zerofill = 0;
  if (opts.machineOpt) {
    if (!std::getenv("SAKU_NO_SR")) sr = opt_.strengthReduction(fns);
    // Zero-fill coalescing runs before `peephole` on purpose: it rewrites the
    // zero-fill stores to use x0, which leaves the `li 0` they came from with
    // no readers, and the peephole round immediately below is what deletes it.
    if (!std::getenv("SAKU_NO_ZEROFILL")) zerofill = opt_.zeroFill(fns);
    if (!std::getenv("SAKU_NO_PEEPHOLE")) peephole = opt_.peephole(fns);
    if (!std::getenv("SAKU_NO_SCHED")) scheduled = opt_.schedule(fns);
    if (!std::getenv("SAKU_NO_BLOCKCSE")) blockcse = opt_.blockLocalCse(fns);
    if (!std::getenv("SAKU_NO_MLICM")) hoisted = opt_.licm(fns);
    // LICM just pulled one `li`/`la` per inlined loop body into the shared
    // preheader, so the block-local CSE has to run once more on the result
    // (this is where the duplicates are now), and the dominator-tree CSE
    // folds the copies that ended up in different blocks.
    if (!std::getenv("SAKU_NO_GLOBALCSE")) globalcse = opt_.globalCse(fns);
    if (!std::getenv("SAKU_NO_BLOCKCSE2")) blockcse2 = opt_.blockLocalCse(fns);
    // Fold register copies (ISel's phi/latch copies in particular) once LICM
    // has settled the loop-invariant parts and before colouring decides the
    // registers, so coalescing at the virtual-register level can retire them.
    if (!std::getenv("SAKU_NO_COALESCE")) coalesced = opt_.coalesceMoves(fns);
  }
  dumpMir("SAKU_MIR_POST_MACHOPT", fns);

  // ---- 3) register allocation (graph-colouring + spills) ---------------
  ra_.allocate(fns);
  dumpMir("SAKU_MIR_POST_RA", fns);

  // ---- 4) second peephole after register allocation --------------------
  // Colouring exposes redundant moves (self / ping-pong / dead-consecutive /
  // move-into-next-store); this round removes them on physical registers.
  // Mirrors the reference SysY backend's SecondPeep (RemoveRedundantMovePass).
  if (opts.machineOpt) postra = opt_.removeRedundantMoves(fns);
  // ---- 5) loop rotation by block layout --------------------------------
  // With physical registers in place the block order is free, so a loop whose
  // header/latch/exit were not laid out for the mid-end `loop-rotate` pass can
  // still lose its unconditional back edge.  Disable with SAKU_NO_LAYOUT_ROTATE.
  if (opts.machineOpt && !std::getenv("SAKU_NO_LAYOUT_ROTATE"))
    rotated = opt_.rotateLoopsByLayout(fns);
  dumpMir("SAKU_MIR_POST_LAYOUT", fns);

  if (opts.stats)
    *opts.stats << "backend-strength-reduction: " << sr
                << " const div/rem/mul rewritten\n"
                << "backend-zerofill: " << zerofill
                << " zero stores coalesced\n"
                << "backend-peephole: " << peephole
                << " instructions eliminated\n"
                << "backend-schedule: " << scheduled << " loads hoisted\n"
                << "backend-blockcse: " << blockcse
                << " instructions eliminated\n"
                << "backend-licm: " << hoisted << " instructions hoisted\n"
                << "backend-global-cse: " << globalcse
                << " materialisations folded\n"
                << "backend-postlicm-cse: " << blockcse2
                << " instructions eliminated\n"
                << "backend-move-coalesce: " << coalesced
                << " copies folded\n"
                << "backend-postra-peephole: " << postra
                << " moves removed\n"
                << "backend-layout-rotate: " << rotated
                << " loops rotated\n";

  // ---- 5) assembly emission ---------------------------------------------
  return writer_.write(mod, fns);
}

} // namespace backend
} // namespace sakura
