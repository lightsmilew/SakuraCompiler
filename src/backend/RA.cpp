// Graph-colouring register allocator implementation: liveness on each block,
// interference graph building, colouring with a spill fallback that rewrites
// virtual registers through explicit spill-slot loads/stores, then fixing up
// callee-saved register usage and call-clobbered temporaries.
#include "RA.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>

#include "../common/Common.h"

using namespace sakura::backend;
using namespace sakura::ir;

namespace sakura {
namespace backend {

namespace {

// ---------------------------------------------------------------------------
// palettes: color index -> physical register number
// ---------------------------------------------------------------------------
const int kXFullPhys[] = {
    8, 9,                             // s0, s1
    10, 11, 12, 13, 14, 15, 16, 17,   // a0..a7
    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, // s2..s11
    7,                                // t2
    28, 29, 30, 31                    // t3..t6
};
constexpr int kXFull = 25;
const int kXCalleePhys[] = {8, 9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27};
constexpr int kXCallee = 12;

const int kFFullPhys[] = {
    8, 9,                             // fs0, fs1
    10, 11, 12, 13, 14, 15, 16, 17,   // fa0..fa7
    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, // fs2..fs11
    2, 3, 4, 5, 6, 7,                 // ft2..ft7
    28, 29, 30, 31                    // ft8..ft11
};
constexpr int kFFull = 30;
const int kFCalleePhys[] = {8, 9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27};
constexpr int kFCallee = 12;

inline bool isCalleeX(int phys) {
  for (int r : kXCalleePhys)
    if (r == phys) return true;
  return false;
}
inline bool isCalleeF(int phys) {
  for (int r : kFCalleePhys)
    if (r == phys) return true;
  return false;
}
// ---------------------------------------------------------------------------
struct Allocator {
  MachineFunc &f;
  int nreg = 0; // current virtual register count

  explicit Allocator(MachineFunc &func) : f(func) {}

  int maxReg() const {
    int m = 0;
    auto upd = [&](int r) {
      if (r >= 0 && r + 1 > m) m = r + 1;
    };
    for (auto &b : f.blocks)
      for (auto &inst : b.instrs) {
        upd(inst.dst);
        upd(inst.a);
        upd(inst.b);
        for (auto &a : inst.args) upd(a.vreg);
      }
    return m;
  }

  std::vector<char> isF;

  void computeFiles() {
    int n = maxReg();
    isF.assign((size_t)n, 0);
    auto setf = [&](int r, bool fileF) {
      if (r >= 0 && r < n) isF[(size_t)r] = fileF;
    };
    for (auto &b : f.blocks)
      for (auto &inst : b.instrs) {
        OpSlots sl = slots(inst);
        setf(sl.def, sl.defFile == RF_F);
        for (int k = 0; k < 3; ++k) setf(sl.use[k], sl.useFile[k] == RF_F);
        for (auto &a : inst.args)
          if (a.vreg >= 0) setf(a.vreg, a.ty == Type::F32);
      }
  }

  // ---- dead pure instruction elimination --------------------------------
  void trimDead() {
    for (int guard = 0; guard < 32; ++guard) {
      int n = maxReg();
      std::vector<int32_t> use((size_t)n, 0);
      auto touch = [&](int r) {
        if (r >= 0 && r < n) ++use[(size_t)r];
      };
      for (auto &b : f.blocks)
        for (auto &inst : b.instrs) {
          OpSlots sl = slots(inst);
          for (int k = 0; k < 3; ++k) touch(sl.use[k]);
          for (auto &a : inst.args) touch(a.vreg);
        }
      bool changed = false;
      for (auto &b : f.blocks) {
        auto &v = b.instrs;
        for (size_t i = 0; i < v.size();) {
          MInst &m = v[i];
          OpSlots sl = slots(m);
          bool pure = false;
          switch (m.op) {
          case MOp::Li: case MOp::LiF: case MOp::MoveX: case MOp::MoveF:
          case MOp::LeaFrame: case MOp::LeaGlobal:
          case MOp::EntryInt: case MOp::EntryFlt:
          case MOp::EntryStkInt: case MOp::EntryStkFlt:
          case MOp::IAdd: case MOp::IAddI: case MOp::ISub: case MOp::INeg:
          case MOp::IMul: case MOp::IDiv: case MOp::IRem:
          case MOp::IXor: case MOp::IXorI:
          case MOp::ISlt: case MOp::ISlti: case MOp::ISltu: case MOp::ISltiu:
          case MOp::ISltuZ: case MOp::ICmp: case MOp::FCmp:
          case MOp::FAdd: case MOp::FSub: case MOp::FMul: case MOp::FDiv:
          case MOp::FNeg: case MOp::I2F: case MOp::F2I: case MOp::FMvWX:
            pure = true;
            break;
          default: break;
          }
          if (sl.def >= 0 && pure && use[(size_t)sl.def] == 0) {
            v.erase(v.begin() + i);
            changed = true;
          } else if (m.op == MOp::Call && m.dst >= 0 &&
                     use[(size_t)m.dst] == 0) {
            m.dst = -1;
            changed = true;
            ++i;
          } else {
            ++i;
          }
        }
      }
      if (!changed) break;
    }
  }

  // ---- control flow graph ------------------------------------------------
  std::unordered_map<std::string, int32_t> labelMap;
  std::vector<std::vector<int32_t>> succ;

  void buildCFG() {
    labelMap.clear();
    for (size_t i = 0; i < f.blocks.size(); ++i)
      labelMap[f.blocks[i].name] = (int32_t)i;
    succ.assign(f.blocks.size(), {});
    for (size_t i = 0; i < f.blocks.size(); ++i) {
      auto &v = f.blocks[i].instrs;
      auto addFall = [&]() {
        if (i + 1 < f.blocks.size()) succ[i].push_back((int32_t)(i + 1));
      };
      auto addTgt = [&](const std::string &sym) {
        auto it = labelMap.find(sym);
        if (it != labelMap.end()) succ[i].push_back(it->second);
      };
      // ISel may end a block with a conditional branch *followed by* an
      // unconditional jump (neither target is the in-layout fall-through), so
      // every trailing jump/branch must contribute a successor, not just the
      // very last instruction.
      size_t k = v.size();
      while (k > 0) {
        MOp op = v[k - 1].op;
        if (op == MOp::Jmp || op == MOp::BrNz || op == MOp::BrZ ||
            op == MOp::BrCmp)
          --k;
        else
          break;
      }
      if (k == v.size()) {
        // No jump/branch at the tail.  An empty block or one that merely runs
        // into the next one in layout falls through; a Ret stops control flow.
        if (v.empty() || v.back().op != MOp::Ret) addFall();
        continue;
      }
      for (size_t t = k; t < v.size(); ++t) addTgt(v[t].sym);
      if (v.back().op != MOp::Jmp && v.back().op != MOp::Ret) addFall();
    }
  }

  // ---- liveness ----------------------------------------------------------
  std::vector<std::vector<char>> liveIn, liveOut;

  // transfer live-out through the instructions of block `bidx`
  void transfer(int32_t bidx, std::vector<char> &live) {
    int n = (int32_t)live.size();
    auto &insts = f.blocks[(size_t)bidx].instrs;
    for (size_t k = insts.size(); k-- > 0;) {
      MInst &m = insts[k];
      OpSlots sl = slots(m);
      auto touch = [&](int r) {
        if (r >= 0 && r < n) live[(size_t)r] = 1;
      };
      for (int j = 0; j < 3; ++j) touch(sl.use[j]);
      for (auto &a : m.args) touch(a.vreg);
      auto kill = [&](int r) {
        if (r >= 0 && r < n) live[(size_t)r] = 0;
      };
      kill(sl.def);
    }
  }

  // returns crossing flags (values live across at least one call)
  void computeLiveness(std::vector<char> &crossing) {
    int32_t nb = (int32_t)f.blocks.size();
    int n = maxReg();
    crossing.assign((size_t)n, 0);
    liveIn.assign((size_t)nb, std::vector<char>((size_t)n, 0));
    liveOut.assign((size_t)nb, std::vector<char>((size_t)n, 0));

    bool changed = true;
    int guard = 0;
    int cap = nb + 8; // a pass propagates at least one CFG edge; nb passes suffice
    while (changed && guard++ < cap) {
      changed = false;
      for (int32_t bi = nb - 1; bi >= 0; --bi) {
        std::vector<char> out((size_t)n, 0);
        for (int32_t s : succ[(size_t)bi])
          for (int32_t r = 0; r < n; ++r)
            out[(size_t)r] |= liveIn[(size_t)s][(size_t)r];
        if (out != liveOut[(size_t)bi]) {
          liveOut[(size_t)bi] = out;
          changed = true;
        }
        std::vector<char> in = out;
        transfer(bi, in);
        // A live-in change (e.g. a value first made live by this pass) can
        // widen a *predecessor's* live-out on the next pass, so the fixed
        // point must keep iterating while live-in moves too.
        if (in != liveIn[(size_t)bi]) {
          liveIn[(size_t)bi] = in;
          changed = true;
        }
      }
    }

    for (int32_t bi = 0; bi < nb; ++bi) {
      auto &insts = f.blocks[(size_t)bi].instrs;
      std::vector<char> live = liveOut[(size_t)bi];
      for (size_t k = insts.size(); k-- > 0;) {
        MInst &m = insts[k];
        OpSlots sl = slots(m);
        if (m.op == MOp::Call) {
          for (int32_t r = 0; r < n; ++r)
            if (live[(size_t)r] && r != m.dst) crossing[(size_t)r] = 1;
        }
        auto touch = [&](int r) {
          if (r >= 0 && r < n) live[(size_t)r] = 1;
        };
        for (int j = 0; j < 3; ++j) touch(sl.use[j]);
        for (auto &a : m.args) touch(a.vreg);
        if (sl.def >= 0 && sl.def < n) live[(size_t)sl.def] = 0;
      }
    }
  }

  // ---- interference graph ------------------------------------------------
  std::vector<int32_t> xNodes, fNodes;
  std::vector<int32_t> xIdx, fIdx; // vreg -> graph index (or -1)
  std::vector<std::vector<char>> adjX, adjF;

  bool appears(int32_t r) const {
    for (auto &b : f.blocks)
      for (auto &m : b.instrs) {
        if (m.dst == r || m.a == r || m.b == r) return true;
        for (auto &a : m.args)
          if (a.vreg == r) return true;
      }
    return false;
  }

  void buildGraph() {
    int n = maxReg();
    computeFiles();
    xIdx.assign((size_t)n, -1);
    fIdx.assign((size_t)n, -1);
    xNodes.clear();
    fNodes.clear();
    for (int32_t r = 0; r < n; ++r) {
      if (!appears(r)) continue;
      if (isF[(size_t)r]) {
        fIdx[(size_t)r] = (int32_t)fNodes.size();
        fNodes.push_back(r);
      } else {
        xIdx[(size_t)r] = (int32_t)xNodes.size();
        xNodes.push_back(r);
      }
    }
    size_t nx = xNodes.size(), nf = fNodes.size();
    adjX.assign(nx, std::vector<char>(nx, 0));
    adjF.assign(nf, std::vector<char>(nf, 0));

    for (size_t bi = 0; bi < f.blocks.size(); ++bi) {
      auto &insts = f.blocks[bi].instrs;
      std::vector<char> live = liveOut[bi];
      for (size_t k = insts.size(); k-- > 0;) {
        MInst &m = insts[k];
        OpSlots sl = slots(m);

        // Distinct virtual registers read by the *same* instruction must land
        // in distinct physical registers (e.g. `add t, x, y` with x!=y).
        std::vector<int32_t> reads;
        for (int j = 0; j < 3; ++j) {
          int32_t r = sl.use[j];
          if (r >= 0 && r < n) reads.push_back(r);
        }
        for (auto &a : m.args)
          if (a.vreg >= 0 && a.vreg < n) reads.push_back(a.vreg);
        auto cliqueEdge = [&](int32_t r1, int32_t r2) {
          if (r1 == r2) return;
          bool f1 = isF[(size_t)r1];
          if (f1 != isF[(size_t)r2]) return;
          if (f1) {
            int32_t i = fIdx[(size_t)r1], j2 = fIdx[(size_t)r2];
            if (i < 0 || j2 < 0) return;
            adjF[(size_t)i][(size_t)j2] = 1;
            adjF[(size_t)j2][(size_t)i] = 1;
          } else {
            int32_t i = xIdx[(size_t)r1], j2 = xIdx[(size_t)r2];
            if (i < 0 || j2 < 0) return;
            adjX[(size_t)i][(size_t)j2] = 1;
            adjX[(size_t)j2][(size_t)i] = 1;
          }
        };
        for (size_t p = 0; p < reads.size(); ++p)
          for (size_t q = p + 1; q < reads.size(); ++q)
            cliqueEdge(reads[p], reads[q]);

        int32_t def = sl.def;
        if (def >= 0) {
          if (isF[(size_t)def]) {
            int32_t di = fIdx[(size_t)def];
            if (di >= 0)
              for (size_t r = 0; r < nf; ++r)
                if ((size_t)r != (size_t)di && live[(size_t)fNodes[r]]) {
                  adjF[(size_t)di][r] = 1;
                  adjF[r][(size_t)di] = 1;
                }
          } else {
            int32_t di = xIdx[(size_t)def];
            if (di >= 0)
              for (size_t r = 0; r < nx; ++r)
                if ((size_t)r != (size_t)di && live[(size_t)xNodes[r]]) {
                  adjX[(size_t)di][r] = 1;
                  adjX[r][(size_t)di] = 1;
                }
          }
        }
        auto touch = [&](int r) {
          if (r >= 0 && r < n) live[(size_t)r] = 1;
        };
        for (int j = 0; j < 3; ++j) touch(sl.use[j]);
        for (auto &a : m.args) touch(a.vreg);
        if (def >= 0 && def < n) live[(size_t)def] = 0;
      }
    }
  }

  // ---- graph colouring ----------------------------------------------------
  // nodes: vregs of one file.  Fills color[vreg] with a physical register.
  // forbid[i] lists physical registers (as a 32-bit char mask) that node i
  // must not be assigned (used for entry argument registers that a later
  // Entry instruction still has to read).  Returns false when any node could
  // not be coloured (those keep color -1).
  bool colorFile(const std::vector<int32_t> &nodes,
                 std::vector<std::vector<char>> &adj,
                 const std::vector<char> &crossing,
                 const std::vector<std::vector<char>> &forbid,
                 std::vector<int32_t> &color, const int *calPhys, int Kcal,
                 const int *fullPhys, int Kfull) {
    int32_t n = (int32_t)nodes.size();
    std::vector<char> removed((size_t)n, 0);
    std::vector<char> cand((size_t)n, 0);
    std::vector<int32_t> stack;
    stack.reserve((size_t)n);

    auto degree = [&](int32_t i) {
      int32_t d = 0;
      for (int32_t j = 0; j < n; ++j)
        if (i != j && !removed[(size_t)j] && adj[(size_t)i][(size_t)j]) ++d;
      return d;
    };
    auto Kof = [&](int32_t i) {
      return crossing[(size_t)nodes[(size_t)i]] ? Kcal : Kfull;
    };

    for (;;) {
      int32_t pick = -1;
      for (int32_t i = 0; i < n; ++i) {
        if (!removed[(size_t)i] && degree(i) < Kof(i)) {
          pick = i;
          break;
        }
      }
      if (pick < 0) {
        int32_t maxd = -1;
        for (int32_t i = 0; i < n; ++i) {
          if (removed[(size_t)i]) continue;
          int32_t d = degree(i);
          if (d > maxd) {
            maxd = d;
            pick = i;
          }
        }
        if (pick < 0) break;
        cand[(size_t)pick] = 1;
      }
      removed[(size_t)pick] = 1;
      stack.push_back(pick);
    }

    std::vector<int32_t> c((size_t)n, -1);
    bool ok = true;
    for (int32_t si = (int32_t)stack.size() - 1; si >= 0; --si) {
      int32_t i = stack[(size_t)si];
      int32_t K = Kof(i);
      std::vector<char> used(32, 0);
      for (int32_t j = 0; j < n; ++j)
        if (adj[(size_t)i][(size_t)j] && c[(size_t)j] >= 0)
          used[(size_t)c[(size_t)j]] = 1;
      for (int pr = 0; pr < 32; ++pr)
        if (forbid[(size_t)i][(size_t)pr]) used[(size_t)pr] = 1;
      const int *pal = crossing[(size_t)nodes[(size_t)i]] ? calPhys : fullPhys;
      int32_t ch = -1;
      for (int32_t ci = 0; ci < K; ++ci) {
        if (!used[(size_t)pal[ci]]) {
          ch = pal[ci];
          break;
        }
      }
      if (ch < 0) {
        ok = false; // actual spill
        continue;
      }
      c[(size_t)i] = ch;
    }
    for (int32_t i = 0; i < n; ++i)
      if (c[(size_t)i] >= 0) color[(size_t)nodes[(size_t)i]] = c[(size_t)i];
    return ok;
  }

  // Per-file mask (one 32-bit char row per graph node) of physical argument
  // registers that the node's value must not occupy.  At function entry each
  // register-passed parameter is still sitting in its ABI register until its
  // Entry instruction reads it, so the destination of an earlier Entry may
  // not reuse an argument register that a later Entry still has to read.
  std::vector<std::vector<char>> entryForbid(
      const std::vector<int32_t> &nodes, const std::vector<int32_t> &idx,
      bool isF) const {
    std::vector<std::vector<char>> out((size_t)nodes.size(),
                                       std::vector<char>(32, 0));
    std::vector<int32_t> defs, srcs;
    for (auto &b : f.blocks)
      for (auto &m : b.instrs) {
        bool e = isF ? m.op == MOp::EntryFlt : m.op == MOp::EntryInt;
        if (!e || m.dst < 0) continue;
        defs.push_back(m.dst);
        srcs.push_back(m.imm);
      }
    for (size_t i = 0; i < defs.size(); ++i) {
      int32_t di = idx[(size_t)defs[(size_t)i]];
      if (di < 0) continue;
      for (size_t k = i + 1; k < srcs.size(); ++k)
        if (srcs[k] >= 0 && srcs[k] < 32) out[(size_t)di][(size_t)srcs[k]] = 1;
    }
    return out;
  }

  // insert spill loads/stores for vregs marked in `spilled`.  Spill slot
  // numbers keep growing across rounds (f.spillCount) so that values spilled
  // in earlier rounds never share a memory slot with later spills.
  void spillRound(const std::vector<char> &spilled) {
    int n = maxReg();
    std::vector<int32_t> slot((size_t)n, -1);
    int nextSlot = f.spillCount;
    for (int32_t r = 0; r < n; ++r)
      if (spilled[(size_t)r]) slot[(size_t)r] = nextSlot++;

    for (auto &b : f.blocks) {
      std::vector<MInst> out;
      out.reserve(b.instrs.size() * 2 + 4);
      for (auto &m : b.instrs) {
        std::unordered_map<int32_t, int32_t> tmpFor;
        auto loadUse = [&](int32_t r) -> int32_t {
          if (r < 0 || slot[(size_t)r] < 0) return r;
          auto it = tmpFor.find(r);
          if (it != tmpFor.end()) return it->second;
          MInst t;
          bool fileF = isF[(size_t)r];
          t.op = fileF ? MOp::SpillFlw : MOp::SpillLw;
          t.dst = nreg++;
          t.imm = slot[(size_t)r];
          t.ty = fileF ? Type::F32 : Type::I32;
          out.push_back(t);
          tmpFor[r] = t.dst;
          return t.dst;
        };

        MInst m2 = m;
        m2.a = loadUse(m.a);
        m2.b = loadUse(m.b);
        for (auto &a : m2.args) a.vreg = loadUse(a.vreg);
        out.push_back(m2);

        if (m2.dst >= 0 && slot[(size_t)m2.dst] >= 0) {
          bool fileF = isF[(size_t)m2.dst];
          int32_t t = nreg++;
          out.back().dst = t;
          MInst st;
          st.op = fileF ? MOp::SpillFsw : MOp::SpillSw;
          st.a = t;
          st.imm = slot[(size_t)m.dst];
          st.ty = fileF ? Type::F32 : Type::I32;
          out.push_back(st);
        }
      }
      b.instrs = std::move(out);
    }
    f.spillCount = nextSlot;
  }
};

} // namespace

// ---------------------------------------------------------------------------
void runRegisterAlloc(std::vector<MachineFunc> &fns) {
  for (auto &f : fns) {
    Allocator A(f);
    A.trimDead();
    if (getenv("SAKURA_DUMP_RA")) {
      fprintf(stderr, "== func %s ==\n", f.name.c_str());
      for (auto &b : f.blocks) {
        fprintf(stderr, "-- block %s --\n", b.name.c_str());
        for (auto &m : b.instrs) {
          fprintf(stderr, "  op=%-3d dst=%d a=%d b=%d imm=%d sym=%s args=[",
                  (int)m.op, m.dst, m.a, m.b, m.imm, m.sym.c_str());
          for (auto &x : m.args) fprintf(stderr, "%d ", x.vreg);
          fprintf(stderr, "]\n");
        }
      }
    }
    A.nreg = A.maxReg();
    A.buildCFG();

    std::vector<int32_t> color; // vreg -> physical register
    int round = 0;
    for (; round < 24; ++round) {
      A.nreg = A.maxReg();
      A.computeFiles();
      std::vector<char> crossing;
      A.computeLiveness(crossing);
      A.buildCFG();
      A.buildGraph();

      int n = A.maxReg();
      color.assign((size_t)n, -1);
      std::vector<int32_t> cX((size_t)n, -1), cF((size_t)n, -1);
      auto fX = A.entryForbid(A.xNodes, A.xIdx, false);
      auto fF = A.entryForbid(A.fNodes, A.fIdx, true);
      bool okX = A.colorFile(A.xNodes, A.adjX, crossing, fX, cX,
                             kXCalleePhys, kXCallee, kXFullPhys, kXFull);
      bool okF = A.colorFile(A.fNodes, A.adjF, crossing, fF, cF,
                             kFCalleePhys, kFCallee, kFFullPhys, kFFull);
      if (getenv("SAKURA_DUMP_RA")) {
        fprintf(stderr, "-- colors round %d (okX=%d okF=%d)\n", round, okX,
                okF);
        for (size_t i = 0; i < A.xNodes.size(); ++i) {
          int32_t r = A.xNodes[i];
          fprintf(stderr, "  x%-3d -> %d\n", r, cX[(size_t)r]);
          fprintf(stderr, "    neighbors:");
          for (size_t j = 0; j < A.xNodes.size(); ++j)
            if (A.adjX[i][j]) fprintf(stderr, " x%d", A.xNodes[j]);
          fprintf(stderr, "\n");
        }
      }
      if (okX && okF) {
        for (int32_t r = 0; r < n; ++r)
          color[(size_t)r] = cX[(size_t)r] >= 0 ? cX[(size_t)r] : cF[(size_t)r];
        break;
      }
      std::vector<char> spilled((size_t)n, 0);
      for (int32_t r : A.xNodes)
        if (cX[(size_t)r] < 0) spilled[(size_t)r] = 1;
      for (int32_t r : A.fNodes)
        if (cF[(size_t)r] < 0) spilled[(size_t)r] = 1;
      A.spillRound(spilled);
    }

    // record used callee-saved registers (from colouring)
    for (int32_t r = 0; r < (int32_t)color.size(); ++r) {
      if (color[(size_t)r] < 0) continue;
      if (A.isF[(size_t)r]) {
        if (isCalleeF(color[(size_t)r]))
          f.usedCSF.push_back(color[(size_t)r]);
      } else if (isCalleeX(color[(size_t)r])) {
        f.usedCSX.push_back(color[(size_t)r]);
      }
    }
    std::sort(f.usedCSX.begin(), f.usedCSX.end());
    f.usedCSX.erase(std::unique(f.usedCSX.begin(), f.usedCSX.end()),
                    f.usedCSX.end());
    std::sort(f.usedCSF.begin(), f.usedCSF.end());
    f.usedCSF.erase(std::unique(f.usedCSF.begin(), f.usedCSF.end()),
                    f.usedCSF.end());

    // bake physical registers into the machine code
    for (auto &b : f.blocks)
      for (auto &m : b.instrs) {
        auto mapOp = [&](int32_t &r) {
          if (r >= 0 && r < (int32_t)color.size()) r = color[(size_t)r];
        };
        mapOp(m.dst);
        mapOp(m.a);
        mapOp(m.b);
        for (auto &a : m.args) mapOp(a.vreg);
      }
    for (auto &b : f.blocks)
      for (auto &m : b.instrs)
        if (m.op == MOp::Call) f.hasCall = true;
  }
}

} // namespace backend
} // namespace sakura
