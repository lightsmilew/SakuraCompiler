// Graph-colouring register allocator implementation: liveness on each block,
// interference graph building, colouring with a spill fallback that rewrites
// virtual registers through explicit spill-slot loads/stores, then fixing up
// callee-saved register usage and call-clobbered temporaries.
#include "RA.h"

#include <algorithm>
#include <cassert>
#include <climits>
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

  // Instruction kinds whose result can be recomputed from scratch anywhere in
  // the function without reading any register or memory: integer / float
  // constant materialisation.  Recomputation is preferred over spilling for
  // these - `li` is one cheap instruction with no memory access, whereas a
  // spill costs a store plus a load (and, in a loop, the reload repeats on
  // every iteration).  `LeaGlobal` / `LeaFrame` are deliberately *not* here:
  // an `la` expands to two instructions, so keeping the address alive in a
  // register is better than rematerialising it inside a loop.
  static bool isRematerializable(MOp op) {
    switch (op) {
    case MOp::Li:
    case MOp::LiWide:
    case MOp::LiF:
      return true;
    default:
      return false;
    }
  }

  // vreg -> a copy of its *single* defining instruction, for vregs whose
  // definition is rematerialisable.  A vreg with several definitions (ISel
  // reuses a phi's vreg across predecessor copies) is excluded: recomputing
  // one of them would not stand in for the others.  The defining instructions
  // are copied because spillRound rewrites the block vectors in place.
  std::unordered_map<int32_t, MInst> rematCandidates() const {
    std::unordered_map<int32_t, int> defCount;
    std::unordered_map<int32_t, MInst> def;
    for (auto &b : f.blocks)
      for (auto &m : b.instrs) {
        if (m.dst < 0) continue;
        ++defCount[m.dst];
        if (isRematerializable(m.op))
          def[m.dst] = m;
        else
          def.erase(m.dst);
      }
    // Reading a vreg costs the same as redefining it, so only clone where the
    // result is actually used; and cap the fan-out so a value read all over a
    // loop body is not expanded into a long run of `li`s.
    std::unordered_map<int32_t, int> reads;
    auto countReads = [&](const MInst &m) {
      OpSlots s = slots(m);
      for (int i = 0; i < 3; ++i)
        if (s.use[i] >= 0) ++reads[s.use[i]];
      if (m.op == MOp::Ret && m.a >= 0) ++reads[m.a];
      if (m.op == MOp::Call)
        for (const MArg &a : m.args)
          if (a.vreg >= 0) ++reads[a.vreg];
    };
    for (auto &b : f.blocks)
      for (auto &m : b.instrs) countReads(m);
    std::unordered_map<int32_t, MInst> out;
    for (auto &kv : def) {
      int32_t r = kv.first;
      if (defCount[r] != 1) continue;
      auto it = reads.find(r);
      if (it == reads.end() || it->second < 1 || it->second > 8) continue;
      out[r] = kv.second;
    }
    return out;
  }

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
          case MOp::Li: case MOp::LiWide: case MOp::LiF: case MOp::MoveX:
          case MOp::MoveF:
          case MOp::LeaFrame: case MOp::LeaGlobal:
          case MOp::EntryInt: case MOp::EntryFlt:
          case MOp::EntryStkInt: case MOp::EntryStkFlt:
          case MOp::IAdd: case MOp::IAddI: case MOp::ISub: case MOp::INeg:
          case MOp::IMul: case MOp::IDiv: case MOp::IRem: case MOp::Mulh:
          case MOp::Mulhu: case MOp::Mul64:
          case MOp::SllI: case MOp::SrlI: case MOp::SraI:
          case MOp::Shl64I: case MOp::Shr64I: case MOp::Sar64I:
          case MOp::IXor: case MOp::IXorI:
          case MOp::IAnd: case MOp::IAndI:
          case MOp::IOr:
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

  // ---- spill cost --------------------------------------------------------
  // Estimated cost of spilling each virtual register.  LLVM's greedy allocator
  // orders spill candidates by `weight / degree`, where the weight of a live
  // range is the sum over its defs and uses of a loop-depth multiplier
  // (1 << depth, capped).  The point is the *exponential* factor: a value that
  // is live across a loop body is reloaded on every iteration, so spilling it
  // costs #iterations x more than spilling a value used once in the entry
  // block.  Without that factor a Chaitin-Briggs simplify phase that gets
  // stuck picks the highest-degree node, which is usually the hottest address
  // or accumulator, and the reloads land straight inside the innermost loop.
  std::vector<int64_t> spillWeight;

  // Loop nesting depth per block, from the DFS back edges.  An edge u->v where
  // v is still on the current DFS path closes a natural loop; every block on
  // the path from v to u lies in its body and gets one more level of depth.
  // This over-approximates when loops share a header (the on-path blocks are
  // all marked), which only makes the *relative* weights more conservative --
  // blocks outside every loop still end up with a strictly smaller multiplier.
  void computeLoopDepth(std::vector<int> &depth) const {
    const int32_t nb = (int32_t)f.blocks.size();
    depth.assign((size_t)nb, 0);
    std::vector<char> state((size_t)nb, 0); // 0 = new, 1 = on path, 2 = done
    std::vector<int32_t> path;
    struct Frame {
      int32_t u;
      size_t next;
    };
    std::vector<Frame> st;
    for (int32_t root = 0; root < nb; ++root) {
      if (state[(size_t)root]) continue;
      state[(size_t)root] = 1;
      path.push_back(root);
      st.push_back({root, 0});
      while (!st.empty()) {
        Frame &fr = st.back();
        const auto &sc = succ[(size_t)fr.u];
        if (fr.next < sc.size()) {
          int32_t v = sc[fr.next++];
          if (v < 0 || v >= nb) continue;
          if (state[(size_t)v] == 1) {
            for (size_t k = path.size(); k-- > 0;) {
              ++depth[(size_t)path[k]];
              if (path[k] == v) break;
            }
          } else if (state[(size_t)v] == 0) {
            state[(size_t)v] = 1;
            path.push_back(v);
            st.push_back({v, 0});
          }
        } else {
          state[(size_t)fr.u] = 2;
          path.pop_back();
          st.pop_back();
        }
      }
    }
  }

  void computeSpillWeights() {
    spillWeight.assign((size_t)std::max(1, nreg), 0);
    if (f.blocks.empty()) return;
    // With the weights all zero the stuck branch degenerates to "highest
    // degree wins", i.e. the behaviour before loop-depth weighting, so this
    // doubles as an exact A/B switch for benchmarking.
    static const bool off = std::getenv("SAKU_NO_SPILLW") != nullptr;
    if (off) return;
    std::vector<int> depth;
    computeLoopDepth(depth);
    for (size_t bi = 0; bi < f.blocks.size(); ++bi) {
      int d = depth[bi];
      if (d > 12) d = 12;
      const int64_t w = (int64_t)1 << d;
      for (const auto &m : f.blocks[bi].instrs) {
        OpSlots sl = slots(m);
        if (sl.def >= 0 && sl.def < (int32_t)spillWeight.size())
          spillWeight[(size_t)sl.def] += w;
        for (int k = 0; k < 3; ++k)
          if (sl.use[k] >= 0 && sl.use[k] < (int32_t)spillWeight.size())
            spillWeight[(size_t)sl.use[k]] += w;
      }
    }
  }

  int64_t weightOf(int32_t r) const {
    if (r < 0 || r >= (int32_t)spillWeight.size()) return 0;
    return spillWeight[(size_t)r];
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
      // live_in = (live_out - def) + uses.  The order matters: an instruction
      // may read and write the *same* virtual register (a coalesced in-place
      // induction update `addiw iv, iv, 4`, or any two-address op).  Killing
      // the definition must therefore happen before the uses are added back,
      // otherwise the self-use is erased and the value looks dead before the
      // instruction, which under-approximates interference and can colour two
      // simultaneously-live values into one register.
      auto kill = [&](int r) {
        if (r >= 0 && r < n) live[(size_t)r] = 0;
      };
      kill(sl.def);
      auto touch = [&](int r) {
        if (r >= 0 && r < n) live[(size_t)r] = 1;
      };
      for (int j = 0; j < 3; ++j) touch(sl.use[j]);
      for (auto &a : m.args) touch(a.vreg);
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
        // same order as transfer(): kill the definition, then add the uses,
        // so an in-place read-modify-write keeps its source live.
        if (sl.def >= 0 && sl.def < n) live[(size_t)sl.def] = 0;
        auto touch = [&](int r) {
          if (r >= 0 && r < n) live[(size_t)r] = 1;
        };
        for (int j = 0; j < 3; ++j) touch(sl.use[j]);
        for (auto &a : m.args) touch(a.vreg);
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
        // kill the definition first, then re-add uses (see transfer()): an
        // instruction that reads and writes the same register keeps that
        // register live across the instruction.
        if (def >= 0 && def < n) live[(size_t)def] = 0;
        for (int j = 0; j < 3; ++j) touch(sl.use[j]);
        for (auto &a : m.args) touch(a.vreg);
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
                 std::vector<int32_t> &color,
                 const std::vector<int32_t> &idx,
                 const std::unordered_map<int32_t, std::vector<int32_t>> &partners,
                 const int *calPhys, int Kcal, const int *fullPhys,
                 int Kfull, bool degreeRule) {
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
        // No node can be simplified, so one of them has to live in memory.
        // Two candidate rules:
        //
        //  * cost (default): cheapest cost per interference edge removed,
        //    LLVM's greedy-allocator ordering by weight/degree.  The cost of a
        //    live range grows with how often it is touched, which is
        //    exponential in loop depth (a value live across a loop is reloaded
        //    once per trip), so this keeps the accumulator and the innermost
        //    address in registers.
        //  * degree (fallback): the classical highest-degree rule.  The cost
        //    rule is only a heuristic and can stall: spilling the cheapest
        //    node may remove no edges at all, so the retry loop keeps
        //    uncolouring a new set of vregs while the spill code it inserts
        //    adds more pressure than the spill removed (measured on
        //    functional/87_many_params: +10 vregs per round, never
        //    converging).  The caller switches to this rule as soon as a retry
        //    fails to shrink the uncoloured set.
        int32_t best = -1;
        if (degreeRule) {
          int32_t maxd = -1;
          for (int32_t i = 0; i < n; ++i) {
            if (removed[(size_t)i]) continue;
            int32_t d = degree(i);
            if (d > maxd) {
              maxd = d;
              best = i;
            }
          }
        } else {
          int64_t bestW = 0;
          int32_t bestD = -1;
          for (int32_t i = 0; i < n; ++i) {
            if (removed[(size_t)i]) continue;
            int64_t w = weightOf(nodes[(size_t)i]);
            int32_t d = degree(i);
            if (d < 1) d = 1;
            if (best < 0) {
              best = i;
              bestW = w;
              bestD = d;
              continue;
            }
            // cross-multiplied w/d < bestW/bestD
            bool better = w * (int64_t)bestD < bestW * (int64_t)d;
            if (!better && w == bestW && d > bestD) better = true;
            if (better) {
              best = i;
              bestW = w;
              bestD = d;
            }
          }
        }
        pick = best;
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
      // Copy-coalescing hint: try a register already chosen for a value this
      // one is copied to/from.  If they end up in the same register the copy
      // vanishes (self move).  This only reorders the choice among colours
      // that are already legal (unused by an interfering neighbour and not
      // forbidden), so it can never introduce an interference violation.
      auto pit = partners.find(nodes[(size_t)i]);
      if (pit != partners.end()) {
        for (int32_t pv : pit->second) {
          if (pv < 0 || pv >= (int32_t)idx.size()) continue;
          int32_t pj = idx[(size_t)pv];
          if (pj < 0) continue;
          int32_t pc = c[(size_t)pj];
          if (pc < 0 || pc >= 32 || used[(size_t)pc]) continue;
          bool inPal = false;
          for (int32_t ci = 0; ci < K; ++ci)
            if (pal[ci] == pc) {
              inPal = true;
              break;
            }
          if (inPal) {
            ch = pc;
            break;
          }
        }
      }
      for (int32_t ci = 0; ch < 0 && ci < K; ++ci) {
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
  //
  // Values in `remat` are not spilled at all: their single definition is a
  // constant or an address materialisation, so it is cheaper (and, in a loop,
  // removes a memory access from every iteration) to recompute the value at
  // each use than to reload it from a slot.  Every read of such a vreg is
  // replaced by a fresh copy of the defining instruction and the original
  // definition is dropped, so the vreg disappears exactly like a spilled one.
  void spillRound(const std::vector<char> &spilled,
                  const std::unordered_map<int32_t, MInst> &cands) {
    int n = maxReg();
    // Rematerialise only what would otherwise go to a spill slot.  This is
    // LLVM's rule (a clone stands in for a reload, it is not a substitute for
    // keeping the value in a register) and it matters a lot in loops: machine
    // LICM hoists one `li` per constant into the preheader so the loop body
    // reads a register, and cloning every read would put the `li` straight
    // back inside the body - undoing the hoist and paying for the *definition*
    // of the constant on every iteration.
    std::unordered_map<int32_t, MInst> remat;
    for (const auto &kv : cands)
      if (kv.first >= 0 && kv.first < (int32_t)spilled.size() &&
          spilled[(size_t)kv.first])
        remat.emplace(kv.first, kv.second);
    std::vector<int32_t> slot((size_t)n, -1);
    int nextSlot = f.spillCount;
    for (int32_t r = 0; r < n; ++r)
      if (spilled[(size_t)r] && !remat.count(r)) slot[(size_t)r] = nextSlot++;

    for (auto &b : f.blocks) {
      std::vector<MInst> out;
      out.reserve(b.instrs.size() * 2 + 4);
      for (auto &m : b.instrs) {
        std::unordered_map<int32_t, int32_t> tmpFor;
        auto loadUse = [&](int32_t r) -> int32_t {
          if (r < 0) return r;
          auto rit = remat.find(r);
          if (rit != remat.end()) {
            auto it = tmpFor.find(r);
            if (it != tmpFor.end()) return it->second;
            MInst c = rit->second; // clone the cheap definition
            c.dst = nreg++;
            out.push_back(c);
            tmpFor[r] = c.dst;
            return c.dst;
          }
          if (slot[(size_t)r] < 0) return r;
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
    // every read of a rematerialised vreg was rewritten, so its original
    // definition (which no longer has a slot) is now dead
    if (!remat.empty())
      for (auto &b : f.blocks) {
        auto &v = b.instrs;
        for (size_t i = 0; i < v.size();) {
          if (v[i].dst >= 0 && remat.count(v[i].dst))
            v.erase(v.begin() + (long)i);
          else
            ++i;
        }
      }
    f.spillCount = nextSlot;
  }
};

} // namespace

// ---------------------------------------------------------------------------
void RegisterAllocator::allocate(std::vector<MachineFunc> &fns) {
  for (auto &f : fns) {
    Allocator A(f);
    A.trimDead();
    A.nreg = A.maxReg();
    A.buildCFG();

    std::vector<int32_t> color; // vreg -> physical register
    int prevUncolored = INT32_MAX;
    // One colouring attempt.  Fills `color` and returns true on success; on
    // failure it rewrites the function to spill what could not be coloured
    // (`forceAll` spills every vreg, the last resort) and reports the number
    // of uncoloured vregs through `uncolored`.
    auto attempt = [&](bool forceAll, bool degreeRule,
                       int *uncolored) -> bool {
      A.nreg = A.maxReg();
      A.computeFiles();
      std::vector<char> crossing;
      A.computeLiveness(crossing);
      A.buildCFG();
      A.computeSpillWeights();
      A.buildGraph();

      int n = A.maxReg();
      color.assign((size_t)n, -1);
      std::vector<int32_t> cX((size_t)n, -1), cF((size_t)n, -1);
      auto fX = A.entryForbid(A.xNodes, A.xIdx, false);
      auto fF = A.entryForbid(A.fNodes, A.fIdx, true);
      // Move partners (both directions) so colouring can keep a copy's two
      // ends in one register and retire the copy.
      std::unordered_map<int32_t, std::vector<int32_t>> partners;
      for (auto &b : f.blocks)
        for (auto &m : b.instrs)
          if ((m.op == MOp::MoveX || m.op == MOp::MoveF) && m.dst >= 0 &&
              m.a >= 0) {
            partners[m.dst].push_back(m.a);
            partners[m.a].push_back(m.dst);
          }
      bool okX = A.colorFile(A.xNodes, A.adjX, crossing, fX, cX, A.xIdx,
                             partners, kXCalleePhys, kXCallee, kXFullPhys,
                             kXFull, degreeRule);
      bool okF = A.colorFile(A.fNodes, A.adjF, crossing, fF, cF, A.fIdx,
                             partners, kFCalleePhys, kFCallee, kFFullPhys,
                             kFFull, degreeRule);
      if (okX && okF) {
        for (int32_t r = 0; r < n; ++r)
          color[(size_t)r] = cX[(size_t)r] >= 0 ? cX[(size_t)r] : cF[(size_t)r];
        return true;
      }
      std::vector<char> spilled((size_t)n, forceAll ? 1 : 0);
      int nUncolored = 0;
      if (!forceAll) {
        for (int32_t r : A.xNodes)
          if (cX[(size_t)r] < 0) {
            spilled[(size_t)r] = 1;
            ++nUncolored;
          }
        for (int32_t r : A.fNodes)
          if (cF[(size_t)r] < 0) {
            spilled[(size_t)r] = 1;
            ++nUncolored;
          }
      }
      if (uncolored) *uncolored = nUncolored;
      A.spillRound(spilled, forceAll ? std::unordered_map<int32_t, MInst>{}
                                     : A.rematCandidates());
      return false;
    };

    bool converged = false;
    bool degreeRule = false;
    for (int round = 0; round < 24 && !converged; ++round) {
      int uncolored = 0;
      converged = attempt(false, degreeRule, &uncolored);
      // Livelock guard: if spilling did not shrink the uncoloured set, the
      // cost heuristic is not breaking whatever clique is blocking us (it can
      // pick a node of degree 0 and remove nothing).  Switch to the classical
      // highest-degree rule, which maximises the edges removed per spill.
      // Without this the retry loop can grow the function by ~10 vregs per
      // round and never converge.
      if (!converged && uncolored >= prevUncolored) degreeRule = true;
      prevUncolored = uncolored;
    }
    // Belt and braces: if even 24 retries failed there is no colouring at all
    // (`color` is still all -1), and baking that in would print -1 as a
    // register number.  Spill everything instead: every value is then loaded
    // immediately before its use and stored immediately after, so every live
    // range spans a single instruction and the next attempt must succeed.
    // The result is slow code, but only for a function the allocator already
    // could not handle, and it is correct.
    int fallbackGuard = 0;
    while (!converged && fallbackGuard++ < 8) {
      int uncolored = 0;
      converged = attempt(true, true, &uncolored);
    }
    assert(converged && "register allocation failed to converge");

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
