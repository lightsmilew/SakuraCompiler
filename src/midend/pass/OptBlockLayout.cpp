// Static weighted trace placement (Pettis/Hansen's top-down algorithm).
// Loop membership dominates the weight; connected call-rich blocks break ties.
// SSA definition order is an additional hard constraint of this backend.
#include "Opt.h"
#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace sakura { namespace ir { namespace {
bool layout(Function &f) {
  const size_t n = f.blocks.size();
  if (n < 3 || n > 2048) return false;
  std::unordered_map<Value *, size_t> owner;
  for (size_t b = 0; b < n; ++b) {
    owner[f.blocks[b].get()] = b;
    for (auto &i : f.blocks[b]->instrs) owner[i.get()] = b;
  }
  std::vector<std::vector<size_t>> succ(n), pred(n), dep(n);
  std::vector<int> calls(n), depth(n), pending(n);
  for (size_t b = 0; b < n; ++b) {
    auto &bb = *f.blocks[b];
    for (auto &i : bb.instrs) calls[b] += i->op == Op::Call;
    if (bb.instrs.empty()) return false;
    auto &t = *bb.instrs.back();
    if (t.op == Op::Br || t.op == Op::CondBr) {
      for (size_t p = t.op == Op::Br ? 0 : 1; p < t.ops.size(); ++p) {
        auto it = owner.find(t.ops[p]); if (it == owner.end()) return false;
        size_t s = it->second;
        if (std::find(succ[b].begin(), succ[b].end(), s) == succ[b].end()) {
          succ[b].push_back(s); pred[s].push_back(b);
        }
      }
    }
  }
  std::vector<uint8_t> reached(n); std::vector<size_t> work{0}; reached[0] = 1;
  while (!work.empty()) { size_t b = work.back(); work.pop_back();
    for (size_t s : succ[b]) if (!reached[s]) { reached[s] = 1; work.push_back(s); } }
  // Dominators as bit sets, then union natural loops sharing a header. Counting
  // each backedge separately would spuriously inflate multi-latch loop heat.
  size_t words = (n + 63) / 64;
  std::vector<std::vector<uint64_t>> dom(n, std::vector<uint64_t>(words, ~uint64_t(0)));
  std::fill(dom[0].begin(), dom[0].end(), 0); dom[0][0] = 1;
  for (size_t b = 1; b < n; ++b) if (!reached[b]) std::fill(dom[b].begin(), dom[b].end(), 0);
  bool change = true;
  for (size_t round = 0; change && round < n; ++round) {
    change = false;
    for (size_t b = 1; b < n; ++b) if (reached[b]) {
      std::vector<uint64_t> d(words, ~uint64_t(0));
      for (size_t p : pred[b]) if (reached[p]) for (size_t w = 0; w < words; ++w) d[w] &= dom[p][w];
      d[b / 64] |= uint64_t(1) << (b % 64);
      if (d != dom[b]) { dom[b] = std::move(d); change = true; }
    }
  }
  if (change) return false;
  auto dominates = [&](size_t a, size_t b) { return (dom[b][a / 64] >> (a % 64)) & 1; };
  for (size_t h = 0; h < n; ++h) if (reached[h]) {
    std::vector<uint8_t> loop(n); loop[h] = 1; work.clear();
    for (size_t p : pred[h]) if (dominates(h, p) && p != h) { loop[p] = 1; work.push_back(p); }
    bool isLoop = !work.empty() || std::find(succ[h].begin(), succ[h].end(), h) != succ[h].end();
    while (!work.empty()) { size_t b = work.back(); work.pop_back();
      for (size_t p : pred[b]) if (!loop[p] && reached[p]) { loop[p] = 1; work.push_back(p); } }
    if (isLoop) for (size_t b = 0; b < n; ++b) depth[b] += loop[b];
  }
  auto dependency = [&](size_t a, size_t b) {
    if (a == b) return;
    if (std::find(dep[a].begin(), dep[a].end(), b) == dep[a].end()) {
      dep[a].push_back(b); ++pending[b];
    }
  };
  // Dominators first: preserve CFG structure even if a block uses no values.
  for (size_t b = 1; b < n; ++b) if (reached[b])
    for (size_t a = 0; a < n; ++a) if (a != b && dominates(a, b)) dependency(a, b);
  for (size_t b = 0; b < n; ++b) for (auto &u : f.blocks[b]->instrs) {
    auto &i = *u;
    for (size_t p = 0; p < i.ops.size(); ++p) {
      auto *def = dynamic_cast<Instruction *>(i.ops[p]);
      if (!def || def->op == Op::Phi) continue; // Phi registers are prebound.
      auto it = owner.find(def); if (it == owner.end()) continue;
      size_t target = b;
      if (i.op == Op::Phi) {
        if (p % 2 == 0) continue;
        auto predIt = owner.find(i.ops[p - 1]); if (predIt == owner.end()) return false;
        target = predIt->second; // Incoming copies are selected in the predecessor.
      }
      dependency(it->second, target);
    }
  }
  if (pending[0]) return false;
  std::vector<size_t> order; order.reserve(n); std::vector<uint8_t> placed(n);
  while (order.size() < n) {
    size_t best = n; int64_t bestScore = INT64_MIN;
    size_t last = order.empty() ? 0 : order.back();
    for (size_t b = 0; b < n; ++b) if (!placed[b] && !pending[b]) {
      if (order.empty() && b != 0) continue;
      bool adjacent = std::find(succ[last].begin(), succ[last].end(), b) != succ[last].end();
      // Call count is a bounded tie breaker, never a substitute for a profile.
      // Original order supplies deterministic ties and handles unknown paths.
      int64_t score = (reached[b] ? 1000000 : 0) + std::min(depth[b], 32) * 10000 +
        (adjacent ? 1000 : 0) + (adjacent ? std::min(calls[b] + calls[last], 8) * 32 : 0);
      if (b == last + 1) score += 1;
      if (score > bestScore) { best = b; bestScore = score; }
    }
    // Cyclic definition constraints or unsupported CFG: preserve the full
    // original layout atomically, rather than emit a partially valid order.
    if (best == n) return false;
    placed[best] = 1; order.push_back(best);
    for (size_t b : dep[best]) --pending[b];
  }
  bool changed = false;
  for (size_t i = 0; i < n; ++i) changed |= order[i] != i;
  if (!changed) return false;
  decltype(f.blocks) blocks; blocks.reserve(n);
  for (size_t b : order) blocks.push_back(std::move(f.blocks[b]));
  f.blocks = std::move(blocks); // Object identities, CFG and phi inputs survive.
  return true;
}
class BlockLayoutPass final : public Pass {
public:
  const char *name() const override { return "block-layout"; }
  Layer inLayer() const override { return Layer::Cf; }
  bool run(Module &m) override { bool changed = false;
    for (auto &f : m.functions()) if (!f->isLib) changed |= layout(*f);
    return changed; }
};
} // namespace
std::unique_ptr<Pass> makeBlockLayoutPass() { return std::make_unique<BlockLayoutPass>(); }
} }
