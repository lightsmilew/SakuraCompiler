# Development tools

All repository tooling lives here. Paths are resolved from each script, so
commands work from another working directory as well as the repository root.

| Directory | Purpose | Entry points |
| --- | --- | --- |
| `build/` | Configure and build the compiler | `build.sh` |
| `compile/` | Compile sources and produce reference assembly | `compile_dir.sh`, `llvm_ref.sh`, `run.sh`, `sysy_prelude.h` |
| `test/` | Host/guest correctness checks | `test.sh`, `qemu_verify.sh`, `guest_test.sh` |
| `benchmark/` | Validated timing and allocator comparisons | `benchmark.py`, `benchmark_guest.py`, `benchmark_report.py`, `register_pressure.py`, `move_audit.py`, `branch_audit.py` |
| `frontend/` | Parser generation dependency | ANTLR jar |
| `common/` | Shared tooling support | `memjobs.sh` |

Build artifacts and full executable payloads default to `build/benchmarks/`.
Publish comparison tables, measurements, compiler/input hashes and pressure
reports under `results/`. Keep `docs/` for architecture and optimization design.
Each focused recheck should use its own output directory to preserve the full
suite's manifest and raw measurements.
`BUILD_DIR` is shared by the build, compile and test entries (relative to the
repository or absolute); `COMP` overrides the compiler for compile/test entries.

```sh
bash tools/build/build.sh
bash tools/compile/compile_dir.sh cases/optimization -o build/assembly -O2
bash tools/compile/llvm_ref.sh cases/optimization/12_pointer_stream.sy build/reference.s
bash tools/test/test.sh functional h_functional performance2026 tensor optimization regression
python3 tools/benchmark/benchmark.py --variant before=build/compiler.before \
  --variant after=build/compiler --llvm clang-18 --library build/libsysy_riscv.a
python3 tools/benchmark/register_pressure.py build/benchmarks/performance2026 \
  --out results/pointer-strength-reduction/pressure
python3 tools/benchmark/move_audit.py build/benchmarks/performance2026 \
  --variants before after llvm --out results/move-optimization/copies
python3 tools/benchmark/branch_audit.py build/benchmarks/performance2026 \
  --variants before after llvm --out results/assembly-layout/branches
```

`move_audit.py` separates `mv`/`fmv.s` from cross-file bit materialisation,
and validates their counts against `backend-copies` diagnostics when available.
Older compiler/LLVM logs without these diagnostics report unknown origins.
Register argument shuffle loads/stores are reported separately from RA spills.

`branch_audit.py` checks local branch symbols and counts conditional branches,
unconditional jumps, numeric labels and jumps to the next textual instruction. These
are static counts; labels and QEMU timings do not measure hardware prediction.
Implicit assembler alignment padding is not counted, so such a jump in LLVM
output need not be redundant in the final machine code.
