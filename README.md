# SakuraCompiler

A self-contained **SysY → RISC-V64gc** compiler written in C++17.

* **Front-end**: ANTLR4 lexer/parser (grammar vendored in `src/SysY.g4`) with a
  visitor AST builder and a full semantic analyser (scopes, typing, constant
  folding, array-initialiser flattening).
* **Mid-end**: an MLIR-style SSA IR organised into dialects
  (`func` / `arith` / `cf` / `scf` / `affine` / `memref` / `tensor`).  Loops
  stay first-class (`affine.for`, then `scf.while` region ops) until a final
  flattening to a flat CFG.  A pipeline of conversion + optimisation passes
  lowers the IR through three layers — **affine → scf → cf** — and a
  `verify-cf-only` gate guarantees that only flat-cf ops reach the back-end.
* **Back-end**: RISC-V instruction selection, a machine-level optimisation
  stage (constant strength reduction, peephole, load scheduling, block-local
  CSE, LICM), graph-colouring register allocation with iterative
  spill/rematerialisation, a post-allocation redundant-move pass, and GNU
  `as`-compatible assembly emission.

No third-party dependency beyond the vendored ANTLR4 runtime.  The mid- and
back-end "general optimisations" implemented here are the ones found in the
reference SysY compiler used during development (this workspace) but missing
in this project; they were ported into the same pass framework
(`src/midend/pass/Opt*.cpp`, `src/backend/MachineOpt.cpp`) and validated for
correctness against the shared test suite.

Verified end-to-end on a QEMU RISC-V64 Ubuntu guest (assembly linked against
the reference `libsysy_riscv.a`):

| suite                 | cases | status        |
|-----------------------|------:|---------------|
| `cases/functional`    |   100 | 100% passing  |
| `cases/h_functional`  |    40 | 100% passing  |
| `cases/performance2026`|   60 | 100% passing  |
| `cases/tensor`        |    25 | 100% passing (TensorType extension) |

---

## 1. Quick start

```bash
./scripts/build.sh                            # build the compiler
./scripts/compile_dir.sh cases/functional -o /tmp/out     # compile a suite to asm
./build/compiler cases/functional/00_main.sy -S           # single file -> stdout

# one-off mid-end IR views
./build/compiler in.sy --dump-affine
./build/compiler in.sy --dump-scf -o /dev/null > in.scf.mlir
./build/compiler in.sy --dump-cf  -o /dev/null > in.cf.mlir

# end-to-end correctness on the QEMU VM (must be running, see §6)
./scripts/test.sh
```

## 2. Build

Requirements: **CMake ≥ 3.10** and a **C++17** compiler.  ANTLR4 is vendored
under `3rd_party/`, so nothing is downloaded.

| command | effect |
|---|---|
| `./scripts/build.sh` | configure + incremental build |
| `./scripts/build.sh clean` | wipe `build/` and rebuild from scratch |
| `cmake --build build -j$(nproc)` | incremental rebuild of an existing tree |

The driver is produced at `build/compiler`.  `scripts/build.sh` honours the
`BUILD_DIR` and `JOBS` environment variables.

## 3. Command line

```
build/compiler <input.sy> [-S] [-o <out>] [-O0|-O1|-O2]
               [--dump-ir | --dump-affine | --dump-scf | --dump-cf | --dump-final]
               [--pass-stats]
```

| flag | meaning |
|---|---|
| `<input.sy>` | source file (SysY).  Exactly one input required. |
| `-S` | emit assembly (the default behaviour; flag kept for compatibility). |
| `-o <out>` | write the assembly to `<out>` instead of stdout. |
| `-O0` / `-O1` / `-O2` | optimisation level.  Default `-O2`. |
| `--dump-affine` | print the module at the affine layer (raw IRBuilder output) to **stdout**. |
| `--dump-scf` | print the module when the pipeline first enters the scf layer (after `affine.for` → `scf.while` lowering, before scf-layer optimisation). |
| `--dump-cf` | print the module when the pipeline first enters the flat-cf layer (right after loop flattening, before cf-layer optimisation). |
| `--dump-final` | print the module right before instruction selection (post-optimisation, pre-ISel). |
| `--dump-ir` | print **all three** layer views; suppresses assembly output. |
| `--pass-stats` | print a per-pass "changed/unchanged" report plus back-end stage counters. |

Notes on the layer views:

* The driver always runs the back-end, even when a `--dump-*` layer flag is
  given.  Assembly is therefore appended to stdout unless you pass `-o` (or
  use `--dump-ir`, which skips the assembly output).  To capture *only* an IR
  view of a file, redirect stdout and park the assembly:
  `./build/compiler in.sy --dump-cf -o /dev/null > in.cf.mlir`.
* `-O0` keeps only the structural conversions (tensor expansion,
  affine→scf, scf→cf) and disables the back-end machine-opt stage; `-O1`
  runs the same pass sequence as `-O2` with a lighter cleanup schedule.

### 3.1 Output format

Assembly is RISC-V64gc GNU `as` text with SysY runtime symbols unresolved
(`getint`, `putch`, `getfarray`, …).  Link it against the runtime library,
e.g. with `riscv64-linux-gnu-gcc -march=rv64gc -static`.

The IR printer emits MLIR-style text.  Every value is SSA and typed
(`i32` / `f32` / `ptr`); the same `while` loop looks like this at each layer:

```text
// ---------- affine layer (IRBuilder output) ----------
affine.for %1 = %5 to %4 step 1 {
  ^bb1:
    %6 = memref.load %2 : i32
    %7 = memref.load %1 : i32
    %8 = arith.addi %6, %7 : i32
    memref.store %8, %2 : i32
    affine.yield
}

// ---------- scf layer (structured loops as regions) ----------
scf.while {
  ^bb1:
    %4 = memref.load %0 : i32
    %5 = arith.cmpi slt, %4, %2 : i32
    scf.condition %5
} do {
  ^bb2:
    %6 = memref.load %1 : i32
    %7 = memref.load %0 : i32
    %8 = arith.addi %6, %7 : i32
    memref.store %8, %1 : i32
    cf.br ^bb3
  ^bb3:
    %9 = memref.load %0 : i32
    %10 = arith.addi %9, %c1 : i32
    memref.store %10, %0 : i32
    scf.yield
}

// ---------- cf layer (flat CFG) ----------
cf.br ^bb1
^bb1:
  %3 = memref.load %0 : i32
  %4 = arith.cmpi slt, %3, %2 : i32
  cf.cond_br %4, ^bb2, ^bb4
^bb2:
  ...
^bb3:
  %8 = arith.addi %6, %c1 : i32
  memref.store %8, %0 : i32
  cf.br ^bb1
```

## 4. Helper scripts

| script | purpose |
|---|---|
| `scripts/build.sh` | build (or clean-rebuild) the compiler. |
| `scripts/compile_dir.sh` | batch-compile every `.sy` in a directory, emitting assembly **or** an affine/scf/cf/final IR dump per file. |
| `scripts/run.sh` | legacy helper: `-build`, `-rebuild`, `-S file.sy`, `-qemu-test [suite...]`. |
| `scripts/test.sh` | build + end-to-end QEMU correctness gate over the suites. |
| `tools/qemu_verify.sh` | host side of the correctness gate (compile, push to VM, cache bookkeeping). |
| `tools/guest_test.sh` | guest side of the correctness gate (as + link + run + diff). |

### 4.1 `scripts/compile_dir.sh`

```
./scripts/compile_dir.sh DIR... -o OUTDIR [MODE] [compiler flags]
```

Modes (exactly one, default `--asm`):

| mode | artifact per file |
|---|---|
| `--asm` | `OUTDIR/<dir>/<base>.s` — RISC-V64 assembly |
| `--affine` | `OUTDIR/<dir>/<base>.affine.mlir` — affine-layer IR view |
| `--scf` | `OUTDIR/<dir>/<base>.scf.mlir` — scf-layer IR view |
| `--cf` | `OUTDIR/<dir>/<base>.cf.mlir` — cf-layer IR view |
| `--final` | `OUTDIR/<dir>/<base>.final.mlir` — pre-ISel module |
| `--ir` | `OUTDIR/<dir>/<base>.ir.txt` — all three layer views in one run |

Files land under `OUTDIR/<label>/`, where `<label>` is the basename of the
input directory, so compiling several suites into one `OUTDIR` never mixes
files.  Compiler flags (`-O0/-O1/-O2`, …) may be given anywhere and are
forwarded; IR-dump modes discard the always-generated assembly.

```bash
./scripts/compile_dir.sh cases/functional -o /tmp/out            # 100 .s files
./scripts/compile_dir.sh cases/tensor cases/h_functional \
    -o /tmp/out --cf -O1                                          # cf IR at O1
./scripts/compile_dir.sh cases/functional -o /tmp/out --asm -O0   # O0 assembly
```

`JOBS` (parallelism, default nproc ≤ 16), `COMP` (compiler path),
`KEEP_GOING` and `TIMEOUT_SECONDS` (default 180) are environment overrides.
Exit status is non-zero if any file fails; failed runs print per-file logs
(`OUTDIR/<label>/.failures.txt`).

## 5. Optimisation pipeline

### 5.1 IR layers and conversions

The IRBuilder produces the **affine** layer: counting loops remain
`affine.for`, whole-tensor ops are still present.  The pipeline then runs
`expand-tensor-ops` (lowers tensor ops to loop nests), `lower-affine-to-scf`
(`affine.for` → `scf.while` region ops), and `canonicalize-control-flow`
(flattens every structured loop into a flat `cf` CFG).  A final
`verify-cf-only` gate aborts if any tensor/affine/scf op survived, so the
module handed to the back-end is guaranteed flat cf.

### 5.2 Mid-end optimisation passes

Registered in `src/midend/pass/Pipeline.cpp`; each pass is a class in
`src/midend/pass/Opt*.cpp`.

| pass | layer | effect |
|---|---|---|
| `const-fold` | all | constant folding of arithmetic/comparison ops; folds `cond_br`/`scf.condition`/`affine.for` on constant conditions (pruning stale phi entries). |
| `algebraic` | all | algebraic simplifications (0/1 identities, `x−x`, …). |
| `mem-cse` | all | straight-line common-subexpression elimination over loads/stores. |
| `dead-code` | all | dead store / unreachable-result elimination. |
| `normalize-cmp` | cf | canonicalise comparisons. |
| `add-chain` | cf | reassociate address/induction add chains. |
| `licm` | affine, scf | loop-invariant code motion on structured loops. |
| `loop-unroll` | affine | full unrolling of small constant-trip `affine.for`. |
| `cfg-simplify` | cf | constant/identical-arm branch folding, empty-thunk collapsing, jump merging, unreachable-block removal. |
| `inline-small` | cf | inline small leaf helpers. |
| `tail-rec-elim` | cf | turn self tail calls into parameter-slot loops. |
| `inline-general` | cf | inline general non-recursive multi-block helpers (recursion reduced to a flat loop above becomes a plain inlineable body). |
| `mem2reg` | cf | SSA register promotion: lift non-escaping scalar slots out of memory. |
| `dom-cse` | cf | whole-CFG dominance CSE over the promoted phi/SSA values. |
| `remove-unused` | affine, cf | drop unused globals/functions/values. |
| `expand-tensor-ops` | affine | tensor-op → loop-nest lowering (see §7). |
| `lower-affine-to-scf`, `canonicalize-control-flow`, `verify-cf-only` | — | layer conversions / final gate. |

### 5.3 Back-end stages

`src/backend/Pipeline.cpp` runs five stages inside `RISCVBackend::run`:

1. **Instruction selection** (`ISel.cpp`) — pure-cf IR → machine IR, phi
   copies scheduled as parallel copies at edge ends.
2. **Machine optimisation** (`MachineOpt.cpp`, off at `-O0`) —
   *strength-reduce* (constant `div`/`rem`/`mul` → multiply-shift magic),
   *peephole*, *schedule* (load hoisting), *blockcse* (block-local CSE),
   *licm*.
3. **Register allocation** (`RA.cpp`) — Briggs-style graph colouring with
   iterative spill/rematerialise rounds; `a0-a7` / `fa0-fa7` argument
   registers reserved; 16-byte stack alignment; ABI outbound argument layout
   (register streams + stack overflow).
4. **Post-RA peephole** — redundant-move removal on physical registers.
5. **Assembly emission** (`Asm.cpp`).

`--pass-stats` prints counters for the mid-end passes and for the back-end
stages (`backend-strength-reduction`, `backend-peephole`,
`backend-schedule`, `backend-blockcse`, `backend-licm`,
`backend-postra-peephole`).

## 6. End-to-end testing on the QEMU VM

The correctness gate compiles every case with the local compiler, pushes the
`.s`/`.in`/`.out` files to a RISC-V64 Ubuntu guest (QEMU), and there assembles,
links against `libsysy_riscv.a`, runs, and diffs stdout + exit code against
the reference `.out`.

Prerequisites: the VM is up and reachable at `ssh ubuntu@localhost:2222`
(password is the `QEMU_PASS` default in `tools/qemu_verify.sh`); the guest has
the cross toolchain (`riscv64-linux-gnu-*`) and the reference runtime library
at `~/libsysy_riscv.a` (or `~/riscv/libsysy_riscv.a`).

```bash
./scripts/test.sh                          # build + functional + h_functional + performance2026
./scripts/test.sh tensor                   # single suite
./scripts/run.sh -qemu-test h_functional performance2026
./tools/qemu_verify.sh functional h_functional performance2026 tensor
```

**Verification cache.** Repeat runs are cheap: a case is re-run on the guest
only when its freshly produced assembly differs from the last verified one, or
when the compiler binary / source / `.in` / `.out` changed (the cache key).
Verified assembly and keys live under `build/.verify_cache/<suite>/`; a FAIL,
`TIMEOUT`, assemble/link failure or compile failure drops the stale `.ok`
marker so the case is retried next time.  During development, cache it by
keeping the compiler binary unchanged for unchanged `.s` files — cases whose
assembly did not change are reported `CACHED` and skip the guest entirely.

## 7. Design notes

* **SysY data model**: scalars and arrays are memory-resident at first —
  `memref.alloca` allocates a stack object, `memref.gep` folds byte offsets,
  `memref.load`/`memref.store` access memory — and `mem2reg` later promotes
  non-escaping scalar slots to SSA registers.  Large array initialisers are
  lowered to a runtime zero-fill loop instead of one store per element, so
  compile time stays linear in source size.
* **Structured loops**: `while` loops whose body cannot `return` / `break` /
  `continue` out of the loop keep the MLIR-style `scf.while` region form
  (`scf.condition` / `scf.yield`), so loop structure is available to LICM and
  unrolling.  Loops with `break`/`continue`/`return` stay as plain CFG.
* **TensorType extension**: multi-dimensional `tensor int/float` values are
  first-class.  A tensor is a buffer (alloca/global/pointer) plus a static
  shape carried on the op; whole-tensor ops write into an explicit
  destination buffer so semantics stay whole-array (and a future vectoriser
  could fuse the loops).  Scalar operands stay scalar and are broadcast by
  the back-end loop.  A function returning a tensor gets an sret-style
  signature (extra leading pointer argument).
* **Call optimisation interplay**: `tail-rec-elim` runs before
  `inline-general`; recursion reduced to a flat loop becomes a plain
  inlineable body, and recursion is never inlined (self-recursive callees are
  rejected by the inliner).
* **Verification philosophy**: every transformation is a pass on a
  `PassManager`, and the back-end gate `verify-cf-only` enforces the
  mid-end → back-end contract (no tensor/affine/scf op survives into
  instruction selection).

## 8. Directory layout

```
CMakeLists.txt                    build definition (ANTLR4 vendored as a subdir)
README.md                         this file
src/
  SysY.g4                         ANTLR4 grammar for SysY
  main.cpp                        driver: flags, layer views, stage wiring
  frontend/                       ASTBuilder + SemanticAnalysis (typed AST)
  midend/
    ir/                           IR types: module/function/block/instruction,
                                  dialects & opcode enum, textual printer
    irbuild/                      AST -> affine-layer IR lowering (IRBuilder)
    pass/                         PassManager + Pipeline + Opt*.cpp passes
  backend/                        ISel / MachineOpt / RA / Asm / Pipeline
  common/                         shared errors & helpers
3rd_party/antlr4-runtime/         vendored ANTLR4 C++ runtime
cases/                            functional, h_functional, performance2026, tensor
vm/                               QEMU guest disk image
docs/architecture.md              deeper design notes
scripts/                          build.sh, compile_dir.sh, run.sh, test.sh
tools/                            qemu_verify.sh (host), guest_test.sh (guest)
```
