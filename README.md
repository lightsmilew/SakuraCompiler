# SakuraCompiler

A self-contained SysY -> RISC-V64gc compiler. The frontend parses the SysY
grammar (ANTLR4, visitor pattern) with a full semantic analyser, the mid-end
builds an MLIR-style SSA IR (dialect-organised `func` / `arith` / `cf` / `scf`
/ `memref` / `tensor` ops; structured loops stay as `scf.while` region ops),
and the back-end selects RISC-V instructions, performs graph-colouring
register allocation and emits GNU `as`-compatible assembly.
No third-party dependencies beyond the vendored ANTLR4 runtime.

Verified end-to-end on a QEMU RISC-V Ubuntu guest (compiled assembly linked
against the reference `libsysy_riscv.a`):

- `cases/functional` -- 100% passing
- `cases/h_functional` -- 100% passing
- `cases/performance2026` -- 100% passing
- `cases/tensor` -- 100% passing (TensorType extension)

## Layout

```
CMakeLists.txt                  build definition
src/
  SysY.g4                       ANTLR4 grammar
  frontend/                     ANTLR4 visitor AST builder + semantic analyser
  midend/ir/                    MLIR-style IR (module / function / block / op)
  midend/irbuild/               SysY AST -> IR lowering (affine layer)
  midend/pass/                  PassManager + pass pipeline (Pipeline/Passes)
  backend/                      ISel, graph-colouring RA, assembly emission
3rd_party/antlr4-runtime/       vendored ANTLR4 C++ runtime
scripts/
  run.sh                        build / compile / QEMU-test helper
  test.sh                       full 3-suite correctness gate
tools/
  qemu_verify.sh                host side: compile cases, push to VM, run guest script
  guest_test.sh                 guest side: as + link + run + diff per case
cases/                          functional, h_functional, performance2026, tensor
vm/                             QEMU disk image
docs/                           design notes
```

## Build

```bash
./scripts/run.sh -rebuild        # clean full build (or: cmake --build build)
```

Requires CMake >= 3.10 and a C++17 compiler. ANTLR4 is vendored under
`3rd_party`, so nothing needs installing.

## Compile a source file

```bash
./build/compiler input.sy -S -o out.s        # emit RISC-V assembly
./build/compiler input.sy --dump-ir          # print the MLIR-style mid-end IR
```

The output is RISC-V64 assembly ready for `riscv64-linux-gnu-gcc -static`.

## End-to-end testing on the QEMU VM

Start the VM (ssh `ubuntu@localhost:2222`, password in `tools/qemu_verify.sh`),
then:

```bash
./scripts/test.sh                       # functional + h_functional + performance2026
./scripts/test.sh functional            # single suite
./scripts/test.sh tensor                # TensorType suite
./scripts/run.sh -qemu-test h_functional performance2026
```

Repeat runs are cached: a case is re-run under QEMU only when its freshly
produced assembly differs from the last verified one (or its sources / `.in` /
`.out` / the compiler binary changed).  Verified assembly lives in
`build/.verify_cache`, so unchanged runs skip the QEMU guest workload entirely.

## Design notes

- **Frontend**: ANTLR4 lexer/parser generated from `src/SysY.g4`; a visitor
  builds typed AST nodes; the semantic analyser resolves scopes, checks types,
  folds constants and lowers array initialisers to flat element lists. SysY
  builtins (`getint`, `putch`, `getfloat`, `starttime`, ...) are declared and
  mapped to the runtime's exported names (`_sysy_starttime` etc.).
- **Mid-end**: an MLIR-style SSA IR. Every opcode belongs to a dialect
  (`func`, `arith`, `cf`, `scf`, `affine`, `memref`, `tensor`) and `--dump-ir`
  prints dialect-qualified ops (`func.call`, `arith.addi`, `cf.cond_br`,
  `memref.load`, `tensor.addi`, ...) in a standard MLIR-like textual form,
  with SSA values typed `i32` / `f32` / `ptr`. SysY scalars and arrays stay
  memory-resident: `memref.alloca` allocates a stack object, `memref.gep`
  folds byte offsets, and loads/stores access memory through addresses. The
  flat opcode enum is preserved as the single dispatch key used by back-end
  instruction selection.
- **Structured loops**: `while` loops whose body cannot `return` / `break` /
  `continue` out of the loop are preserved in the IR as MLIR-style
  `scf.while` region ops (`scf.while { ^cond ... scf.condition %c } do {
  ^body ... scf.yield }`), so loop structure stays first-class for later
  mid-end passes. Loops with `break`/`continue`/`return` keep the plain
  `cf.br`/`cf.cond_br` CFG form.
- **Pass pipeline**: the mid-end is organised in three control-flow layers —
  `affine` (counting loops stay `affine.for`), `scf` (generic `scf.while`)
  and `cf` (flat CFG). `runMidEndPipeline()` (`midend/pass/Pipeline.cpp`)
  owns a `PassManager` that runs `expand-tensor-ops`, `lower-affine-to-scf`,
  `canonicalize-control-flow` and a trailing `verify-cf-only` back-end gate;
  per-layer optimisation passes plug into the same sequence. The driver only
  calls `runMidEndPipeline()` — the module handed to instruction selection /
  register allocation / assembly is guaranteed pure cf.
- **TensorType**: multi-dimensional `tensor int/float` values are first-class
  in the IR. A tensor is a buffer (alloca/global/pointer) plus a static
  shape carried on the op; whole-tensor ops (`tensor.addi`, `tensor.muli`,
  `tensor.matmul`, ...) write their element-wise / matmul result into an
  explicit destination buffer so the semantics stay whole-array (a future
  vectorising pass can fuse the loops later). Scalar promotion is handled by
  the IR builder (scalar operands stay scalar and are broadcast by the
  back-end loop). A function returning a tensor is lowered to an sret-style
  signature: an extra leading pointer argument receives the result buffer.
- **Back-end**:
  - instruction selection lowers IR ops to RISC-V `MInst`s and computes the
    machine CFG;
  - liveness analysis feeds an interference graph;
  - register allocation uses Briggs-style graph colouring, iterating
    spill/rematerialise rounds until the graph is colour-able, with the
    `a0-a7` / `fa0-fa7` argument registers reserved appropriately; spill slots
    are assigned to the stack frame;
  - the frame layout keeps 16-byte stack alignment, saves/restores callee-saved
    registers, and passes outbound arguments per the RISC-V calling convention
    (register streams `a0-a7` / `fa0-fa7`, overflow to the shared stack area).
- Large array initialisers are lowered to a runtime zero-fill loop instead of
  one store per element, keeping compile time linear in source size.
