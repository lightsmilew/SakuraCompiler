# SakuraCompiler architecture

A single-pass, self-contained SysY -> RISC-V64gc compiler. The front-end uses
ANTLR4 (visitor pattern), the mid-end is an MLIR-style SSA IR, and the
back-end does RISC-V instruction selection, graph-colouring register
allocation and GNU-`as`-compatible emission. There are no third-party
dependencies other than the vendored ANTLR4 C++ runtime.

```
 .sy source
    |
    v
 [frontend]   ANTLR4 lexer/parser  (src/SysY.g4 -> generate/)
    |         ASTBuilder: parse tree -> typed AST (ASTNode.h)
    v
 [frontend]   SemanticAnalysis: scopes, types, const folding,
    |         array-init flattening, builtin registration
    v
 [midend]     IRBuilder: AST -> MLIR-style IR (midend/ir)
    |         - module / function / basic block / instruction
    |         - dialect metadata: func, arith, cf, memref, tensor
    v
 [backend]    ISel: IR ops -> RISC-V MInsts on virtual regs
    |         (frame slots 4B; flat contiguous arrays)
    v
 [backend]    liveness + interference graph
    |         graph-colouring RA (Briggs-style) with spilling
    v
 [backend]    Asm: colours + frame layout -> RISC-V .s text
    v
 riscv64-linux-gnu-gcc -static (on the QEMU VM) -> executable
```

## Source layout

| Path | Role |
| ---- | ---- |
| `src/SysY.g4` | ANTLR4 grammar (lexer + parser). |
| `src/common/Common.h` | `TypeCat`, `ConstVal`, IEEE-754 helpers, `CompileError`. |
| `src/frontend/ASTNode.h` | Typed AST + analysis annotations. |
| `src/frontend/ASTBuilder.*` | ANTLR visitor building the AST. |
| `src/frontend/SemanticAnalysis.*` | Semantic pass over the AST. |
| `src/midend/ir/IR.h/.cpp` | IR data structures, dialect metadata, MLIR-style printer. |
| `src/midend/irbuild/IRBuilder.*` | AST -> IR lowering. |
| `src/backend/Machine.*` | Machine IR (`MOp`, `MInst`, `MachineFunc`) + reg names. |
| `src/backend/ISel.*` | Instruction selection. |
| `src/backend/RA.*` | Liveness + graph-colouring allocator. |
| `src/backend/Asm.*` | Assembly text emission. |
| `src/main.cpp` | Driver: `compiler <in.sy> [-S] [-o out.s] [--dump-ir]`. |

## Front-end

ANTLR4 parse trees are consumed by a visitor in `ASTBuilder`. The grammar
mirrors the SysY spec with the TensorType extension: expression layers
(`cond -> lor -> land -> eq -> rel -> add -> mul -> unary -> primary`,
with `@` matrix-multiply at the `mul` layer), declaration/init lists,
`if/while/break/continue/return`, function definitions, and
`tensor int/float` types usable as a basic type, a function return type, and
(parameter with `[]` first dimension) as a formal parameter.

`SemanticAnalysis` then annotates the AST in place:

- **Scopes** - a stack of name->`VarSymbol` tables is maintained; blocks push
  and pop scopes, function bodies are checked with their parameters in scope.
- **Types** - every expression gets an `evalType` (`int`/`float`); scalar
  values and plain arrays behave as in SysY, while whole-tensor expressions
  are flagged (`isTensorVal`) and annotated with a static `tShape`. Tensors
  support element-wise `+ - * / %` (with scalar promotion and same-shape
  requirements), unary `+ -`, and rank-2 `@` matrix multiply; relation /
  logical operators on tensors are type errors.
- **Tensor variables** - a `tensor` declaration stores its element type and
  dimension list; init brace lists are validated against the rank and
  flattened row-major. Tensor parameters keep `[]` for the first dimension
  (decayed pointer, second and later dims must be static); tensor-returning
  functions must return a static-shape tensor whose shape is recorded on the
  function.
- **Constant folding** - every expression tries to fold to a `ConstVal`
  (`valid` flag); dimensions, array sizes and const initialisers must fold.
- **Initialiser flattening** - const and non-const initialiser brace trees are
  validated and, for consts, flattened to one `ConstVal` per element so that
  IRBuilder can emit flat element lists (row-major, zero-filled).
- **Builtins** - SysY library functions (`getint`, `putch`, `getfloat`,
  `getarray`, `putarray`, `starttime`, `stoptime`, ...) are declared with the
  reference runtime signatures. `starttime`/`stoptime` map onto the
  underscore-exported `_sysy_starttime`/`_sysy_stoptime`.

## Mid-end: an MLIR-style IR

The mid-end is deliberately *not* textual LLVM IR. It is an SSA CFG IR
modelled on MLIR:

- Every opcode belongs to a **dialect**: `func` (`func.call`, `func.return`),
  `arith` (`arith.constant`, `arith.addi/subi/muli/divsi/remsi`,
  `arith.addf/subf/mulf/divf`, `arith.cmpi/cmpf`, `arith.sitofp/fptosi`,
  `arith.xori`), `cf` (`cf.br`, `cf.cond_br`), `memref`
  (`memref.alloca/load/store/gep`) and `tensor` (whole-tensor ops such as
  `tensor.copy`, `tensor.addi/addf`, `tensor.muli/mulf`, `tensor.divi/divf`,
  `tensor.remi`, `tensor.negi/negf`, `tensor.matmul/matmul_f`).
- The flat `Op` enum stays the single dispatch key used by the back-end; the
  dialect layer (`dialectOf` / `opName`) is what makes the IR read as MLIR.
- SSA values carry MLIR scalar types (`i32`, `f32`) plus a pointer type
  (`ptr`).
- `Module::toString()` prints standard-MLIR-style text:

```
module {
  func.func private @putint(i32)
  func.func @main() -> i32 {
    %0 = arith.constant 10 : i32
    %1 = memref.alloca : i32
    memref.store %0, %1 : i32
    %2 = memref.load %1 : i32
    %3 = arith.addi %2, %0 : i32
    func.return %3
  }
}
```

**Memory model.** SysY scalars/arrays are memory-resident. Local variables
become `memref.alloca` stack objects; globals become `memref.global` objects;
array parameters are raw `ptr` arguments. `memref.gep` performs byte-offset
address arithmetic (all scalar elements are 4 bytes), and
`memref.load`/`memref.store` access memory through addresses. This keeps
address computation and the 4-byte frame-slot model identical all the way to
RISC-V.

**Whole-tensor ops.** A tensor value is a buffer (alloca / global / array
parameter pointer) plus a static shape recorded on the op; it is never split
into scalars in the mid-end, so the tensor semantics stay first-class and a
future vectorising pass can fuse whole-array expressions. Tensor ops use an
explicit destination buffer as their first operand:

```
%t = memref.alloca : [4 x i32]
tensor.muli %t, @b, %c3 : (memref<4xi32>, i32) -> memref<4xi32>
tensor.addi @c, @a, %t : (memref<131072xi32>, memref<131072xi32>) -> memref<131072xi32>
```

Scalar operands are broadcast (scalar promotion) by the expansion, not
copied into full temporaries. A function returning a tensor is lowered with a
hidden `sret` result buffer: the IR signature gains a leading pointer
parameter, `return x` inside it lowers to `tensor.copy` into that buffer, and
call sites pass the caller's destination buffer first. Inside a tensor
function the result buffer is threaded to `return` statements only; all
intermediate results are normal whole-tensor buffers.

## Back-end

### Instruction selection (`ISel.*`)

Each IR instruction is lowered to one or more `MInst`s on virtual registers.
Important peepholes:

- an alloca's frame slot is resolved lazily on first use (`PtrVal` keeps a
  frame offset);
- stores into never-read allocas are dropped;
- `x - 0` becomes `neg`, `0.0 - x` becomes `fneg`;
- fully constant comparisons are folded at selection time.

**Tensor expansion.** Whole-tensor ops are expanded into explicit machine
loop nests (`ISel.cpp`, `lowerTensorElementwise` / `lowerTensorMatmul`). Each
expansion uses a small number of stack slots for its induction counters and
emits a structured loop whose blocks each end with their control transfer
(header block: conditional branch to the exit + fall-through into the body;
body block: `j` back to the header) so the machine CFG -- and therefore
liveness / colouring -- sees the exact control flow. Element-wise ops
(including copies and scalar-promoted binary ops) become a one-dimensional
loop over the flat row-major buffer; matrix multiply becomes an `i/j/k`
triple-nested loop over the `{M, N, P}` shape carried by the op. Sources that
alias the destination are copied to a temporary first (required for matmul,
where reads are repeated).

### Register allocation (`RA.*`)

Liveness is computed per machine function; two vregs interfere when one is
live at the definition of the other. An interference graph feeds a
Briggs-style graph colouring allocator:

- pre-coloured physical registers (`a0-a7`/`fa0-fa7`, `t`/`ft` scratch,
  callee-saved `s`/`fs`) participate in coalescing;
- the allocator iterates simplify/spill rounds; uncolourable nodes spill to
  stack slots and a new round runs until the graph colours;
- caller-saved and callee-saved sets are derived from the colouring so the
  prologue/epilogue only save registers the function actually uses;
- instruction scheduling decisions keep the machine CFG/liveness exact.

### Assembly emission (`Asm.*`)

The writer emits GNU `as` syntax for the RISC-V64gc subset. Frame layout
keeps 16-byte stack alignment, reserves outbound argument space before calls
(`ArgCursor` distributes `a0-a7`/`fa0-fa7`, then 8-byte stack slots), and
folds frame/global displacements directly into `lw`/`sw` (and FP) forms.

## Testing

`cases/{functional,h_functional,performance2026,tensor}` are compiled on the
host by `build/compiler`, then the assembly is pushed to a QEMU RISC-V Ubuntu
VM, assembled/linked against `libsysy_riscv.a`, run, and stdout + exit code
are diffed against reference `.out` files. `scripts/test.sh` and
`scripts/run.sh -qemu-test` drive the whole loop.
