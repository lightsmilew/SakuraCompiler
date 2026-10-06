# Loop splitting by index sets

`loop-split` runs at the Affine layer at `-O2`, after polyhedral scheduling and
before LICM/unrolling. Set `SAKU_NO_LOOP_SPLIT=1` to compare the same compiler
with this pass disabled. It handles a counted loop's branch CFG before lowering
loses the explicit induction slot, bounds and region ownership.

## Literature and choice of transformation

Barton, Tal, Blainey and Amaral's
[Generalized Index-Set Splitting (CC 2005)](https://webdocs.cs.ualberta.ca/~amaral/papers/BartonCC05.pdf)
describes partitioning a loop's index range to simplify conditions that change
with the induction variable. Its examples require clipped bounds; its algorithm
also budgets code growth. This implementation follows the domain partition and
specialization principle, using sorted constant cut points and conservative
size caps rather than implementing the paper's full index sub-range tree and
greedy selection algorithm. Removing a conditional per iteration also exposes
straight-line bodies to existing optimizations. A monotone branch can already
be predictable, so branch removal does not establish a higher prediction hit
rate; the measurements here concern elapsed time and generated code.

[GCC's `-fsplit-loops`](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html#index-fsplit-loops)
likewise splits iteration space where a condition becomes uniformly true or
false. This differs from
[LLVM LoopDistribute](https://llvm.org/doxygen/LoopDistribute_8cpp_source.html),
which separates groups of statements to isolate memory dependence cycles and
enable vectorization. Moving all executions of one branch before another can
reorder memory effects and floating-point reductions. Consecutive index-set
splitting retains their original order and therefore fits this request safely.

## Supported domains and predicates

The pass requires `affine.for iv = lb to ub step 1` with signed `iv < ub`, a
private nonescaping i32 induction alloca, and an acyclic body CFG ending in
`affine.yield`. Nested loops are visited independently; a body containing region
operations, arrays, phis, early exits, cyclic edges or escaping SSA results
is not cloned. Private scalar temporaries are supported only when a CFG
must-initialization analysis proves every read follows a store in that
iteration, and their addresses have only direct load/store uses. This also
covers boolean slots introduced by short-circuit conditions. Stores to the
induction slot and pointer escape reject the loop.

The two operands of a signed comparison can be integer affine expressions of
the current IV and constants. All six comparisons, reversed operands and
boolean negation are supported. Each intermediate add/subtract/multiply must
fit i32 throughout the possible original domain. In particular, a final affine
expression alone does not prove that its intermediate computations avoid wrap.
Captured loads of the induction slot from outside the body are not current-IV
reads. Equality and inequality use the root and the following index as cuts;
ordered comparisons use a monotone binary search for the transition.

Lower and upper loop bounds may be runtime values; cut points are currently
constant. Unrecognized data-dependent conditions remain in the cloned bodies.
There are at most eight parts, 192 original instructions and 1024 conservatively
estimated cloned instructions. Known loops with fewer than eight iterations
are skipped. If these caps are exceeded the loop stays intact; there is no
profile-based or greedy partial selection yet.

## Correctness

For sorted cuts `c1 < ... < ck`, construct shared bounds

```
b0 = lb
bj = max(lb, min(cj, ub))       (1 <= j <= k)
b(k+1) = max(lb, ub)
```

Each generated loop visits `[bj, b(j+1))` in order. When `lb < ub` their
concatenation is exactly `[lb,ub)` without gaps, repeats or reordered iterations.
When `ub <= lb` all loops are empty and leave the induction slot at `lb`, as the
original loop would. This is stronger than merely clipping the iteration sets:
reusing an observable induction slot requires empty loops to reseed it safely.

Within a part, every modeled comparison has constant truth. Only its selected
CFG edges survive. Reachability pruning, a retained-definition check and
unique-predecessor path merging remove unused arms while retaining the order
of all executed operations. No reassociation, alias-based memory reordering or
speculation of unselected statements is performed. Consequently scalar
recurrences, cross-iteration RAW dependences, opaque calls and floating-point
reductions can be retained in their original order.

Splitting exposes small loops to the existing unroller. The accompanying repair
writes the final IV after full unrolling, widens static trip-count subtraction,
and guards nonrepresentable induction exits. Partial unrolling now clamps empty
domains and computes the distance modulo four from endpoint residues, avoiding
overflow in `ub-lb` and preserving the final IV for empty dynamic domains.

## Validation

`tests/loop_split_regression.cpp` uses an independent structured-IR interpreter
to compare memory, final induction values and the complete ordered call trace.
It covers all comparisons, affine slopes of both signs, negation, runtime
bounds, signed extrema, equality singleton ranges, retained data conditions,
code-growth limits and unsafe-region rejection. It also checks full and partial
unrolling's observable IV state. Run it through CTest with
`SAKURA_BUILD_TESTS=ON`.

The bound-recovery regression also covers induction initialization in a
predecessor block. A failed local seed lookup must terminate and preserve the
runtime bound, rather than revisiting its load indefinitely. This was caught by
the complete source regression (`60_sort_test6`), and fixed before final timing.

`cases/optimization/07`–`11` add source-level effect-order, memory-dependence,
nonassociative floating-point and two timed branch-heavy array cases. Expected
outputs are independently generated with GCC `-O0 -ffp-contract=off`. RISC-V
execution compares these against the prior compiler and LLVM, with repeated
timings for the two hot cases. All 236 programs and 1361 executions passed;
the two hot cases improved by 2.307x and 1.462x in QEMU. Real performance2026
cases showed mixed results and no stable suite improvement. See
[measurements and the seven-run recheck](loop-split-results.md) for the complete
results, hashes and limits of these measurements.
