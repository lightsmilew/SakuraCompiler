# RISC-V 复制生成与寄存器分配

`mv` / `fmv.s` 的来源有两类。机器 IR 的 `MoveX` / `MoveF` 包含
Phi 消除、循环递推以及 CSE 产生的复制；入口参数、调用参数、调用结果和
返回值的 ABI 复制则由汇编输出阶段生成，不是独立的 MIR Move。
因此只改窥孔匹配无法覆盖所有复制。

`fmv.w.x` 是整数位模式到浮点寄存器的搬运，例如浮点常量物化。
它跨越寄存器文件，不能通过把两端分配到相同编号的寄存器来消除。

## 复制感知的干涉图

对 `d = copy s`，仅因为 `s` 在复制后仍然活跃，不为 `d` 与 `s` 增加
冲突边。复制后两者值相同，允许已有的 move partner 着色偏好选用同一
物理寄存器，再由分配后的清理去掉自复制。

此规则仅跳过当前复制所产生的那一条边，不删除其他指令已经建立的边。
源或目标的重定义仍建立正常冲突，同时读取的值也保留保守的冲突检查。
机器 IR 已不是 SSA，Phi 的多处定义及并行复制环尤其需要这些约束。

## ABI 着色偏好

使用与汇编输出相同的 `ArgCursor`，为入口参数、调用参数、调用结果和
返回值累计物理寄存器偏好。循环深度使用现有分配器的估算，权重为
`2^min(depth,12)`。这些是软偏好，仍检查真实冲突、入口未读取参数的
保护和寄存器集合；跨调用活跃值继续使用 callee-saved 集合。

ABI 偏好仅在其最大估算收益大于该值相邻 MIR 复制的累计成本时，才优先
于已有的复制亲和关系。否则一次冷路径输出复制可能把外层 Phi 固定在
`fa0`，而循环内的临时值也偏好 `fa0`，导致原来共用寄存器的内外层 Phi
每次循环都要复制。热循环的减少成本不能由出口的静态指令数替代。

保守着色不会保证消除所有复制：必须跨调用保存的地址仍要从 `s` 寄存器
搬入 ABI 参数寄存器；同一值作为两个不同位置的参数时，仍需准备两个
位置；有真实活跃区间冲突的 Phi 也需要复制。

## 调用参数的并行搬运

旧输出阶段只要发现一个源寄存器也出现在目标集合，就先存栈再读回。
例如调用结果在 `a0`，下一次调用需要它作为第四个参数，同时第一个
参数来自 `s3`，旧规则生成 `sd a0; mv a0,s3; ld a3`。这只是参数
shuffle，寄存器分配可以完全没有 spill；无环依赖直接按
`mv a3,a0; mv a0,s3` 的顺序处理即可。

新的串行化器反复选择“目标已不被任何待执行复制读取”的项。
只有剩余复制构成环时，才把一个源保存到分配器预留的 `t1` 或 `ft1`，
替换所有对该源的待执行读取，再继续处理。整数和浮点文件分别检查依赖，
重复源、identity copy 与多个独立环均支持。`mv` 保留完整 RV64 指针；
`fmv.s` 保留 F32 位模式。串行化过程中没有访存或其它 scratch 使用。
`fmv.s` 对应 `fsgnj.s rd,rs,rs`；该操作不把 NaN 规范化，负零和 NaN
payload 可随复制保留，见 [RISC-V F 扩展规范](https://docs.riscv.org/reference/isa/unpriv/f-st-ext.html)。

参数搬运不再分配临时栈区；真正的出栈参数、spill、局部变量以及
callee-saved 保存区仍由统一 frame layout 决定。复制来源统计包含环的
临时寄存器复制，shuffle 读写次数为零。测试直接解释实际输出汇编，
遍历两个文件各 7! 个排列及固定随机种子的混合/重复源输入，检查调用点
每个 ABI 寄存器的原始位模式。

## 分配后复制到 store 的折叠

`Xn` 与 `Fn` 共享编号，但属于不同寄存器文件。`mv X10,X11; fsw F10,0(X10)`
不能因为 store 的浮点值编号也为 10，就改成 `fsw F11,0(X10)` 并删除整数
复制；这会同时改变地址和写入值。折叠必须检查 `OpSlots.useFile[0]`，
确认 store 的值操作数与复制处于同一文件。使用 `isMoveF(store)` 判断会
把所有 store 都判成非浮点复制，既会错误匹配整数复制，也会漏掉真正的
浮点复制到 store。定义覆盖判断同样直接使用 `defFile`。

独立内存解释器为错误匹配检查实际写入地址和结果，为正确的浮点折叠
检查负零及 NaN 位模式，保证删除复制时不改变可观察内存。

## 诊断与验证

`--pass-stats` 增加每个函数的 `backend-copies`：整数/浮点分别统计
`mir`、`entry`、`argument`、`result`、`return`、`address` 来源，另列
参数 shuffle 的静态读写次数。`tools/benchmark/move_audit.py` 从实际
汇编重新统计复制数量，并核对这些来源的总和；没有该诊断的旧编译器和
LLVM 产物保留未知来源。`register_pressure.py` 独立核对活跃峰值和 spill。

`SAKU_NO_RA_COPY_EDGE=1` 恢复复制上的保守冲突边；
`SAKU_NO_RA_ABI_HINT=1` 关闭 ABI 着色偏好。两者一起用于构造旧行为的
消融版本，必须核对其汇编与冻结基线逐字节相同后才能归因。

后端独立解释器验证源重定义、分支路径、Phi 交换循环、调用 clobber、
整数/浮点文件隔离、负零和 NaN 位模式；浮点归约用例还检查冷 ABI
偏好没有增加循环内复制。源级 `cases/regression/move_abi_phi.sy` 覆盖
递归调用、整数寄存器/栈参数、浮点参数和循环 Phi，预期结果由 GCC -O0
独立核对。测量记录及对比产物放在 [results](../results/move-optimization/)。

## 原理参考

- [LLVM RegisterCoalescer](https://llvm.org/doxygen/RegisterCoalescer_8h_source.html)：
  合并复制两端的寄存器，使复制成为 identity copy。
- [LLVM TargetRegisterInfo::getRegAllocationHints](https://llvm.org/doxygen/TargetRegisterInfo_8cpp_source.html)：
  分配提示用于帮助消除复制，同时检查物理寄存器是否合法、是否保留以及
  是否属于可分配集合。
- [LLVM 寄存器分配研究，LCPC 2005](https://llvm.org/pubs/2005-10-20-LCPC-RegAlloc.pdf)：
  复制消除与寄存器分配相互影响，合并必须服从活跃区间与干涉信息。

当前实现沿用 Sakura 的图着色分配器，没有引入 LLVM 的 LiveIntervals
合并和 live-range splitting。文献和当前 LLVM 源码提供原理参考，性能
对照使用 benchmark manifest 中记录的实际 Clang 版本与目标选项。
