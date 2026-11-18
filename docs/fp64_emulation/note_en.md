# FP64 emulation study plan (short version)

2026-11-13. Task list: `todo.md`. Chinese version: `note_zh.md`.

## 1. Motivation

- AI-oriented GPUs from NVIDIA and AMD have cut FP64 sharply. B300/GB300 delivers about 1.2–1.3 TFLOPS, and MI455X is also weak in FP64.
- Vendor Ozaki emulation in cuBLAS covers dense GEMM only.
- AMD (arXiv:2611.13332) estimates that GEMM emulation helps less than 10% of HPC workloads. The same paper estimates that about 100 of 131 kernels cross the FP64 ridge on MI455X, and most of these kernels are not GEMM.
- LQCD stencils and BLAS fall into this gap.
- We use present RTX cards (ridge about 0.6 flop/B) as a measurable proxy for future hardware.

## 2. Approach

- The target is FP64 accuracy. We do not aim beyond FP64.
- The weakest link in the computation chain sets the accuracy. Every link therefore keeps at least 48 bits: native double or float-float, with no FP32 link.
- We decide per kernel whether to emulate. We compare the kernel intensity I (flop/B, counted in double) with the hardware ridge R = P64 / BW:
  - If I > R, the kernel is limited by FP64 throughput. It computes in float-float inside the kernel and keeps double storage.
  - If I < R, the kernel is bandwidth-bound and stays in native double.
- On RTX only a few kernels need emulation. On B300 almost all of them do. On a chip without FP64 units all of them do. Full-chain float-float is the limiting case of this scheme at the weakest-FP64 end.
- Mixed precision uses a low-precision solve as a preconditioner. It acts at the preconditioning level, so it is orthogonal to emulation and the two can be combined.

## 3. Hardware ridge points

| GPU | FP64 (TFLOPS) | Bandwidth (TB/s) | R (flop/B) |
|-|-|-|-|
| B300 / GB300 | 1.2–1.3 | 7.7 | ~0.16 |
| MI455X | n/a | n/a | ~0.21 |
| RTX 3080 | ~0.47 | 0.76 | ~0.61 (approximate) |
| RTX 4090 | ~1.3 | 1.0 | ~1.3 (approximate) |
| Rubin | ~33 | 22 | ~1.5 |
| B200 | 37–40 | 7.7 | ~4.8 |
| A30 | ~5.2 | 0.93 | ~5.6 |
| H100 | 34 | 3.2 | ~10.6 |

We measure mainly on RTX. The A30 serves as a strong-FP64 control. It shows that the emulate-or-native decision for the same kernel changes with the hardware.

## 4. Earlier result: full-chain multiword in bridge_cuda

On an RTX 3080, full-chain QDW takes about 1.0× the wall time of native FP64, and full-chain QTW takes about 1.66×.

| Item | FP32 | QDW | QTW | FP64 |
|-|-|-|-|-|
| Wall time (relative to FP64) | n/a | ~1.0× | ~1.66× | 1× |
| True residual floor ‖r‖²/‖b‖² (mq 0.10, 16⁴) | 3.3e-13 | 9.5e-28 | 6.9e-34 | 3.1e-30 |
| Iterations to 1e-12 (real configuration) | 140 | 140 | 140 | 140 |

Four causes explain the result:
1. Memory traffic does not drop. QDW uses 8 B per real number, the same as double. QTW uses 12 B.
2. CG normalizes s, x, r and p in every iteration. This adds four full-vector read/write passes.
3. The eo hop intensity (about 0.5–0.7 flop/B) is only slightly above the 3080 ridge. The QTW hopb kernel also uses 167 registers with a 2304 B spill, so it does not reach the bandwidth limit.
4. The iteration count does not change with precision. Fewer iterations cannot recover the cost per iteration.

## 5. Speedup bound

The speedup bound per kernel is

g = min( I / R , 64 / k ).

- Here k is the number of FP32 instructions per FP64 operation in float-float. We estimate 10–20 and still need to measure it.
- Float-float moves the same bytes as double. The bound is the double bandwidth limit, about twice the native FP32 time. It is far below 64×.
- The total speedup follows Amdahl's law: S = 1 / (Σ f_bw + Σ f_cb / g). Here f_bw and f_cb are the time fractions of bandwidth-bound and compute-bound kernels.

The following estimates use intensities that we have not measured:

| Kernel | I (flop/B) | 3080 | B300 |
|-|-|-|-|
| Bridge++ eo hop | ~0.5–0.7 | ~1× | ~3–4× |

## 6. Scope and code

- The scope is eo-DWF CGNE. AMG is future work.
- The Bridge++ code is the `feat/no-amg` branch of the bridge_cuda CUDA fork. This branch removes all AMG code from main and keeps QDW/QTW. We do not port it to Bridge 2.1.3 OpenACC.
- One Bridge++ DdagD call launches about 14 kernels: 4 each of 5dir, hopb and LU_inv, plus copy and axpy. The 5dir and hopb kernels are separate. γ5 is not on the CGNE path.
- We do not fuse kernels. Emulation aims to bring a kernel that is FP64-compute-bound on RTX back to the bandwidth limit. A fused kernel has higher register pressure and spills more easily in float-float. The spill traffic to local memory adds to the bytes of a kernel that is already memory-bound, so it runs slower. We therefore regard fusion as unsuitable together with emulation. This is an inference that we have not measured.

## 7. Hypotheses

- **H1 (classification).** If I > R, the speedup bound is about I / R. If I < R, there is no speedup.
- **H2 (composition).** Amdahl's law predicts the total speedup from the kernel time fractions within 20%.
- **H3 (accuracy).** With selective emulation, the true residual floor stays within 2⁵ of FP64 in ‖r‖ (48 against 53 bits). The iteration count and observables match FP64.
- **H4 (implementation).** A float-float kernel reaches the bandwidth limit only if it uses no more registers than the double kernel and does not spill.

## 8. Experiments

| Experiment | Comparison | Quantities |
|-|-|-|
| E1 | FP64, lowendgpu (double storage, float-float arithmetic in hopb only) and full-chain float-float (QDW), CGNE with the same lattice, configuration, mq and Ls | D, DdagD, time per iteration, full solve |
| E2 | Same low-precision preconditioner, outer FP64 arithmetic native or emulated | Whether the two combine |
| E3 | Native FP64 against selective emulation over a mass scan | True residual history, observables, HMC ΔH |

The work has four phases:
1. We profile with ncu on RTX under native Linux, with the A30 as control. The output is a kernel × hardware decision table.
2. We implement the float-float kernels. The gate is no register spill.
3. We measure speed on RTX and AI GPUs and verify accuracy. The gate is H2 within 20% and H3 satisfied.
4. We write a Lattice 2027 proceedings paper first and a journal paper later.

The lattice size (24⁴ or 24³×48) and Ls (8 or 16) are still open.

## 9. Related work

| Work | Content | Difference from this study |
|-|-|-|
| Swirydowicz et al. (AMD), arXiv:2611.13332 | Ozaki-II GEMM measurements; GEMM emulation helps < 10% of HPC; Ozaki-II is slower than native for small matrices | We treat non-GEMM kernels that become FP64-compute-bound on future hardware |
| Mukunoki & Ozaki, arXiv:2510.13536 | Full-chain quasi multi-word CG on CPU | GPU, preconditioned DWF, selective emulation |
| Matsuoka, arXiv:2606.06510 | Model and projection of Ozaki II for memory-bound workloads | Measurements; float-float for stencils |
| Kouya, arXiv:2609.35844 | Branch-free pair renormalizer; binary32 pair arithmetic is bandwidth-bound on GB10 | Its future work asks for a selection guideline per accelerator, which this study provides; it has no stencil or LQCD work |
| cuBLAS Ozaki | Dense DGEMM only | We treat kernels other than GEMM |
| QUDA mixed precision | Low-precision preconditioner with FP64 outer solver | Orthogonal; can be combined |

## 10. Main risks

- Float-float kernels may need too many registers and spill.
- We have not measured the intensities or the hardware figures, so the decision table may change.
- Without a B300 machine, the main scenario rests on projection, and the paper must state this.
- Float-float has 5 fewer mantissa bits than double and the exponent range of float (about 1e-38). Very small values may underflow.
- We must verify with a mass scan whether 48 bits suffice at small mass and in ill-conditioned cases.
