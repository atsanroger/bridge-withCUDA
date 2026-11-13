# FP64 emulation 研究計畫(精簡版)

2026-11-13。待辦清單見 `todo.md`。英文版見 `note_en.md`。

## 1. 動機

- NVIDIA 和 AMD 的 AI 取向 GPU 把 FP64 大幅縮減:B300/GB300 約 1.2–1.3 TFLOPS,MI455X 也很弱。
- 廠商提供的 Ozaki emulation(cuBLAS)只涵蓋 dense GEMM。
- AMD(arXiv:2611.13332)估計 GEMM emulation 只幫得上不到 10% 的 HPC。它也估計 MI455X 上 131 個 kernel 中約有 100 個會跨過 FP64 損益點，而這些 kernel 大多不是 GEMM。
- LQCD 的 stencil 和 BLAS 正好落在這個缺口。
- 今天的 RTX(損益點約 0.6 flop/B)當作未來硬體的可量測代理。

## 2. 主張

- 目標是 FP64 精度，不追求超過 FP64。
- 精度由計算鏈上最弱的一環決定，所以每一環都要 ≥ 48 bit:原生 double 或 float-float,不允許 FP32 環節。
- 哪些 kernel 要 emulate,依各 kernel 的強度 I(flop/B,以 double 計)對照硬體損益點 R = P64 / BW 決定:
  - I > R:FP64 算力受限。在 kernel 內部用 float-float 計算，儲存仍是 double。
  - I < R:頻寬受限，留在原生 double。
- 硬體光譜:RTX 上只有少數 kernel 要 emulate,B300 上幾乎全部要，沒有 FP64 單元的晶片全部要。全程 float-float 是這個框架在最弱那一端的特例。
- 混合精度(低精度 solve 當 preconditioner)是前處理層的技術，和 emulation 正交，可以疊加。

## 3. 硬體損益點

| GPU | FP64 (TFLOPS) | 頻寬 (TB/s) | R (flop/B) |
|---|---|---|---|
| B300 / GB300 | 1.2–1.3 | 7.7 | ~0.16 |
| MI455X | — | — | ~0.21 |
| RTX 3080 | ~0.47 | 0.76 | ~0.61(近似值) |
| RTX 4090 | ~1.3 | 1.0 | ~1.3(近似值) |
| Rubin | ~33 | 22 | ~1.5 |
| B200 | 37–40 | 7.7 | ~4.8 |
| A30 | ~5.2 | 0.93 | ~5.6 |
| H100 | 34 | 3.2 | ~10.6 |

量測主要在 RTX 上做。A30 當 FP64 強的對照組，用來顯示同一個 kernel 的決策會隨硬體反轉。

## 4. 前期結果:bridge_cuda 全程多字為什麼輸

在 RTX 3080 上，全程 QDW 的牆鐘時間約為原生 FP64 的 1.0×,全程 QTW 約 1.66×。

| 項目 | FP32 | QDW | QTW | FP64 |
|---|---|---|---|---|
| 牆鐘(對 FP64) | — | ~1.0× | ~1.66× | 1× |
| 真殘差 floor ‖r‖²/‖b‖²(mq 0.10,16⁴) | 3.3e-13 | 9.5e-28 | 6.9e-34 | 3.1e-30 |
| 到 1e-12 的迭代數(真 config) | 140 | 140 | 140 | 140 |

原因:
1. 讀寫量沒有減少。QDW 每個實數 8 B,和 double 相同;QTW 是 12 B。
2. CG 每個 iteration 對 s、x、r、p 各做一次 normalize,多出 4 次整向量讀寫。
3. eo hop 的強度(約 0.5–0.7 flop/B)只勉強高於 3080 的損益點。QTW hopb 又有 167 個暫存器和 2304 B spill,跑不到頻寬上限。
4. 迭代數不隨精度改變，所以不能靠減少迭代補回單步成本。

## 5. 速度上限

每個 kernel 的加速上限:

g = min( I / R , 64 / k )

- k 是每個 FP64 運算換成 float-float 後的 FP32 指令數，估計 10–20,要量測。
- float-float 的讀寫量和 double 相同，所以上限是 double 的頻寬上限(約為原生 FP32 時間的 2 倍),不是 64 倍。
- 整體加速依 Amdahl:S = 1 / (Σ f_bw + Σ f_cb / g)。

估計值(強度未量測):

| kernel | I (flop/B) | 3080 | B300 |
|---|---|---|---|
| Ruttice 1-pass MDWF | ~1.2 | ~2× | ~7× |
| Bridge++ eo hop | ~0.5–0.7 | ~1× | ~3–4× |

## 6. 範圍與程式

- 範圍:eo-DWF CGNE。Bridge++ 沒有 RutticeQCD 那套 AMG,AMG 留作 future work。
- Bridge++ 程式:舊 bridge_cuda CUDA fork 的 `feat/no-amg` branch(main 去掉所有 AMG,保留 QDW/QTW)。不移植到 Bridge 2.1.3 OpenACC。
- 第二個實驗軸:RutticeQCD(大量 kernel fusion)對照 Bridge++ alt_Accel(不 fuse)。
- Bridge++ 的 DdagD 每次約 14 個 kernel:5dir、hopb、LU_inv 各 4 次，加上 copy/axpy。5dir 和 hopb 沒有 merge;γ5 不在 CGNE 路徑上。

## 7. 假說

- **H1(分類)**:I > R 時加速上限約 I / R;I < R 時沒有加速。
- **H2(組合)**:整體加速可由各 kernel 時間佔比以 Amdahl 預測，誤差 ≤ 20%。
- **H3(精度)**:選擇性 emulation 的真殘差 floor 和 FP64 差距在 ‖r‖ 的 2⁵ 倍內(48 對 53 bit),迭代數相同，觀測量一致。
- **H4(實作)**:float-float kernel 的暫存器數 ≤ double 版且不 spill,才能跑到頻寬上限。
- **H5(fusion × emulation)**:fusion 減少位元組、不減 flop,會提高 I,所以融合版從 emulation 獲益較多;但暫存器較多，較容易 spill。

## 8. 實驗

| 實驗 | 對照 | 量什麼 |
|---|---|---|
| E1 | Ruttice 1-pass 對 2-pass MDWF,各有 double 與 float-float | I、時間、暫存器 |
| E2 | Ruttice 對 Bridge++,同格點、config、mq、Ls 的 CGNE | D、DdagD、每 iteration 時間、完整 solve |
| E3 | 低精度前處理不變，只換 outer 的 FP64 算術 | 驗證兩者可疊加 |
| E4 | 原生 FP64 對選擇性 emulation,mq 掃描 | 真殘差曲線、觀測量、HMC ΔH |

階段:
1. RTX 上 ncu 剖析(原生 Linux),A30 對照，產出 kernel × 硬體決策表。
2. 實作 float-float kernel。閘門:無 spill。
3. RTX / AI GPU 實測和精度驗證。閘門:H2 誤差 ≤ 20%,H3 通過。
4. 撰稿：先投 Lattice 2027 proceedings,完整版投期刊。

尚未決定：格點大小(24⁴ 或 24³×48)、Ls(8 或 16)。

## 9. 相關工作

| 工作 | 內容 | 本計畫的不同 |
|---|---|---|
| Swirydowicz et al.(AMD),arXiv:2611.13332 | Ozaki-II GEMM 實測;GEMM emulation 只幫得上 < 10% HPC;Ozaki-II 小矩陣比原生慢 | 處理非 GEMM、在未來硬體上 FP64 算力受限的 kernel |
| Mukunoki & Ozaki,arXiv:2510.13536 | CPU 上全程 quasi multi-word CG | GPU、預條件 DWF、選擇性 emulation |
| Matsuoka,arXiv:2606.06510 | Ozaki II 用於 memory-bound 工作負載的模型與投影 | 實測;stencil 用 float-float |
| Kouya,arXiv:2609.35844 | branch-free pair renormalizer;GB10 上 binary32 pair 頻寬受限 | 其 future work 要求「依加速器的選擇準則」,即本計畫的貢獻;它沒有 stencil 或 LQCD |
| cuBLAS Ozaki | 只有 dense DGEMM | 處理 GEMM 以外的 kernel |
| QUDA 混合精度 | 低精度前處理 + FP64 outer | 正交，可疊加 |

## 10. 主要風險

- float-float kernel 暫存器過多而 spill(Ruttice double `hop_gemm_kernel` 已用 168 個)。
- 強度和規格數字都還沒量測，決策表可能要改。
- 沒有 B300 機器時，主要情境只能投影，文中要標明。
- float-float 少 5 bit,指數範圍和 float 相同(約 1e-38),極小值可能 underflow。
- 小質量、病態情況下 48 bit 是否足夠，要用 mq 掃描驗證。
