# FP64 emulation TODO(eo-DWF CGNE,branch `main`)

計畫內容見 `note_zh.md`(英文版 `note_en.md`)。

目標：在 FP64 精度下，讓 CGNE 在 RTX / AI GPU 上從 FP64 算力受限變成頻寬受限。
整條計算鏈都要 ≥ 48 bit(原生 double 或 float-float,不能有 FP32 環節)。
哪些 kernel 用 emulation,依各 kernel 的強度 I 對照硬體 ridge R = P64/BW 決定。

量測平台:RTX 3080(修好後)。A30 只當 FP64 強的對照組。
ncu 要在原生 Linux 上跑，不能用 WSL2。

## 目前的程式結構(已讀過程式碼)

- `DdagD`(`src/lib_alt_Accel/Fopr/afopr_Domainwall_5din_eo-tmpl.h:1058`)每次呼叫約 14 個 kernel:
  4 次 5dir/5dirdag、4 次 hopb、4 次 LU_inv/LUdag_inv、2 次 copy+axpy。
- 5dir 和 hopb **沒有** merge:5dir 把 y 寫到 global 的 `m_w1`,hopb 再讀回來。LU_inv 也是獨立 kernel。
- hopb 有 `jgm5` 參數可以把 γ5 併進去，但 `D_eo` / `Ddag_eo` 都傳 0。CGNE 的 DdagD 路徑沒有用到 γ5。
- 保持不 fuse,**不要動手 merge**。emulation 是要讓 kernel 回到 memory bound;融合後暫存器壓力更大，用 float-float 更容易 spill,spill 的流量會讓已經 memory bound 的 kernel 更慢(推論，未量測)。

## 各 kernel 強度估計(未量測)

每 5D site,FP64:

| kernel | flop | 強度 (flop/B) | 3080 (R≈0.61) | B300 (R≈0.16) |
|---|---|---|---|---|
| hopb | ~1320 | ~0.5–0.7 | 邊界附近 | FP64 算力受限 |
| 5dir / 5dirdag | 168 | ~0.44 | 頻寬受限，接近 ridge | FP64 算力受限(約 2.7×) |
| LU_inv / LUdag_inv | ~264 | ~0.35–0.7 | 接近 ridge | FP64 算力受限 |
| copy / axpy / dot / norm | 2 | 0.08–0.12 | 頻寬受限 | 頻寬受限 |

在 RTX 上可能只需要改 hop,在 B300 上 5dir 和 LU_inv 也要改。BLAS 在所有硬體上都保持原生 double。

## 待辦

### 0. 先決定
- [ ] 晶格大小:24⁴ 或 24³×48
- [ ] Ls:8 或 16(會影響 hopb 的 gauge 攤提和 LU_inv 的強度)

### 1. RTX 3080 上 FP64 DdagD 的 ncu profiling
- [ ] 每個 kernel 記錄:時間佔比、DRAM 吞吐率、FP64 pipe 使用率、達到的 flop/B
- [ ] 用量測值取代上表的估計，決定哪些 kernel 要做 emulation
- [ ] A30 跑同一組設定當對照

### 2. hybrid hopb(double 儲存 + float-float 計算)
- [ ] 從 `mult_Domainwall_5din_eo_cuda_qdw-inc.h:1399` 的 `hopb_qdw<EXT, RECON>` 改寫
- [ ] 輸入/輸出都是 double,在暫存器裡拆成 float hi/lo,計算用 float-float,寫回時合成 double(不需要另外的 normalize pass)
- [ ] `cuobjdump --dump-resource-usage` 檢查暫存器和 spill(TW hopb 用到 167 暫存器、spill 2304 B;DW 的數字還沒記錄)
- [ ] 精度驗證(可以先在 A30 上做):和 FP64 比 ‖D_hyb x − D_fp64 x‖ / ‖D x‖
- [ ] 在 3080 上量 kernel 時間，和原生 FP64 hopb 比

### 3. 5dir / LU_inv 的 hybrid 版本(依第 1 步結果決定)
- [ ] 若在 3080 上是 FP64 算力受限就做
- [ ] 即使 3080 不需要，為了推論 B300 也可能要做

### 4. 完整 CGNE solve
- [ ] 比較 FP64、hybrid、全程 QDW 三者:iteration 數、真殘差 floor、總時間
- [ ] 用 roofline 模型預測加速，目標誤差在 20% 以內

## 延後(先不做)

- lowendgpu 在多 process(有 halo)時，邊界那幾段 kernel 仍是 double,而且不會報錯(QDW/QTW 會 exit)。單 GPU 研究不受影響;要用 MPI 前，先比照 QDW/QTW 加上報錯，或寫 hyb 版的邊界 kernel。
- `hyb_split` 在 |x| < 2⁻⁷⁵ 時丟掉 lo(只剩 24 bit),|x| < 2⁻¹²⁶ 時歸零。就 norm 而言可以忽略;可能受影響的只有本身就小於約 1e-23 的分量(例如離 source 很遠的 propagator)。推論，未測試。
- hyb 的 `double * hyb_ff` 只在係數是 ±1、2 時精確。在 hop include 裡加入其他係數(例如 κ)之前，要先改這個運算子。
- 3080 上重量 hopb v1 對 v2(`BRIDGE_HOPB_V2=1`),決定預設。A30 上 v1 較快。

- 測 QTW `Renormalize3`(`bridgeACC_AField.h:190`)的 non-overlap 違反率。它是 3 次 TwoSum,少了 Kouya RenormBF-pair(arXiv:2609.35844)最上面那一步 FastTwoSum(h, m)。
  用 Kouya 的 [II] 類輸入，和 MPFR 比對。
  和小質量 QTW floor 變差(mq 0.025:1.3e-28,mq 0.10:5e-37)有沒有關係，目前只是假設。
  double 儲存的選擇性設計不受影響，因為寫回 double 本身就是一次 renormalization。

## 升級到 Bridge++ 2.1.3(延後，等 3080 量測做完)

2026-11-18 在 A30 上做過一次試驗升級，沒有 commit。
做法：從 main(d1b326e)開 detached worktree,用 `/home/chen/red_gpu/bridge213/bridge-2.1.3/src` 覆蓋 `src/`。
fork 的 import base 是 commit 5ad12a5(VERSION 2.0.0-mod.93)。

### 規模(base 5ad12a5、2.1.3、main 三方比較，只算 `src/`)

- base 1512 個檔，2.1.3 931 個檔。
- base 和 2.1.3 都有、內容不同:406 個。只在 base(2.1.3 沒發佈):617 個。2.1.3 新增:36 個。
- 上游改動最多的目錄:`lib`(175 檔，約 17.9k 行，fork 只改了其中 1 檔)、`BridgeOpenACC`(32 檔，約 12.1k 行)、QXS inline(約 21k 行)、`lib_alt_Accel/Fopr`(43 檔，約 5.8k 行)。
- fork 改過的 161 個檔:112 個只在 base(BridgeCUDA、BridgeNVSHMEM 等 fork 自己的樹，2.1.3 沒有),19 個是 fork 新增，**30 個是雙方都改過的真正衝突**。
- CMake 用 GLOB 收集原始碼，2.1.3 新增的檔案會自動編進去，不用改 `CMakeLists.txt` 的清單。

### 試驗中為了編過而做的處理

1. 覆蓋 865 個檔、新增 36 個;30 個衝突檔暫時全部保留 fork 版(上游在這些檔裡的改動沒有收進來)。
2. `lib_alt_Accel/inline/define_params.h` 保留 fork 版。2.1.3 版缺 `NS_MAX`、`WARP_LENGTH`、`MAX_THREAD_PER_BLOCK`、`getNumSMs`,BridgeCUDA 需要它們(不處理時 263 個編譯錯誤)。
3. 產生 `src/version.h`。2.1.3 的 `configure.h` 會 include 它，但只有上游 Makefile 的 `version_h` 規則會產生(`#define BRIDGE_VERSION "..."`),CMake 沒有。正式升級要在 CMake 加一條 configure_file。
4. 刪掉 76 個只在 base、2.1.3 沒發佈的舊開發檔(例如 `lib/Force/Fermion/aforce_F_Smeard.cpp`、`force_F_Overlap_Nf2`、`fopr_Overlap_5d`、`lib_alt/Force/Fermion/aforce_F_*`、`lib/Eigen/*.res`),它們和新 API 對不上。
   例外：留下 `lib_alt/Solver/asolver_FGMRES*`(`lib_alt_Accel/Solver/asolver_FGMRES.cpp` 用到),以及 fork 版的 `astaple_lex*`(Accel/SIMD2/Vector 三份;fork 的 CloverTerm 用到 `upper_th`/`lower_th`)。
5. `lib_alt_Accel/inline/define_available.h` 加 `#if defined(USE_ACCEL_CUDA) || defined(USE_ACCEL_NVSHMEM)` 區塊，定義 Wilson、Wilson_eo、Clover、Clover_eo、Domainwall、Domainwall_eo、Domainwall_5din、Domainwall_5din_eo 的 `ACCEL_FOPR_*_AVAILABLE`。
   這是 2.1.3 留給其他後端的掛鉤，原本只有 OpenACC 區塊。不加時所有 AFopr 的 `*_double.cpp`/`*_float.cpp` 都被 `#ifdef` 掉，link 時缺 vtable(193 個 undefined reference)。Staggered 沒有加，因為 BridgeCUDA 沒有 staggered kernel。
6. `afopr_Clover*`、`afopr_CloverTerm*` 保留 fork 版。2.1.3 版呼叫的 `mult_csw_dirac/chiral`、`mult_clover_D_dirac/chiral`、`mult_wilson_1_dirac/chiral` 參數和 BridgeCUDA 的宣告不同。
7. 錯誤數:263 → 16 → 5 → 193(link)→ 12 → 0。只 build 了 `bridge_test_alt_spectrum`(preset `cuda-ampere-mpi`)。

### 試驗結果(A30,12⁴,Ls 8,mq 0.10,`BRIDGE_NSOLVE=2`、`BRIDGE_PERF_NREP=20`,單 GPU)

小設定，只用來比對數值，不是效能量測。

| 模式 | corelib diff2(trial) | 和 main 比 | Main 總時間 main / trial |
|---|---|---|---|
| FP | 2.7e-30、3.4e-30 | 數值輸出相同 | 5.10 s / 5.12 s |
| lowendgpu | 3.4e-28、2.5e-28 | 數值輸出相同 | 8.59 s / 8.63 s |
| DW | 4.5e-13(float operator 檢查) | 數值輸出相同 | 9.60 s / 9.64 s |
| TW | 4.5e-13(float operator 檢查) | 數值輸出相同 | 11.50 s / 11.67 s |

「數值輸出相同」指去掉時間、GFlops、記憶體和 log 文字後 diff 為 0 行。
不同的只有 log:版本字串、多印 plaq、Smear/Stout 建構訊息、多印 `alpha`。
因為 30 個衝突檔都用 fork 版，這個結果只說明 2.1.3 的 corelib 和其他檔不改變 emulation 路徑的數值，不代表合併後的版本正確。

### 30 個衝突檔(上游改動行數 / fork 改動行數，對 base 5ad12a5)

| 檔案(`src/` 下) | 上游 | fork |
|---|---|---|
| `lib_alt_Accel/Fopr/afopr_Wilson-tmpl.h` | 658 | 22 |
| `lib_alt_Accel/Fopr/afopr_Domainwall_5din_eo-tmpl.h` | 273 | 522 |
| `lib_alt_Accel/Fopr/afopr_Domainwall_5din-tmpl.h` | 209 | 63 |
| `lib_alt/Measurements/Fermion/fprop_alt_Standard_eo-tmpl.h` | 159 | 188 |
| `lib_alt_Accel/Fopr/afopr.cpp` | 88 | 12 |
| `lib_alt_Accel/Field/afield-tmpl.h` | 73 | 284 |
| `lib_alt_Accel/Field/afield-inc.h` | 53 | 121 |
| `lib_alt_Accel/Field/afield_double.cpp` | 52 | 4 |
| `lib_alt_Accel/Fopr/afopr_Wilson.h` | 50 | 15 |
| `lib_alt_QXS/Field/aindex_eo.h` | 44 | 4 |
| `lib_alt/Measurements/Fermion/fprop_alt_Standard_eo.h` | 32 | 5 |
| `lib_alt/Solver/asolver_CG-tmpl.h` | 31 | 374 |
| `lib_alt_Accel/Fopr/afopr_Domainwall_5din.h` | 29 | 25 |
| `lib_alt_Accel/Field/afield_float.cpp` | 24 | 4 |
| `lib_alt_QXS/Field/aindex_eo-inc.h` | 23 | 6 |
| `lib_alt_Accel/BridgeOpenACC/bridgeACC_Domainwall_float.cpp` | 20 | 1 |
| `lib_alt_Accel/BridgeOpenACC/bridgeACC_Domainwall_double.cpp` | 20 | 1 |
| `lib_alt_Accel/Field/aindex_eo-inc.h` | 16 | 30 |
| `lib_alt_Accel/BridgeOpenACC/bridgeACC_Wilson_float.cpp` | 15 | 1 |
| `lib_alt_Accel/BridgeOpenACC/bridgeACC_Wilson_double.cpp` | 15 | 1 |
| `lib_alt/Solver/asolver_CG.h` | 15 | 19 |
| `lib_alt_Accel/Fopr/afopr_Domainwall_5din_eo.h` | 13 | 30 |
| `lib_alt_Accel/BridgeOpenACC/bridgeACC_Clover_double.cpp` | 13 | 1 |
| `lib_alt_Accel/BridgeOpenACC/bridgeACC_AField_float.cpp` | 13 | 1 |
| `lib_alt_Accel/Field/aindex_eo.h` | 11 | 6 |
| `lib_alt_Accel/BridgeOpenACC/bridgeACC_Clover_float.cpp` | 11 | 1 |
| `lib_alt_Accel/BridgeOpenACC/bridgeACC_AField_double.cpp` | 11 | 1 |
| `lib/Fopr/afopr.h` | 6 | 4 |
| `lib_alt_Accel/Field/afield.h` | 4 | 59 |
| `lib_alt_Accel/inline/define_index.h` | 2 | 2 |

上游改動的主要內容:
- `afopr_Domainwall_5din_eo-tmpl.h`:detailed timer(`START_TIMER`/`STOP_TIMER`,pack/bulk/boundary/comm)、`require_communication` 參數、`m_Nbdsize` 改用 `ceil_nwp`、`set_parameters` 回寫 `gamma_matrix_type`/`code_implementation`。
- `afopr_Wilson-tmpl.h`:Wilson hopping 拆成各方向 kernel(`mult_wilson_xp1/xm2/…/tp2`)、`mult_DorH`、`require_communication`、新的 copy_to/from_device 呼叫。
- fork 的改動:QDW/QTW/hyb(lowendgpu)、hopb v1/v2 與 RECON、記憶體瘦身(fprop)、CG 的 multiword 路徑(`asolver_CG-tmpl.h` 374 行)。

### BridgeACC API 差異(2.1.3 BridgeOpenACC 的 header 對 base)

- 2.1.3 新增:`mult_domainwall_5din_LUinv_dirac`、`mult_domainwall_5din_LUdaginv_dirac`、`mult_domainwall_5din_hopb_dirac_4d`、`mult_domainwall_5din_hopb_dirac_5d`、`mult_staggered_phase`,以及 Wilson 各方向 kernel。
- 2.1.3 移除(fork 的 BridgeCUDA 仍有):`block_*_eo`、`mult_clover_coarse_*`、`mult_clover_H_*`、`mult_clover_dd_dirac`、`mult_domainwall_5din_L_inv/U_inv/…` 等約 50 個名稱。
- 要用 2.1.3 版的 Wilson/Clover/DWF Fopr,BridgeCUDA 需要補上新增的函式並對齊參數。

### 待辦
- [ ] 從 2.1.3 開新分支，把 fork 的 patch 依功能分批搬過去(BridgeCUDA 後端 → define_params/define_available/version.h → AField/CG multiword → DWF eo Fopr emulation → fprop 瘦身),不要把 2.1.3 直接 merge 進 main。
- [ ] 30 個衝突檔做三方合併。估計 2–4 天(估計，未驗證),最重的是 `afopr_Domainwall_5din_eo-tmpl.h` 和 `afopr_Wilson-tmpl.h`。
- [ ] BridgeCUDA 補 2.1.3 新增的 API(LUinv/LUdaginv、hopb 4d/5d、Wilson 方向 kernel),或決定永久保留 fork 版的 Wilson/Clover Fopr。估計 2–5 天，看補多少。
- [ ] CMake 產生 `version.h`。
- [ ] `use_accel_impl = cuda` 的正式接法:2.1.3 的 `USE_ACCEL_CUDA` 會 include 不存在的 `deviceManager_CUDA.h`,要用 fork 的版本補上。
- [ ] 其他 target(HMC、eigen 等 test)全部 build;確認刪掉的 76 個舊檔沒有別人用。估計 1–2 天。
- [ ] 驗證:四種模式 × su3_reconstruction true/false 和 main 比數值;24³×48 Ls 8 的記憶體用量不變。
- [ ] 沒測過：MPI 多 rank、NVSHMEM 後端、上游 Makefile build。
