# FP64 emulation TODO(eo-DWF CGNE,branch `feat/no-amg`)

計畫書:https://claude.ai/artifact/RTZY7fZ7BVT3tBS7AJAzYe

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
- Bridge++ 這一側保持不 fuse,作為和 RutticeQCD(大量 kernel fusion)對照的那一軸。**不要在 Bridge++ 裡動手 merge。**

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

- 測 QTW `Renormalize3`(`bridgeACC_AField.h:190`)的 non-overlap 違反率。它是 3 次 TwoSum,少了 Kouya RenormBF-pair(arXiv:2609.35844)最上面那一步 FastTwoSum(h, m)。
  用 Kouya 的 [II] 類輸入，和 MPFR 比對。
  和小質量 QTW floor 變差(mq 0.025:1.3e-28,mq 0.10:5e-37)有沒有關係，目前只是假設。
  double 儲存的選擇性設計不受影響，因為寫回 double 本身就是一次 renormalization。
