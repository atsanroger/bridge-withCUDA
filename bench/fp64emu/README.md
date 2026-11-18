# eo-DWF CGNR benchmark(FP64 emulation 研究用)

計畫與待辦見 `docs/fp64_emulation/`。這個目錄只放跑測試和 profiling 的腳本。

## 前置

- binary:`build/cuda-ampere-mpi/bridge_test_alt_spectrum`(preset `cuda-ampere-mpi`,sm_80;86,3080 是 sm_86)
  ```
  cmake --preset cuda-ampere-mpi && cmake --build build/cuda-ampere-mpi -j --target bridge_test_alt_spectrum
  ```
- config:Bridge++ `Binary`(big-endian double,沒有 header)。這台機器上的位置與 plaquette 見
  `/home/chen/red_gpu/affr-develop-latest/gauge_field/ENSEMBLES.md`(12⁴、16⁴、24⁴、24³×48、32⁴)。
  `make_run.sh` 會檢查檔案大小是否等於 V × 4 × 18 × 8 B。
- ncu 要在原生 Linux 上跑(WSL2 拿不到計數器),而且要有計數器權限。

## 用法

```
G=/home/chen/red_gpu/affr-develop-latest/gauge_field
./make_run.sh r24_fp FP 24x24x24x24 16 $G/conf_quench_24x24x24x24_v2.bin   # 產生 run 目錄
./run.sh r24_fp                                                            # 一般執行
BRIDGE_NSOLVE=1 BRIDGE_SKIP_CHECK=1 ./prof_nsys.sh r24_fp                  # 各 kernel 時間佔比
BRIDGE_PERF_NREP=2 BRIDGE_NSOLVE=1 BRIDGE_SKIP_CHECK=1 ./prof_ncu.sh r24_fp_ncu   # 各 kernel 計數器
```

- MODE 就是 yaml 的 `Solver: multiword_mode`,一個鍵決定整組精度設定:
  - `FP`:原生 double。
  - `lowendgpu`:double 儲存，只有 hopb 用 float-float 計算，其餘(5dir、LU_inv、BLAS)是原生 double。
  - `DW`:全程 float-float。`TW`:全程 float-triple。

  `field_type` 和 `extended_precision` 由它推導，yaml 裡不用寫;寫了而且矛盾時，程式會報錯。
  `su3_reconstruction` 和 `multiword_mode` 無關，四種模式都照它執行:`false`(預設)讀完整的 link(18 個實數),`true` 讀 12 個再重建第三行，用的是各模式自己的算術。
- `make_run.sh` 的選用環境變數:`MQ`(0.10)、`M0`(1.0)、`TOL`(1.0e-24,對 ‖r‖²/‖b‖²)、
  `MAXITER`(100)× `NRESTART`(40)= 總迭代上限。Möbius (b, c) = (1.5, 0.5),α = 1。
  ncu 用的 run 目錄建議 `MAXITER=20 NRESTART=1`,只看幾個 iteration。
- solver 是 CGNR(alt_Accel 只有 CGNR)。它和 CGNE 用的 kernel 相同(D†D 對 DD†),每個 iteration 的成本相同。

## test 程式的環境變數開關

不設定時，行為和原本完全相同。

| 變數 | 預設 | 作用 |
|---|---|---|
| `BRIDGE_PERF_NREP` | 200 | `mult_performance` 的重複次數(Dee_inv、Doe、D、Ddag、DdagD 各量一次) |
| `BRIDGE_NSOLVE` | 12 | 解幾個 color-spin source;少於 12 時跳過 correlator 與 m_res |
| `BRIDGE_SKIP_CHECK` | 不設 | 跳過開頭和 CPU corelib 的 operator 比對(24⁴×16 上約 2 分鐘) |
| `BRIDGE_ONLY_5DIN` | `run.sh` 會設 1 | 不跑後面的 PVprec 2pt test |
| `BRIDGE_HOPB_V2` | 不設 | 設 1 時 hopb 改用 v2(gauge 放 shared memory),只在 `su3_reconstruction: true` 且 Ls 是 8 的倍數時生效，原生與 lowendgpu 都一樣。不設時兩者都用 v1;A30 上 v1 較快(24³×48、Ls 8,原生 2.45 對 2.69 ms/call,lowendgpu 32.4 對 32.7 ms/call),3080 上待量 |

## 輸出

- `prof_nsys.sh`:`RUNDIR/nsys.nsys-rep`,以及依時間排序的 kernel 表(`kern_table.py`)。
- `prof_ncu.sh`:`RUNDIR/ncu.ncu-rep`、`ncu_raw.csv`,以及每個 kernel 的平均值(`ncu_table.py`):
  時間、DRAM MB、GFLOP/s(FP64 與 FP32,FMA 算 2)、I64 = FP64 flop / DRAM byte、DRAM GB/s、
  FP64 pipe 使用率、DRAM 使用率、occupancy、register 數。
- Bridge++ 自己印的 GFlops 以 FP64 flop 計。lowendgpu 的 hopb 實際執行的是 FP32 指令，所以那是「FP64 等效」的數字，要和 ncu 的 GF32 分開看。
- 晶格太小時資料會留在 L2 裡(A30 是 24 MB,3080 是 5 MB),DRAM byte 會偏低、I64 會偏高。正式量測用 24⁴ 以上。
