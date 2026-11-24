# CUDA Matrix Multiplication (SGEMM) Optimization Report

## 1. Problem & Objective
* **Problem:** Compute the matrix multiplication $C = A \times B$ for single-precision floating-point (`FP32`) matrices of size $N \times N$ ($2048 \times 2048$).
* **Target:** Optimize a custom CUDA kernel to achieve performance comparable to NVIDIA's standard **cuBLAS** library.
* **Hardware Environment:** NVIDIA GeForce RTX 2080 Ti (Compute Capability 7.5, 68 SMs).
* **Evaluation Metrics:**
  * Kernel execution time (`ms`)
  * Compute throughput (`GFLOPS`)
  * Relative performance compared to cuBLAS (`%`)

---

## 2. Development Milestones (23/Nov/2026)

### Milestone 1: Baseline (Basic Shared Memory Tiling)
* **Approach:** Partitioned matrices $A$ and $B$ into sub-matrices (tiles) and loaded them iteratively into on-chip Shared Memory to reduce global memory bandwidth bottlenecks. Applied `__syncthreads()` after each phase to prevent race conditions.
* **Verification:** Output verified against cuBLAS with zero functional discrepancy.
* **Result:** Achieved **7,055.35 GFLOPS** (`2.43501 ms`), reaching **65.68%** of cuBLAS performance.

### Milestone 2: Tile Size & Resource Tuning (Current)
* **Approach:** Tuned tile dimensions and thread block configurations (`16x16` grid/block layout) to better utilize on-chip Shared Memory (`16.38 KB/block`) and register files (`124 registers/thread`).
* **Result:** Increased throughput to **7,545.39 GFLOPS** (`2.27687 ms`), reaching **70.74%** of cuBLAS performance (+5.06% improvement over Baseline).

---

## 3. Performance Benchmark ($N = 2048$)

| Implementation | Execution Time (ms) | Throughput (GFLOPS) | % of cuBLAS | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Milestone 1: Baseline (Shared Mem)** | 2.43501 | 7,055.35 | 65.68% | Passed |
| **Milestone 2: Tuned Tile Size** | **2.27687** | **7,545.39** | **70.74%** | **Current** |
| **NVIDIA cuBLAS (Reference)** | 1.61067 | 10,666.30 | 100.00% | Reference |

---

## 4. Bottleneck Analysis (via NVIDIA Nsight Compute)

Profiling the current `matrixMulKernel` (`2.24 ms` kernel duration, `73.11%` Compute & Memory throughput) revealed three primary hardware bottlenecks preventing higher performance:

1. **Tail Effect (Est. Speedup: up to 50.00%)**
   * **Observation:** With `50%` theoretical occupancy (2 blocks/SM) across 68 SMs, the GPU executes `136 blocks` per full wave. Launching `256 blocks` results in `1.88` waves per SM (1 full wave of 136 blocks + 1 partial wave of 120 blocks), leaving several SMs idle during the second wave.
2. **Shared Memory Bank Conflicts (Est. Speedup: 26.50%)**
   * **Observation:** Shared memory store operations suffer from an average **1.5-way bank conflict** (`2,097,152` conflicts across `4,194,304` requests), causing `Stall MIO Throttle` to be the dominant warp stall reason.
3. **Low Occupancy & Latency Hiding (Est. Speedup: 26.89%)**
   * **Observation:** High register usage (`124 registers/thread`) limits theoretical occupancy to `50%` (4 warps/scheduler instead of 8). Consequently, schedulers issue instructions on only `48.97%` of cycles (`51.03%` cycles have no eligible warps).

---

## 5. Next Steps

- [ ] **Eliminate Bank Conflicts:** Add padding to shared memory tile arrays (e.g., `tile[TILE_SIZE][TILE_SIZE + 1]`) to resolve column-major access conflicts.
- [ ] **Reduce Register Pressure:** Refactor inner loop unrolling or test `--maxrregcount` compiler flags to increase active warps per SM without triggering local memory spilling.
- [ ] **Mitigate Tail Effect:** Implement **2D Block-Tiling / Register Tiling** (each thread computes multiple output elements) and adjust grid/block dimensions to align with multiples of `68 SMs`.
- [ ] **Vectorized Memory Access:** Use `float4` loads/stores to maximize global memory transaction efficiency.