# Parallel LU Factorization solver using OpenMP

**Course Project:** ID5130 — Parallel Scientific Computing  
[cite_start]**Authors:** Karthik Puranam & Abhinav Singh [cite: 2, 3]  
[cite_start]**Date:** May 5, 2026 [cite: 4]

---

## 📌 Project Overview
[cite_start]This repository contains a high-performance C++ implementation and performance analysis of a parallel LU factorization solver using **OpenMP** on shared-memory multi-core CPUs[cite: 6]. 

[cite_start]The system matrix $A$ is generated from a fully implicit backward-Euler finite difference discretization of the time-dependent **2D Heat Equation** on a uniform grid[cite: 13, 17, 23]:

[cite_start]$$\frac{\partial u}{\partial t}=\alpha\left(\frac{\partial^{2}u}{\partial x^{2}}+\frac{\partial^{2}u}{\partial y^{2}}\right)$$ [cite: 19]

[cite_start]The resulting linear system $(I-\lambda A_{h})u^{t+1}=u^{t}$ yields a large, sparse, block-tridiagonal stencil matrix[cite: 25, 30]. [cite_start]While the initial matrix is sparse, severe fill-in during factorization makes the resulting $L$ and $U$ factors dense, introducing heavy computational and memory transfer overheads[cite: 31].

---

## 📂 Repository Structure
The repository is kept completely flat and modularized for clarity:

* [cite_start]`ID5130_Project.pdf`: Comprehensive project report covering mathematical formulations, scaling charts, and deep architectural analysis[cite: 6, 74, 196].
* [cite_start]`heat_gen.cpp`: Code to assemble the sparse 5-point Laplacian stencil matrix representing the implicit physical grid domain[cite: 14, 26, 30].
* [cite_start]`lu_naive.cpp`: Standard LU factorization parallelized via loop-level `#pragma omp parallel for` constructs[cite: 7].
* [cite_start]`lu_blocked.cpp`: Cache-optimized tiled implementation that processes data in $b \times b$ blocks, dramatically reducing memory bandwidth pressure and synchronization overhead[cite: 7, 34].

---

## ⚙️ Mathematical & Complexity Profile
[cite_start]Standard sequential and loop-level naive implementations scale aggressively with matrix size[cite: 40]:
* [cite_start]**Time Complexity:** $O(N^3)$ — Total floating-point operations $\approx \frac{2}{3}N^3$[cite: 41].
* [cite_start]**Space Complexity:** $O(N^2)$ — Due to dense fill-in[cite: 42].
* [cite_start]**Memory Stalls:** Naive parallel routines transfer $O(N^3)$ data between main memory and cache, choking performance on shared-memory multi-core CPUs[cite: 12, 45].

### The Blocked Optimization Advantage
[cite_start]By packing memory into an optimized $b \times b$ tile architecture, the blocked version achieves[cite: 34]:
* [cite_start]**Data Transfer Reduction:** Cuts total memory transfers significantly down to $O(N^3/b)$[cite: 46].
* [cite_start]**Barrier Amortization:** Synchronization overhead drops drastically from $N$ loop barriers down to $N_b = \lceil N/b \rceil$ block barriers[cite: 47].

---

## 📊 Performance Benchmarks
[cite_start]All implementations were thoroughly benchmarked across the following parameter combinations[cite: 15]:
* [cite_start]**Matrix Size ($N$):** $\{484, 1024, 2025, 4096\}$ [cite: 15]
* [cite_start]**Block Size ($b$):** $\{64, 128, 256\}$ [cite: 15]
* [cite_start]**Thread Count ($p$):** $\{1, 2, 4, 8\}$ [cite: 15]

### Key Discoveries
* [cite_start]**Absolute Speedup:** At the largest matrix size ($N = 4096$) utilizing 8 execution threads, the Blocked OpenMP architecture completes execution in just **3.73 seconds**, representing an absolute **15.75x speedup** over the standalone sequential baseline ($58.8$ seconds)[cite: 192].
* [cite_start]**Cache Reuse Multiplier:** Running completely single-threaded, the blocked solver ($b=128$) records a **3.71x speedup** over sequential naive execution[cite: 184, 188]. [cite_start]This confirms that optimizing cache-line reuse is a fundamental prerequisite to scalable parallel engineering[cite: 189].
* [cite_start]**Synchronization Penalties:** For small grid matrices ($N \le 484$), multi-threaded naive loop scheduling is actively detrimental[cite: 103, 190]. [cite_start]The synchronization costs of thread fork/join operations and step barriers overpower the actual compute payload, making 8-thread naive routines slower than running on a single thread[cite: 103, 191].

---

## 🚀 Compilation & Execution

Each implementation file can be built independently using any standard compiler with OpenMP flag triggers.

### 1. Compile Code
```bash
# Compile Matrix Generator
g++ -O3 heat_gen.cpp -o heat_gen

# Compile Naive Loop Parallel Solver
g++ -O3 -fopenmp lu_naive.cpp -o lu_naive

# Compile Optimized Blocked Parallel Solver
g++ -O3 -fopenmp lu_blocked.cpp -o lu_blocked
