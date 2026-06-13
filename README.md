# Parallel LU Factorization solver using OpenMP

**Course Project:** ID5130 — Parallel Scientific Computing  
**Authors:** Karthik Puranam & Abhinav Singh 
**Date:** May 5, 2026 


## Project Overview
This repository contains a high-performance C++ implementation and performance analysis of a parallel LU factorization solver using **OpenMP** on shared-memory multi-core CPUs. 

The system matrix $A$ is generated from a fully implicit backward-Euler finite difference discretization of the time-dependent **2D Heat Equation** on a uniform grid

$$\frac{\partial u}{\partial t}=\alpha\left(\frac{\partial^{2}u}{\partial x^{2}}+\frac{\partial^{2}u}{\partial y^{2}}\right)$$ 

The resulting linear system $(I-\lambda A_{h})u^{t+1}=u^{t}$ yields a large, sparse, block-tridiagonal stencil matrix. While the initial matrix is sparse, severe fill-in during factorization makes the resulting $L$ and $U$ factors dense, introducing heavy computational and memory transfer overheads.


## Repository Structure
The repository is kept completely flat and modularized for clarity:

* `ID5130_Project.pdf`: Comprehensive project report covering mathematical formulations, scaling charts, and deep architectural analysis.
* `heat_gen.cpp`: Code to assemble the sparse 5-point Laplacian stencil matrix representing the implicit physical grid domain.
* `lu_naive.cpp`: Standard LU factorization parallelized via loop-level `#pragma omp parallel for` constructs.
* `lu_blocked.cpp`: Cache-optimized tiled implementation that processes data in $b \times b$ blocks, dramatically reducing memory bandwidth pressure and synchronization overhead

## Mathematical & Complexity Profile
Standard sequential and loop-level naive implementations scale aggressively with matrix size:
* **Time Complexity:** $O(N^3)$ — Total floating-point operations $\approx \frac{2}{3}N^3$.
* **Space Complexity:** $O(N^2)$ — Due to dense fill-in.
* **Memory Stalls:** Naive parallel routines transfer $O(N^3)$ data between main memory and cache, choking performance on shared-memory multi-core CPUs.

### The Blocked Optimization Advantage
By packing memory into an optimized $b \times b$ tile architecture, the blocked version achieves:
* **Data Transfer Reduction:** Cuts total memory transfers significantly down to $O(N^3/b)$.
* **Barrier Amortization:** Synchronization overhead drops drastically from $N$ loop barriers down to $N_b = \lceil N/b \rceil$ block barriers.



