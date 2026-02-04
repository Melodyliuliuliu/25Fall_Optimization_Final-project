# 25Fall_Optimization_Final-project
This is my final project for the course DATS-SHU 200-001 Topics in Machine Learning: Optimization for Data Science and Machine Learning. This project presents an empirical comparison of FISTA and ADMM for solving $l_1$-regularized optimization problems.
While the primary focus is on the **LASSO** problem, the project also touches upon Logistic Regression and Generalized LASSO. Through controlled experiments on synthetic data, this study investigates:
* **Convergence Behavior:** Transition to local linear convergence upon support identification.
* **Computational Efficiency:** Trade-offs between iteration count and wall-clock time.
* **Parameter Sensitivity:** Robustness of ADMM to the penalty parameter $\rho$ and FISTA to step size estimation.

> **Note:** This repository contains the code and implementation details for the report *"Comparing FISTA and ADMM for l1-Regularized Optimization: A LASSO-Centric Study"* (Xingtong Liu, Dec 2025).

## 🚀 Key Implementations

The repository implements the following algorithms in Python:

1.  **ISTA (Baseline):** Standard proximal gradient method.
2.  **FISTA (Accelerated):**
    * Standard implementation with constant step size ($1/L$).
    * Adaptive implementation using **Backtracking Line Search** (removing the dependency on pre-computing $L$).
3.  **ADMM (Split-Variable):**
    * **Naive Implementation:** Uses standard matrix inversion ($O(n^3)$) per update.
    * **Optimized Implementation:** Leverages the **Matrix Inversion Lemma** (Sherman-Morrison-Woodbury formula) to reduce the per-iteration complexity to $O(mn)$, significantly improving runtime for high-dimensional features.

## 📊 Experimental Highlights

The experiments were conducted using synthetic data ($m=50, n=100$, sparsity $K=10$). Key findings include:

### 1. Convergence & Support Identification
* **Sign Stability:** We empirically verified that algorithms trigger **Local Linear Convergence** once the "active support set" (non-zero indices) is correctly identified.
* **Performance:** ADMM identifies the support set significantly earlier ($k \approx 30$) compared to FISTA ($k \approx 107$) and ISTA ($k \approx 200$).

### 2. Wall-Clock Time vs. Iterations
* While **ADMM** converges in fewer iterations, **FISTA** demonstrates superior computational efficiency (lower wall-clock time) for this problem size due to cheap matrix-vector multiplications.
* The optimized ADMM implementation bridges this gap but remains computationally heavier per step than FISTA.

### 3. Parameter Sensitivity
* **ADMM:** Empirical analysis shows optimal convergence with penalty parameter $\rho \approx 1.0 \sim 2.0$. Performance degrades significantly when $\rho < 0.1$ (oscillations) or $\rho > 10$ (slow progress).
