# A Comparative Analysis of Variational and Kernel Quantum Classifiers Under Depolarizing Gate Stress in IBM Qiskit 2.5.2

**Authors:** Rajnish Singh, Utkrisht Patel, Rakesh Kumar  
*International Journal of Computational Intelligence and Applications*

This repository contains the official reproducible benchmark suite, datasets, and analysis scripts for our study evaluating the resilience of Quantum Support Vector Classifiers (QSVC) against Variational Quantum Classifiers (VQC) under depolarizing gate noise across 5 benchmark datasets.

---

## 1. Abstract
Quantum machine learning (QML) models are expected to perform well on high-dimensional classification tasks, but few benchmarks compare them with classical models under stress-test levels of simulated gate noise. We compare a variational quantum classifier (VQC), trained once with Constrained Optimization BY Linear Approximation (COBYLA) and once with Simultaneous Perturbation Stochastic Approximation (SPSA), and a Quantum Support Vector Classifier (QSVC) with a classical Radial Basis Function Support Vector Machine (RBF-SVM). The quantum models were implemented in IBM Qiskit 2.5.2 and run without noise and under two levels of depolarizing noise (low stress and high stress). Five binary datasets were used: Iris, make_moons, make_circles, Breast Cancer, and Wine. Each was reduced to two features by principal component analysis (PCA) and encoded in a two-qubit ZZFeatureMap circuit. Each result is the mean and standard deviation of five random seeds.

The RBF-SVM was the most accurate model on every dataset (0.893 to 0.993). Among the quantum models, the QSVC was the most accurate on four of the five datasets, and a VQC was the most accurate on make_circles (trained with COBYLA in the ideal setting and with SPSA under noise). Noise changed the accuracy only slightly: in 29 of the 30 model–dataset–noise comparisons the change was smaller than the larger of the two standard deviations involved (the exception is a rise of 0.053 for the VQC trained with COBYLA on make_moons at high stress), so we cannot conclude that either architecture is more robust to noise. Whether precision or recall was higher depended on the dataset. The QSVC had higher recall than precision on Breast Cancer, Wine, and make_circles, and slightly higher precision than recall on Iris and make_moons. After transpilation, the VQC circuit (depth 15) and the QSVC kernel circuit (depth 18) have similar sizes, with six controlled-NOT (CNOT) gates each. We found no quantum advantage on these tasks. The experiments do not test parameter efficiency, so the parameter counts are given for reference only.

---

## 2. Research Questions Addressed
* **RQ1:** What is the difference in performance of variational and kernel-based quantum classifiers under depolarizing noise across various evaluation metrics?
* **RQ2:** Is there a classification asymmetry in terms of precision and recall that is not captured by aggregate accuracy for quantum classifiers?
* **RQ3:** How do the numbers of trainable parameters of the quantum models compare with those of classical models, and do these experiments support any claim of parameter efficiency?

---

## 3. Reproducibility & Verification Guide

### All Runs Are Deterministically Seeded
To ensure 100% exact numerical reproducibility:
* **Seeds Used:** `[42, 100, 2023, 777, 1234]`
* **Sampling & Simulator Seeding:**
  - `Ideal`: `StatevectorSampler(default_shots=1024, seed=seed)`
  - `Noisy`: `AerSamplerV2(default_shots=1024, options={'backend_options': {'noise_model': noise_model, 'seed_simulator': seed}})`
  - A fresh, unshared sampler instance is instantiated for every model.
* **Random Generators:** `np.random.seed(seed)`, `qiskit_algorithms.algorithm_globals.random_seed = seed`, and `qiskit_machine_learning.utils.algorithm_globals.random_seed = seed` are set prior to model execution.
* **VQC Parameter Initialization:** Fixed deterministically for each seed across all noise settings via `init_point = np.random.default_rng(seed).random(ansatz.num_parameters)`.

### How to Rerun
1. **Quick Reproducibility Check:**
   ```bash
   python reproducibility_check.py
   ```
   * Verifies that two independent training runs of VQC-COBYLA-300 and VQC-SPSA-100 on Iris (Seed 42, Ideal) converge to identical weights.
   * Reruns make_moons VQC-COBYLA-300 High_Stress across all 5 seeds and checks against `results_revision/per_seed_results.csv`.

2. **Run Pytest Test Suite:**
   ```bash
   pytest tests/
   ```
   *Note: Tests validate dataset ingestion, preprocessing bounds, noise model definitions, circuit transpilation metrics (Table 6), and CSV integrity; they do not run the full quantum training experiments.*

3. **Regenerate Manuscript Plots & Summary Tables:**
   ```bash
   python plot_combined.py
   ```

4. **Full End-to-End Experiment Execution (Resumable):**
   ```bash
   python robustness_study.py
   ```

### Expected Output Files
* `results_revision/per_seed_results.csv`: Granular per-seed results (250 rows: 5 datasets × 5 seeds × (3 noise levels × 3 quantum models + 1 classical baseline)).
* `results_revision/Oo_Paper_Summary_Table.csv`: Summary Table reporting Mean ± Sample SD (ddof=1) for Accuracy, Precision, Recall, and F1.
* `results_revision/Oo_circuit_metrics.csv`: Gate-level circuit depth, 1Q gate count, and CNOT counts (Table 6).
* `results_revision/Oo_noise_vs_ideal_ttest.csv`: Paired t-tests between ideal and noisy settings.
* `results_revision/plots/fig_accuracy_multi.png`: Figure 2 in manuscript (Accuracy catplot with error bars).
* `results_revision/plots/fig_f1_multi.png`: Figure 3 in manuscript (F1 catplot with error bars).

### Tested with:
* `Python 3.14.6`
* `qiskit == 2.5.2`
* `qiskit-aer == 0.17.2`
* `qiskit-machine-learning == 0.9.1`
* `qiskit-algorithms == 0.4.0`
* `scikit-learn == 1.9.0`
* `pandas == 3.0.3`
* `numpy == 2.4.6`
* `scipy == 1.18.0`
* `matplotlib == 3.11.0`
* `seaborn == 0.13.2`

---

## 4. Repository Structure

```text
Noise-Resilience-Quantum-Classifiers/
│
├── README.md                      # Comprehensive reproducibility and replication guide
├── requirements.txt               # Compatible dependencies for Qiskit 2.5.2 environment
├── pytest.ini                     # Pytest configuration
├── robustness_study.py            # Main experimental pipeline (resumable, seeded)
├── reproducibility_check.py       # Reproducibility verification test script
├── plot_combined.py               # Generates Figure 2, Figure 3, and summary tables
│
├── dataset/                       # Benchmark dataset exports
│   ├── Binary_Iris.csv
│   ├── Breast_Cancer.csv
│   ├── Make_Circles.csv
│   ├── Make_Moons.csv
│   └── Wine.csv
│
├── tests/                         # Automated unit test suite
│   └── test_noise_study.py
│
└── results_revision/              # Experimental results and figures
    ├── per_seed_results.csv       # 250-row fully seeded dataset
    ├── Oo_Paper_Summary_Table.csv # Paper Summary Table (Mean ± SD)
    ├── Oo_circuit_metrics.csv     # Transpiled circuit metrics (Table 6)
    ├── Oo_noise_vs_ideal_ttest.csv# Paired t-test statistics
    └── plots/                     # Output figures (Figures 2 & 3)
        ├── fig_accuracy_multi.png
        └── fig_f1_multi.png
```

---

## 5. Experimental Methodology

### Datasets & Preprocessing (Algorithm 1)
Five binary classification datasets: Binary Iris, Make Moons, Make Circles, Breast Cancer, and Wine.
1. Stratified 70%/30% Train/Test partition (`random_state=seed`)
2. `StandardScaler` (zero mean, unit variance)
3. `PCA (n_components=2)`
4. `MinMaxScaler([0, π])`

### Quantum Models
* **ZZFeatureMap:** 2 qubits, linear entanglement, `reps=2`.
* **VQC:** `RealAmplitudes` ansatz (`reps=2`, 6 parameters), trained using `COBYLA(maxiter=300)` and `SPSA(maxiter=100)`.
* **QSVC:** Fidelity kernel matrix via `ComputeUncompute` with classical SVM solver.

### Noise Stress Levels
* **Ideal:** Depolarizing $\lambda_1 = 0.0$, $\lambda_2 = 0.0$ (`StatevectorSampler`, 1024 shots).
* **Low Stress:** $\lambda_1 = 0.01$, $\lambda_2 = 0.04$ ($p_1 = 0.0075, p_2 = 0.0375$).
* **High Stress:** $\lambda_1 = 0.02$, $\lambda_2 = 0.08$ ($p_1 = 0.0150, p_2 = 0.0750$).

---

## 6. License & Citation
If you find this work or codebase helpful in your research, please cite:
```bibtex
@article{singh2026comparative,
  title={A Comparative Analysis of Variational and Kernel Quantum Classifiers Under Depolarizing Gate Stress in IBM Qiskit 2.5.2},
  author={Singh, Rajnish and Patel, Utkrisht and Kumar, Rakesh},
  journal={International Journal of Computational Intelligence and Applications},
  year={2026}
}
```
