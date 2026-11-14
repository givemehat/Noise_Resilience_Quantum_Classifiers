"""
Unit tests for QML Noise Resilience Study.
Validates:
1. Benchmark dataset loading, binary target partitioning, and stratification.
2. Data preprocessing pipeline (Train/Test split, PCA, [0, pi] range normalization).
3. Qiskit noise model synthesis (depolarizing single- and two-qubit error rates).
4. Quantum feature map & ansatz parameter consistency.
"""

import numpy as np
import pytest
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_aer.noise import NoiseModel

from robustness_study import get_datasets, preprocess_data, create_noise_model


class TestDatasetIngestion:
    """Validate benchmark dataset loading and format integrity."""

    def test_get_datasets_keys(self):
        datasets = get_datasets()
        expected = ["Binary_Iris", "Make_Moons", "Make_Circles", "Breast_Cancer", "Wine"]
        for key in expected:
            assert key in datasets
            X, y = datasets[key]
            assert isinstance(X, np.ndarray)
            assert isinstance(y, np.ndarray)
            assert len(X) == len(y)
            # Binary classification constraint: labels must be in {0, 1}
            assert set(np.unique(y)).issubset({0, 1})


class TestPreprocessingPipeline:
    """Validate Algorithm 1 preprocessing constraints."""

    def test_preprocessing_output_bounds(self):
        datasets = get_datasets()
        X, y = datasets["Make_Moons"]
        X_train, X_test, y_train, y_test = preprocess_data(X, y, random_state=42, n_features=2)

        # Output feature dimension
        assert X_train.shape[1] == 2
        assert X_test.shape[1] == 2

        # 70/30 train-test partition
        assert len(X_train) == 70
        assert len(X_test) == 30

        # Features must be scaled strictly within [0, pi] for quantum angle encoding
        assert np.all(X_train >= -1e-6)
        assert np.all(X_train <= np.pi + 1e-6)
        assert not np.isnan(X_train).any()
        assert not np.isnan(X_test).any()


class TestNoiseModelSynthesis:
    """Validate Qiskit Aer noise model generation."""

    def test_ideal_noise_model(self):
        model = create_noise_model(0.0, 0.0)
        assert model is None or len(model.to_dict().get("errors", [])) == 0

    def test_stress_noise_model(self):
        model = create_noise_model(p1=0.01, p2=0.04)
        assert isinstance(model, NoiseModel)
        assert not model.is_ideal()


class TestQuantumCircuitConstructs:
    """Validate quantum feature maps and ansatz parameters."""

    def test_zz_feature_map(self):
        n_qubits = 2
        feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2, entanglement="linear")
        assert feature_map.num_qubits == n_qubits
        assert feature_map.num_parameters == n_qubits

    def test_real_amplitudes_ansatz(self):
        n_qubits = 2
        ansatz = RealAmplitudes(num_qubits=n_qubits, reps=2, entanglement="linear")
        assert ansatz.num_qubits == n_qubits
        # RealAmplitudes with reps=2 and 2 qubits has (reps + 1) * n_qubits = 6 parameters
        assert ansatz.num_parameters == 6


class TestCircuitMetricsAndResults:
    """Validate circuit transpilation metrics and per-seed results integrity."""

    def test_circuit_metrics_table_6(self):
        from qiskit_aer import AerSimulator
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from robustness_study import extract_circuit_metrics

        pm = generate_preset_pass_manager(optimization_level=1, backend=AerSimulator())
        dummy_fm = ZZFeatureMap(2, reps=2)
        dummy_ans = RealAmplitudes(2, reps=2)
        vqc_circ = dummy_fm.compose(dummy_ans)
        kernel_circ = dummy_fm.compose(dummy_fm.inverse())

        vqc_depth, vqc_1q, vqc_cnot = extract_circuit_metrics(vqc_circ, pm)
        k_depth, k_1q, k_cnot = extract_circuit_metrics(kernel_circ, pm)

        # Table 6 values in manuscript
        assert (vqc_depth, vqc_1q, vqc_cnot) == (15, 16, 6)
        assert (k_depth, k_1q, k_cnot) == (18, 20, 6)

    def test_per_seed_results_integrity(self):
        import os
        import pandas as pd

        csv_path = "results_revision/per_seed_results.csv"
        if not os.path.exists(csv_path):
            pytest.skip("per_seed_results.csv not found")

        df = pd.read_csv(csv_path)
        assert len(df) == 250
        assert not df.isnull().values.any()

        # Check expected models and datasets
        assert set(df["Dataset"].unique()) == {"Binary_Iris", "Make_Moons", "Make_Circles", "Breast_Cancer", "Wine"}
        assert set(df["Noise"].unique()) == {"Ideal", "Low_Stress", "High_Stress"}
        assert set(df["Model"].unique()) == {"Classical_SVM", "VQC_COBYLA_300", "VQC_SPSA_100", "QSVC"}

        # Classical_SVM should only appear under Ideal (5 datasets x 5 seeds = 25 rows)
        assert len(df[df["Model"] == "Classical_SVM"]) == 25
        assert set(df[df["Model"] == "Classical_SVM"]["Noise"].unique()) == {"Ideal"}
