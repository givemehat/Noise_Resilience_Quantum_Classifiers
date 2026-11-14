import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons, make_circles, load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import SPSA, COBYLA
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_algorithms.utils import algorithm_globals as alg_globals
from qiskit_machine_learning.utils import algorithm_globals as ml_globals
from qiskit_machine_learning.algorithms import VQC, QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
from qiskit.primitives import StatevectorSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import warnings

warnings.filterwarnings('ignore')

SEEDS = [42, 100, 2023, 777, 1234]

NOISE_LEVELS = {
    'Ideal': (0.0, 0.0),
    'Low_Stress': (0.01, 0.04),
    'High_Stress': (0.02, 0.08)
}

def get_datasets():
    """
    Loads and prepares the 5 benchmark datasets used in the study.
    Applies filtering for binary classification and subsamples larger datasets to 100 instances.
    """
    datasets = {}
    
    # 1. Binary Iris
    iris = load_iris()
    X, y = iris.data, iris.target
    idx = y < 2
    datasets['Binary_Iris'] = (X[idx], y[idx])
    
    # 2. Make Moons
    X, y = make_moons(n_samples=100, noise=0.15, random_state=42)
    datasets['Make_Moons'] = (X, y)
    
    # 3. Make Circles
    X, y = make_circles(n_samples=100, noise=0.1, factor=0.5, random_state=42)
    datasets['Make_Circles'] = (X, y)
    
    # 4. Breast Cancer (downsampled for simulation constraints)
    bc = load_breast_cancer()
    X, y = bc.data, bc.target
    X, _, y, _ = train_test_split(X, y, train_size=100, stratify=y, random_state=42)
    datasets['Breast_Cancer'] = (X, y)
    
    # 5. Wine (Binary subset, downsampled)
    wine = load_wine()
    X, y = wine.data, wine.target
    idx = y < 2
    X_wine, y_wine = X[idx], y[idx]
    X_wine, _, y_wine, _ = train_test_split(X_wine, y_wine, train_size=100, stratify=y_wine, random_state=42)
    datasets['Wine'] = (X_wine, y_wine)
    
    return datasets

def preprocess_data(X, y, random_state, n_features=2):
    """
    Preprocesses the data exactly as described in Algorithm 1 of the paper.
    Steps: Train/Test Split (70/30) -> StandardScaler -> PCA -> MinMaxScaler (0 to pi).
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=random_state)
    
    std_scaler = StandardScaler()
    X_train_std = std_scaler.fit_transform(X_train)
    X_test_std = std_scaler.transform(X_test)
    
    pca = PCA(n_components=n_features)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)
    
    angle_scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_final = angle_scaler.fit_transform(X_train_pca)
    X_test_final = angle_scaler.transform(X_test_pca)
    
    return X_train_final, X_test_final, y_train, y_test

def create_noise_model(p1: float, p2: float, gate_1q=None, gate_2q=None) -> NoiseModel:
    """
    Synthesizes a hardware-calibrated Pauli depolarizing noise model.
    """
    if p1 == 0.0 and p2 == 0.0:
        return None
    if gate_1q is None:
        gate_1q = ['u', 'p', 'r', 'sx', 'x', 'rz']
    if gate_2q is None:
        gate_2q = ['cx', 'ecr', 'cz']
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p1, 1), gate_1q)
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p2, 2), gate_2q)
    return noise_model

def make_fresh_sampler(noise_name, seed, gate_1q=None, gate_2q=None):
    """Creates a distinct, non-shared sampler instance for each model."""
    if noise_name == 'Ideal':
        return StatevectorSampler(default_shots=1024, seed=seed)
    else:
        p1, p2 = NOISE_LEVELS[noise_name]
        nm = create_noise_model(p1, p2, gate_1q, gate_2q)
        return AerSamplerV2(
            default_shots=1024,
            options={'backend_options': {'noise_model': nm, 'seed_simulator': seed}}
        )

def run_experiments(output_csv='results_revision/per_seed_results.csv'):
    """
    Executes the reproducible experimental pipeline across all datasets, models, seeds, and noise levels.
    Resumable: skips already-computed (Dataset, Seed, Noise, Model) tuples.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Load existing progress if available
    if os.path.exists(output_csv):
        df_existing = pd.read_csv(output_csv)
        completed_keys = set(zip(
            df_existing['Dataset'],
            df_existing['Seed'],
            df_existing['Noise'],
            df_existing['Model']
        ))
        print(f"Resuming: found {len(completed_keys)} already completed runs in {output_csv}.")
    else:
        completed_keys = set()
        # Initialize file with header
        pd.DataFrame(columns=['Dataset', 'Seed', 'Noise', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1']).to_csv(output_csv, index=False)

    datasets = get_datasets()
    pm = generate_preset_pass_manager(optimization_level=1, backend=AerSimulator())
    
    # Determine gate lists dynamically from transpiled circuits
    dummy_feature_map = ZZFeatureMap(2, reps=2)
    dummy_ansatz = RealAmplitudes(2, reps=2)
    t_vqc = pm.run(dummy_feature_map.compose(dummy_ansatz))
    t_kernel = pm.run(dummy_feature_map.compose(dummy_feature_map.inverse()))
    
    all_ops = set(t_vqc.count_ops().keys()).union(set(t_kernel.count_ops().keys()))
    all_ops.discard('barrier')
    all_ops.discard('measure')
    gate_1q = [g for g in all_ops if g != 'cx']
    gate_2q = ['cx']
    
    for ds_name, (X, y) in datasets.items():
        for seed in SEEDS:
            X_train, X_test, y_train, y_test = preprocess_data(X, y, random_state=seed)
            n_qubits = X_train.shape[1]
            
            feature_map = ZZFeatureMap(n_qubits, reps=2)
            ansatz = RealAmplitudes(n_qubits, reps=2)
            
            # Same deterministic starting point for this seed across all noise settings
            init_point = np.random.default_rng(seed).random(ansatz.num_parameters)
            
            # Classical RBF-SVM Baseline (Ideal only)
            key_svm = (ds_name, seed, 'Ideal', 'Classical_SVM')
            if key_svm not in completed_keys:
                svm = SVC(kernel='rbf', random_state=seed)
                svm.fit(X_train, y_train)
                y_pred = svm.predict(X_test)
                row_svm = {
                    'Dataset': ds_name, 'Seed': seed, 'Noise': 'Ideal', 'Model': 'Classical_SVM',
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred, zero_division=0),
                    'Recall': recall_score(y_test, y_pred, zero_division=0),
                    'F1': f1_score(y_test, y_pred, zero_division=0)
                }
                pd.DataFrame([row_svm]).to_csv(output_csv, mode='a', header=False, index=False)
                completed_keys.add(key_svm)
            
            for noise_name in ['Ideal', 'Low_Stress', 'High_Stress']:
                # 1. QSVC
                key_qsvc = (ds_name, seed, noise_name, 'QSVC')
                if key_qsvc not in completed_keys:
                    np.random.seed(seed)
                    alg_globals.random_seed = seed
                    ml_globals.random_seed = seed
                    
                    sampler_qsvc = make_fresh_sampler(noise_name, seed, gate_1q, gate_2q)
                    fidelity = ComputeUncompute(sampler=sampler_qsvc, transpiler=pm)
                    qsvc = QSVC(quantum_kernel=FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity))
                    qsvc.fit(X_train, y_train)
                    y_pred_qsvc = qsvc.predict(X_test)
                    
                    row_qsvc = {
                        'Dataset': ds_name, 'Seed': seed, 'Noise': noise_name, 'Model': 'QSVC',
                        'Accuracy': accuracy_score(y_test, y_pred_qsvc),
                        'Precision': precision_score(y_test, y_pred_qsvc, zero_division=0),
                        'Recall': recall_score(y_test, y_pred_qsvc, zero_division=0),
                        'F1': f1_score(y_test, y_pred_qsvc, zero_division=0)
                    }
                    pd.DataFrame([row_qsvc]).to_csv(output_csv, mode='a', header=False, index=False)
                    completed_keys.add(key_qsvc)
                    print(f"[{ds_name} | Seed {seed} | {noise_name}] QSVC done: Acc={row_qsvc['Accuracy']:.3f}")
                
                # 2. VQC Models (COBYLA & SPSA)
                for opt_name, make_opt in [
                    ('VQC_COBYLA_300', lambda: COBYLA(maxiter=300)),
                    ('VQC_SPSA_100', lambda: SPSA(maxiter=100))
                ]:
                    key_vqc = (ds_name, seed, noise_name, opt_name)
                    if key_vqc not in completed_keys:
                        np.random.seed(seed)
                        alg_globals.random_seed = seed
                        ml_globals.random_seed = seed
                        
                        sampler_vqc = make_fresh_sampler(noise_name, seed, gate_1q, gate_2q)
                        vqc = VQC(
                            feature_map=feature_map,
                            ansatz=ansatz,
                            optimizer=make_opt(),
                            sampler=sampler_vqc,
                            pass_manager=pm,
                            initial_point=init_point
                        )
                        vqc.fit(X_train, np.eye(2)[y_train])
                        y_pred_vqc = np.argmax(vqc.predict(X_test), axis=1)
                        
                        row_vqc = {
                            'Dataset': ds_name, 'Seed': seed, 'Noise': noise_name, 'Model': opt_name,
                            'Accuracy': accuracy_score(y_test, y_pred_vqc),
                            'Precision': precision_score(y_test, y_pred_vqc, zero_division=0),
                            'Recall': recall_score(y_test, y_pred_vqc, zero_division=0),
                            'F1': f1_score(y_test, y_pred_vqc, zero_division=0)
                        }
                        pd.DataFrame([row_vqc]).to_csv(output_csv, mode='a', header=False, index=False)
                        completed_keys.add(key_vqc)
                        print(f"[{ds_name} | Seed {seed} | {noise_name}] {opt_name} done: Acc={row_vqc['Accuracy']:.3f}")

    print(f"\nAll experiments complete. Results saved in {output_csv}.")
    save_circuit_metrics()

def extract_circuit_metrics(circuit, pm):
    """Calculates circuit depth, 1Q gates, and CNOT count after transpilation."""
    transpiled = pm.run(circuit)
    ops = transpiled.count_ops()
    cnot_count = ops.get('cx', 0)
    gates_1q = sum(c for g, c in ops.items() if g not in ['cx', 'barrier', 'measure'])
    return transpiled.depth(), gates_1q, cnot_count

def save_circuit_metrics(output_csv='results_revision/Oo_circuit_metrics.csv'):
    """Extracts gate-level metrics for VQC and QSVC kernel circuits and saves to CSV."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    pm = generate_preset_pass_manager(optimization_level=1, backend=AerSimulator())
    dummy_fm = ZZFeatureMap(2, reps=2)
    dummy_ans = RealAmplitudes(2, reps=2)
    vqc_circ = dummy_fm.compose(dummy_ans)
    kernel_circ = dummy_fm.compose(dummy_fm.inverse())

    vqc_depth, vqc_1q, vqc_cnot = extract_circuit_metrics(vqc_circ, pm)
    k_depth, k_1q, k_cnot = extract_circuit_metrics(kernel_circ, pm)

    datasets = get_datasets()
    rows = []
    for ds_name in datasets.keys():
        rows.append({
            'Dataset': ds_name,
            'Model': 'VQC_Circuit',
            'Depth': vqc_depth,
            '1Q_Gates': vqc_1q,
            'CNOTs': vqc_cnot
        })
        rows.append({
            'Dataset': ds_name,
            'Model': 'ComputeUncompute_Kernel',
            'Depth': k_depth,
            '1Q_Gates': k_1q,
            'CNOTs': k_cnot
        })
    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(output_csv, index=False)
    print(f"Circuit metrics saved to {output_csv}.")
    return df_metrics

if __name__ == '__main__':
    run_experiments()

