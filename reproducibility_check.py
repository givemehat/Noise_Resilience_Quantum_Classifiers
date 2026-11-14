#!/usr/bin/env python3
"""
Reproducibility Verification Script for QML Noise Resilience Study.

Checks:
1. Deterministic weight convergence on Binary_Iris (Seed 42, Ideal) across two
   independent training runs for both VQC_COBYLA_300 and VQC_SPSA_100.
2. Exact numerical metric reproduction of Make_Moons VQC_COBYLA_300 at High_Stress
   across all 5 random seeds against results_revision/per_seed_results.csv.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_algorithms.utils import algorithm_globals as alg_globals
from qiskit_machine_learning.utils import algorithm_globals as ml_globals
from qiskit_machine_learning.algorithms import VQC
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

import robustness_study as rs

def check_weight_determinism():
    print("=" * 70)
    print("CHECK 1: VQC Weight Determinism (Binary_Iris, Seed 42, Ideal)")
    print("=" * 70)
    
    datasets = rs.get_datasets()
    X, y = datasets['Binary_Iris']
    X_train, X_test, y_train, y_test = rs.preprocess_data(X, y, random_state=42)
    
    pm = generate_preset_pass_manager(optimization_level=1, backend=AerSimulator())
    dummy_fm = ZZFeatureMap(2, reps=2)
    dummy_ans = RealAmplitudes(2, reps=2)
    t_vqc = pm.run(dummy_fm.compose(dummy_ans))
    t_k = pm.run(dummy_fm.compose(dummy_fm.inverse()))
    all_ops = set(t_vqc.count_ops().keys()).union(set(t_k.count_ops().keys()))
    all_ops.discard('barrier')
    all_ops.discard('measure')
    gate_1q = [g for g in all_ops if g != 'cx']
    gate_2q = ['cx']
    
    def train_run(opt_name, maxiter):
        np.random.seed(42)
        alg_globals.random_seed = 42
        ml_globals.random_seed = 42
        
        fm = ZZFeatureMap(2, reps=2)
        ans = RealAmplitudes(2, reps=2)
        init_point = np.random.default_rng(42).random(ans.num_parameters)
        sampler = rs.make_fresh_sampler('Ideal', 42, gate_1q, gate_2q)
        
        if opt_name == 'COBYLA':
            opt = COBYLA(maxiter=maxiter)
        else:
            opt = SPSA(maxiter=maxiter)
            
        vqc = VQC(
            feature_map=fm,
            ansatz=ans,
            optimizer=opt,
            sampler=sampler,
            pass_manager=pm,
            initial_point=init_point
        )
        vqc.fit(X_train, np.eye(2)[y_train])
        return np.array(vqc.weights)

    # 1. COBYLA 300
    print("Training VQC_COBYLA_300 (Run 1)...")
    w_cobyla_1 = train_run('COBYLA', 300)
    print("Training VQC_COBYLA_300 (Run 2)...")
    w_cobyla_2 = train_run('COBYLA', 300)
    cobyla_identical = np.array_equal(w_cobyla_1, w_cobyla_2)
    print(f"VQC_COBYLA_300 Run 1 vs Run 2 identical: {cobyla_identical}")
    assert cobyla_identical, "VQC_COBYLA_300 weights are not identical across runs!"

    # 2. SPSA 100
    print("Training VQC_SPSA_100 (Run 1)...")
    w_spsa_1 = train_run('SPSA', 100)
    print("Training VQC_SPSA_100 (Run 2)...")
    w_spsa_2 = train_run('SPSA', 100)
    spsa_identical = np.array_equal(w_spsa_1, w_spsa_2)
    print(f"VQC_SPSA_100 Run 1 vs Run 2 identical: {spsa_identical}")
    assert spsa_identical, "VQC_SPSA_100 weights are not identical across runs!"

    print("\n--> CHECK 1 PASSED: Perfect deterministic convergence achieved.")
    return True

REPO_ROOT = Path(__file__).resolve().parent

def check_make_moons_exception(csv_path=None):
    if csv_path is None:
        csv_path = REPO_ROOT / 'results_revision' / 'per_seed_results.csv'
    print("\n" + "=" * 70)
    print("CHECK 2: Reproduction of Make_Moons VQC_COBYLA_300 (High_Stress, 5 Seeds)")
    print("=" * 70)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Reference file not found: {csv_path}")

    df_ref = pd.read_csv(csv_path)
    datasets = rs.get_datasets()
    X, y = datasets['Make_Moons']
    
    pm = generate_preset_pass_manager(optimization_level=1, backend=AerSimulator())
    dummy_fm = ZZFeatureMap(2, reps=2)
    dummy_ans = RealAmplitudes(2, reps=2)
    t_vqc = pm.run(dummy_fm.compose(dummy_ans))
    t_k = pm.run(dummy_fm.compose(dummy_fm.inverse()))
    all_ops = set(t_vqc.count_ops().keys()).union(set(t_k.count_ops().keys()))
    all_ops.discard('barrier')
    all_ops.discard('measure')
    gate_1q = [g for g in all_ops if g != 'cx']
    gate_2q = ['cx']

    seeds = [42, 100, 2023, 777, 1234]
    records = []
    all_match = True

    for seed in seeds:
        X_train, X_test, y_train, y_test = rs.preprocess_data(X, y, random_state=seed)
        
        np.random.seed(seed)
        alg_globals.random_seed = seed
        ml_globals.random_seed = seed
        
        fm = ZZFeatureMap(2, reps=2)
        ans = RealAmplitudes(2, reps=2)
        init_point = np.random.default_rng(seed).random(ans.num_parameters)
        sampler = rs.make_fresh_sampler('High_Stress', seed, gate_1q, gate_2q)
        
        vqc = VQC(
            feature_map=fm,
            ansatz=ans,
            optimizer=COBYLA(maxiter=300),
            sampler=sampler,
            pass_manager=pm,
            initial_point=init_point
        )
        vqc.fit(X_train, np.eye(2)[y_train])
        preds = np.argmax(vqc.predict(X_test), axis=1)
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        
        ref_row = df_ref[
            (df_ref['Dataset'] == 'Make_Moons') &
            (df_ref['Seed'] == seed) &
            (df_ref['Noise'] == 'High_Stress') &
            (df_ref['Model'] == 'VQC_COBYLA_300')
        ].iloc[0]
        
        match = (
            abs(acc - ref_row['Accuracy']) < 1e-9 and
            abs(prec - ref_row['Precision']) < 1e-9 and
            abs(rec - ref_row['Recall']) < 1e-9 and
            abs(f1 - ref_row['F1']) < 1e-9
        )
        if not match:
            all_match = False
            
        records.append({
            'Seed': seed,
            'Old_Acc': ref_row['Accuracy'],
            'New_Acc': acc,
            'Old_Prec': ref_row['Precision'],
            'New_Prec': prec,
            'Old_Rec': ref_row['Recall'],
            'New_Rec': rec,
            'Old_F1': ref_row['F1'],
            'New_F1': f1,
            'Match': match
        })
        print(f"Seed {seed:4d} | Acc: {acc:.4f} (exp: {ref_row['Accuracy']:.4f}) | Match: {match}")

    df_comp = pd.DataFrame(records)
    print("\nComparison Table:")
    print(df_comp.to_string(index=False))
    print(f"\nAll 5 seeds exact match: {all_match}")
    assert all_match, "Mismatch found in Make_Moons High_Stress reproduction!"
    print("--> CHECK 2 PASSED: 100% exact reproduction across all 5 seeds.")
    return True

if __name__ == '__main__':
    check_weight_determinism()
    check_make_moons_exception()
