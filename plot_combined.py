import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# --- Load new seeded run ---
df = pd.read_csv(REPO_ROOT / 'results_revision' / 'per_seed_results.csv')

# --- Generate Paper Summary Table (mean +/- std with ddof=1) ---
summary = df.groupby(['Dataset', 'Noise', 'Model']).agg({
    'Accuracy': ['mean', 'std'],
    'Precision': ['mean', 'std'],
    'Recall': ['mean', 'std'],
    'F1': ['mean', 'std']
})
summary.columns = ['_'.join(col) for col in summary.columns.values]

formatted_table = pd.DataFrame()
formatted_table['Dataset'] = summary.index.get_level_values('Dataset')
formatted_table['Noise'] = summary.index.get_level_values('Noise')
formatted_table['Model'] = summary.index.get_level_values('Model')

for metric in ['Accuracy', 'Precision', 'Recall', 'F1']:
    formatted_table[metric] = summary.apply(
        lambda row: f"{row[f'{metric}_mean']:.3f} ± {row[f'{metric}_std']:.3f}"
        if pd.notnull(row[f'{metric}_std']) else f"{row[f'{metric}_mean']:.3f}",
        axis=1
    ).values

formatted_table.to_csv(REPO_ROOT / 'results_revision' / 'Oo_Paper_Summary_Table.csv', index=False)
print("Saved Oo_Paper_Summary_Table.csv")

# --- Explicit ordering ---
DATASET_ORDER = ['Binary_Iris', 'Make_Moons', 'Make_Circles', 'Breast_Cancer', 'Wine']
MODEL_ORDER   = ['Classical_SVM', 'VQC_COBYLA_300', 'VQC_SPSA_100', 'QSVC']
NOISE_ORDER   = ['Ideal', 'Low_Stress', 'High_Stress']

# Restrict to known categories so seaborn respects order
df['Dataset'] = pd.Categorical(df['Dataset'], categories=DATASET_ORDER, ordered=True)
df['Model']   = pd.Categorical(df['Model'],   categories=MODEL_ORDER,   ordered=True)
df['Noise']   = pd.Categorical(df['Noise'],   categories=NOISE_ORDER,   ordered=True)

os.makedirs(REPO_ROOT / 'results_revision' / 'plots', exist_ok=True)

# --- Verification printout ---
print("\n--- Verification: bar heights (mean) and error (sd, ddof=1) from raw CSV ---")
verify_cases = [
    ('Binary_Iris',  'Ideal',       'Accuracy'),
    ('Make_Circles', 'Ideal',       'Accuracy'),
    ('Wine',         'High_Stress', 'Accuracy'),
    ('Make_Moons',   'High_Stress', 'Accuracy'),
]
for ds, noise, metric in verify_cases:
    print(f"\n  {ds} | {noise} | {metric}")
    for model in MODEL_ORDER:
        sub = df[(df['Dataset'] == ds) & (df['Noise'] == noise) & (df['Model'] == model)]
        if sub.empty:
            continue
        m = sub[metric].mean()
        s = sub[metric].std(ddof=1)
        print(f"    {model:<20}: mean={m:.3f}  sd={s:.3f}")

# --- Generate Plots ---
sns.set_theme(style="whitegrid", font_scale=1.1)

for metric in ['Accuracy', 'F1']:
    # VERTICAL/PORTRAIT version — the one that goes in the paper
    g_tall = sns.catplot(
        data=df,
        kind="bar",
        x="Dataset",
        y=metric,
        hue="Model",
        row="Noise",
        hue_order=MODEL_ORDER,
        row_order=NOISE_ORDER,
        order=DATASET_ORDER,
        capsize=.1,
        errorbar="sd",
        height=3.5,
        aspect=1.8,
        palette="viridis",
    )
    g_tall.set_axis_labels("", f"{metric} Score")
    g_tall.set_titles("{row_name} Noise", size=14, pad=10)

    for ax in g_tall.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(25)
            label.set_horizontalalignment('right')

    sns.move_legend(g_tall, "center left", bbox_to_anchor=(1.02, 0.5))
    g_tall.fig.subplots_adjust(hspace=0.4)

    out_name = f"fig_accuracy_multi.png" if metric == "Accuracy" else "fig_f1_multi.png"
    g_tall.savefig(REPO_ROOT / 'results_revision' / 'plots' / out_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: results_revision/plots/{out_name}")

    # LANDSCAPE version (optional, not in paper)
    g_wide = sns.catplot(
        data=df,
        kind="bar",
        x="Dataset",
        y=metric,
        hue="Model",
        col="Noise",
        hue_order=MODEL_ORDER,
        col_order=NOISE_ORDER,
        order=DATASET_ORDER,
        capsize=.1,
        errorbar="sd",
        height=5,
        aspect=1.3,
        palette="viridis",
    )
    g_wide.set_axis_labels("", f"{metric} Score")
    g_wide.set_titles("{col_name} Noise", size=14, pad=10)
    for ax in g_wide.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment('right')
    sns.move_legend(g_wide, "center left", bbox_to_anchor=(1.02, 0.5))
    g_wide.fig.subplots_adjust(wspace=0.1)
    landscape_name = f"Oo_Combined_{metric}_Landscape.png"
    g_wide.savefig(REPO_ROOT / 'results_revision' / 'plots' / landscape_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: results_revision/plots/{landscape_name}  (landscape, optional)")

print("\nAll plots generated successfully.")
