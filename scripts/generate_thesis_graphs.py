import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set premium aesthetic
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Roboto', 'Arial', 'DejaVu Sans']

def save_plot(name):
    plt.tight_layout()
    plt.savefig(f"{name}.png", dpi=300, bbox_inches='tight')
    print(f"Saved {name}.png")

# --- OMR RESULTS DATA ---
omr_data = {
    'Run': ['Run 0', 'Run 69', 'Run 71', 'Run 77', 'Run 78'],
    'Accuracy': [94.50, 96.68, 97.71, 98.14, 98.37],
    'Model': ['Prototype (TF)', 'DiamondCNN (v1)', 'DiamondCNN (v2)', 'DiamondCNN (Final)', 'AscendingCNN (SOTA)']
}
df_omr = pd.DataFrame(omr_data)

# --- NLP RESULTS DATA ---
# Using Audited QWK for NLP as they represent the calibrated performance
nlp_data = {
    'Run': ['Run 1', 'Run 2', 'Run 4', 'Run 6', 'Run 7', 'Run 8', 'Run 9'],
    'QWK': [0.5205, 0.4481, 0.8025, 0.4075, 0.8174, 0.8484, 0.8512],
    'Model': ['XGBoost (Linguistic)', 'BERT-Base', 'BERT (Patched)', 'DistilBERT', 'DeBERTa-v3', 'Hybrid (SOTA)', 'Grand Ensemble']
}
df_nlp = pd.DataFrame(nlp_data)

# --- PLOT 1: OMR EVOLUTION ---
plt.figure(figsize=(10, 6))
palette = sns.color_palette("viridis", as_cmap=False, n_colors=len(df_omr))
ax = sns.lineplot(data=df_omr, x='Run', y='Accuracy', marker='o', markersize=10, linewidth=3, color='#2ecc71')

# Add labels to points
for i, txt in enumerate(df_omr['Accuracy']):
    ax.annotate(f"{txt}%", (df_omr['Run'][i], df_omr['Accuracy'][i]), 
                textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')

plt.title('SwiftGrade OMR Classification Accuracy Evolution', fontsize=16, pad=20, fontweight='bold')
plt.xlabel('Training Iteration', fontsize=12)
plt.ylabel('Validation Accuracy (%)', fontsize=12)
plt.ylim(94, 99)
save_plot("OMR_Accuracy_Evolution")

# --- PLOT 2: NLP EVOLUTION ---
plt.figure(figsize=(10, 6))
ax = sns.lineplot(data=df_nlp, x='Run', y='QWK', marker='s', markersize=10, linewidth=3, color='#3498db')

# Highlight the SOTA
sota_idx = df_nlp[df_nlp['Run'] == 'Run 9'].index[0]
plt.scatter(df_nlp['Run'][sota_idx], df_nlp['QWK'][sota_idx], color='red', s=200, zorder=5, label='SOTA Champion')

# Add labels to points
for i, txt in enumerate(df_nlp['QWK']):
    ax.annotate(f"{txt:.4f}", (df_nlp['Run'][i], df_nlp['QWK'][i]), 
                textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')

plt.title('SwiftGrade NLP Scoring Performance (QWK)', fontsize=16, pad=20, fontweight='bold')
plt.xlabel('Training Iteration', fontsize=12)
plt.ylabel('Quadratic Weighted Kappa (QWK)', fontsize=12)
plt.ylim(0.35, 0.95)
save_plot("NLP_QWK_Evolution")

# --- PLOT 3: COMBINED COMPARISON (BAR CHART) ---
# For a thesis, sometimes a bar chart comparing baseline vs SOTA is more punchy
comparison_data = {
    'Module': ['OMR Accuracy', 'OMR Accuracy', 'NLP QWK', 'NLP QWK'],
    'State': ['Baseline', 'Production (SOTA)', 'Baseline', 'Production (SOTA)'],
    'Score': [94.50, 98.37, 0.5205, 0.8512]
}
df_comp = pd.DataFrame(comparison_data)

# Split into two for different scales
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# OMR Bar
sns.barplot(data=df_comp[df_comp['Module'] == 'OMR Accuracy'], x='State', y='Score', ax=ax1, palette='Greens_d')
ax1.set_title('OMR Accuracy Improvement', fontsize=14, fontweight='bold')
ax1.set_ylim(90, 100)
ax1.set_ylabel('Accuracy (%)')

# NLP Bar
sns.barplot(data=df_comp[df_comp['Module'] == 'NLP QWK'], x='State', y='Score', ax=ax2, palette='Blues_d')
ax2.set_title('NLP QWK Improvement', fontsize=14, fontweight='bold')
ax2.set_ylim(0.4, 1.0)
ax2.set_ylabel('Quadratic Weighted Kappa')

plt.suptitle('SwiftGrade System Performance: Baseline vs. Production', fontsize=18, fontweight='bold', y=1.02)
save_plot("Performance_Comparison_Summary")
