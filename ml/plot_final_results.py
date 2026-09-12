import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUMMARIES_DIR = PROJECT_ROOT / "results" / "summaries"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "final"

def generate_plots():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Plot Training Size Sensitivity
    train_size_csv = SUMMARIES_DIR / "training_size_sensitivity.csv"
    if train_size_csv.exists():
        df_size = pd.read_csv(train_size_csv)
        df_size = df_size.sort_values('training_size')
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()
        
        ax1.plot(df_size['training_size'], df_size['accuracy'] * 100, 'b-o', label='Classification Accuracy (%)')
        ax2.plot(df_size['training_size'], df_size['average_regret'], 'r-s', label='Average Regret (%)')
        
        ax1.set_xlabel('Training Dataset Size (Workloads)')
        ax1.set_ylabel('Accuracy (%)', color='b')
        ax2.set_ylabel('Average Regret (%)', color='r')
        
        plt.title('Impact of Training Data Size on Performance and Regret')
        fig.tight_layout()
        plt.savefig(FIGURES_DIR / "training_size_sensitivity.png")
        plt.close()

    # 2. Plot Feature Ablation
    ablation_csv = SUMMARIES_DIR / "feature_ablation.csv"
    if ablation_csv.exists():
        df_ablation = pd.read_csv(ablation_csv)
        df_ablation = df_ablation.sort_values('average_regret', ascending=False)
        
        plt.figure(figsize=(12, 6))
        plt.bar(df_ablation['removed_feature'], df_ablation['average_regret'], color='salmon')
        plt.axhline(y=df_ablation[df_ablation['removed_feature'] == 'none']['average_regret'].values[0], color='k', linestyle='--', label='Baseline (All Features)')
        
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Average Regret (%)')
        plt.title('Feature Ablation: Impact on Regret when Removing a Single Feature')
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "feature_ablation.png")
        plt.close()

    # 3. Plot Objective Weight Sensitivity
    weights_csv = SUMMARIES_DIR / "objective_weight_sensitivity.csv"
    if weights_csv.exists():
        df_weights = pd.read_csv(weights_csv)
        
        plt.figure(figsize=(10, 6))
        x = np.arange(len(df_weights['configuration']))
        plt.bar(x, df_weights['average_regret'], color='teal')
        
        plt.xticks(x, df_weights['configuration'], rotation=15)
        plt.ylabel('Average Regret (%)')
        plt.title('Sensitivity of ML Scheduler to Shifting Objective Weights')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "objective_weight_sensitivity.png")
        plt.close()

    # 4. Compare Schedulers (from generalization_results.csv or performance_summary.csv)
    perf_csv = SUMMARIES_DIR / "generalization_results.csv"
    if perf_csv.exists():
        df_gen = pd.read_csv(perf_csv).iloc[0]
        
        labels = ['Rule-Based', 'ML-Based']
        regrets = [df_gen['rule_based_average_regret'], df_gen['ml_average_regret']]
        
        plt.figure(figsize=(8, 6))
        plt.bar(labels, regrets, color=['gray', 'blue'])
        plt.ylabel('Average Regret (%)')
        plt.title('Generalization (Unseen Data): Rule-Based vs ML-Based Regret')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "generalization_comparison.png")
        plt.close()
        
    print(f"Final plots successfully saved to {FIGURES_DIR}")

if __name__ == "__main__":
    generate_plots()
