import pandas as pd
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUMMARIES_DIR = PROJECT_ROOT / "results" / "summaries"

def generate_final_summaries():
    # 1. Final Results CSV
    perf_path = SUMMARIES_DIR / "performance_summary.csv"
    if perf_path.exists():
        df_perf = pd.read_csv(perf_path)
        # Assuming df_perf has rows for each scheduler
        # Let's write it to final_results.csv as requested
        df_perf.to_csv(PROJECT_ROOT / "results" / "final_results.csv", index=False)
        print("Generated final_results.csv")
    
    # 2. Final Experiment Summary CSV
    # Aggregate from generalization, ablation, weights, training size
    gen_path = SUMMARIES_DIR / "generalization_results.csv"
    if gen_path.exists():
        df_gen = pd.read_csv(gen_path).iloc[0]
        
        # We construct final_experiment_summary.csv
        exp_summary = [{
            "experiment_name": "Generalization (Unseen Seed 2026)",
            "workload_count": df_gen['num_workloads'],
            "model_name": "RandomForest_Baseline",
            "classification_accuracy": df_gen.get('ml_accuracy', "NA"),
            "weighted_precision": df_gen.get('ml_weighted_precision', "NA"),
            "weighted_recall": df_gen.get('ml_weighted_recall', "NA"),
            "weighted_f1": df_gen.get('ml_weighted_f1', "NA"),
            "average_regret": df_gen.get('ml_average_regret', "NA"),
            "median_regret": df_gen.get('ml_median_regret', "NA"),
            "worst_regret": df_gen.get('ml_worst_regret', "NA"),
            "zero_regret_percent": df_gen.get('ml_zero_regret_percent', "NA"),
            "within_5_percent": df_gen.get('ml_within_5_percent', "NA"),
            "within_10_percent": df_gen.get('ml_within_10_percent', "NA"),
            "rule_based_average_regret": df_gen.get('rule_based_average_regret', "NA")
        }]
        
        pd.DataFrame(exp_summary).to_csv(PROJECT_ROOT / "results" / "final_experiment_summary.csv", index=False)
        print("Generated final_experiment_summary.csv")

if __name__ == "__main__":
    generate_final_summaries()
