import sys
from pathlib import Path

# Add project root to path so we can import ml module
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ml.config import DATASET_PATH, SCALER_PATH, LABEL_MAPPING_PATH, FEATURE_COLUMNS, RANDOM_STATE
from ml.preprocessing import preprocess_dataset

def print_distribution(y_series, name):
    counts = y_series.value_counts()
    percentages = y_series.value_counts(normalize=True) * 100
    print(f"\n--- {name} Target Distribution ---")
    for cls in counts.index:
        print(f"{cls:10}: {counts[cls]:4} workloads ({percentages[cls]:5.1f}%)")

def main():
    print("==================================================")
    print("STEP 10 — ML PREPROCESSING + FEATURE ENGINEERING")
    print("==================================================\n")
    
    print(f"Loading dataset from: {DATASET_PATH}")
    try:
        results = preprocess_dataset(DATASET_PATH, SCALER_PATH, LABEL_MAPPING_PATH)
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return

    print("Dataset validated successfully.\n")
    
    df_raw = results["df_raw"]
    print(f"Full Dataset dimensions: {df_raw.shape[0]} workloads, {df_raw.shape[1]} columns")
    
    print("\nExact 10 Feature Columns:")
    for i, col in enumerate(FEATURE_COLUMNS, 1):
        print(f"  {i}. {col}")
        
    print(f"\nData Split (random_state={RANDOM_STATE}):")
    print(f"  Training: {results['X_train'].shape[0]} workloads")
    print(f"  Testing:  {results['X_test'].shape[0]} workloads")
    
    print_distribution(results['df_raw']['target_scheduler'], "Full Dataset")
    print_distribution(results['y_train_raw'], "Training Dataset")
    print_distribution(results['y_test_raw'], "Testing Dataset")
    
    print("\nTarget Label Encoding Mapping:")
    import json
    with open(LABEL_MAPPING_PATH, 'r') as f:
        mapping = json.load(f)
    for k, v in mapping.items():
        print(f"  {k:10} -> {v}")
        
    print("\nScaling:")
    print("  StandardScaler fitted ONLY on training data.")
    print("  Training and testing data transformed.")
    
    print("\nSaved Artifacts:")
    print(f"  {SCALER_PATH}")
    print(f"  {LABEL_MAPPING_PATH}")
    
    print("\nSample Processed Training Data (First 3 rows, scaled features):")
    print(results['X_train'].head(3))
    
    print("\nSample Encoded Training Labels:")
    print(results['y_train'].head(3).to_string())
    
    print("\n==================================================")
    print("SUCCESS: Preprocessing completed successfully.")
    print("NOTE: ML model training (Step 11) is NOT implemented yet.")
    print("==================================================")

if __name__ == "__main__":
    main()
