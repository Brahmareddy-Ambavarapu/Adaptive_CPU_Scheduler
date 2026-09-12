import sys
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ml.config import DATASET_PATH, SCALER_PATH, LABEL_MAPPING_PATH, RANDOM_STATE, TEST_SIZE, FEATURE_COLUMNS, LABEL_MAPPING
from ml.preprocessing import preprocess_dataset
from ml.model_utils import (
    get_logistic_regression, get_decision_tree, get_random_forest,
    save_model, save_metadata
)
from ml.evaluate_models import (
    evaluate_model, generate_confusion_matrix, save_classification_report,
    run_cross_validation, extract_feature_importance, create_model_comparison_plot,
    generate_predictions_csv, generate_best_model_reports, report_class_distribution,
    RESULTS_DIR
)
from sklearn.metrics import accuracy_score, f1_score

def main():
    print("==================================================")
    print("STEP 11 — ML TRAINING & EVALUATION PIPELINE")
    print("==================================================\n")
    
    # 1. Reuse Step 10 Preprocessing
    print(f"Loading and preprocessing dataset from: {DATASET_PATH}...")
    try:
        prep_results = preprocess_dataset(DATASET_PATH, SCALER_PATH, LABEL_MAPPING_PATH)
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return
        
    X_train = prep_results["X_train"]
    X_test = prep_results["X_test"]
    y_train = prep_results["y_train"]
    y_test = prep_results["y_test"]
    df_raw = prep_results["df_raw"]
    y_train_raw = prep_results["y_train_raw"]
    y_test_raw = prep_results["y_test_raw"]
    
    # 2. Report Class Distribution
    report_class_distribution(df_raw["target_scheduler"], y_train_raw, y_test_raw)
    
    # 3. Initialize Models
    models = {
        "Logistic Regression": get_logistic_regression(),
        "Decision Tree": get_decision_tree(),
        "Random Forest": get_random_forest()
    }
    
    # 4. Cross Validation on Training Data
    print("Running 5-fold Stratified CV on training data...")
    cv_results = run_cross_validation(models, X_train, y_train)
    print(cv_results)
    
    # 5. Train & Evaluate Models on Test Data
    print("\nTraining and evaluating models on test set...")
    evaluation_results = []
    train_test_perf = []
    
    for name, model in models.items():
        # Train on FULL training set
        model.fit(X_train, y_train)
        
        # Predict on Test set
        y_pred = model.predict(X_test)
        
        # Predict on Train set (for overfitting check)
        y_train_pred = model.predict(X_train)
        
        # Metrics
        res = evaluate_model(name, y_test, y_pred)
        evaluation_results.append(res)
        
        # Train vs Test
        train_test_perf.append({
            "model": name,
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "test_accuracy": res["accuracy"],
            "train_weighted_f1": f1_score(y_train, y_train_pred, average="weighted", zero_division=0),
            "test_weighted_f1": res["weighted_f1"]
        })
        
        # Artifacts
        generate_confusion_matrix(name, y_test, y_pred)
        save_classification_report(name, y_test, y_pred)
        
        if name in ["Random Forest", "Logistic Regression"]:
            extract_feature_importance(model, name)
            
        # Save model
        save_model(model, f"{name.lower().replace(' ', '_')}.joblib")
        print(f"  {name} evaluated.")
        
    df_results = pd.DataFrame(evaluation_results)
    df_results.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    
    df_train_test = pd.DataFrame(train_test_perf)
    df_train_test.to_csv(RESULTS_DIR / "train_test_performance.csv", index=False)
    
    # 6. Generate Predictions CSV
    generate_predictions_csv(models, X_test, y_test)
    
    # 7. Select Best Model
    # Sort by weighted_f1 desc, then accuracy desc
    df_sorted = df_results.sort_values(by=["weighted_f1", "accuracy"], ascending=[False, False])
    best_name = df_sorted.iloc[0]["model"]
    best_model = models[best_name]
    best_f1 = df_sorted.iloc[0]["weighted_f1"]
    best_acc = df_sorted.iloc[0]["accuracy"]
    
    print(f"\nBest Model Selected: {best_name}")
    print(f"  Weighted F1: {best_f1:.4f}")
    print(f"  Accuracy:    {best_acc:.4f}")
    
    # Save Best Model
    save_model(best_model, "best_model.joblib")
    
    # Best model predictions & errors
    # We need the raw original features from X_test index for the error analysis
    # Unfortunately prep_results["X_test"] is scaled, we can pull from df_raw
    X_test_raw = df_raw.loc[X_test.index, FEATURE_COLUMNS]
    generate_best_model_reports(best_model, best_name, X_test, y_test, X_test_raw)
    
    # Save Metadata
    metadata = {
        "selected_model": best_name,
        "random_state": RANDOM_STATE,
        "feature_names": FEATURE_COLUMNS,
        "label_mapping": LABEL_MAPPING,
        "test_size": TEST_SIZE,
        "selection_criterion": "weighted_f1 -> accuracy",
        "training_size": len(X_train),
        "testing_size": len(X_test),
        "model_parameters": best_model.get_params(),
        "all_model_metrics": df_results.to_dict(orient="records")
    }
    save_metadata(metadata)
    
    # 8. Visualizations
    create_model_comparison_plot(df_results, "accuracy", "Model Accuracy Comparison", "model_accuracy_comparison.png")
    create_model_comparison_plot(df_results, "weighted_f1", "Model Weighted F1 Comparison", "model_f1_comparison.png")

    print("\n==================================================")
    print("SUCCESS: Step 11 Pipeline completed successfully.")
    print("==================================================")

if __name__ == "__main__":
    main()
