import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

from .config import RANDOM_STATE, LABEL_MAPPING, REVERSE_LABEL_MAPPING, PROJECT_ROOT, FEATURE_COLUMNS

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Order classes consistently
CLASS_ORDER = ["FCFS", "SJF", "RR", "Priority"]
CLASS_INDICES = [LABEL_MAPPING[c] for c in CLASS_ORDER]

def evaluate_model(model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculates evaluation metrics for a single model."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    return {
        "model": model_name,
        "accuracy": accuracy,
        "weighted_precision": precision,
        "weighted_recall": recall,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1
    }

def generate_confusion_matrix(model_name: str, y_true: np.ndarray, y_pred: np.ndarray):
    """Generates and saves a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_INDICES)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.colorbar()
    
    tick_marks = np.arange(len(CLASS_ORDER))
    plt.xticks(tick_marks, CLASS_ORDER, rotation=45)
    plt.yticks(tick_marks, CLASS_ORDER)
    
    plt.ylabel('Actual Scheduler')
    plt.xlabel('Predicted Scheduler')
    
    for i in range(len(CLASS_ORDER)):
        for j in range(len(CLASS_ORDER)):
            plt.text(j, i, str(cm[i, j]), horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max() / 2. else "black")
                     
    plt.tight_layout()
    
    safe_name = model_name.lower().replace(" ", "_")
    filename = FIGURES_DIR / f"confusion_matrix_{safe_name}.png"
    plt.savefig(filename)
    plt.close()

def save_classification_report(model_name: str, y_true: np.ndarray, y_pred: np.ndarray):
    """Saves classification report."""
    report = classification_report(
        y_true, y_pred, 
        labels=CLASS_INDICES, 
        target_names=CLASS_ORDER, 
        zero_division=0
    )
    safe_name = model_name.lower().replace(" ", "_")
    path = RESULTS_DIR / f"classification_report_{safe_name}.txt"
    with open(path, 'w') as f:
        f.write(f"Classification Report for {model_name}\n\n")
        f.write(report)

def run_cross_validation(models_dict: dict, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Performs 5-fold stratified CV on training data using weighted F1."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    results = []
    
    # We must ensure there's enough data for CV
    if y_train.value_counts().min() < 5:
         print("Warning: Some classes have fewer than 5 samples. CV might be noisy or fail.")
         
    for name, model in models_dict.items():
        fold_f1s = []
        try:
            for train_idx, val_idx in cv.split(X_train, y_train):
                # We need to clone the model to avoid fitting the same instance
                from sklearn.base import clone
                cloned_model = clone(model)
                
                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_val = y_train.iloc[val_idx]
                
                cloned_model.fit(X_fold_train, y_fold_train)
                preds = cloned_model.predict(X_fold_val)
                fold_f1 = f1_score(y_fold_val, preds, average="weighted", zero_division=0)
                fold_f1s.append(fold_f1)
                
            results.append({
                "model": name,
                "mean_weighted_f1": np.mean(fold_f1s),
                "std_weighted_f1": np.std(fold_f1s)
            })
        except Exception as e:
            print(f"CV Failed for {name}: {e}")
            results.append({
                "model": name,
                "mean_weighted_f1": 0.0,
                "std_weighted_f1": 0.0
            })
            
    df_cv = pd.DataFrame(results)
    df_cv.to_csv(RESULTS_DIR / "cross_validation.csv", index=False)
    return df_cv

def extract_feature_importance(model, model_name: str):
    """Extracts and saves feature importances/coefficients."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        df_imp = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "importance": importances
        }).sort_values(by="importance", ascending=False)
        
        safe_name = model_name.lower().replace(" ", "_")
        df_imp.to_csv(RESULTS_DIR / f"feature_importance_{safe_name}.csv", index=False)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.barh(df_imp["feature"][::-1], df_imp["importance"][::-1])
        plt.xlabel("Importance")
        plt.title(f"Feature Importance - {model_name}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{safe_name}_feature_importance.png")
        plt.close()
        
    elif hasattr(model, "coef_"):
        coefs = model.coef_
        df_coef = pd.DataFrame({"feature": FEATURE_COLUMNS})
        for i, cls in enumerate(model.classes_):
             # Some models (like binary logreg) might only have 1 row for coefs
             if coefs.shape[0] > i:
                 cls_name = REVERSE_LABEL_MAPPING.get(cls, f"Class_{cls}")
                 df_coef[f"coef_{cls_name}"] = coefs[i]
        
        safe_name = model_name.lower().replace(" ", "_")
        df_coef.to_csv(RESULTS_DIR / f"{safe_name}_coefficients.csv", index=False)

def create_model_comparison_plot(df_results: pd.DataFrame, metric: str, title: str, filename: str):
    """Creates a bar chart comparing models on a specific metric."""
    plt.figure(figsize=(8, 5))
    bars = plt.bar(df_results["model"], df_results[metric], color=['skyblue', 'lightgreen', 'salmon'])
    plt.ylim(0, 1.1)
    plt.ylabel(metric)
    plt.title(title)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f"{yval:.4f}", va='bottom', ha='center')
        
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename)
    plt.close()

def generate_predictions_csv(models_dict: dict, X_test: pd.DataFrame, y_test: pd.Series):
    """Generates predictions and probabilities CSV."""
    df_preds = pd.DataFrame()
    df_preds["sample_index"] = y_test.index
    df_preds["actual_scheduler"] = y_test.map(REVERSE_LABEL_MAPPING).values
    
    for name, model in models_dict.items():
        safe_name = name.lower().replace(" ", "_")
        preds = model.predict(X_test)
        df_preds[f"{safe_name}_prediction"] = pd.Series(preds).map(REVERSE_LABEL_MAPPING).values
        df_preds[f"{safe_name}_correct"] = (preds == y_test.values)
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)
            for i, cls_id in enumerate(CLASS_INDICES):
                cls_name = CLASS_ORDER[i]
                # Find which column in probs corresponds to cls_id
                # Model classes might not be strictly [0,1,2,3] if it didn't see all in training
                if cls_id in model.classes_:
                    idx = np.where(model.classes_ == cls_id)[0][0]
                    df_preds[f"{safe_name}_prob_{cls_name.lower()}"] = probs[:, idx]
                else:
                    df_preds[f"{safe_name}_prob_{cls_name.lower()}"] = 0.0

    df_preds.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)
    
def generate_best_model_reports(best_model, best_name: str, X_test: pd.DataFrame, y_test: pd.Series, X_test_raw: pd.DataFrame):
    """Generates predictions and error analysis for the best model."""
    preds = best_model.predict(X_test)
    
    df_preds = pd.DataFrame()
    df_preds["sample_index"] = y_test.index
    df_preds["actual_scheduler"] = y_test.map(REVERSE_LABEL_MAPPING).values
    df_preds["predicted_scheduler"] = pd.Series(preds).map(REVERSE_LABEL_MAPPING).values
    
    if hasattr(best_model, "predict_proba"):
        probs = best_model.predict_proba(X_test)
        df_preds["prediction_confidence"] = np.max(probs, axis=1)
    else:
        df_preds["prediction_confidence"] = 1.0 # Fallback
        
    df_preds.to_csv(RESULTS_DIR / "best_model_predictions.csv", index=False)
    
    # Error analysis
    df_errors = df_preds[df_preds["actual_scheduler"] != df_preds["predicted_scheduler"]].copy()
    if not df_errors.empty:
        # Include original features for errors
        for col in FEATURE_COLUMNS:
            df_errors[col] = X_test_raw.loc[df_errors["sample_index"], col].values
            
    df_errors.to_csv(RESULTS_DIR / "best_model_errors.csv", index=False)

def report_class_distribution(y_full: pd.Series, y_train: pd.Series, y_test: pd.Series):
    """Saves class distribution across splits."""
    dist = []
    
    for y, split in [(y_full, "Full"), (y_train, "Train"), (y_test, "Test")]:
        counts = y.value_counts()
        total = len(y)
        for cls_name, cls_id in LABEL_MAPPING.items():
            count = counts.get(cls_id, counts.get(cls_name, 0))
            dist.append({
                "split": split,
                "scheduler": cls_name,
                "count": count,
                "percentage": (count / total) * 100 if total > 0 else 0
            })
            
    df_dist = pd.DataFrame(dist)
    df_dist.to_csv(RESULTS_DIR / "class_distribution.csv", index=False)
