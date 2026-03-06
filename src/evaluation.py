import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, labels: list) -> dict:
    """
    Evaluates predictions and returns Accuracy, Precision, Recall, F1, Macro-F1,
    and the Confusion Matrix. Assumes labels are provided in the order you'd like 
    them to appear in the matrix.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = accuracy_score(y_true, y_pred)
    
    # Calculate Precision, Recall, F1-Score (per class)
    # the warning is to avoid division by zero
    precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    
    # Macro averages
    macro_precision = precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    
    # Weighted averages
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)

    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)

    results = {
        'confusion_matrix': cm,
        'accuracy': acc,
        'precision_per_class': dict(zip(labels, precision)),
        'recall_per_class': dict(zip(labels, recall)),
        'f1_per_class': dict(zip(labels, f1)),
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'classification_report': report
    }
    
    return results

def print_evaluation(results: dict, labels: list):
    """
    Utility function to pretty print the evaluation results.
    """
    print("-" * 50)
    print("Classification Report:")
    print(results['classification_report'])
    print("-" * 50)
    print(f"Accuracy:    {results['accuracy']:.4f}")
    print(f"Macro-F1:    {results['macro_f1']:.4f}")
    print(f"Weighted-F1: {results['weighted_f1']:.4f}")
    print("-" * 50)
    
    print("\nConfusion Matrix:")
    df_cm = pd.DataFrame(results['confusion_matrix'], index=labels, columns=labels)
    print(df_cm)
    print("-" * 50)
