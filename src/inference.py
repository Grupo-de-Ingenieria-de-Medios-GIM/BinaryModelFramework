import numpy as np
import pandas as pd
from typing import Dict, Any

def two_step_inference(X_test: pd.DataFrame, models: Dict[str, Any], normal_class: str) -> np.ndarray:
    """
    Implements Algorithm 2: Two-StepInference from the paper.
    
    Step 1: Attack Detection (using the Normal binary model).
    Step 2: Attack Classification (using the remaining binary models).
    
    Args:
        X_test (pd.DataFrame): The test features.
        models (Dict[str, Any]): Dictionary of trained models per class.
        normal_class (str): The label of the 'Normal' or legitimate traffic class.
        
    Returns:
        np.ndarray: Array of predicted labels for X_test.
    """
    predictions = []
    
    normal_model = models.get(normal_class)
    if not normal_model:
        raise ValueError(f"Normal class model '{normal_class}' not found in the trained models.")

    # We can optimize this by performing batch predictions, but for aligning exactly with the algorithm logic:
    # We will do batch prediction for Step 1, then masked batch prediction for Step 2.
    
    # --- Step 1: Attack Detection ---
    # The normal model predicts probability of class 1 (Normal) and class 0 (Attack)
    # The models were trained with target class as 1 and rest as 0.
    probas_normal_model = normal_model['model'].predict_proba(X_test)
    
    # Identify which instances are classified as Normal
    # probas_normal_model[:, 1] is P(Normal)
    # probas_normal_model[:, 0] is P(Attack)
    
    is_normal_mask = probas_normal_model[:, 1] >= probas_normal_model[:, 0]
    
    # Initialize the final predictions array with a placeholder
    final_preds = np.empty(len(X_test), dtype=object)
    final_preds[is_normal_mask] = normal_class
    
    # --- Step 2: Attack Classification ---
    # Filter only the instances classified as attacks
    is_attack_mask = ~is_normal_mask
    X_attack = X_test[is_attack_mask]
    
    if len(X_attack) > 0:
        # Collect probabilities from all other models
        attack_classes = [c for c in models.keys() if c != normal_class]
        
        # Matrix to hold positive probabilities for each attack model: shape (len(X_attack), num_attack_classes)
        attack_probas = np.zeros((len(X_attack), len(attack_classes)))
        
        for idx, attack_cls in enumerate(attack_classes):
            model = models[attack_cls]['model']
            # P(y = c_j | x, m_j) which is the probability of the positive class (index 1)
            probas = model.predict_proba(X_attack)[:, 1]
            attack_probas[:, idx] = probas
            
        # Select the class with the highest probability
        best_attack_idx = np.argmax(attack_probas, axis=1)
        best_attack_classes = np.array(attack_classes)[best_attack_idx]
        
        # Assign back to final predictions
        final_preds[is_attack_mask] = best_attack_classes

    return final_preds
