import numpy as np
import pandas as pd
from typing import Dict, Any, List
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

def generate_binary_dataset(X: pd.DataFrame, y: pd.Series, target_class: str) -> tuple:
    """
    Implements Algorithm 1: BinaryDatasetGeneration from the paper.
    Generates a balanced binary dataset for a target class using OvR and SMOTE.
    """
    classes = y.unique()
    N = len(classes)
    
    # Positive class: all samples of target category
    pos_mask = (y == target_class)
    X_pos = X[pos_mask]
    y_pos = pd.Series([1] * len(X_pos), index=X_pos.index)
    n_j = len(X_pos)
    
    if N > 1:
        n_samples_c = n_j // (N - 1)
    else:
        n_samples_c = 0

    X_neg_list = []
    y_neg_list = []
    
    # Negative class: samples from non-target categories
    for c_k in classes:
        if c_k == target_class:
            continue
            
        neg_mask = (y == c_k)
        X_k = X[neg_mask]
        
        if len(X_k) >= n_samples_c:
            # Randomly sample n_samples_c
            X_sampled = X_k.sample(n=n_samples_c, random_state=42)
            X_neg_list.append(X_sampled)
        else:
            # All available samples + SMOTE
            if len(X_k) == 0:
                continue
            
            # Since SMOTE requires at least 2 classes, we temporarily combine X_k with a dummy class 
            # to synthesize data, or better yet, since we only need to synthesize points for X_k, 
            # we can just use SMOTE directly if we format it correctly.
            # However, SMOTE syntheizes minority class in a binary (or multi) dataset.
            # To just oversample X_k to reach n_samples_c, we can combine X_k with X_pos 
            # and oversample X_k to match the desired count (n_samples_c).
            
            X_temp = pd.concat([X_pos, X_k])
            y_temp = pd.Series([1]*len(X_pos) + [0]*len(X_k))
            
            # SMOTE parameters: k_neighbors=5, but if len(X_k) <= 5, we need to adjust k_neighbors
            k_neighbors = min(5, len(X_k) - 1)
            if k_neighbors < 1:
                # Cannot use SMOTE with 1 sample, just pad with dupes (RandomOverSampler behavior)
                X_sampled = X_k.sample(n=n_samples_c, replace=True, random_state=42)
                X_neg_list.append(X_sampled)
                continue
            
            # Use sampling_strategy to exactly reach n_samples_c for the temporary class 0
            smote = SMOTE(sampling_strategy={0: n_samples_c, 1: len(X_pos)}, 
                          k_neighbors=k_neighbors, random_state=42)
            
            X_res, y_res = smote.fit_resample(X_temp, y_temp)
            # Extract just the synthesized negative class
            X_neg_k = X_res[y_res == 0]
            X_neg_list.append(X_neg_k)
            
    if X_neg_list:
        X_neg = pd.concat(X_neg_list)
        y_neg = pd.Series([0] * len(X_neg), index=X_neg.index)
        
        X_binary = pd.concat([X_pos, X_neg]).reset_index(drop=True)
        y_binary = pd.concat([y_pos, y_neg]).reset_index(drop=True)
    else:
        # Fallback if no negative samples
        X_binary = X_pos.reset_index(drop=True)
        y_binary = y_pos.reset_index(drop=True)

    return X_binary, y_binary


def get_algorithms_and_params() -> dict:
    """
    Returns the algorithms and their hyperparameter grids as defined in Table 3.
    """
    return {
        'KNN': {
            'estimator': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [1, 5, 10, 15, 20, 30],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree']
            }
        },
        'SVM': {
            'estimator': SVC(probability=True, random_state=42),
            'params': {
                'C': [1, 10, 100, 1000],
                'gamma': [1e-5, 1e-3, 0.1, 1]
            }
        },
        'DT': {
            'estimator': DecisionTreeClassifier(random_state=42),
            'params': {
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 5, 15, 25, 35]
            }
        },
        'MLP': {
            'estimator': MLPClassifier(random_state=42, early_stopping=True),
            'params': {
                'activation': ['relu', 'tanh'],
                'learning_rate_init': [1e-5, 1e-3, 0.1, 1],
                'max_iter': [20, 60, 100, 200],
                'batch_size': [32, 128, 256]
            }
        }
    }


def train_best_binary_model(X_binary: pd.DataFrame, y_binary: pd.Series) -> tuple:
    """
    Performs GridSearchCV over KNN, SVM, DT, and MLP to find the best model
    for the given binary dataset optimizing for Macro-F1.
    """
    macro_f1_scorer = make_scorer(f1_score, average='macro')
    algorithms = get_algorithms_and_params()
    
    best_overall_model = None
    best_overall_score = -1
    best_overall_name = ""

    for name, config in algorithms.items():
        grid_search = GridSearchCV(
            estimator=config['estimator'],
            param_grid=config['params'],
            cv=5,
            scoring=macro_f1_scorer,
            n_jobs=-1
        )
        grid_search.fit(X_binary, y_binary)
        
        if grid_search.best_score_ > best_overall_score:
            best_overall_score = grid_search.best_score_
            best_overall_model = grid_search.best_estimator_
            best_overall_name = name

    return best_overall_model, best_overall_name, best_overall_score


def generate_all_binary_models(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Loops through each class, generates the binary dataset, and finds the best model.
    """
    classes = y.unique()
    models = {}
    
    for c_j in classes:
        print(f"Generating binary dataset and training model for class: {c_j}")
        X_bin, y_bin = generate_binary_dataset(X, y, target_class=c_j)
        best_model, best_algo, best_score = train_best_binary_model(X_bin, y_bin)
        
        models[c_j] = {
            'model': best_model,
            'algorithm': best_algo,
            'macro_f1_val': best_score
        }
        print(f"  -> Best model for {c_j}: {best_algo} (Macro-F1: {best_score:.4f})")
    
    return models
