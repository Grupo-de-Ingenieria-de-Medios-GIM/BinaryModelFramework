import yaml
import warnings
from src.dataset import load_and_preprocess
from src.model_generation import generate_all_binary_models
from src.inference import two_step_inference
from src.evaluation import evaluate_predictions, print_evaluation

# Suppress warnings from sklearn and imblearn for cleaner output
warnings.filterwarnings('ignore')

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    print("Loading configuration...")
    config = load_config()
    
    train_path = config.get('DATASET_TRAIN', 'train.csv')
    test_path = config.get('DATASET_TEST', 'test.csv')
    labels = config.get('LABELS', [])
    
    # We assume 'Normal': legit traffic or Benign
    normal_labels = ['Normal', 'Benign', 'normal']
    normal_class = next((label for label in labels if label in normal_labels), None)
    
    if normal_class is None:
        raise ValueError("Normal class not found in the LABELS provided in config.yaml.")

    # Define features to drop, categorical, and target based on standard datasets (this can be configured further if needed)
    columns_to_drop = ['id', 'ID', 'srcip', 'sport', 'dstip', 'dsport', 'Stime', 'stcpb', 'dtcpb', 'tcprtt', 'synack', 'ackdat']
    categorical_cols = ['proto', 'state', 'service']  # Example categorical cols
    target_col = 'attack_cat'  # Adjust based on dataset. In UNSW-NB15 it's attack_cat or label.

    # Detect file type based on extension
    file_type = 'npy' if train_path.endswith('.npy') else 'csv'
    
    print(f"Loading and preprocessing data from {train_path} and {test_path}...")
    try:
        X_train, y_train, X_test, y_test = load_and_preprocess(
            train_path=train_path,
            test_path=test_path,
            file_type=file_type,
            categorical_cols=categorical_cols,
            columns_to_drop=columns_to_drop,
            target_col=target_col
        )
    except Exception as e:
        print(f"Error loading datasets. Please check paths in config.yaml and the target_col variable. Exception: {str(e)}")
        print("Note: If you have different column patterns, please adjust target_col and categorical_cols in main.py.")
        return

    print("\n--- Phase 1: Binary Model Generation ---")
    print("This will create a balanced Binary Dataset (OvR + SMOTE) for each class and run GridSearchCV across 4 algorithms.")
    models = generate_all_binary_models(X_train, y_train)

    print("\n--- Phase 2: Two-Step Inference ---")
    print("Running Attack Detection Model followed by Attack Classification Models.")
    y_pred = two_step_inference(X_test, models, normal_class=normal_class)

    print("\n--- Phase 3: Evaluation ---")
    # Get labels from models to match confusion matrix format exactly
    evaluation_labels = list(models.keys())
    # You might want to sort labels so Normal is at the end or as per LABELS order
    sorted_labels = [label for label in labels if label in evaluation_labels]
    
    results = evaluate_predictions(y_test, y_pred, labels=sorted_labels)
    print_evaluation(results, labels=sorted_labels)

    print("\nProcess finished successfully.")

if __name__ == '__main__':
    main()
