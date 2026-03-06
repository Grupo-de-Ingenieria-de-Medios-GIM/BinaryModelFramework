import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

def load_data(filepath, file_type='csv'):
    """
    Loads dataset from a given filepath.
    Supports 'csv' and 'npy' extensions.
    """
    if file_type == 'npy':
        return pd.DataFrame(np.load(filepath, allow_pickle=True))
    elif file_type == 'csv':
        return pd.read_csv(filepath)
    else:
        raise ValueError("Unsupported file type. Use 'csv' or 'npy'.")


def drop_non_informative_features(df, columns_to_drop):
    """
    Drops non-informative columns like IP addresses, timestamps, etc.
    """
    # ensure columns exist
    cols = [col for col in columns_to_drop if col in df.columns]
    return df.drop(columns=cols)


def preprocess_data(df, categorical_cols, target_col):
    """
    Preprocess the data mimicking the paper's methodology:
    - Label or One-Hot Encoding for categorical features.
    - Standard Scaling (z-score) for numerical features.
    """
    # Separate features and target
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # Identify numerical columns
    numerical_cols = [col for col in X.columns if col not in categorical_cols]

    # Handle Categorical Columns (One-Hot Encoding for this implementation)
    if categorical_cols:
        # Using pd.get_dummies for simplicity and to match the one-hot encoding
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

    # Handle Numerical Columns (Standard Scaling)
    if numerical_cols:
         scaler = StandardScaler()
         X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    return X, y

def load_and_preprocess(train_path, test_path, file_type, categorical_cols, columns_to_drop, target_col):
    """
    Wrapper to load both train and test data, drop columns, and preprocess them together to ensure identical feature encoding.
    """
    df_train = load_data(train_path, file_type)
    df_test = load_data(test_path, file_type)

    df_train['is_train'] = True
    df_test['is_train'] = False

    df_combined = pd.concat([df_train, df_test], ignore_index=True)

    # Drop non-informative features
    df_combined = drop_non_informative_features(df_combined, columns_to_drop)
    
    # Save is_train mask
    is_train_mask = df_combined['is_train'].copy()
    df_combined = df_combined.drop(columns=['is_train'])

    # Preprocess (Scale & One-Hot Encode)
    X_combined, y_combined = preprocess_data(df_combined, categorical_cols, target_col)
    
    # Split back to train and test
    X_train = X_combined[is_train_mask == True]
    y_train = y_combined[is_train_mask == True]
    
    X_test = X_combined[is_train_mask == False]
    y_test = y_combined[is_train_mask == False]
    
    return X_train, y_train, X_test, y_test
