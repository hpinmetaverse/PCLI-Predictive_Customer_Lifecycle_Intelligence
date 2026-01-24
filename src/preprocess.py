# Data Preprocessing Module for Customer Churn Prediction
# Minor Project AK7 — JUET Guna (MP)
# Team: Harsh Vardhan Chauhan, Himanshu S. Patil, Rudransh Srivastava

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE


def load_data(filepath: str = './data/WA_Fn-UseC_-Telco-Customer-Churn.csv') -> pd.DataFrame:
    """
    Load the IBM Telco Customer Churn dataset.

    Args:
        filepath: Path to the CSV file.

    Returns:
        Raw DataFrame.
    """
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def explore_data(df: pd.DataFrame) -> None:
    """Print basic EDA information about the dataset."""
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"\nShape: {df.shape}")
    print(f"\nColumn types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nChurn distribution:\n{df['Churn'].value_counts()}")
    print(f"\nChurn rate: {df['Churn'].value_counts(normalize=True)['Yes']:.2%}")
    print("\nFirst 5 rows:")
    print(df.head())


def preprocess_data(df: pd.DataFrame) -> tuple:
    """
    Full preprocessing pipeline for the Telco churn dataset.

    Steps:
    1. Drop customerID (irrelevant identifier)
    2. Fix TotalCharges (convert to numeric, fill missing)
    3. Binary encode Yes/No columns
    4. Encode gender
    5. One-hot encode multi-class categoricals
    6. Encode target variable

    Args:
        df: Raw DataFrame from load_data()

    Returns:
        (X, y, feature_columns) tuple
    """
    df = df.copy()

    # Step 1: Drop irrelevant column
    df.drop('customerID', axis=1, inplace=True)

    # Step 2: Fix TotalCharges — contains spaces, convert to float
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Fill missing TotalCharges with MonthlyCharges * tenure
    mask = df['TotalCharges'].isna()
    df.loc[mask, 'TotalCharges'] = df.loc[mask, 'MonthlyCharges'] * df.loc[mask, 'tenure']

    # Step 3: Binary encode Yes/No columns
    binary_cols = [
        'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn'
    ]
    for col in binary_cols:
        df[col] = (df[col] == 'Yes').astype(int)

    # Step 4: Encode gender (Male=0, Female=1)
    df['gender'] = (df['gender'] == 'Female').astype(int)

    # Step 5: Binary encode service columns (No/Yes but also "No phone service" / "No internet service")
    service_cols = [
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    for col in service_cols:
        df[col] = df[col].apply(lambda x: 1 if x == 'Yes' else 0)

    # Step 6: SeniorCitizen is already 0/1, no change needed

    # Step 7: One-hot encode multi-category columns
    df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod'], drop_first=False)

    # Remove redundant dummy columns (keep meaningful ones)
    # InternetService_DSL is the reference -> drop it
    if 'InternetService_DSL' in df.columns:
        df.drop('InternetService_DSL', axis=1, inplace=True)
    # Contract_Month-to-month is reference -> drop
    if 'Contract_Month-to-month' in df.columns:
        df.drop('Contract_Month-to-month', axis=1, inplace=True)
    # PaymentMethod_Bank transfer (automatic) is reference -> drop
    if 'PaymentMethod_Bank transfer (automatic)' in df.columns:
        df.drop('PaymentMethod_Bank transfer (automatic)', axis=1, inplace=True)

    # Step 8: Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    print(f"\nPreprocessing complete!")
    print(f"Features: {X.shape[1]} columns")
    print(f"Feature names: {list(X.columns)}")
    print(f"Class distribution: {y.value_counts().to_dict()}")

    return X, y, list(X.columns)


def apply_smote(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42):
    """
    Apply SMOTE to handle class imbalance in training data.

    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed for reproducibility

    Returns:
        (X_resampled, y_resampled)
    """
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print(f"\nSMOTE Applied:")
    print(f"  Before: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  After:  {dict(zip(*np.unique(y_res, return_counts=True)))}")

    return X_res, y_res


def get_train_test_split(X, y, test_size: float = 0.2, random_state: int = 42, apply_oversampling: bool = True):
    """
    Split data into train/test sets and optionally apply SMOTE.

    Args:
        X: Feature DataFrame or array
        y: Target series or array
        test_size: Proportion of test set
        random_state: Random seed
        apply_oversampling: Whether to apply SMOTE on training data

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\nTrain/Test Split:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")

    if apply_oversampling:
        X_train, y_train = apply_smote(X_train.values if hasattr(X_train, 'values') else X_train,
                                        y_train.values if hasattr(y_train, 'values') else y_train,
                                        random_state=random_state)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Example usage
    df = load_data()
    explore_data(df)
    X, y, feature_names = preprocess_data(df)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    print("\nPreprocessing pipeline complete. Ready for model training.")
