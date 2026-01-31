# Model Training & Evaluation Module for Customer Churn Prediction
# Minor Project AK7 — JUET Guna (MP)
# Team: Harsh Vardhan Chauhan, Himanshu S. Patil, Rudransh Srivastava

import os
import pickle
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

import shap
import warnings
warnings.filterwarnings('ignore')


# ==============================================================================
# Classical ML Models
# ==============================================================================

def train_logistic_regression(X_train, y_train, random_state: int = 42):
    """Train Logistic Regression baseline model."""
    model = LogisticRegression(max_iter=1000, random_state=random_state, C=0.1)
    model.fit(X_train, y_train)
    print("✓ Logistic Regression trained.")
    return model


def train_decision_tree(X_train, y_train, random_state: int = 42):
    """Train Decision Tree classifier."""
    model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    print("✓ Decision Tree trained.")
    return model


def train_random_forest(X_train, y_train, random_state: int = 42):
    """Train Random Forest classifier (primary model for deployment)."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("✓ Random Forest trained.")
    return model


def train_xgboost(X_train, y_train, random_state: int = 42):
    """Train XGBoost classifier."""
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("✓ XGBoost trained.")
    return model


# ==============================================================================
# Evaluation
# ==============================================================================

def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    """
    Evaluate a trained model on the test set.

    Returns:
        dict with accuracy, precision, recall, f1, roc_auc
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }

    print(f"\n{'=' * 50}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'=' * 50}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

    return metrics


def compare_models(results: list) -> pd.DataFrame:
    """
    Create a comparison DataFrame from a list of evaluation result dicts.

    Args:
        results: List of dicts returned by evaluate_model()

    Returns:
        Sorted DataFrame comparing all models.
    """
    df = pd.DataFrame(results)
    df = df.set_index('model')
    df = df.sort_values('f1', ascending=False)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (sorted by F1 Score)")
    print("=" * 60)
    print(df.round(4).to_string())
    return df


# ==============================================================================
# SHAP Explainability
# ==============================================================================

def compute_shap_explainer(model, X_train, model_type: str = 'tree'):
    """
    Compute SHAP explainer and save it for later use.

    Args:
        model: Trained sklearn/XGBoost model
        X_train: Training features (used for background data)
        model_type: 'tree' for RF/XGBoost, 'linear' for LogReg

    Returns:
        SHAP explainer object
    """
    if model_type == 'tree':
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.LinearExplainer(model, X_train)

    print("✓ SHAP explainer created.")
    return explainer


def get_shap_values(explainer, X):
    """Compute SHAP values for a set of samples."""
    shap_values = explainer.shap_values(X)
    print(f"✓ SHAP values computed for {len(X)} samples.")
    return shap_values


# ==============================================================================
# Model Persistence
# ==============================================================================

def save_model(model, filepath: str) -> None:
    """Save a trained model using pickle."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved to: {filepath}")


def save_explainer(explainer, filepath: str) -> None:
    """Save a SHAP explainer using joblib (compressed)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(explainer, filename=filepath)
    print(f"✓ SHAP explainer saved to: {filepath}")


def load_model(filepath: str):
    """Load a pickled model."""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"✓ Model loaded from: {filepath}")
    return model


def load_explainer(filepath: str):
    """Load a SHAP explainer."""
    explainer = joblib.load(filepath)
    print(f"✓ Explainer loaded from: {filepath}")
    return explainer


# ==============================================================================
# Main Training Pipeline
# ==============================================================================

def run_full_pipeline(X_train, X_test, y_train, y_test, feature_names: list,
                      save_dir: str = './models/'):
    """
    Run the complete training and evaluation pipeline for all models,
    save best model (Random Forest) and its SHAP explainer.

    Args:
        X_train, X_test, y_train, y_test: Data splits
        feature_names: List of feature column names
        save_dir: Directory to save trained models

    Returns:
        (best_model, comparison_df)
    """
    print("\n" + "=" * 60)
    print("TRAINING ALL MODELS")
    print("=" * 60)

    # Train all models
    models = {
        'Logistic Regression': train_logistic_regression(X_train, y_train),
        'Decision Tree': train_decision_tree(X_train, y_train),
        'Random Forest': train_random_forest(X_train, y_train),
        'XGBoost': train_xgboost(X_train, y_train),
    }

    # Evaluate all models
    results = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, model_name=name)
        results.append(metrics)

    # Compare
    comparison = compare_models(results)

    # Save best model (Random Forest) and SHAP explainer
    rf_model = models['Random Forest']
    save_model(rf_model, os.path.join(save_dir, 'model_rfc.pkl'))

    # SHAP explainer for Random Forest
    explainer = compute_shap_explainer(rf_model, X_train, model_type='tree')
    save_explainer(explainer, os.path.join(save_dir, 'explainer_rfc.bz2'))

    return rf_model, comparison


if __name__ == "__main__":
    from preprocess import load_data, preprocess_data, get_train_test_split

    df = load_data()
    X, y, feature_names = preprocess_data(df)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)

    best_model, results = run_full_pipeline(
        X_train, X_test, y_train, y_test, feature_names
    )
    print("\nPipeline complete! Models saved to ./models/")
