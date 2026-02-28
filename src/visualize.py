# Visualization & EDA Module for Customer Churn Prediction
# Minor Project AK7 — JUET Guna (MP)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
import shap
import warnings
warnings.filterwarnings('ignore')

# Set consistent plot style
sns.set_style("darkgrid")
plt.rcParams.update({
    'figure.facecolor': '#1e293b',
    'axes.facecolor': '#1e293b',
    'axes.edgecolor': '#475569',
    'axes.labelcolor': '#94a3b8',
    'text.color': '#f1f5f9',
    'xtick.color': '#94a3b8',
    'ytick.color': '#94a3b8',
    'grid.color': '#334155',
    'font.family': 'DejaVu Sans',
    'font.size': 11,
})

COLORS = {
    'primary': '#6366f1',
    'accent': '#06b6d4',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'churn_yes': '#ef4444',
    'churn_no': '#10b981',
}


def plot_churn_distribution(df: pd.DataFrame, save_path: str = None):
    """Plot overall churn distribution as a bar chart and pie chart."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Customer Churn Distribution', fontsize=16, fontweight='bold', color='#f1f5f9')

    churn_counts = df['Churn'].value_counts()

    # Bar chart
    bars = axes[0].bar(['No Churn', 'Churn'], churn_counts.values,
                       color=[COLORS['churn_no'], COLORS['churn_yes']],
                       edgecolor='none', alpha=0.85)
    axes[0].set_title('Count', color='#f1f5f9')
    axes[0].set_ylabel('Number of Customers')
    for bar, val in zip(bars, churn_counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                     str(val), ha='center', fontweight='bold', color='#f1f5f9')

    # Pie chart
    axes[1].pie(
        churn_counts.values,
        labels=['No Churn', 'Churn'],
        colors=[COLORS['churn_no'], COLORS['churn_yes']],
        autopct='%1.1f%%', startangle=90,
        textprops={'color': '#f1f5f9', 'fontsize': 12},
        wedgeprops={'edgecolor': '#0f172a', 'linewidth': 2}
    )
    axes[1].set_title('Proportion', color='#f1f5f9')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    plt.show()


def plot_feature_distributions(df: pd.DataFrame, save_path: str = None):
    """Plot distribution of key numeric features by churn status."""
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Key Feature Distributions by Churn Status', fontsize=14, fontweight='bold',
                 color='#f1f5f9')

    churn_yes = df[df['Churn'] == 'Yes'] if df['Churn'].dtype == object else df[df['Churn'] == 1]
    churn_no = df[df['Churn'] == 'No'] if df['Churn'].dtype == object else df[df['Churn'] == 0]

    fix_col = lambda col: col if col in df.columns else col.lower()

    for ax, feat in zip(axes, numeric_features):
        col = fix_col(feat)
        ax.hist(churn_no[col].dropna(), bins=40, alpha=0.7,
                color=COLORS['churn_no'], label='No Churn', edgecolor='none')
        ax.hist(churn_yes[col].dropna(), bins=40, alpha=0.7,
                color=COLORS['churn_yes'], label='Churn', edgecolor='none')
        ax.set_title(feat, color='#f1f5f9', fontweight='bold')
        ax.set_xlabel(feat)
        ax.set_ylabel('Count')
        ax.legend(framealpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    plt.show()


def plot_categorical_churn(df: pd.DataFrame, save_path: str = None):
    """Plot churn rate by key categorical features."""
    cat_features = ['Contract', 'InternetService', 'PaymentMethod', 'gender']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Churn Rate by Categorical Features', fontsize=14, fontweight='bold',
                 color='#f1f5f9')
    axes = axes.flatten()

    for ax, feat in zip(axes, cat_features):
        if feat not in df.columns:
            continue

        churn_col = 'Churn'
        churn_rate = df.groupby(feat)[churn_col].apply(
            lambda x: (x == 'Yes').mean() if x.dtype == object else x.mean()
        ).reset_index()
        churn_rate.columns = [feat, 'churn_rate']
        churn_rate = churn_rate.sort_values('churn_rate', ascending=False)

        bars = ax.bar(churn_rate[feat], churn_rate['churn_rate'],
                      color=COLORS['primary'], alpha=0.85, edgecolor='none')
        ax.set_title(f'Churn Rate by {feat}', color='#f1f5f9', fontweight='bold')
        ax.set_ylabel('Churn Rate')
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=15)

        for bar, rate in zip(bars, churn_rate['churn_rate']):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{rate:.1%}', ha='center', fontsize=9, color='#f1f5f9')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    plt.show()


def plot_correlation_heatmap(X: pd.DataFrame, save_path: str = None):
    """Plot correlation heatmap for all features."""
    corr = X.corr()

    fig, ax = plt.subplots(figsize=(16, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr, mask=mask, annot=False, cmap='coolwarm',
        center=0, vmin=-1, vmax=1,
        linewidths=0.5, linecolor='#0f172a',
        ax=ax
    )
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', color='#f1f5f9')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    plt.show()


def plot_model_comparison(results_df: pd.DataFrame, save_path: str = None):
    """Bar chart comparing model performance metrics."""
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']

    x = np.arange(len(results_df.index))
    width = 0.15

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold', color='#f1f5f9')

    palette = [COLORS['primary'], COLORS['accent'], COLORS['success'], COLORS['warning'], COLORS['danger']]

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        offset = (i - len(metrics) / 2) * width
        bars = ax.bar(x + offset, results_df[metric], width, label=label,
                      color=palette[i], alpha=0.85, edgecolor='none')

    ax.set_xticks(x)
    ax.set_xticklabels(results_df.index, rotation=15)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.legend(framealpha=0.3, loc='upper right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    plt.show()


def plot_shap_summary(shap_values, X, feature_names: list, save_path: str = None):
    """
    Plot SHAP summary (beeswarm) plot for global feature importance.

    Args:
        shap_values: SHAP values array (for positive class for RF)
        X: Feature array
        feature_names: Column names
    """
    plt.figure(figsize=(10, 8))
    # For RandomForest, shap_values is a list; take the churn class (index 1)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    shap.summary_plot(sv, X, feature_names=feature_names, show=False,
                      plot_type='dot', color_bar=True)
    plt.title('SHAP Feature Importance (Global)', fontsize=14, fontweight='bold', color='#f1f5f9')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()


def plot_shap_bar(shap_values, feature_names: list, save_path: str = None):
    """Plot SHAP bar chart of mean absolute feature importance."""
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    mean_shap = np.abs(sv).mean(axis=0)
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': mean_shap})
    importance_df = importance_df.sort_values('importance', ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(importance_df['feature'], importance_df['importance'],
            color=COLORS['primary'], alpha=0.85, edgecolor='none')
    ax.set_title('Top 15 Features by Mean |SHAP| Value', fontsize=14,
                 fontweight='bold', color='#f1f5f9')
    ax.set_xlabel('Mean |SHAP Value|')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    plt.show()


def plot_confusion_matrix(y_test, y_pred, save_path: str = None):
    """Plot styled confusion matrix."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred)
    labels = ['No Churn', 'Churn']

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                ax=ax, linewidths=2, linecolor='#0f172a',
                annot_kws={'size': 18, 'weight': 'bold'})

    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', color='#f1f5f9')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    plt.show()


def plot_roc_curve(models_dict: dict, X_test, y_test, save_path: str = None):
    """
    Plot ROC curve for multiple models on the same axes.

    Args:
        models_dict: {'Model Name': trained_model, ...}
    """
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Chance')

    palette = [COLORS['primary'], COLORS['accent'], COLORS['success'], COLORS['warning']]
    for (name, model), color in zip(models_dict.items(), palette):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f'{name} (AUC = {roc_auc:.3f})')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — Model Comparison', fontsize=14, fontweight='bold', color='#f1f5f9')
    ax.legend(loc='lower right', framealpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    plt.show()
