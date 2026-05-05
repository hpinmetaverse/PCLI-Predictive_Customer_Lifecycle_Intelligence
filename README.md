 Predictive Customer Churn Analysis using ML
---

## Project Overview

This project focuses on **Predictive Customer Churn Analysis using Machine Learning**. Customer churn - the loss of clients to competitors - is a critical metric in subscription-based businesses like telecom or streaming services.

The goal is to build an interpretable ML pipeline that:
- Predicts whether a customer is likely to churn
- Explains *why* a prediction was made using Explainable AI (XAI) techniques
- Deploys as a web application for real-time predictions

---

## Dataset

**IBM Telco Customer Churn Dataset** (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)

- **Size**: 7,043 customer records
- **Features**: 21 features including tenure, monthly charges, contract type, services used, payment method, etc.
- **Target**: Binary classification — `Churn` (Yes/No)

### Key Features
| Feature | Description |
|---|---|
| `tenure` | Number of months the customer has stayed |
| `MonthlyCharges` | Monthly billing amount |
| `TotalCharges` | Total amount charged |
| `Contract` | Contract type: Month-to-Month, One year, Two year |
| `InternetService` | DSL / Fiber optic / No |
| `PaymentMethod` | Bank transfer / Credit card / Electronic check / Mailed check |
| `SeniorCitizen` | Whether the customer is a senior citizen |

---

## Methodology

### 1. Data Preprocessing
- Dropped irrelevant `customerID` column
- Handled missing/inconsistent values in `TotalCharges`
- Binary encoding for boolean features (Partner, Dependents, PhoneService, etc.)
- One-hot encoding for categorical features (InternetService, Contract, PaymentMethod)

### 2. Exploratory Data Analysis (EDA)
- Identified key patterns and anomalies in the data
- Analyzed churn distribution (class imbalance)
- Feature correlation analysis

### 3. Feature Engineering
- Engineered new features and transformed existing variables
- Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to handle class imbalance

### 4. Models Implemented
| Model | Description |
|---|---|
| Logistic Regression | Baseline linear classifier |
| Decision Tree | Rule-based classifier |
| **Random Forest** | Ensemble model - **best performing** (used in deployment) |
| XGBoost | Gradient boosting classifier |
| MLP (PyTorch) | Custom multi-layer perceptron neural network |

### 5. Model Evaluation
- Evaluated using: **Accuracy, Precision, Recall, F1 Score, ROC-AUC**
- Compared all models to identify best-performing algorithm
- Final model: **Random Forest Classifier**

### 6. Explainable AI (XAI)
- **SHAP** (SHapley Additive exPlanations):
  - Global feature importance via summary plots
  - Local per-prediction explanations via force plots
- **LIME** (Local Interpretable Model-agnostic Explanations):
  - Local explanations for individual customer predictions

---

## Project Structure

```
minor-project-churn/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # IBM Telco Dataset
├── models/
│   ├── model_rfc.pkl                            # Trained Random Forest model
│   ├── mlp_model.pkl                            # Trained MLP model
│   └── explainer_rfc.bz2                       # Pre-computed SHAP explainer
├── templates/
│   └── index.html                              # Flask web app template
├── notebooks/
│   └── main.ipynb                              # Main analysis notebook
├── app.py                                      # Flask web application
├── requirements.txt                            # Python dependencies
├── Pipfile                                     # Pipenv config
└── README.md                                   # Project documentation
```

---

## Setup & Installation

### Prerequisites
- Python 3.9+
- pip or pipenv

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd minor-project-churn

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Web App

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

### Running the Notebook

```bash
jupyter notebook notebooks/main.ipynb
```

---

## Results

- **Best Model**: Random Forest Classifier
- **Key Churn Predictors**: Contract type, Tenure, Monthly Charges, Internet Service type
- **XAI Integration**: SHAP force plots provide per-customer explanation; LIME provides local model transparency
- **Deployment**: FastAPI web app with real-time churn probability gauge and SHAP explanation

---

## Technologies Used

| Category | Tools |
|---|---|
| **Language** | Python 3.9 |
| **ML Libraries** | scikit-learn, XGBoost, imbalanced-learn (SMOTE) |
| **Deep Learning** | PyTorch |
| **XAI** | SHAP, LIME |
| **Web Framework** | FastAPI|
| **Data Processing** | pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Notebook** | Jupyter |

---

## References

1. IBM Telco Customer Churn Dataset - [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)
2. SHAP: Lundberg & Lee (2017) -"A Unified Approach to Interpreting Model Predictions"
3. LIME: Ribeiro et al. (2016) -"Why Should I Trust You?"
4. Scikit-learn documentation - https://scikit-learn.org
