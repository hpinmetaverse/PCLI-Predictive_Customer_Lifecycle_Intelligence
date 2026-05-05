import pandas as pd
import numpy as np
import pickle
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap

# Load dataset from data folder
df = pd.read_csv('./data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Preprocess
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df['Churn'] = (df['Churn'] == 'Yes').astype(int)

# Encode binary columns
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines',
               'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
               'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0, 'No phone service': 0, 'No internet service': 0})

df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

# One-hot encode
df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod'])

# Debug: print actual column names
print("Columns after get_dummies:")
print([c for c in df.columns if 'Internet' in c or 'Contract' in c or 'Payment' in c])

# Rename to match app.py expected columns
rename_map = {}
for col in df.columns:
    if 'InternetService_No' in col:
        rename_map[col] = 'InternetService_No'
    if 'InternetService_Fiber' in col:
        rename_map[col] = 'InternetService_Fiber optic'
    if 'Contract_One' in col:
        rename_map[col] = 'Contract_One year'
    if 'Contract_Two' in col:
        rename_map[col] = 'Contract_Two year'
    if 'Credit card' in col:
        rename_map[col] = 'PaymentMethod_Credit card (automatic)'
    if 'Electronic' in col:
        rename_map[col] = 'PaymentMethod_Electronic check'
    if 'Mailed' in col:
        rename_map[col] = 'PaymentMethod_Mailed check'

df.rename(columns=rename_map, inplace=True)
print("\nRenamed columns:", list(rename_map.values()))

COLUMNS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
    'InternetService_Fiber optic', 'InternetService_No',
    'Contract_One year', 'Contract_Two year',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

X = df[COLUMNS]
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(f"\nAccuracy: {model.score(X_test, y_test):.2%}")

# SHAP explainer
explainer = shap.TreeExplainer(model)

# Save models
os.makedirs('models', exist_ok=True)
pickle.dump(model, open('./models/model.pkl', 'wb'))
joblib.dump(explainer, './models/explainer.bz2', compress=('bz2', 9))
print("✅ Models saved to ./models/")