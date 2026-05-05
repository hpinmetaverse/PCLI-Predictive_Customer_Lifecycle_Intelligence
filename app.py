import io
import base64
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS # type: ignore
import numpy as np
import pandas as pd
import pickle
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle
import shap  # type: ignore

app = Flask(__name__)
CORS(app)  # allow calls from the Lovable edge function

# --- Load models ---
model = pickle.load(open('./models/model.pkl', 'rb'))
ENCODED_COLUMNS = model.feature_names_in_.tolist()
explainer = joblib.load(filename="./models/explainer.bz2")

# Optional survival model (for cumulative hazard + survival curve)
try:
    survival_model = pickle.load(open('./models/survivemodel.pkl', 'rb'))
except Exception:
    survival_model = None


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


def _parse_inputs(form):
    """Parse form-style or JSON-style inputs into the model feature vector."""
    def flag(name):
        v = form.get(name)
        if v is None:
            return 0
        if isinstance(v, bool):
            return 1 if v else 0
        return 1 if str(v).lower() in ('1', 'true', 'on', 'yes') else 0

    gender = 1 if str(form.get("gender", "0")) == "1" else 0
    SeniorCitizen = flag('SeniorCitizen')
    Partner = flag('Partner')
    Dependents = flag('Dependents')
    PaperlessBilling = flag('PaperlessBilling')
    PhoneService = flag('PhoneService')
    MultipleLines = 1 if (flag('MultipleLines') and PhoneService) else 0

    MonthlyCharges = float(form.get("MonthlyCharges", 0) or 0)
    Tenure = int(float(form.get("Tenure", 0) or 0))
    TotalCharges = MonthlyCharges * Tenure

    internet = str(form.get("InternetService", "0"))
    InternetService_Fiberoptic = 1 if internet == "2" else 0
    InternetService_No = 1 if internet == "0" else 0
    has_internet = (InternetService_No == 0)

    OnlineSecurity = 1 if (flag('OnlineSecurity') and has_internet) else 0
    OnlineBackup = 1 if (flag('OnlineBackup') and has_internet) else 0
    DeviceProtection = 1 if (flag('DeviceProtection') and has_internet) else 0
    TechSupport = 1 if (flag('TechSupport') and has_internet) else 0
    StreamingTV = 1 if (flag('StreamingTV') and has_internet) else 0
    StreamingMovies = 1 if (flag('StreamingMovies') and has_internet) else 0

    contract = str(form.get("Contract", "0"))
    Contract_Oneyear = 1 if contract == "1" else 0
    Contract_Twoyear = 1 if contract == "2" else 0

    payment = str(form.get("PaymentMethod", "0"))
    PaymentMethod_CreditCard = 1 if payment == "1" else 0
    PaymentMethod_ElectronicCheck = 1 if payment == "2" else 0
    PaymentMethod_MailedCheck = 1 if payment == "3" else 0

    features = [
        gender, SeniorCitizen, Partner, Dependents, Tenure,
        PhoneService, MultipleLines, OnlineSecurity, OnlineBackup,
        DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
        PaperlessBilling, MonthlyCharges, TotalCharges,
        InternetService_Fiberoptic, InternetService_No,
        Contract_Oneyear, Contract_Twoyear,
        PaymentMethod_CreditCard, PaymentMethod_ElectronicCheck,
        PaymentMethod_MailedCheck
    ]
    return features, MonthlyCharges, Tenure


def _fig_to_b64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    plt.close('all')
    return base64.b64encode(buf.getvalue()).decode()


def _create_gauge(probability):
    labels = ['LOW', 'MEDIUM', 'HIGH', 'EXTREME']
    colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
    N = len(labels)
    colors_r = list(reversed(colors))
    labels_r = list(reversed(labels))

    fig, ax = plt.subplots(figsize=(6, 4))
    start = np.linspace(0, 180, N + 1, endpoint=True)[:-1]
    end = np.linspace(0, 180, N + 1, endpoint=True)[1:]
    ang_range = np.c_[start, end]
    mid = start + ((end - start) / 2.0)

    for ang, c in zip(ang_range, colors_r):
        ax.add_patch(Wedge((0., 0.), .4, *ang, facecolor='w', lw=2))
        ax.add_patch(Wedge((0., 0.), .4, *ang, width=0.10, facecolor=c, lw=2, alpha=0.75))

    for m, lab in zip(mid, labels_r):
        rot = np.degrees(np.radians(m) * np.pi / np.pi - np.radians(90))
        ax.text(0.35 * np.cos(np.radians(m)), 0.35 * np.sin(np.radians(m)),
                lab, ha='center', va='center', fontsize=11, fontweight='bold', rotation=rot)

    ax.add_patch(Rectangle((-0.4, -0.1), 0.8, 0.1, facecolor='w', lw=2))
    ax.text(0, -0.05, 'Churn Probability {:.2f}'.format(probability),
            ha='center', va='center', fontsize=14, fontweight='bold')

    needle = (1 - probability) * 180
    ax.arrow(0, 0, 0.225 * np.cos(np.radians(needle)), 0.225 * np.sin(np.radians(needle)),
             width=0.04, head_width=0.09, head_length=0.1, fc='#2c3e50', ec='#2c3e50')
    ax.add_patch(Circle((0, 0), radius=0.02, facecolor='#2c3e50'))
    ax.add_patch(Circle((0, 0), radius=0.01, facecolor='white', zorder=11))

    ax.set_frame_on(False)
    ax.axes.set_xticks([])
    ax.axes.set_yticks([])
    ax.axis('equal')
    plt.tight_layout()
    return _fig_to_b64()


def _create_shap(features):
    shap_values = explainer.shap_values(np.array([features]))

    # Shape is (1, 23, 2) — take sample 0, all features, class 1 (churn)
    sv = np.array(shap_values)[0, :, 1]  # shape → (23,)

    if isinstance(explainer.expected_value, (list, np.ndarray)):
        ev = float(explainer.expected_value[1])
    else:
        ev = float(explainer.expected_value)

    # sv and COLUMNS are both 23 now ✓
    shap.force_plot(ev, sv, COLUMNS, matplotlib=True, show=False)
    return _fig_to_b64()


def _survival_curves(tenure):
    """Cumulative hazard + survival probability. Falls back to a synthetic curve
    if no survival model is loaded so the UI always renders 4 images."""
    months = np.arange(0, 73)
    if survival_model is not None:
        try:
            sf = survival_model.predict_survival_function(np.zeros((1, len(COLUMNS))))
            surv = np.array([float(sf[0](t)) for t in months])
        except Exception:
            surv = np.exp(-0.05 * months)
    else:
        surv = np.exp(-0.05 * months)
    surv = np.clip(surv, 1e-6, 1.0)
    hazard = -np.log(surv)

    # --- Cumulative hazard ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(months, hazard, color='#e74c3c', label='Hazard')
    ax.axvline(tenure, linestyle='--', color='#3498db', label='Current Position')
    ax.set_title('Cumulative Hazard Over Time')
    ax.set_xlabel('Tenure'); ax.set_ylabel('Cumulative Hazard')
    ax.legend()
    hazard_b64 = _fig_to_b64()

    # --- Survival probability ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(months, surv, color='#e74c3c', label='Survival Function')
    ax.axvline(tenure, linestyle='--', color='#3498db', label='Current Position')
    ax.set_title('Survival Probability Over Time')
    ax.set_xlabel('Tenure'); ax.set_ylabel('Survival Probability')
    ax.legend()
    survival_b64 = _fig_to_b64()

    # Lifetime value estimate = expected remaining months * monthly charges
    expected_remaining = float(np.trapz(surv[tenure:], months[tenure:])) if tenure < len(months) else 0.0
    return hazard_b64, survival_b64, expected_remaining


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Original HTML endpoint (kept for backward compatibility)."""
    features, monthly, tenure = _parse_inputs(request.form)
    prob = float(model.predict_proba([np.array(features)])[0, 1])
    return render_template('index.html',
                           prediction_text='Churn Probability: {:.1%}'.format(prob),
                           url1=_create_gauge(prob),
                           url2=_create_shap(features))


@app.route('/predict_json', methods=['POST'])
def predict_json():
    """JSON endpoint used by the Lovable edge function."""
    payload = request.get_json(silent=True) or request.form
    features, monthly, tenure = _parse_inputs(payload)

    prob = float(model.predict_proba([np.array(features)])[0, 1])
    gauge_b64 = _create_gauge(prob)
    shap_b64 = _create_shap(features)
    hazard_b64, survival_b64, expected_remaining = _survival_curves(tenure)
    ltv = round(monthly * expected_remaining, 2)

    return jsonify({
        "churn_probability": round(prob, 2),
        "lifetime_value": ltv,
        "gauge_image": gauge_b64,
        "shap_image": shap_b64,
        "hazard_image": hazard_b64,
        "survival_image": survival_b64,
    })


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)

# python3 app.py