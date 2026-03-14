import io
import base64
from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import joblib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle
import shap
shap.initjs()

app = Flask(__name__)

# Load the trained Random Forest model
model = pickle.load(open('./models/model_rfc.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle form submission and return churn prediction with SHAP explanation.
    Input features: gender, SeniorCitizen, Partner, Dependents, Tenure,
                    PhoneService, MultipleLines, OnlineSecurity, OnlineBackup,
                    DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
                    PaperlessBilling, MonthlyCharges, InternetService,
                    Contract, PaymentMethod
    """

    # --- Binary features (checkboxes) ---
    gender = 1 if request.form.get("gender") == "1" else 0
    SeniorCitizen = 1 if 'SeniorCitizen' in request.form else 0
    Partner = 1 if 'Partner' in request.form else 0
    Dependents = 1 if 'Dependents' in request.form else 0
    PaperlessBilling = 1 if 'PaperlessBilling' in request.form else 0
    PhoneService = 1 if 'PhoneService' in request.form else 0
    MultipleLines = 1 if ('MultipleLines' in request.form and PhoneService == 1) else 0

    # --- Numeric features ---
    MonthlyCharges = float(request.form["MonthlyCharges"])
    Tenure = int(request.form["Tenure"])
    TotalCharges = MonthlyCharges * Tenure

    # --- Internet Service (one-hot encoded) ---
    internet_service = request.form.get("InternetService", "0")
    InternetService_Fiberoptic = 1 if internet_service == "2" else 0
    InternetService_No = 1 if internet_service == "0" else 0

    # Internet-dependent services
    has_internet = (InternetService_No == 0)
    OnlineSecurity = 1 if ('OnlineSecurity' in request.form and has_internet) else 0
    OnlineBackup = 1 if ('OnlineBackup' in request.form and has_internet) else 0
    DeviceProtection = 1 if ('DeviceProtection' in request.form and has_internet) else 0
    TechSupport = 1 if ('TechSupport' in request.form and has_internet) else 0
    StreamingTV = 1 if ('StreamingTV' in request.form and has_internet) else 0
    StreamingMovies = 1 if ('StreamingMovies' in request.form and has_internet) else 0

    # --- Contract type (one-hot encoded) ---
    contract = request.form.get("Contract", "0")
    Contract_Oneyear = 1 if contract == "1" else 0
    Contract_Twoyear = 1 if contract == "2" else 0

    # --- Payment method (one-hot encoded) ---
    payment = request.form.get("PaymentMethod", "0")
    PaymentMethod_CreditCard = 1 if payment == "1" else 0
    PaymentMethod_ElectronicCheck = 1 if payment == "2" else 0
    PaymentMethod_MailedCheck = 1 if payment == "3" else 0

    # --- Assemble feature vector ---
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

    columns = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
        'InternetService_Fiber optic', 'InternetService_No',
        'Contract_One year', 'Contract_Two year',
        'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
    ]

    final_features = [np.array(features)]

    # --- Model prediction ---
    prediction = model.predict_proba(final_features)
    churn_probability = prediction[0, 1]

    # --- SHAP Explanation ---
    explainer = joblib.load(filename="./models/explainer_rfc.bz2")
    shap_values = explainer.shap_values(np.array(final_features))

    shap_img = io.BytesIO()
    shap.force_plot(
        explainer.expected_value[1],
        shap_values[1],
        columns,
        matplotlib=True,
        show=False
    ).savefig(shap_img, bbox_inches="tight", format='png')
    shap_img.seek(0)
    shap_url = base64.b64encode(shap_img.getvalue()).decode()
    plt.close('all')

    # --- Churn Probability Gauge ---
    gauge_url = _create_gauge(probability=churn_probability)

    return render_template(
        'index.html',
        prediction_text='Churn Probability: {:.1%}'.format(churn_probability),
        churn_label=_get_risk_label(churn_probability),
        url1=gauge_url,
        url2=shap_url
    )


def _get_risk_label(probability):
    """Return a risk category label based on churn probability."""
    if probability < 0.25:
        return "LOW RISK"
    elif probability < 0.50:
        return "MEDIUM RISK"
    elif probability < 0.75:
        return "HIGH RISK"
    else:
        return "EXTREME RISK"


def _create_gauge(probability=0.5):
    """
    Create a semicircular gauge chart showing churn probability.

    Args:
        probability: float between 0 and 1 representing churn probability

    Returns:
        base64-encoded PNG image string
    """
    labels = ['LOW', 'MEDIUM', 'HIGH', 'EXTREME']
    colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']

    N = len(labels)
    colors_reversed = list(reversed(colors))
    labels_reversed = list(reversed(labels))

    gauge_img = io.BytesIO()
    fig, ax = plt.subplots(figsize=(6, 4))

    # Create angle ranges for each segment
    start = np.linspace(0, 180, N + 1, endpoint=True)[:-1]
    end = np.linspace(0, 180, N + 1, endpoint=True)[1:]
    ang_range = np.c_[start, end]
    mid_points = start + ((end - start) / 2.0)

    # Draw gauge wedges
    for ang, c in zip(ang_range, colors_reversed):
        ax.add_patch(Wedge((0., 0.), .4, *ang, facecolor='w', lw=2))
        ax.add_patch(Wedge((0., 0.), .4, *ang, width=0.10, facecolor=c, lw=2, alpha=0.75))

    # Add labels on gauge
    for mid, lab in zip(mid_points, labels_reversed):
        rotation = np.degrees(np.radians(mid) * np.pi / np.pi - np.radians(90))
        ax.text(
            0.35 * np.cos(np.radians(mid)),
            0.35 * np.sin(np.radians(mid)),
            lab,
            ha='center', va='center', fontsize=11, fontweight='bold',
            rotation=rotation
        )

    # Bottom rectangle and probability text
    ax.add_patch(Rectangle((-0.4, -0.1), 0.8, 0.1, facecolor='w', lw=2))
    ax.text(
        0, -0.05,
        'Churn Probability: {:.1%}'.format(probability),
        ha='center', va='center', fontsize=14, fontweight='bold'
    )

    # Draw needle
    needle_angle = (1 - probability) * 180
    ax.arrow(
        0, 0,
        0.225 * np.cos(np.radians(needle_angle)),
        0.225 * np.sin(np.radians(needle_angle)),
        width=0.04, head_width=0.09, head_length=0.1,
        fc='#2c3e50', ec='#2c3e50'
    )

    ax.add_patch(Circle((0, 0), radius=0.02, facecolor='#2c3e50'))
    ax.add_patch(Circle((0, 0), radius=0.01, facecolor='white', zorder=11))

    ax.set_frame_on(False)
    ax.axes.set_xticks([])
    ax.axes.set_yticks([])
    ax.axis('equal')
    plt.tight_layout()

    plt.savefig(gauge_img, format='png', dpi=100)
    gauge_img.seek(0)
    url = base64.b64encode(gauge_img.getvalue()).decode()
    plt.close('all')
    return url


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
