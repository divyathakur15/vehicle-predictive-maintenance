import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Vehicle Health Predictor",
    page_icon="🚗",
    layout="wide"
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("models/vehicle_failure_model.pkl")


# --------------------------------------------------
# Load Model Metrics
# --------------------------------------------------

@st.cache_resource
def load_metrics():
    return joblib.load("models/model_metrics.pkl")


model = load_model()
model_metrics = load_metrics()


# --------------------------------------------------
# Prediction History
# --------------------------------------------------

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🚗 Vehicle Health & Predictive Maintenance")

st.markdown(
    "### Machine Learning-based vehicle maintenance risk prediction"
)

st.info(
    "Enter vehicle telemetry below to estimate the probability "
    "that maintenance may be required."
)


# --------------------------------------------------
# Sidebar - Vehicle Inputs
# --------------------------------------------------

st.sidebar.header("Vehicle Telemetry")

engine_temperature = st.sidebar.slider(
    "Engine Temperature (°C)",
    min_value=60.0,
    max_value=130.0,
    value=90.0
)

oil_pressure = st.sidebar.slider(
    "Oil Pressure (bar)",
    min_value=1.0,
    max_value=6.0,
    value=3.5
)

rpm = st.sidebar.slider(
    "Engine RPM",
    min_value=800,
    max_value=5000,
    value=2500
)

vehicle_speed = st.sidebar.slider(
    "Vehicle Speed (km/h)",
    min_value=0,
    max_value=140,
    value=60
)

vibration = st.sidebar.slider(
    "Vibration (mm/s)",
    min_value=0.2,
    max_value=8.0,
    value=2.5
)

brake_temperature = st.sidebar.slider(
    "Brake Temperature (°C)",
    min_value=50.0,
    max_value=250.0,
    value=120.0
)

battery_voltage = st.sidebar.slider(
    "Battery Voltage (V)",
    min_value=10.0,
    max_value=14.0,
    value=12.5
)

mileage = st.sidebar.slider(
    "Vehicle Mileage (km)",
    min_value=1000,
    max_value=150000,
    value=60000
)

coolant_temperature = st.sidebar.slider(
    "Coolant Temperature (°C)",
    min_value=60.0,
    max_value=120.0,
    value=85.0
)


# --------------------------------------------------
# Create Input DataFrame
# --------------------------------------------------

input_data = pd.DataFrame({
    "engine_temperature": [engine_temperature],
    "oil_pressure": [oil_pressure],
    "rpm": [rpm],
    "vehicle_speed": [vehicle_speed],
    "vibration": [vibration],
    "brake_temperature": [brake_temperature],
    "battery_voltage": [battery_voltage],
    "mileage": [mileage],
    "coolant_temperature": [coolant_temperature]
})


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔍 Predict Vehicle Health", use_container_width=True):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    probability_percentage = probability * 100

    health_score = 100 - probability_percentage


    # --------------------------------------------------
    # Determine Risk Level
    # --------------------------------------------------

    if probability >= 0.70:
        risk_level = "HIGH RISK"
    elif probability >= 0.40:
        risk_level = "MODERATE RISK"
    else:
        risk_level = "LOW RISK"


    # --------------------------------------------------
    # Prediction Result
    # --------------------------------------------------

    st.subheader("Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Maintenance Risk",
            f"{probability_percentage:.1f}%"
        )

    with col2:
        st.metric(
            "Vehicle Health Score",
            f"{health_score:.1f}/100"
        )

    with col3:
        st.metric(
            "Model",
            "Random Forest"
        )


    # --------------------------------------------------
    # Maintenance Status
    # --------------------------------------------------

    if prediction == 1:
        st.error("⚠️ Maintenance Required")
    else:
        st.success("✅ Vehicle Healthy")


    # --------------------------------------------------
    # Maintenance Risk
    # --------------------------------------------------

    st.subheader("Maintenance Risk")

    gauge_col1, gauge_col2, gauge_col3 = st.columns([1, 2, 1])

    with gauge_col2:

        st.markdown(
            f"""
<div style="text-align:center;">
<div style="font-size:48px;font-weight:bold;margin-bottom:5px;">
{probability_percentage:.1f}%
</div>
<div style="font-size:18px;margin-bottom:10px;">
Maintenance Risk
</div>
</div>
""",
            unsafe_allow_html=True
        )

        st.progress(
            min(max(probability, 0.0), 1.0)
        )

        st.markdown(
            f"""
<div style="text-align:center;font-weight:bold;font-size:18px;margin-top:8px;">
{risk_level}
</div>
""",
            unsafe_allow_html=True
        )


    # --------------------------------------------------
    # Vehicle Health Score
    # --------------------------------------------------

    st.subheader("Vehicle Health")

    st.progress(
        min(max(health_score / 100, 0.0), 1.0)
    )

    if health_score >= 70:

        st.success(
            f"Good vehicle health based on the predicted "
            f"maintenance risk of {probability_percentage:.1f}%."
        )

    elif health_score >= 40:

        st.warning(
            f"Moderate vehicle health. The predicted "
            f"maintenance risk is {probability_percentage:.1f}%."
        )

    else:

        st.error(
            f"Low vehicle health score. The predicted "
            f"maintenance risk is {probability_percentage:.1f}%."
        )


    # --------------------------------------------------
    # Current Sensor Readings
    # --------------------------------------------------

    st.subheader("📡 Current Sensor Readings")

    sensor_col1, sensor_col2, sensor_col3 = st.columns(3)

    with sensor_col1:

        st.metric(
            "Engine Temperature",
            f"{engine_temperature:.1f} °C"
        )

        st.metric(
            "Oil Pressure",
            f"{oil_pressure:.2f} bar"
        )

        st.metric(
            "Engine RPM",
            f"{rpm:,}"
        )

    with sensor_col2:

        st.metric(
            "Vehicle Speed",
            f"{vehicle_speed:.0f} km/h"
        )

        st.metric(
            "Vibration",
            f"{vibration:.2f} mm/s"
        )

        st.metric(
            "Brake Temperature",
            f"{brake_temperature:.1f} °C"
        )

    with sensor_col3:

        st.metric(
            "Battery Voltage",
            f"{battery_voltage:.2f} V"
        )

        st.metric(
            "Vehicle Mileage",
            f"{mileage:,} km"
        )

        st.metric(
            "Coolant Temperature",
            f"{coolant_temperature:.1f} °C"
        )


    # --------------------------------------------------
    # Feature Importance
    # --------------------------------------------------

    st.subheader("📊 Feature Importance")

    feature_names = {
        "engine_temperature": "Engine Temperature",
        "oil_pressure": "Oil Pressure",
        "rpm": "Engine RPM",
        "vehicle_speed": "Vehicle Speed",
        "vibration": "Vibration",
        "brake_temperature": "Brake Temperature",
        "battery_voltage": "Battery Voltage",
        "mileage": "Mileage",
        "coolant_temperature": "Coolant Temperature"
    }

    importance = pd.DataFrame({
        "Feature": input_data.columns,
        "Importance": model.feature_importances_
    })

    importance["Feature"] = importance["Feature"].map(feature_names)

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    st.bar_chart(
        importance.set_index("Feature")
    )


    # --------------------------------------------------
    # Vehicle Telemetry Data
    # --------------------------------------------------

    st.subheader("Vehicle Telemetry Data")

    st.dataframe(
        input_data,
        use_container_width=True
    )


    # --------------------------------------------------
    # Model Performance
    # --------------------------------------------------

    st.markdown("---")

    st.subheader("🤖 Model Performance")

    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

    with metric_col1:
        st.metric(
            "Accuracy",
            f"{model_metrics['accuracy'] * 100:.2f}%"
        )

    with metric_col2:
        st.metric(
            "Precision",
            f"{model_metrics['precision'] * 100:.2f}%"
        )

    with metric_col3:
        st.metric(
            "Recall",
            f"{model_metrics['recall'] * 100:.2f}%"
        )

    with metric_col4:
        st.metric(
            "F1 Score",
            f"{model_metrics['f1_score'] * 100:.2f}%"
        )

    with metric_col5:
        st.metric(
            "ROC-AUC",
            f"{model_metrics['roc_auc'] * 100:.2f}%"
        )

    st.caption(
        f"Evaluation performed on {model_metrics['test_samples']:,} "
        "simulated test samples."
    )


    # --------------------------------------------------
    # Visual Confusion Matrix
    # --------------------------------------------------

    st.subheader("Confusion Matrix")

    cm_values = model_metrics["confusion_matrix"]

    cm = pd.DataFrame(
        cm_values,
        index=["Actual Healthy", "Actual Maintenance"],
        columns=["Predicted Healthy", "Predicted Maintenance"]
    )


    # Styled confusion matrix
    def highlight_cells(value):
        return (
            "background-color: #d4edda; "
            "font-weight: bold; "
            "text-align: center;"
        )


    styled_cm = cm.style.map(highlight_cells)

    st.dataframe(
        styled_cm,
        use_container_width=True
    )


    # --------------------------------------------------
    # Confusion Matrix Explanation
    # --------------------------------------------------

    tn = cm_values[0][0]
    fp = cm_values[0][1]
    fn = cm_values[1][0]
    tp = cm_values[1][1]

    cm_col1, cm_col2, cm_col3, cm_col4 = st.columns(4)

    with cm_col1:
        st.metric(
            "True Negatives",
            f"{tn:,}"
        )

    with cm_col2:
        st.metric(
            "False Positives",
            f"{fp:,}"
        )

    with cm_col3:
        st.metric(
            "False Negatives",
            f"{fn:,}"
        )

    with cm_col4:
        st.metric(
            "True Positives",
            f"{tp:,}"
        )


    # --------------------------------------------------
    # Save Prediction to History
    # --------------------------------------------------

    prediction_record = {
        "Prediction": len(st.session_state.prediction_history) + 1,
        "Maintenance Risk": round(probability_percentage, 1),
        "Health Score": round(health_score, 1),
        "Risk Level": risk_level,
        "Status": (
            "Maintenance Required"
            if prediction == 1
            else "Vehicle Healthy"
        )
    }

    st.session_state.prediction_history.append(
        prediction_record
    )


# --------------------------------------------------
# Prediction History
# --------------------------------------------------

if len(st.session_state.prediction_history) > 0:

    st.markdown("---")

    st.subheader("📈 Prediction History")

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------
    # Risk Trend
    # --------------------------------------------------

    st.subheader("Maintenance Risk Trend")

    risk_chart = history_df[
        ["Prediction", "Maintenance Risk"]
    ].set_index("Prediction")

    st.line_chart(
        risk_chart
    )


    # --------------------------------------------------
    # History Summary
    # --------------------------------------------------

    history_col1, history_col2, history_col3 = st.columns(3)

    with history_col1:
        st.metric(
            "Predictions",
            len(history_df)
        )

    with history_col2:
        st.metric(
            "Average Risk",
            f"{history_df['Maintenance Risk'].mean():.1f}%"
        )

    with history_col3:
        st.metric(
            "Highest Risk",
            f"{history_df['Maintenance Risk'].max():.1f}%"
        )


    # --------------------------------------------------
    # Clear History
    # --------------------------------------------------

    if st.button("🗑️ Clear Prediction History"):

        st.session_state.prediction_history = []

        st.rerun()


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("---")

st.caption(
    "This project uses simulated vehicle telemetry data "
    "for machine learning demonstration purposes."
)