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


model = load_model()


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

        if prediction == 1:
            st.error("⚠️ Maintenance Required")
        else:
            st.success("✅ Vehicle Healthy")

    with col3:
        st.metric(
            "Model",
            "Random Forest"
        )


    # --------------------------------------------------
    # Progress Bar
    # --------------------------------------------------

    st.progress(
        min(probability, 1.0)
    )


    if probability >= 0.70:

        st.warning(
            "High maintenance risk detected. "
            "The vehicle telemetry indicates conditions "
            "associated with increased maintenance requirements."
        )

    elif probability >= 0.40:

        st.warning(
            "Moderate maintenance risk detected. "
            "Consider inspecting the vehicle telemetry."
        )

    else:

        st.success(
            "Low maintenance risk detected based on "
            "the provided telemetry."
        )


    # --------------------------------------------------
    # Feature Importance
    # --------------------------------------------------

    st.subheader("📊 Feature Importance")

    importance = pd.DataFrame({
        "Feature": input_data.columns,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    st.bar_chart(
        importance.set_index("Feature")
    )


    # --------------------------------------------------
    # Input Summary
    # --------------------------------------------------

    st.subheader("Vehicle Telemetry")

    st.dataframe(
        input_data,
        use_container_width=True
    )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("---")

st.caption(
    "This project uses simulated vehicle telemetry data "
    "for machine learning demonstration purposes."
)