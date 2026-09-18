import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Vehicle Health Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# Global Styling
# --------------------------------------------------

st.markdown(
    """
<style>
footer {visibility: hidden;}

.app-header {
    background: linear-gradient(135deg, #161B22 0%, #0E1117 100%);
    border: 1px solid #30363D;
    border-radius: 14px;
    padding: 22px 28px;
    margin-bottom: 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
}
.app-header h1 { margin: 0; font-size: 26px; }
.app-header p { margin: 4px 0 0 0; color: #8b949e; font-size: 14px; }
.status-badge {
    background: rgba(34,197,94,0.15);
    color: #22c55e;
    border: 1px solid rgba(34,197,94,0.4);
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    white-space: nowrap;
}

.kpi-grid, .sensor-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
}
.kpi-card, .sensor-card {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 12px;
    padding: 14px 16px;
}
.kpi-card .kpi-label, .sensor-card .s-label {
    color: #8b949e;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}
.kpi-card .kpi-value {
    font-size: 22px;
    font-weight: 700;
    margin-top: 4px;
}
.sensor-card .s-value {
    font-size: 19px;
    font-weight: 700;
    margin-top: 2px;
}
.sensor-card.flagged {
    border-color: rgba(239,68,68,0.55);
    background: rgba(239,68,68,0.07);
}
.sensor-card.flagged .s-value { color: #ef4444; }

.risk-pill {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 14px;
    margin-top: 6px;
}
.risk-low { background: rgba(34,197,94,0.15); color: #22c55e; border: 1px solid rgba(34,197,94,0.4); }
.risk-moderate { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.4); }
.risk-high { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.4); }
</style>
""",
    unsafe_allow_html=True
)


# --------------------------------------------------
# Load Model + Metrics
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("models/vehicle_failure_model.pkl")


@st.cache_resource
def load_metrics():
    return joblib.load("models/model_metrics.pkl")


model = load_model()
metrics = load_metrics()

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    f"""
<div class="app-header">
    <div>
        <h1>🚗 Vehicle Health Monitor</h1>
        <p>Predictive Maintenance — Machine Learning Diagnostics</p>
    </div>
    <div class="status-badge">● MODEL ONLINE</div>
</div>
""",
    unsafe_allow_html=True
)


# --------------------------------------------------
# Model-level KPI strip (always visible)
# --------------------------------------------------

st.markdown(
    f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">Model Accuracy</div>
        <div class="kpi-value">{metrics['accuracy'] * 100:.1f}%</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">ROC-AUC</div>
        <div class="kpi-value">{metrics['roc_auc'] * 100:.1f}%</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">F1 Score</div>
        <div class="kpi-value">{metrics['f1_score'] * 100:.1f}%</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Algorithm</div>
        <div class="kpi-value" style="font-size:18px;">Random Forest</div>
    </div>
</div>
""",
    unsafe_allow_html=True
)


# --------------------------------------------------
# Sidebar - Vehicle Inputs
# --------------------------------------------------

st.sidebar.header("🔧 Vehicle Telemetry")

with st.sidebar.expander("Engine & Powertrain", expanded=True):
    engine_temperature = st.slider("Engine Temperature (°C)", 60.0, 130.0, 90.0)
    oil_pressure = st.slider("Oil Pressure (bar)", 1.0, 6.0, 3.5)
    rpm = st.slider("Engine RPM", 800, 5000, 2500)
    coolant_temperature = st.slider("Coolant Temperature (°C)", 60.0, 120.0, 85.0)

with st.sidebar.expander("Chassis & Braking", expanded=True):
    vehicle_speed = st.slider("Vehicle Speed (km/h)", 0, 140, 60)
    vibration = st.slider("Vibration (mm/s)", 0.2, 8.0, 2.5)
    brake_temperature = st.slider("Brake Temperature (°C)", 50.0, 250.0, 120.0)

with st.sidebar.expander("Electrical & Usage", expanded=True):
    battery_voltage = st.slider("Battery Voltage (V)", 10.0, 14.0, 12.5)
    mileage = st.slider("Vehicle Mileage (km)", 1000, 150000, 60000)

analyze_clicked = st.sidebar.button("⚡ Analyze Vehicle", use_container_width=True, type="primary")


# --------------------------------------------------
# Input DataFrame (column order must match training data)
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

# Thresholds mirrored from generate_data.py's synthetic risk rule —
# used only to flag sensor cards, not to override the model's prediction.
sensor_flags = {
    "🌡️ Engine Temperature": (f"{engine_temperature:.1f} °C", engine_temperature > 105),
    "🛢️ Oil Pressure": (f"{oil_pressure:.2f} bar", oil_pressure < 2.5),
    "⚙️ Engine RPM": (f"{rpm:,}", rpm > 3800),
    "🏎️ Vehicle Speed": (f"{vehicle_speed:.0f} km/h", False),
    "📳 Vibration": (f"{vibration:.2f} mm/s", vibration > 4.5),
    "🛑 Brake Temperature": (f"{brake_temperature:.1f} °C", brake_temperature > 180),
    "🔋 Battery Voltage": (f"{battery_voltage:.2f} V", battery_voltage < 11.5),
    "📏 Mileage": (f"{mileage:,} km", mileage > 100000),
    "❄️ Coolant Temperature": (f"{coolant_temperature:.1f} °C", coolant_temperature > 100),
}


# --------------------------------------------------
# Run Prediction
# --------------------------------------------------

if analyze_clicked:
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    probability_percentage = probability * 100
    health_score = 100 - probability_percentage

    if probability >= 0.70:
        risk_level, risk_class, risk_color = "HIGH RISK", "risk-high", "#ef4444"
    elif probability >= 0.40:
        risk_level, risk_class, risk_color = "MODERATE RISK", "risk-moderate", "#f59e0b"
    else:
        risk_level, risk_class, risk_color = "LOW RISK", "risk-low", "#22c55e"

    st.session_state.last_result = {
        "prediction": int(prediction),
        "probability_percentage": probability_percentage,
        "health_score": health_score,
        "risk_level": risk_level,
        "risk_class": risk_class,
        "risk_color": risk_color,
        "sensor_flags": sensor_flags,
        "input_data": input_data,
    }

    st.session_state.prediction_history.append({
        "Prediction": len(st.session_state.prediction_history) + 1,
        "Maintenance Risk": round(probability_percentage, 1),
        "Health Score": round(health_score, 1),
        "Risk Level": risk_level,
        "Status": "Maintenance Required" if prediction == 1 else "Vehicle Healthy"
    })


# --------------------------------------------------
# Tabs
# --------------------------------------------------

tab_diagnostics, tab_analytics, tab_history = st.tabs(
    ["🔍 Diagnostics", "📊 Model Analytics", "📈 History"]
)


# ==================================================
# TAB 1 — DIAGNOSTICS
# ==================================================

with tab_diagnostics:

    result = st.session_state.last_result

    if result is None:
        st.info("Set the vehicle telemetry values in the sidebar, then click **⚡ Analyze Vehicle** to run a prediction.")
    else:
        status_col, gauge_col, health_col = st.columns([1, 1.3, 1.3])

        with status_col:
            st.markdown("#### Status")
            if result["prediction"] == 1:
                st.error("⚠️ Maintenance Required")
            else:
                st.success("✅ Vehicle Healthy")
            st.markdown(
                f'<span class="risk-pill {result["risk_class"]}">{result["risk_level"]}</span>',
                unsafe_allow_html=True
            )
            st.caption("Model: Random Forest Classifier")

        with gauge_col:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result["probability_percentage"],
                number={"suffix": "%", "font": {"size": 34}},
                title={"text": "Maintenance Risk", "font": {"size": 14}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": result["risk_color"]},
                    "steps": [
                        {"range": [0, 40], "color": "rgba(34,197,94,0.20)"},
                        {"range": [40, 70], "color": "rgba(245,158,11,0.20)"},
                        {"range": [70, 100], "color": "rgba(239,68,68,0.20)"},
                    ],
                }
            ))
            fig_gauge.update_layout(
                height=240, margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)", font={"color": "#F3F4F6"}
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with health_col:
            fig_health = go.Figure(data=[go.Pie(
                labels=["Health", "Gap"],
                values=[result["health_score"], 100 - result["health_score"]],
                hole=0.72,
                marker_colors=["#3b82f6", "#30363D"],
                textinfo="none",
                sort=False
            )])
            fig_health.update_layout(
                height=240, margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
                annotations=[
                    dict(text=f"{result['health_score']:.0f}", x=0.5, y=0.52,
                         font_size=30, font_color="#F3F4F6", showarrow=False),
                    dict(text="HEALTH SCORE", x=0.5, y=0.36,
                         font_size=10, font_color="#8b949e", showarrow=False),
                ]
            )
            st.plotly_chart(fig_health, use_container_width=True)

        st.markdown("#### 📡 Sensor Telemetry")
        cards_html = ""
        flagged_names = []
        for label, (value, flagged) in result["sensor_flags"].items():
            css_class = "sensor-card flagged" if flagged else "sensor-card"
            cards_html += (
                f'<div class="{css_class}"><div class="s-label">{label}</div>'
                f'<div class="s-value">{value}</div></div>'
            )
            if flagged:
                flagged_names.append(label)
        st.markdown(f'<div class="sensor-grid">{cards_html}</div>', unsafe_allow_html=True)

        if flagged_names:
            st.warning("Outside normal range: " + ", ".join(f.split(" ", 1)[1] for f in flagged_names))
        else:
            st.success("All 9 telemetry parameters are within normal operating range.")

        with st.expander("📊 Feature Importance (this model)"):
            importance_df = pd.DataFrame({
                "Feature": [feature_names[c] for c in result["input_data"].columns],
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=True)

            fig_imp = px.bar(
                importance_df, x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale=["#30363D", "#3b82f6"]
            )
            fig_imp.update_layout(
                height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#F3F4F6"}, coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(fig_imp, use_container_width=True)

        with st.expander("Raw input sent to the model"):
            st.dataframe(result["input_data"], use_container_width=True)


# ==================================================
# TAB 2 — MODEL ANALYTICS
# ==================================================

with tab_analytics:

    st.markdown("#### 🤖 Evaluation Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    m2.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
    m3.metric("Recall", f"{metrics['recall'] * 100:.2f}%")
    m4.metric("F1 Score", f"{metrics['f1_score'] * 100:.2f}%")
    m5.metric("ROC-AUC", f"{metrics['roc_auc'] * 100:.2f}%")
    st.caption(
        f"Evaluated on {metrics['test_samples']:,} simulated test samples"
        + (f" · trained {metrics['trained_at']}" if "trained_at" in metrics else "")
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Confusion Matrix**")
        cm_values = metrics["confusion_matrix"]
        fig_cm = px.imshow(
            cm_values, text_auto=True, color_continuous_scale="Blues",
            x=["Pred: Healthy", "Pred: Maintenance"],
            y=["Actual: Healthy", "Actual: Maintenance"]
        )
        fig_cm.update_layout(
            height=340, paper_bgcolor="rgba(0,0,0,0)", font={"color": "#F3F4F6"},
            coloraxis_showscale=False, margin=dict(l=10, r=10, t=20, b=10)
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        tn, fp = cm_values[0][0], cm_values[0][1]
        fn, tp = cm_values[1][0], cm_values[1][1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("TN", f"{tn:,}")
        c2.metric("FP", f"{fp:,}")
        c3.metric("FN", f"{fn:,}")
        c4.metric("TP", f"{tp:,}")

    with chart_col2:
        st.markdown("**ROC Curve**")
        roc_data = metrics.get("roc_curve")
        if roc_data:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=roc_data["fpr"], y=roc_data["tpr"], mode="lines", fill="tozeroy",
                name="Random Forest", line=dict(color="#3b82f6", width=3)
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                line=dict(color="#8b949e", width=1, dash="dash")
            ))
            fig_roc.update_layout(
                height=340, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#F3F4F6"}, legend=dict(orientation="h", y=-0.25),
                margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.info(
                "ROC curve data isn't in model_metrics.pkl yet. Re-run `python train.py` "
                "with the updated script to generate it."
            )


# ==================================================
# TAB 3 — HISTORY
# ==================================================

with tab_history:

    history = st.session_state.prediction_history

    if not history:
        st.info("No predictions yet this session. Run an analysis from the **Diagnostics** tab.")
    else:
        history_df = pd.DataFrame(history)

        st.markdown("#### Maintenance Risk Trend")
        fig_trend = px.line(history_df, x="Prediction", y="Maintenance Risk", markers=True)
        fig_trend.update_traces(line_color="#3b82f6")
        fig_trend.update_layout(
            height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#F3F4F6"}, margin=dict(l=10, r=10, t=20, b=10)
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        h1, h2, h3 = st.columns(3)
        h1.metric("Predictions", len(history_df))
        h2.metric("Average Risk", f"{history_df['Maintenance Risk'].mean():.1f}%")
        h3.metric("Highest Risk", f"{history_df['Maintenance Risk'].max():.1f}%")

        st.markdown("#### Log")
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        if st.button("🗑️ Clear Prediction History"):
            st.session_state.prediction_history = []
            st.session_state.last_result = None
            st.rerun()


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("---")
st.caption(
    "This project uses simulated vehicle telemetry data for machine learning demonstration purposes."
)