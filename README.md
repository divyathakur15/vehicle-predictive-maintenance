# 🚗 Vehicle Health & Predictive Maintenance

An end-to-end machine learning application that predicts whether a vehicle may require maintenance based on simulated vehicle telemetry data.

The project demonstrates a complete machine learning workflow, including synthetic data generation, data preprocessing, model training, model evaluation, prediction, and deployment through an interactive Streamlit dashboard.

## 🌐 Live Demo

👉 **[Open the Live Streamlit Dashboard](https://vehicle-predictive-maintenance-5atsdzc7bjz5asnaujag6f.streamlit.app/)**

## 📌 Project Overview

This project uses machine learning to estimate vehicle maintenance risk using telemetry features such as:

- Engine temperature
- Oil pressure
- Engine RPM
- Vehicle speed
- Vibration
- Brake temperature
- Battery voltage
- Vehicle mileage
- Coolant temperature

The trained model predicts whether maintenance may be required and provides a maintenance-risk probability through an interactive Streamlit dashboard.

The dashboard also provides:

- Vehicle health score
- Current sensor readings
- Feature importance
- Model performance
- Confusion matrix
- Prediction history
- Maintenance risk trend

> **Note:** The telemetry dataset is synthetically generated for learning and demonstration purposes. It does not represent real-world vehicle sensor measurements, vehicle failure rates, or actual maintenance recommendations.

---

## 🧠 Machine Learning

Two classification models were trained and evaluated:

- Logistic Regression
- Random Forest Classifier

### Model Evaluation

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 76.65% | **98.60%** |
| Precision | 27.19% | **89.91%** |
| Recall | 78.22% | **97.03%** |
| F1 Score | 40.36% | **93.33%** |
| ROC-AUC | 85.94% | **99.78%** |

The Random Forest classifier is used by the deployed application.

These results are based on the simulated test dataset and should not be interpreted as real-world vehicle performance.

---

## 🔄 Machine Learning Workflow

~~~text
Vehicle Telemetry
        ↓
Synthetic Data Generation
        ↓
Data Preprocessing
        ↓
Train/Test Split
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Random Forest Model
        ↓
Maintenance Risk Prediction
        ↓
Streamlit Dashboard
~~~

---

## 📊 Dashboard Features

### 1. Maintenance Risk Prediction

The dashboard predicts the probability that vehicle maintenance may be required.

The predicted risk is categorized into:

- **Low Risk**
- **Moderate Risk**
- **High Risk**

### 2. Vehicle Health Score

A vehicle health score ranging from **0–100** is calculated from the predicted maintenance risk.

A higher score represents lower predicted maintenance risk.

### 3. Current Sensor Readings

The dashboard displays the telemetry values used for the current prediction:

- Engine Temperature
- Oil Pressure
- Engine RPM
- Vehicle Speed
- Vibration
- Brake Temperature
- Battery Voltage
- Vehicle Mileage
- Coolant Temperature

### 4. Feature Importance

The application visualizes the Random Forest model's feature importance values to show the relative importance of the telemetry features used by the trained model.

### 5. Model Performance

The dashboard displays the trained model's:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

### 6. Confusion Matrix

The application provides the Random Forest confusion matrix with:

- True Negatives
- False Positives
- False Negatives
- True Positives

### 7. Prediction History

Predictions made during the current application session are stored and displayed with:

- Prediction
- Maintenance Risk
- Health Score
- Risk Level
- Status

### 8. Maintenance Risk Trend

The dashboard visualizes maintenance risk across previous predictions and provides:

- Total Predictions
- Average Risk
- Highest Risk

### 9. Dark Theme

The application uses a dark theme by default to provide a cleaner dashboard experience.

Users can still change the Streamlit appearance settings if they prefer a light theme.

---

## 🛠️ Tech Stack

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Random Forest**
- **Logistic Regression**
- **Matplotlib**
- **Streamlit**
- **Joblib**

---

## 📁 Project Structure

~~~text
vehicle-predictive-maintenance/
│
├── data/
│   └── vehicle_telemetry.csv
│
├── models/
│   ├── vehicle_failure_model.pkl
│   └── model_metrics.pkl
│
├── .streamlit/
│   └── config.toml
│
├── app.py
├── generate_data.py
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
~~~

---

## ▶️ Run Locally

### 1. Clone the Repository

~~~bash
git clone https://github.com/divyathakur15/vehicle-predictive-maintenance.git
cd vehicle-predictive-maintenance
~~~

### 2. Create a Virtual Environment

~~~bash
python -m venv venv
~~~

### 3. Activate the Virtual Environment

For Windows PowerShell:

~~~powershell
venv\Scripts\Activate.ps1
~~~

### 4. Install Dependencies

~~~bash
pip install -r requirements.txt
~~~

### 5. Generate the Dataset

~~~bash
python generate_data.py
~~~

This generates the simulated vehicle telemetry dataset:

~~~text
data/vehicle_telemetry.csv
~~~

### 6. Train the Models

~~~bash
python train.py
~~~

This generates:

~~~text
models/vehicle_failure_model.pkl
models/model_metrics.pkl
~~~

### 7. Run the Streamlit Application

~~~bash
streamlit run app.py
~~~

The application will open in your default web browser.

---

## 📈 Random Forest Results

The Random Forest classifier achieved the following results on the simulated test dataset:

| Metric | Result |
|---|---:|
| Accuracy | **98.60%** |
| Precision | **89.91%** |
| Recall | **97.03%** |
| F1 Score | **93.33%** |
| ROC-AUC | **99.78%** |

The model was evaluated on **2,000 simulated test samples**.

### Confusion Matrix

| | Predicted Healthy | Predicted Maintenance |
|---|---:|---:|
| **Actual Healthy** | 1,776 | 22 |
| **Actual Maintenance** | 6 | 196 |

---

## 🚀 Deployment

The application is deployed using **Streamlit Community Cloud**.

### Live Application

👉 **[Vehicle Health & Predictive Maintenance — Live Demo](https://vehicle-predictive-maintenance-5atsdz7bjz5asnaujag6.streamlit.app/)**

The deployed dashboard allows users to modify vehicle telemetry values and generate maintenance-risk predictions interactively.

---

## 🔮 Future Improvements

Potential future improvements include:

- Real-world vehicle telemetry datasets
- Remaining Useful Life (RUL) prediction
- Time-series based failure prediction
- SHAP-based individual prediction explanations
- Automated maintenance alerts
- Historical vehicle monitoring
- Database integration
- Real-time telemetry ingestion
- Model monitoring and data-drift detection
- Integration with IoT vehicle sensors

---

## ⚠️ Disclaimer

This project is intended for **educational and demonstration purposes**.

The vehicle telemetry data is synthetically generated, and the model predictions should not be used for actual vehicle maintenance, safety, or operational decisions.

---

## 👨‍💻 Author

**Divya Thakur**

B.Tech in Artificial Intelligence & Machine Learning

- GitHub: [divyathakur15](https://github.com/divyathakur15)
- LinkedIn: [Divya Thakur](https://www.linkedin.com/in/divya-thakur-15june2004/)
