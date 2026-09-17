# Vehicle Health & Predictive Maintenance

An end-to-end machine learning project that predicts whether a vehicle may require maintenance based on simulated vehicle telemetry data.

## Project Overview

This project demonstrates a complete machine learning workflow for predictive maintenance in an automotive context.

The system uses vehicle telemetry such as:

- Engine temperature
- Oil pressure
- RPM
- Vehicle speed
- Vibration
- Brake temperature
- Battery voltage
- Mileage
- Coolant temperature

The model predicts whether maintenance may be required and provides a maintenance-risk probability through an interactive Streamlit dashboard.

> **Note:** The telemetry dataset is synthetically generated for learning and demonstration purposes. The model results should not be interpreted as real-world vehicle reliability or failure rates.

## Machine Learning Workflow

1. Generate simulated vehicle telemetry data
2. Perform data preprocessing
3. Split data into training and testing sets
4. Train and compare classification models
5. Evaluate model performance
6. Save the trained model
7. Deploy an interactive Streamlit application

## Models

Two classification models were evaluated:

- Logistic Regression
- Random Forest Classifier

### Random Forest Performance

| Metric | Score |
|---|---:|
| Accuracy | 98.60% |
| Precision | 89.91% |
| Recall | 97.03% |
| F1-Score | 93.33% |
| ROC-AUC | 99.78% |

These results are based on the simulated test dataset.

## Streamlit Dashboard

The application allows users to enter vehicle telemetry values and receive:

- Maintenance risk probability
- Vehicle health status
- Maintenance warning
- Feature importance visualization
- Input telemetry summary

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- Matplotlib
- Streamlit
- Joblib

## Project Structure

```text
vehicle-predictive-maintenance/
│
├── data/
│   └── vehicle_telemetry.csv
│
├── models/
│   └── vehicle_failure_model.pkl
│
├── app.py
├── generate_data.py
├── train.py
├── requirements.txt
├── README.md
└── .gitignore