import pandas as pd
import joblib
from datetime import datetime, timezone

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

data = pd.read_csv("data/vehicle_telemetry.csv")

print("\nDataset Shape:", data.shape)

print("\nTarget Distribution:")
print(data["maintenance_required"].value_counts())


# --------------------------------------------------
# Features and Target
# --------------------------------------------------

X = data.drop("maintenance_required", axis=1)
y = data["maintenance_required"]


# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# Logistic Regression
# --------------------------------------------------

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(
            class_weight="balanced",
            random_state=42,
            max_iter=1000
        )
    )
])

logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_test)
logistic_probabilities = logistic_model.predict_proba(X_test)[:, 1]


logistic_accuracy = accuracy_score(y_test, logistic_predictions)
logistic_precision = precision_score(y_test, logistic_predictions, zero_division=0)
logistic_recall = recall_score(y_test, logistic_predictions, zero_division=0)
logistic_f1 = f1_score(y_test, logistic_predictions, zero_division=0)
logistic_roc_auc = roc_auc_score(y_test, logistic_probabilities)


# --------------------------------------------------
# Logistic Regression Results
# --------------------------------------------------

print("\n" + "=" * 50)
print("LOGISTIC REGRESSION")
print("=" * 50)

print(f"Accuracy  : {logistic_accuracy:.4f}")
print(f"Precision : {logistic_precision:.4f}")
print(f"Recall    : {logistic_recall:.4f}")
print(f"F1 Score  : {logistic_f1:.4f}")
print(f"ROC-AUC   : {logistic_roc_auc:.4f}")


# --------------------------------------------------
# Random Forest
# --------------------------------------------------

random_forest_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

random_forest_model.fit(X_train, y_train)

rf_predictions = random_forest_model.predict(X_test)
rf_probabilities = random_forest_model.predict_proba(X_test)[:, 1]


rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_precision = precision_score(y_test, rf_predictions, zero_division=0)
rf_recall = recall_score(y_test, rf_predictions, zero_division=0)
rf_f1 = f1_score(y_test, rf_predictions, zero_division=0)
rf_roc_auc = roc_auc_score(y_test, rf_probabilities)


# --------------------------------------------------
# Random Forest Results
# --------------------------------------------------

print("\n" + "=" * 50)
print("RANDOM FOREST")
print("=" * 50)

print(f"Accuracy  : {rf_accuracy:.4f}")
print(f"Precision : {rf_precision:.4f}")
print(f"Recall    : {rf_recall:.4f}")
print(f"F1 Score  : {rf_f1:.4f}")
print(f"ROC-AUC   : {rf_roc_auc:.4f}")


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

rf_confusion_matrix = confusion_matrix(y_test, rf_predictions)

print("\nRandom Forest Confusion Matrix:")
print(rf_confusion_matrix)


# --------------------------------------------------
# ROC Curve (for the dashboard's ROC chart)
# --------------------------------------------------

fpr, tpr, _ = roc_curve(y_test, rf_probabilities)

# Downsample to ~100 points so the .pkl stays small and the chart stays smooth
if len(fpr) > 100:
    step = len(fpr) // 100
    fpr = fpr[::step]
    tpr = tpr[::step]

print(f"\nROC curve points captured: {len(fpr)}")


# --------------------------------------------------
# Save Random Forest Model
# --------------------------------------------------

joblib.dump(
    random_forest_model,
    "models/vehicle_failure_model.pkl"
)

print("\nRandom Forest model saved to models/vehicle_failure_model.pkl")


# --------------------------------------------------
# Save Model Evaluation Results
# --------------------------------------------------

model_metrics = {
    "model": "Random Forest",

    "accuracy": rf_accuracy,
    "precision": rf_precision,
    "recall": rf_recall,
    "f1_score": rf_f1,
    "roc_auc": rf_roc_auc,

    "confusion_matrix": rf_confusion_matrix.tolist(),

    "roc_curve": {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist()
    },

    "test_samples": len(y_test),
    "features": list(X.columns),
    "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
}


joblib.dump(
    model_metrics,
    "models/model_metrics.pkl"
)

print("Model evaluation results saved to models/model_metrics.pkl")


# --------------------------------------------------
# Final Summary
# --------------------------------------------------

print("\n" + "=" * 50)
print("TRAINING COMPLETE")
print("=" * 50)

print(f"Random Forest Accuracy : {rf_accuracy * 100:.2f}%")
print(f"Random Forest F1 Score : {rf_f1 * 100:.2f}%")
print(f"Random Forest ROC-AUC  : {rf_roc_auc * 100:.2f}%")

print("\nFiles created:")
print("1. models/vehicle_failure_model.pkl")
print("2. models/model_metrics.pkl")