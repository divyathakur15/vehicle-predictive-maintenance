import pandas as pd
import joblib

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
    classification_report
)

# -----------------------------
# 1. Load Dataset
# -----------------------------

df = pd.read_csv("data/vehicle_telemetry.csv")

print("Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")

# -----------------------------
# 2. Separate Features & Target
# -----------------------------

X = df.drop("maintenance_required", axis=1)
y = df["maintenance_required"]

# -----------------------------
# 3. Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# -----------------------------
# 4. Logistic Regression
# -----------------------------

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        class_weight="balanced",
        random_state=42,
        max_iter=1000
    ))
])

logistic_model.fit(X_train, y_train)

logistic_pred = logistic_model.predict(X_test)
logistic_prob = logistic_model.predict_proba(X_test)[:, 1]

# -----------------------------
# 5. Random Forest
# -----------------------------

random_forest = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

random_forest.fit(X_train, y_train)

rf_pred = random_forest.predict(X_test)
rf_prob = random_forest.predict_proba(X_test)[:, 1]

# -----------------------------
# 6. Evaluation Function
# -----------------------------

def evaluate_model(name, y_true, predictions, probabilities):

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print(f"Accuracy : {accuracy_score(y_true, predictions):.4f}")
    print(f"Precision: {precision_score(y_true, predictions):.4f}")
    print(f"Recall   : {recall_score(y_true, predictions):.4f}")
    print(f"F1 Score : {f1_score(y_true, predictions):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_true, probabilities):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, predictions))


# -----------------------------
# 7. Evaluate Both Models
# -----------------------------

evaluate_model(
    "Logistic Regression",
    y_test,
    logistic_pred,
    logistic_prob
)

evaluate_model(
    "Random Forest",
    y_test,
    rf_pred,
    rf_prob
)

# -----------------------------
# 8. Save Random Forest Model
# -----------------------------

joblib.dump(
    random_forest,
    "models/vehicle_failure_model.pkl"
)

print("\n" + "=" * 50)
print("MODEL SAVED SUCCESSFULLY!")
print("=" * 50)

print("Location:")
print("models/vehicle_failure_model.pkl")