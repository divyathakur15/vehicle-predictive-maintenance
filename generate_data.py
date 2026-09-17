import numpy as np
import pandas as pd

np.random.seed(42)

n = 10000

data = pd.DataFrame({
    "engine_temperature": np.random.normal(90, 12, n).clip(60, 130),
    "oil_pressure": np.random.normal(3.5, 0.8, n).clip(1, 6),
    "rpm": np.random.normal(2500, 800, n).clip(800, 5000),
    "vehicle_speed": np.random.normal(60, 25, n).clip(0, 140),
    "vibration": np.random.normal(2.5, 1.2, n).clip(0.2, 8),
    "brake_temperature": np.random.normal(120, 35, n).clip(50, 250),
    "battery_voltage": np.random.normal(12.5, 0.7, n).clip(10, 14),
    "mileage": np.random.normal(60000, 30000, n).clip(1000, 150000),
    "coolant_temperature": np.random.normal(85, 10, n).clip(60, 120)
})

# Create a maintenance-risk score
risk_score = (
    (data["engine_temperature"] > 105) * 2
    + (data["oil_pressure"] < 2.5) * 2
    + (data["rpm"] > 3800) * 1
    + (data["vibration"] > 4.5) * 3
    + (data["brake_temperature"] > 180) * 2
    + (data["battery_voltage"] < 11.5) * 2
    + (data["mileage"] > 100000) * 2
    + (data["coolant_temperature"] > 100) * 2
)

# Add some randomness so the model doesn't learn a perfect rule
noise = np.random.binomial(1, 0.08, n)

data["maintenance_required"] = (
    ((risk_score + noise) >= 4).astype(int)
)

data.to_csv("data/vehicle_telemetry.csv", index=False)

print("Dataset generated successfully!")
print(f"Rows: {len(data)}")
print("\nClass distribution:")
print(data["maintenance_required"].value_counts())