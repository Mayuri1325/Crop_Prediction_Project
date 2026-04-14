import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

print("Training advanced model...")

# Load dataset
df = pd.read_csv("crop.csv")

# Select columns
df = df[["State_Name", "Season", "Crop", "Area", "Production"]]

# Remove missing values
df = df.dropna()

# Encode categorical data
le_state = LabelEncoder()
le_season = LabelEncoder()
le_crop = LabelEncoder()

df["State_Name"] = le_state.fit_transform(df["State_Name"])
df["Season"] = le_season.fit_transform(df["Season"])
df["Crop"] = le_crop.fit_transform(df["Crop"])

# Features & target
X = df[["State_Name", "Season", "Crop", "Area"]]
y = df["Production"]

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# Save everything
joblib.dump(model, "model.pkl")
joblib.dump(le_state, "state_encoder.pkl")
joblib.dump(le_season, "season_encoder.pkl")
joblib.dump(le_crop, "crop_encoder.pkl")

print("Advanced model trained successfully!")