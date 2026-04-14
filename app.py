from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model and encoders
model = joblib.load("model.pkl")
le_state = joblib.load("state_encoder.pkl")
le_season = joblib.load("season_encoder.pkl")
le_crop = joblib.load("crop_encoder.pkl")


# 🏠 Home Page (loads dropdown values)
@app.route("/")
def home():
    states = list(le_state.classes_)
    seasons = list(le_season.classes_)
    crops = list(le_crop.classes_)

    return render_template(
        "index.html",
        states=states,
        seasons=seasons,
        crops=crops
    )


# 🔮 Prediction
@app.route("/predict", methods=["POST"])
def predict():
    state = request.form["state"]
    season = request.form["season"]
    crop = request.form["crop"]
    area = float(request.form["area"])

    # Convert text → numbers
    state_val = le_state.transform([state])[0]
    season_val = le_season.transform([season])[0]
    crop_val = le_crop.transform([crop])[0]

    # Predict
    prediction = model.predict([[state_val, season_val, crop_val, area]])

    # Send dropdown values again (IMPORTANT)
    states = list(le_state.classes_)
    seasons = list(le_season.classes_)
    crops = list(le_crop.classes_)

    return render_template(
        "index.html",
        result=round(prediction[0], 2),
        states=states,
        seasons=seasons,
        crops=crops
    )


# ▶ Run App
if __name__ == "__main__":
   app.run(debug=True, port=5001)