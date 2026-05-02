from flask import Flask, render_template, request, redirect, url_for
import joblib

app = Flask(__name__)

# Load model and encoders
model = joblib.load("model.pkl")
le_state = joblib.load("state_encoder.pkl")
le_season = joblib.load("season_encoder.pkl")
le_crop = joblib.load("crop_encoder.pkl")


# 🏠 Home Page
@app.route("/")
def home():
    states = list(le_state.classes_)
    seasons = list(le_season.classes_)
    crops = list(le_crop.classes_)

    # Get result from URL (after redirect)
    result = request.args.get("result")

    return render_template(
        "index.html",
        result=result,
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

    # Encode inputs
    state_val = le_state.transform([state])[0]
    season_val = le_season.transform([season])[0]
    crop_val = le_crop.transform([crop])[0]

    # Predict
    prediction = model.predict([[state_val, season_val, crop_val, area]])

    result = round(prediction[0], 2)

    # Redirect to home with result (BEST PRACTICE)
    return redirect(url_for("home", result=result))


# ▶ Run App
if __name__ == "__main__":
    app.run(debug=True, port=5001)