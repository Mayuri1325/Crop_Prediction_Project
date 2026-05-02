# 🌾 Crop Production Prediction System

## 📌 Project Overview
This project is a Machine Learning-based web application that predicts crop production based on:
- State
- Season
- Crop type
- Area

The model is trained using a real agricultural dataset and deployed using Flask.

---

## 🚀 Features
- Real dataset integration
- Machine Learning model (Random Forest)
- Categorical data handling using Label Encoding
- User-friendly web interface with dropdown inputs
- Real-time prediction

---

## 🛠️ Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Flask
- HTML, CSS

---

## 📂 Project Structure

app.py
train_model.py
model.pkl
templates/index.html
crop.csv


---

## ▶️ How to Run

1. Run model training:

python train_model.py


2. Run the Flask app:

python app.py


3. Open browser:

http://127.0.0.1:5001


---

## 🧠 Model Details
- Algorithm: Random Forest Regressor
- Inputs: State, Season, Crop, Area
- Output: Predicted crop production

---

## 📈 Future Improvements
- Add weather data
- Improve UI design
- Deploy online
- Add traffic prediction module

---

## 👩‍💻 Author
Mayuri Potadar