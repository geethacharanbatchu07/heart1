from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model (and scaler if used)
model = pickle.load(open("model.pkl", "rb"))

# OPTIONAL: Uncomment if you used scaling in training
# scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get all inputs from form (IMPORTANT: match training columns exactly)
        input_features = [
            float(request.form.get("age")),
            float(request.form.get("sex")),
            float(request.form.get("cp")),
            float(request.form.get("trestbps")),
            float(request.form.get("chol")),
            float(request.form.get("fbs", 0)),        # optional
            float(request.form.get("restecg", 0)),    # optional
            float(request.form.get("thalach", 0)),    # optional
            float(request.form.get("exang", 0)),      # optional
            float(request.form.get("oldpeak", 0)),    # optional
            float(request.form.get("slope", 0)),      # optional
            float(request.form.get("ca", 0)),         # optional
            float(request.form.get("thal", 0))        # optional
        ]

        # Convert to numpy array
        features = np.array([input_features])

        # OPTIONAL: apply scaler if used
        # features = scaler.transform(features)

        # Prediction
        prediction = model.predict(features)[0]

        # Probability (if classifier supports it)
        try:
            probability = model.predict_proba(features)[0][1]
            confidence = round(probability * 100, 2)
        except:
            confidence = None

        # Result
        if prediction == 1:
            result = "⚠️ High Risk of Heart Disease"
        else:
            result = "✅ Low Risk of Heart Disease"

        if confidence:
            result += f" (Confidence: {confidence}%)"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)