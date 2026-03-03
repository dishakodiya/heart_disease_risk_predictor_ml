from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# -----------------------------
# Load trained model & scaler
# -----------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Feature columns (EXACT ORDER)
# -----------------------------
feature_cols = [
    'age', 'height', 'weight', 'ap_hi', 'ap_lo',
    'gender_2',
    'cholesterol_2', 'cholesterol_3',
    'gluc_2', 'gluc_3',
    'smoke_1', 'alco_1', 'active_1'
]

# Numeric columns to scale
num_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""

    if request.method == "POST":
        try:
            # -----------------------------
            # Read form values
            # -----------------------------
            age = float(request.form["age"])
            height = float(request.form["height"])
            weight = float(request.form["weight"])
            ap_hi = float(request.form["ap_hi"])
            ap_lo = float(request.form["ap_lo"])

            gender = int(request.form["gender"])        # 1 or 2
            cholesterol = int(request.form["cholesterol"])  # 1,2,3
            gluc = int(request.form["gluc"])            # 1,2,3
            smoke = int(request.form["smoke"])          # 0/1
            alco = int(request.form["alco"])            # 0/1
            active = int(request.form["active"])        # 0/1

            # -----------------------------
            # Build feature dict
            # -----------------------------
            data = {
                "age": age,
                "height": height,
                "weight": weight,
                "ap_hi": ap_hi,
                "ap_lo": ap_lo,

                # One-hot encoded fields
                "gender_2": 1 if gender == 2 else 0,

                "cholesterol_2": 1 if cholesterol == 2 else 0,
                "cholesterol_3": 1 if cholesterol == 3 else 0,

                "gluc_2": 1 if gluc == 2 else 0,
                "gluc_3": 1 if gluc == 3 else 0,

                "smoke_1": smoke,
                "alco_1": alco,
                "active_1": active
            }

            # -----------------------------
            # Create DataFrame
            # -----------------------------
            input_df = pd.DataFrame([[data[col] for col in feature_cols]],
                                    columns=feature_cols)

            print("RAW INPUT:")
            print(input_df)

            # -----------------------------
            # Scale numeric columns
            # -----------------------------
            input_df[num_cols] = scaler.transform(input_df[num_cols])

            print("SCALED INPUT:")
            print(input_df)

            # -----------------------------
            # Predict
            # -----------------------------
            prob = model.predict_proba(input_df)[0][1]
            prediction = "YES" if prob >= 0.5 else "NO"

            result = f"Heart Disease Risk: {prediction} (Probability: {prob:.2f})"

        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
