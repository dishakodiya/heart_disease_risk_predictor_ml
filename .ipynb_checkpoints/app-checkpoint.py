# from flask import Flask, render_template, request
# import pandas as pd
# import joblib

# app = Flask(__name__)

# # -----------------------------
# # Load trained model & scaler
# # -----------------------------
# model = joblib.load("model.pkl")
# scaler = joblib.load("scaler.pkl")

# # -----------------------------
# # Feature columns (EXACT ORDER)
# # -----------------------------
# feature_cols = [
#     'age', 'height', 'weight', 'ap_hi', 'ap_lo',
#     'gender_2',
#     'cholesterol_2', 'cholesterol_3',
#     'gluc_2', 'gluc_3',
#     'smoke_1', 'alco_1', 'active_1'
# ]

# # Numeric columns to scale
# num_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']

# @app.route("/", methods=["GET", "POST"])
# def home():
#     result = ""

#     if request.method == "POST":
#         try:
#             # -----------------------------
#             # Read form values
#             # -----------------------------
#             age = float(request.form["age"])
#             height = float(request.form["height"])
#             weight = float(request.form["weight"])
#             ap_hi = float(request.form["ap_hi"])
#             ap_lo = float(request.form["ap_lo"])

#             gender = int(request.form["gender"])        # 1 or 2
#             cholesterol = int(request.form["cholesterol"])  # 1,2,3
#             gluc = int(request.form["gluc"])            # 1,2,3
#             smoke = int(request.form["smoke"])          # 0/1
#             alco = int(request.form["alco"])            # 0/1
#             active = int(request.form["active"])        # 0/1

#             # -----------------------------
#             # Build feature dict
#             # -----------------------------
#             data = {
#                 "age": age,
#                 "height": height,
#                 "weight": weight,
#                 "ap_hi": ap_hi,
#                 "ap_lo": ap_lo,

#                 # One-hot encoded fields
#                 "gender_2": 1 if gender == 2 else 0,

#                 "cholesterol_2": 1 if cholesterol == 2 else 0,
#                 "cholesterol_3": 1 if cholesterol == 3 else 0,

#                 "gluc_2": 1 if gluc == 2 else 0,
#                 "gluc_3": 1 if gluc == 3 else 0,

#                 "smoke_1": smoke,
#                 "alco_1": alco,
#                 "active_1": active
#             }

#             # -----------------------------
#             # Create DataFrame
#             # -----------------------------
#             input_df = pd.DataFrame([[data[col] for col in feature_cols]],
#                                     columns=feature_cols)

#             print("RAW INPUT:")
#             print(input_df)

#             # -----------------------------
#             # Scale numeric columns
#             # -----------------------------
#             input_df[num_cols] = scaler.transform(input_df[num_cols])

#             print("SCALED INPUT:")
#             print(input_df)

#             # -----------------------------
#             # Predict
#             # -----------------------------
#             prob = model.predict_proba(input_df)[0][1]
#             prediction = "YES" if prob >= 0.5 else "NO"

#             result = f"Heart Disease Risk: {prediction} (Probability: {prob:.2f})"

#         except Exception as e:
#             result = f"Error: {str(e)}"

#     return render_template("index.html", result=result)

# if __name__ == "__main__":
#     app.run(debug=True)





from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)

# ✅ CORS - Allow all origins
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": False
    }
})

THRESHOLD = 0.4

# Load model
print("📦 Loading RandomForest model...")
try:
    model = joblib.load("rf_model.pkl")
    print("✅ Model loaded")
except Exception as e:
    print("❌ Model load error:", e)
    model = None

# Load scaler
try:
    scaler = joblib.load("scaler.pkl")
    USE_SCALER = True
    print("✅ Scaler loaded")
except:
    scaler = None
    USE_SCALER = False
    print("⚠️ No scaler found")

# Add after every response
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

# ---------------- API ROUTES ---------------- #

@app.route("/api", methods=["GET"])
def api_home():
    return jsonify({
        "message": "CardioPredict API 🚀",
        "model": "RandomForestClassifier",
        "threshold": THRESHOLD,
        "status": "active"
    })

@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return '', 204
    
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        print(f"📦 Received data: {data}")
        
        required = [
            'age','gender','height','weight',
            'ap_hi','ap_lo','smoke','alco','active',
            'cholesterol_2','cholesterol_3','gluc_2','gluc_3'
        ]
        
        missing = [k for k in required if k not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400
        
        X = pd.DataFrame([data])
        
        if USE_SCALER:
            num_cols = ['age','height','weight','ap_hi','ap_lo']
            X[num_cols] = scaler.transform(X[num_cols])
        
        prob = float(model.predict_proba(X)[0][1])
        prediction = int(prob >= THRESHOLD)
        risk = "Low" if prob < 0.3 else "Moderate" if prob < 0.6 else "High"
        
        result = {
            "prediction": prediction,
            "probability": round(prob, 4),
            "risk": risk
        }
        
        print(f"✅ Prediction result: {result}")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ---------------- FRONTEND ROUTES ---------------- #

@app.route("/")
def home():
    return send_from_directory("frontend", "home.html")

@app.route("/home.html")
def home_page():
    return send_from_directory("frontend", "home.html")

@app.route("/predict.html")
def predict_page():
    return send_from_directory("frontend", "predict.html")

@app.route("/about.html")
def about_page():
    return send_from_directory("frontend", "about.html")

@app.route("/dashboard.html")
def dashboard_page():
    return send_from_directory("frontend", "dashboard.html")

@app.route("/login.html")
def login_page():
    return send_from_directory("frontend", "login.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("frontend", path)

# ---------------- RUN ---------------- #

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚀 CardioPredict API Starting...")
    print(f"🌐 Local: http://localhost:{port}")
    print(f"📊 Predict: http://localhost:{port}/predict.html")
    print(f"🏠 Home: http://localhost:{port}/home.html")
    print(f"🔌 API: http://localhost:{port}/api/predict")
    
    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )