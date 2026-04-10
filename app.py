# -----------------------
# 1. Imports & Setup
# -----------------------
import json, os
import pandas as pd
import joblib

from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------
# 2. Dataset (2 domains)
# -----------------------
os.makedirs("data", exist_ok=True)

sample_data = [
    # Entertainment
    {"text": "Actor confirmed new movie", "label": 0, "domain": "entertainment"},
    {"text": "Celebrity secretly married yesterday", "label": 1, "domain": "entertainment"},
    {"text": "Movie release officially announced", "label": 0, "domain": "entertainment"},
    {"text": "Shocking breakup rumor of actor", "label": 1, "domain": "entertainment"},

    # Finance
    {"text": "Stock market reaches new high", "label": 0, "domain": "finance"},
    {"text": "Earn double money in one day guaranteed", "label": 1, "domain": "finance"},
    {"text": "Bank introduces new savings scheme", "label": 0, "domain": "finance"},
    {"text": "Free bitcoin giveaway for everyone", "label": 1, "domain": "finance"}
]

with open("data/news.json", "w") as f:
    json.dump(sample_data, f)

# -----------------------
# 3. Train Model
# -----------------------
df = pd.DataFrame(sample_data)

X = df["text"]
y = df["label"]

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vec, y)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

# -----------------------
# 4. Flask App
# -----------------------
app = Flask(__name__)

model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# -----------------------
def detect_domain(text):
    text = text.lower()
    
    # Entertainment
    if any(word in text for word in ["actor", "movie", "celebrity", "film", "director"]):
        return "Entertainment"
    
    # Finance
    elif any(word in text for word in ["money", "bitcoin", "bank", "stock", "investment"]):
        return "Finance"
    
    # Politics (NEW)
    elif any(word in text for word in ["government", "election", "minister", "policy", "politics", "party"]):
        return "Politics"
    
    else:
        return "General"
# -----------------------
# 6. Prediction Function
# -----------------------
def predict_rumor(text):
    vec = vectorizer.transform([text])
    
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    
    confidence = max(prob) * 100
    domain = detect_domain(text)

    text_lower = text.lower()

    # 🚨 Rule-based override (important fix)
    if any(word in text_lower for word in ["visited", "met", "attended", "went"]):
        pred = 0  # Not rumor
        reason = "Looks like a normal real-world event"
    
    elif any(word in text_lower for word in ["free", "guaranteed", "shocking", "breaking"]):
        pred = 1  # Rumor
        reason = "Suspicious keywords detected"
    
    else:
        reason = "Model-based prediction"

    return {
        "domain": domain,
        "prediction": "Rumor" if pred == 1 else "True news",
        "confidence": f"{confidence:.2f}%",
        "reason": reason
    
    }

# -----------------------
# 7. Routes
# -----------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_web", methods=["POST"])
def predict_web():
    text = request.form["text"]
    result = predict_rumor(text)
    return render_template("index.html", result=result)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    if not data or "text" not in data:
        return jsonify({"error": "Provide 'text'"})
    
    return jsonify(predict_rumor(data["text"]))

# -----------------------
# 8. Run
# -----------------------
import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))