from flask import Flask, render_template, request, jsonify, send_file
import pickle  
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the sentiment model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Define prediction function
def predict_sentiment(text):
    prediction = model.predict([text])  
    confidence = np.max(model.predict_proba([text])) * 100  
    sentiment = "Positive" if prediction[0] == 1 else "Negative"
    return sentiment, round(confidence, 2)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.form["text"]
    sentiment, confidence = predict_sentiment(text)
    return jsonify({"sentiment": sentiment, "confidence": confidence})

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        df = pd.read_csv(file)

        # Automatically detect review column (case insensitive)
        possible_cols = ["review", "reviews", "text", "comment"]
        review_col = next((col for col in df.columns if col.lower() in possible_cols), None)

        if review_col is None:
            return jsonify({"error": "CSV must contain a column with reviews (e.g., 'review', 'text', 'comment')"}), 400

        # Process each review
        df["sentiment"], df["confidence"] = zip(*df[review_col].astype(str).apply(predict_sentiment))
        df.to_csv("static/results.csv", index=False)

        results = df[[review_col, "sentiment", "confidence"]].to_dict(orient="records")
        return jsonify({"results": results})

    except Exception as e:
        print(e)  # Debug in console
        return jsonify({"error": "Error processing CSV"}), 500


@app.route("/export")
def export():
    return send_file("static/results.csv", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
