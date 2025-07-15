from flask import Flask, render_template, request
import os
import joblib

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'sentiment_classifier.pkl')
model = joblib.load(MODEL_PATH)

@app.route("/", methods={"GET", "POST"})
def index():
    result = None
    if request.method == "POST":
        text = request.form.get("text_box", "")
        if text.strip():
            pred = model.predict([text])[0]
            result = f"Sentiment: {pred}"
        else: 
            result = "Please enter some text to analyze"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)