from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

app = Flask(__name__)

# Load model & tokenizer
model_path = "model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Label reverse mapping
label_map_inv = {0: 'negatif', 1: 'netral', 2: 'positif'}

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_sentiment(text):
    text = preprocess(text)
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return label_map_inv[pred]

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        input_text = request.form["text"]
        result = predict_sentiment(input_text)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
