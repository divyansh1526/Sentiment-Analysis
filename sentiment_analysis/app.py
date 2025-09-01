from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import random
import re

app = Flask(__name__)
sentiment_pipeline = pipeline("sentiment-analysis")

def generate_reply(text, sentiment):
    text_snippet = text[:50] + ("..." if len(text) > 50 else "")

    # Greeting check (rule-based)
    greetings = ["how are you", "hello", "hi", "hey", "good morning", "good evening"]
    if any(greet in text.lower() for greet in greetings):
        return "I'm doing well, thanks for asking! How about you?"

    positive_responses = [
        f"That's wonderful to hear! It sounds like you're feeling great about: '{text_snippet}'",
        f"I'm really happy for you! Your words show a lot of positivity: '{text_snippet}'",
        f"That’s awesome, I can sense the excitement in what you said: '{text_snippet}'"
    ]

    negative_responses = [
        f"I'm sorry you're feeling this way about: '{text_snippet}'. Do you want to share more?",
        f"It sounds tough. Thanks for opening up about: '{text_snippet}'. I'm here for you.",
        f"That must be hard. I hear you when you say: '{text_snippet}'"
    ]

    neutral_responses = [
        f"Thanks for sharing that with me! '{text_snippet}' sounds interesting.",
        f"I hear you. '{text_snippet}' is something worth thinking about.",
        f"Got it! Thanks for letting me know about: '{text_snippet}'."
    ]

    if sentiment == "POSITIVE":
        return random.choice(positive_responses)
    elif sentiment == "NEGATIVE":
        return random.choice(negative_responses)
    else:
        return random.choice(neutral_responses)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    text = data.get("text")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = sentiment_pipeline(text)[0]
    label = result["label"]
    score = result["score"]

    if score < 0.7:  
        sentiment = "NEUTRAL"
    else:
        sentiment = label

    ai_reply = generate_reply(text, sentiment)

    return jsonify({
        "sentiment": sentiment,
        "confidence": score,
        "ai_reply": ai_reply
    })

if __name__ == "__main__":
    app.run(debug=True)
