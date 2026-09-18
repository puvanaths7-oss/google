
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google import genai

from chatbot_config import SYSTEM_PROMPT


load_dotenv()

app = Flask(__name__)

# Load Gemini API key from .env
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Please add your API key to the .env file."
    )

# Create Gemini client
client = genai.Client(api_key=api_key)

# Gemini model
MODEL_NAME = "gemini-3.1-flash-lite"


@app.route("/")
def home():
    """Display the chatbot page."""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """Receive a user message and return a Gemini response."""

    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({
            "error": "Please enter a message."
        }), 400

    try:
        # Combine chatbot instructions with the user's question
        prompt = f"""
{SYSTEM_PROMPT}

User Question:
{user_message}
"""

        # Generate Gemini response
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        answer = (response.text or "").strip()

        if not answer:
            return jsonify({
                "error": "Gemini did not return a response."
            }), 500

        return jsonify({
            "response": answer
        })

    except Exception as error:
        return jsonify({
            "error": f"Gemini API error: {str(error)}"
        }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )

