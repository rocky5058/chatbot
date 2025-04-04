from flask import Flask, request, render_template, jsonify
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables
load_dotenv("apikey.env")

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

def history_to_dict(history):
    """Convert Gemini history to serializable format"""
    return [
        {
            'role': item.role,
            'parts': [part.text for part in item.parts]
        }
        for item in history
    ]

def dict_to_history(history_dict):
    """Convert serialized history back to Gemini format"""
    return [
        {
            'role': item['role'],
            'parts': item['parts']
        }
        for item in history_dict
    ]

@app.route("/")
def home():
    chat = model.start_chat(history=[
        {"role": "user", "parts": ["hello"]},
        {"role": "model", "parts": ["Great to meet you! How can I help?"]},
    ])
    
    return render_template("chat.html",
                         response="Hello! I'm your AI assistant.",
                         chat_history=json.dumps(history_to_dict(chat.history)))

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        user_message = data.get("message")
        history = json.loads(data.get("history", "[]"))
        
        chat = model.start_chat(history=dict_to_history(history))
        response = chat.send_message(user_message)
        
        return jsonify({
            "response": response.text,
            "history": history_to_dict(chat.history)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)