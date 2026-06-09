from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import PyPDF2

app = Flask(__name__)

# ------------------- AI MODEL -------------------
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

chat_history_ids = None


# ------------------- HOME -------------------
@app.route("/")
def home():
    return render_template("index.html")


# ------------------- CHAT -------------------
@app.route("/chat", methods=["POST"])
def chat():
    global chat_history_ids

    user_input = request.json["message"]

    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
    else:
        bot_input_ids = new_input_ids

    with torch.no_grad():
        chat_history_ids = model.generate(
            bot_input_ids,
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id
        )

    reply = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    return jsonify({"response": reply})


# ------------------- PDF UPLOAD -------------------
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]

    if file.filename == "":
        return jsonify({"summary": "❌ No file selected"})

    try:
        reader = PyPDF2.PdfReader(file)
        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""

        if text.strip() == "":
            return jsonify({"summary": "⚠️ No readable text found (maybe scanned PDF)"})

        # simple summary (first 500 chars)
        summary = text[:500]

        return jsonify({"summary": summary})

    except Exception as e:
        return jsonify({"summary": f"❌ Error: {str(e)}"})


# ------------------- RUN -------------------
if __name__ == "__main__":
    app.run(debug=True)