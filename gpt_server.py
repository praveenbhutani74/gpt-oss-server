from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

app = Flask(__name__)
model_name = os.getenv("MODEL_NAME", "gpt2")
port = int(os.getenv("PORT", 5001))

print(f"Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print("Model loaded successfully.")

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json or {}
        prompt = data.get("prompt", "").strip()
        if not prompt:
            return jsonify({"error": "Prompt cannot be empty"}), 400

        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=200)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({"response": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
