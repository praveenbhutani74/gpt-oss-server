from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)
model_name = "gpt2"  # Start with a smaller model like 'gpt2'

print("Loading model. This may take a while...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print("Model loaded successfully.")

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json or {}
    prompt = data.get("prompt", "")
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=200)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

