# from flask import Flask, request, jsonify, render_template
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from flask_cors import CORS

# # Initialize Flask app
# app = Flask(__name__, template_folder="templates")
# CORS(app)  # Enable Cross-Origin Resource Sharing

# # Load pre-trained model and tokenizer
# MODEL_NAME = "gpt2"  # Replace with a lighter model if needed for faster loading
# try:
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#     model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
# except Exception as e:
#     print(f"Error loading model: {e}")
#     tokenizer = None
#     model = None

# # Route to render the HTML page
# @app.route("/")
# def index():
#     return render_template("index.html")  # Ensure index.html is in the 'templates' folder

# # API endpoint to handle chatbot messages
# @app.route("/chat", methods=["POST"])
# def chat():
#     if not model or not tokenizer:
#         return jsonify({"error": "Model is not loaded"}), 500

#     user_input = request.json.get("message", "").strip()  # Get user message from the POST request
#     if not user_input:
#         return jsonify({"error": "Message is required"}), 400

#     try:
#         # Tokenize the user input and generate a response
#         input_ids = tokenizer.encode(user_input, return_tensors="pt")
#         output = model.generate(
#             input_ids,
#             max_length=150,                # Maximum length of the response
#             num_return_sequences=1,        # Number of responses to generate
#             pad_token_id=tokenizer.eos_token_id,  # Use the end-of-sequence token for padding
#             temperature=0.7,               # Controls randomness (lower = more deterministic)
#             top_k=50                       # Top-k sampling to control diversity
#         )

#         # Decode the response and return it
#         response = tokenizer.decode(output[0], skip_special_tokens=True)
#         return jsonify({"response": response})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# # Run the Flask app
# if __name__ == "__main__":
#     app.run(debug=True)









from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__, template_folder="templates")
CORS(app)  # Enable Cross-Origin Resource Sharing

# Load pre-trained model and tokenizer
MODEL_NAME = "gpt2"  # Replace with a lighter model if needed for faster loading
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
except Exception as e:
    print(f"Error loading model: {e}")
    tokenizer = None
    model = None

# Route to render the HTML page
@app.route("/")
def index():
    return render_template("index.html")  # Ensure index.html is in the 'templates' folder

# API endpoint to handle chatbot messages
@app.route("/chat", methods=["POST"])
def chat():
    if not model or not tokenizer:
        return jsonify({"error": "Model is not loaded"}), 500

    user_input = request.json.get("message", "").strip()  # Get user message from the POST request
    if not user_input:
        return jsonify({"error": "Message is required"}), 400

    try:
        # Tokenize the user input and generate a response
        input_ids = tokenizer.encode(user_input, return_tensors="pt")
        output = model.generate(
            input_ids,
            max_length=150,                # Maximum length of the response
            num_return_sequences=1,        # Number of responses to generate
            pad_token_id=tokenizer.eos_token_id,  # Use the end-of-sequence token for padding
            temperature=0.7,               # Controls randomness (lower = more deterministic)
            top_k=50                       # Top-k sampling to control diversity
        )

        # Decode the response and return it
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500




