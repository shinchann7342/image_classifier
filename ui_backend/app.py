from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

# URL of AI backend service
AI_BACKEND_URL = os.getenv("AI_BACKEND_URL", "http://ai_backend:5001/predict")

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Send image to AI backend for processing
    files = {'file': (file.filename, file.stream, file.mimetype)}
    response = requests.post(AI_BACKEND_URL, files=files)

    return jsonify(response.json())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
