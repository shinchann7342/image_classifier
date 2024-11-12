from flask import Flask, request, jsonify
import numpy as np
import cv2
from model import load_model, predict

app = Flask(__name__)
model = load_model()

@app.route('/predict', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    image = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Perform prediction
    results, output_image = predict(model, image)

    # Save output image and send results
    output_path = "outputs/output.jpg"
    cv2.imwrite(output_path, output_image)

    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
