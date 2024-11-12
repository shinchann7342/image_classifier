import cv2
import numpy as np

def load_model():
    # Load a lightweight object detection model, such as MobileNet-SSD
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")
    return net

def predict(model, image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    model.setInput(blob)
    detections = model.forward()

    results = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Append result
            results.append({
                "object_id": idx,
                "confidence": float(confidence),
                "bounding_box": [int(startX), int(startY), int(endX), int(endY)]
            })

            # Draw bounding box on image
            cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 2)

    return results, image
