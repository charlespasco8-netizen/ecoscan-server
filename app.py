from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO
import os

app = Flask(__name__)
model = YOLO("best.pt")

recyclable = ["bottle", "plastic bottle", "can", "paper", "cardboard"]
organic = ["banana peel", "food waste", "leaf", "fruit peel"]
hazardous = ["battery", "bulb", "chemical", "spray can", "laptop"]
non_recyclable = ["wrapper", "styrofoam", "diaper", "sachet"]

@app.route("/", methods=["GET"])
def home():
    return "Server running"

@app.route("/detect-url", methods=["POST"])
def detect_url():
    data = request.get_json(silent=True) or {}
    image_url = data.get("image_url")

    if not image_url:
        return jsonify({
            "success": False,
            "error": "No image_url provided"
        }), 400

    try:
        resp = requests.get(image_url, timeout=20)
        resp.raise_for_status()

        img = Image.open(BytesIO(resp.content)).convert("RGB")

        results = model(img)
        result = results[0]
        names = result.names

        detections = []

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = names[cls_id]

            low = label.lower()
            if low in recyclable:
                waste_type = "Recyclable"
                points = 10
            elif low in organic:
                waste_type = "Organic"
                points = 8
            elif low in hazardous:
                waste_type = "Hazardous"
                points = 15
            else:
                waste_type = "Non-Recyclable"
                points = 5

            detections.append({
                "label": label,
                "confidence": round(conf, 4),
                "waste_type": waste_type,
                "points": points
            })

        if len(detections) == 0:
            return jsonify({
                "success": True,
                "detected": False,
                "message": "No object detected",
                "detections": []
            }), 200

        best = max(detections, key=lambda x: x["confidence"])

        return jsonify({
            "success": True,
            "detected": True,
            "label": best["label"],
            "confidence": best["confidence"],
            "waste_type": best["waste_type"],
            "points": best["points"],
            "detections": detections
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
