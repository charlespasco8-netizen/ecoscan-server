from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge
import requests
from io import BytesIO
import os

app = Flask(__name__)

# Allow larger uploads, but still keep a limit
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

model = YOLO("best.pt")

recyclable = ["bottle", "plastic bottle", "can", "paper", "cardboard"]
organic = ["banana peel", "food waste", "leaf", "fruit peel"]
hazardous = ["battery", "bulb", "chemical", "spray can", "laptop"]
non_recyclable = ["wrapper", "styrofoam", "diaper", "sachet"]


def classify_detection(label: str) -> tuple[str, int]:
    low = label.lower()

    if low in recyclable:
        return "Recyclable", 10
    elif low in organic:
        return "Organic", 8
    elif low in hazardous:
        return "Hazardous", 15
    elif low in non_recyclable:
        return "Non-Recyclable", 5
    else:
        return "Non-Recyclable", 5


def run_detection_on_image(img: Image.Image):
    results = model(img)
    result = results[0]
    names = result.names

    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = names[cls_id]

            waste_type, points = classify_detection(label)

            detections.append({
                "label": label,
                "confidence": round(conf, 4),
                "waste_type": waste_type,
                "points": points
            })

    if len(detections) == 0:
        return {
            "success": True,
            "detected": False,
            "message": "No object detected",
            "detections": []
        }, 200

    best = max(detections, key=lambda x: x["confidence"])

    return {
        "success": True,
        "detected": True,
        "label": best["label"],
        "confidence": best["confidence"],
        "waste_type": best["waste_type"],
        "points": best["points"],
        "detections": detections
    }, 200


@app.route("/", methods=["GET"])
def home():
    return "Server running"


@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return jsonify({
        "success": False,
        "error": "Image too large. Please upload a smaller image."
    }), 413


# FOR MIT App Inventor Web1.PostFile
@app.route("/detect", methods=["POST"])
def detect_file():
    try:
        # Web1.PostFile usually sends raw body, not multipart/form-data
        raw_data = request.get_data()

        if not raw_data:
            return jsonify({
                "success": False,
                "error": "No image data received"
            }), 400

        img = Image.open(BytesIO(raw_data)).convert("RGB")

        response, status_code = run_detection_on_image(img)
        return jsonify(response), status_code

    except UnidentifiedImageError:
        return jsonify({
            "success": False,
            "error": "Uploaded data is not a valid image"
        }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# FOR URL-BASED FLOW (Cloudinary or image link)
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

        response, status_code = run_detection_on_image(img)
        return jsonify(response), status_code

    except UnidentifiedImageError:
        return jsonify({
            "success": False,
            "error": "Downloaded file is not a valid image"
        }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
