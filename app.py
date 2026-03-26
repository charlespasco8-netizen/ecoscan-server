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

# Load YOLO classification model
model = YOLO("best.pt")
print("MODEL NAMES:", model.names)

# Object classes mapped to waste categories
recyclable = [
    "bottle",
    "plastic bottle",
    "plastic_bottle",
    "can",
    "paper",
    "paper cup",
    "paper_cup",
    "cardboard",
    "glass jars",
    "glass_jars"
]

organic = [
    "banana peel",
    "banana_peel",
    "food waste",
    "food_waste",
    "leaf",
    "fruit peel",
    "fruit_peel",
    "egg shell",
    "egg_shell",
    "coffee grounds",
    "ground coffee",
    "ground_coffee"
]

hazardous = [
    "battery",
    "bulb",
    "chemical",
    "spray can",
    "spray_can",
    "laptop",
    "e waste",
    "e_waste",
    "ewaste"
]

non_recyclable = [
    "wrapper",
    "snack wrapper",
    "snack_wrapper",
    "styrofoam",
    "diaper",
    "sachet",
    "disposable cutlery",
    "disposable_cutlery"
]


def normalize_label(label: str) -> str:
    return label.lower().replace("_", " ").replace("-", " ").strip()


def classify_detection(label: str) -> tuple[str, float]:
    low = normalize_label(label)

    recyclable_norm = [normalize_label(x) for x in recyclable]
    organic_norm = [normalize_label(x) for x in organic]
    hazardous_norm = [normalize_label(x) for x in hazardous]
    non_recyclable_norm = [normalize_label(x) for x in non_recyclable]

    if low in recyclable_norm:
        return "Recyclable", 0.5
    elif low in organic_norm:
        return "Organic", 0.5
    elif low in hazardous_norm:
        return "Hazardous", 0.5
    elif low in non_recyclable_norm:
        return "Non-Recyclable", 0.5
    else:
        return "Unknown", 0.0


def run_detection_on_image(img: Image.Image):
    results = model(img)
    result = results[0]

    detections = []

    if result.probs is not None:
        cls_id = int(result.probs.top1)
        conf = float(result.probs.top1conf)
        label = result.names[cls_id]
        normalized_label = normalize_label(label)

        waste_type, points = classify_detection(label)

        detections.append({
            "label": label,
            "normalized_label": normalized_label,
            "confidence": round(conf, 4),
            "waste_type": waste_type,
            "points": points
        })

    if len(detections) == 0:
        return {
            "success": True,
            "detected": False,
            "message": "No waste detected",
            "detections": []
        }, 200

    best = detections[0]

    return {
        "success": True,
        "detected": True,
        "label": best["label"],
        "normalized_label": best["normalized_label"],
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


# FOR URL-BASED FLOW
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
