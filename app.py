from flask import Flask, request, jsonify
from werkzeug.exceptions import RequestEntityTooLarge
from PIL import Image
from ultralytics import YOLO
import numpy as np
import io
import os

app = Flask(__name__)

# Allow larger uploads
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024 

# Load model once
model = YOLO("best.pt")

# CHANGE THESE TO MATCH YOUR MODEL'S CLASS NAMES EXACTLY
recyclable = ["bottle", "plastic bottle", "can", "paper", "cardboard"]
organic = ["banana peel", "food waste", "leaf", "fruit peel"]
hazardous = ["battery", "bulb", "chemical", "spray can", "laptop"]
non_recyclable = ["wrapper", "styrofoam", "diaper", "sachet"]


@app.route("/", methods=["GET"])
def home():
    return "Server running"


@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return jsonify({
        "success": False,
        "error": "Image too large"
    }), 413


@app.route("/detect", methods=["POST"])
def detect():
    try:
        print("\n---- REQUEST RECEIVED ----")
        print("Method:", request.method)
        print("Content-Type:", request.content_type)
        print("Content-Length:", request.content_length)
        print("Files keys:", list(request.files.keys()))
        print("Form keys:", list(request.form.keys()))

        # Get uploaded file from MIT App Inventor Web.PostFile
        if not request.files:
            return jsonify({
                "success": False,
                "error": "No uploaded file received"
            }), 400

        uploaded_file = list(request.files.values())[0]

        if uploaded_file.filename == "":
            return jsonify({
                "success": False,
                "error": "Empty uploaded file"
            }), 400

        # Open image directly from uploaded stream
        img = Image.open(uploaded_file.stream).convert("RGB")
        print("Original image size:", img.size)

        # RESIZE IMAGE HERE TO REDUCE MEMORY USAGE
        img.thumbnail((320, 320))
        print("Resized image size:", img.size)

        # Optional debug save
        img.save("debug_resized.jpg", format="JPEG", quality=85)
        print("Saved debug_resized.jpg:", os.path.getsize("debug_resized.jpg"), "bytes")

        # Convert PIL image to numpy array for YOLO
        img_np = np.array(img)

        # Run YOLO on resized image
        results = model.predict(
            source=img_np,
            imgsz=320,
            conf=0.25,
            verbose=False
        )

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return jsonify({
                "success": True,
                "detected": False,
                "message": "No object detected",
                "category": "Unknown"
            }), 200

        box = result.boxes[0]
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        class_name = str(result.names[class_id]).strip()

        # CATEGORY MAPPING
        if class_name in recyclable:
            category = "Recyclable"
        elif class_name in organic:
            category = "Organic"
        elif class_name in hazardous:
            category = "Hazardous"
        elif class_name in non_recyclable:
            category = "Non-Recyclable"
        else:
            category = "Unknown"

        return jsonify({
            "success": True,
            "detected": True,
            "object": class_name,
            "category": category,
            "confidence": round(confidence, 4)
        }), 200

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    print("Starting debug server...")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
