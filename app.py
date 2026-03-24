from flask import Flask, request, jsonify
from werkzeug.exceptions import RequestEntityTooLarge
from PIL import Image
from ultralytics import YOLO
import numpy as np
import os

app = Flask(_name_)

#Allow larger uploads
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB

#Load model once when server starts
model = YOLO("best.pt")

#CHANGE THESE TO MATCH YOUR MODEL'S CLASS NAMES EXACTLY
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

        # MIT App Inventor Web.PostFile usually sends file in request.files
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

        # Resize image inside Python to reduce memory usage
        img.thumbnail((320, 320))
        print("Resized image size:", img.size)

        # Convert PIL image to numpy array for YOLO
        img_np = np.array(img)

        # Run YOLO prediction
        results = model.predict(
            source=img_np,
            imgsz=320,
            conf=0.25,
            verbose=False
        )

        if not results or len(results) == 0:
            return jsonify({
                "success": True,
                "detected": False,
                "message": "No object detected",
                "category": "Unknown"
            }), 200

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return jsonify({
                "success": True,
                "detected": False,
                "message": "No object detected",
                "category": "Unknown"
            }), 200

        # Get first detection
        box = result.boxes[0]
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        class_name = str(result.names[class_id]).strip()
        class_name_clean = class_name.lower()

        # CATEGORY MAPPING
        if class_name_clean in [x.lower() for x in recyclable]:
            category = "Recyclable"
        elif class_name_clean in [x.lower() for x in organic]:
            category = "Organic"
        elif class_name_clean in [x.lower() for x in hazardous]:
            category = "Hazardous"
        elif class_name_clean in [x.lower() for x in non_recyclable]:
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


if _name_ == "_main_":
    print("Starting server...")
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
