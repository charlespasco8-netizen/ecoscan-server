from flask import Flask, request, jsonify
from PIL import Image
from ultralytics import YOLO
import io
import os

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
model = YOLO("best.pt")

# CHANGE THESE TO MATCH YOUR MODEL'S CLASS NAMES EXACTLY
recyclable = ["bottle", "plastic bottle", "can", "paper", "cardboard"]
organic = ["banana peel", "food waste", "leaf", "fruit peel"]
hazardous = ["battery", "bulb", "chemical", "spray can", "laptop"]
non_recyclable = ["wrapper", "styrofoam", "diaper", "sachet"]

@app.route("/", methods=["GET"])
def home():
    return "Server running"

@app.route("/detect", methods=["POST"])
def detect():
    try:
        print("\n---- REQUEST RECEIVED ----")
        print("Method:", request.method)
        print("Content-Type:", request.content_type)
        print("Content-Length:", request.content_length)

        raw = request.get_data(cache=True, parse_form_data=False)
        print("Raw data length:", len(raw) if raw else 0)

        print("Files keys:", list(request.files.keys()))
        print("Form keys:", list(request.form.keys()))

        image_bytes = None

        if raw and len(raw) > 0:
            image_bytes = raw
            print("Using raw body:", len(image_bytes))

        elif request.files:
            uploaded_file = list(request.files.values())[0]
            image_bytes = uploaded_file.read()
            print("Using request.files:", len(image_bytes))

        if not image_bytes:
            print("No image bytes received.")
            return jsonify({
                "success": False,
                "error": "No image received"
            }), 400

        with open("debug_upload.jpg", "wb") as f:
            f.write(image_bytes)

        print("Saved debug_upload.jpg:", os.path.getsize("debug_upload.jpg"), "bytes")

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        results = model(img)
        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return jsonify({
                "success": True,
                "detected": False,
                "message": "No object detected"
            })

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
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    print("Starting debug server...")
    app.run(host="0.0.0.0", port=5000, debug=True)
