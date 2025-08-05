from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from predict import predict_image
import config
from model import get_model
import torch
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model (lazy load)
model = None  

@app.before_first_request
def load_model():
    """Load the model only when the first request comes in (saves memory during build)."""
    global model
    if model is None:
        print("Loading model for the first request...")
        model = get_model()
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
        model.to(config.DEVICE)
        model.eval()


@app.route("/", methods=["GET", "POST"])
def index():
    global model
    prediction, confidence, image_path = None, None, None

    if request.method == "POST":
        file = request.files.get("image")
        webcam_data = request.form.get("webcam")

        # Handle file upload
        if file and file.filename:
            filename = secure_filename(file.filename)
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(image_path)

        # Handle webcam capture (base64 image)
        elif webcam_data:
            header, encoded = webcam_data.split(",")
            img_data = base64.b64decode(encoded)
            image = Image.open(BytesIO(img_data)).convert("RGB")
            filename = "captured.png"
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            image.save(image_path)

        # Predict only if model is loaded and image is present
        if image_path:
            if model is None:
                load_model()  # Ensure model is loaded (extra safeguard)
            pred_class, probs = predict_image(image_path, model, config.DEVICE)
            prediction = config.CLASS_NAMES[pred_class]
            confidence = f"{probs[pred_class]*100:.2f}%"

    return render_template("index.html", prediction=prediction, confidence=confidence, image_path=image_path)


if __name__ == "__main__":
    app.run(debug=True)
