import os
import numpy as np
import cv2
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import joblib
from flask import Flask, request, render_template, flash
from werkzeug.utils import secure_filename

# Load the trained XGBoost model
model_path = os.path.join('models', 'xgboost_model.pkl')
model = joblib.load(model_path)

# Load VGG16 for feature extraction
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_model = Model(inputs=base_model.input, outputs=base_model.output)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Feature extraction function
def extract_features_from_image(img_path):
    try:
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Failed to load image")

        # Convert grayscale to RGB if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        # Resize and preprocess
        img = cv2.resize(img, (224, 224))
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)
        
        # Extract features
        feature = feature_model.predict(img, verbose=0)
        return feature.flatten().reshape(1, -1)
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

# Flask app setup
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for flash messages

# Define and create upload folder
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Route for image upload and prediction
@app.route("/", methods=["GET", "POST"])
def index():
    try:
        error = None
        if request.method == "POST":
            # Check if a file was uploaded
            if "image" not in request.files:
                error = "No file uploaded"
                return render_template("index.html", error=error)
            
            file = request.files["image"]
            if file.filename == "":
                error = "No file selected"
                return render_template("index.html", error=error)

            if file and allowed_file(file.filename):
                try:
                    # Secure the filename
                    filename = secure_filename(file.filename)
                    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(img_path)

                    # Extract features and predict
                    features = extract_features_from_image(img_path)
                    pred_proba = model.predict_proba(features)[0]
                    pred_class = np.argmax(pred_proba)
                    confidence = pred_proba[pred_class] * 100

                    prediction_label = "Stroke" if pred_class == 1 else "No Stroke"
                    prediction_text = f"{prediction_label} (Confidence: {confidence:.1f}%)"

                    relative_img_path = os.path.join('static', 'uploads', filename)
                    return render_template("index.html", 
                                        prediction=prediction_text, 
                                        image_path=relative_img_path)

                except Exception as e:
                    error = f"Error processing image: {str(e)}"
                    return render_template("index.html", error=error)
            else:
                error = "Invalid file type. Please upload an image file (png, jpg, jpeg, tif, tiff)"
                return render_template("index.html", error=error)
        
        # GET request
        return render_template("index.html", prediction=None, image_path=None)
    
    except Exception as e:
        return render_template("index.html", error=f"Server error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)


