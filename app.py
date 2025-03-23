from flask import Flask, request, jsonify, send_file
import os
from PIL import Image
import torch
import torch.nn as nn
import timm
import torchvision.transforms as transforms
from werkzeug.utils import secure_filename
from flask_cors import CORS
import pymysql
import logging
import csv
from io import StringIO

# Initialize Flask app
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO)

# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Vysh2211', 
    'database': 'hudb'  # Updated to match the frontend's database name
}

def get_db_connection():
    return pymysql.connect(**db_config)

# Cyclone Intensity Prediction Model
class CycloneIntensityModel(nn.Module):
    def __init__(self):
        super(CycloneIntensityModel, self).__init__()
        self.model = timm.create_model('efficientnet_b1.ft_in1k', pretrained=True)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.model(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CycloneIntensityModel().to(device)
model_path = r"cyclone.pth"  # Replace with your model path

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logging.info("Model loaded successfully.")
except FileNotFoundError:
    logging.error("Error: Model file not found.")
    model = None

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_intensity(image):
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(image).item()
    logging.info(f"Predicted intensity: {prediction:.2f}")
    return prediction

def classify_cyclone(intensity):
    if intensity < 39:
        return "Not a Tropical Cyclone: No damage expected."
    elif 39 <= intensity < 74:
        return "Tropical Storm: Minimal damage expected."
    elif 74 <= intensity <= 95:
        return "Less Cyclone (Category 1): Some damage expected."
    elif 96 <= intensity <= 110:
        return "Moderate Cyclone (Category 2): Extensive damage expected."
    elif 111 <= intensity <= 129:
        return "Heavy Cyclone (Category 3): Devastating damage expected."
    elif 130 <= intensity <= 156:
        return "Severe Cyclone (Category 4): Catastrophic damage expected."
    else:
        return "Extreme Cyclone (Category 5): Catastrophic damage expected."

def get_similar_cyclones(intensity):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        query = """
            SELECT name, year, month, day, hour, lat, lon, status, category, wind, pressure
            FROM hurricanes
            WHERE ABS(wind - %s) <= 5  # Adjust the range to ±5 km/hr
            LIMIT 10  # Limit the results to 10 cyclones
        """
        cursor.execute(query, (intensity,))
        data = cursor.fetchall()
        return data
    except pymysql.Error as err:
        logging.error(f"Database error: {err}")
        return []
    finally:
        cursor.close()
        conn.close()

@app.route('/')
def index():
    return "Cyclone Intensity Prediction Backend"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({"error": "Invalid file type. Only image files are allowed."}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        with Image.open(file_path) as image:
            intensity = predict_intensity(image)
            cyclone_type = classify_cyclone(intensity)

            # Query year and region based on intensity
            conn = get_db_connection()
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            query = """
                SELECT year, lat, lon
                FROM hurricanes
                WHERE ABS(wind - %s) <= 5
                LIMIT 1
            """
            cursor.execute(query, (intensity,))
            cyclone_details = cursor.fetchone()
            cursor.close()
            conn.close()

            # Provide fallback values if year and region are unavailable
            year = cyclone_details['year'] if cyclone_details else "Unknown Year"
            region = f"Lat: {cyclone_details['lat']}, Lon: {cyclone_details['lon']}" if cyclone_details else "Unknown Region"

            similar_cyclones = get_similar_cyclones(intensity)

            return jsonify({
                "intensity": f"{intensity:.2f}",
                "cyclone_type": cyclone_type,
                "year": year,
                "region": region,
                "similar_cyclones": similar_cyclones
            })
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/cyclones', methods=['GET'])
def fetch_cyclones():
    search = request.args.get('search', '').strip()  # Get the search input from the user
    try:
        conn = get_db_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        query = """
            SELECT name, year, month, day, hour, lat, lon, status, category, wind, pressure
            FROM hurricanes
        """
        if search:  # If search is provided, filter the data
            query += " WHERE name LIKE %s"
            cursor.execute(query, (f"%{search}%",))
        else:  # Otherwise, fetch all data
            cursor.execute(query)

        data = cursor.fetchall()
        return jsonify(data)
    except pymysql.Error as err:
        logging.error(f"Database error: {err}")
        return jsonify({"error": "Database query failed"}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/download', methods=['GET'])
def download_data():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.execute("""
            SELECT * FROM hurricanes
        """)
        data = cursor.fetchall()

        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        output.seek(0)

        return send_file(
            output, mimetype='text/csv', as_attachment=True,
            attachment_filename='cyclone_data.csv'
        )
    except pymysql.Error as err:
        logging.error(f"Database error: {err}")
        return jsonify({"error": "Database query failed"}), 500
    finally:
        cursor.close()
        conn.close()

if __name__ == '__main__':
    app.run(debug=True)