from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
from clothing_overlay import ClothingOverlay
import os
import base64

app = Flask(__name__)

# Ensure required directories exist
os.makedirs('static/outfits', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Initialize with error handling
try:
    clothing_overlay = ClothingOverlay()
except Exception as e:
    print(f"Error initializing ClothingOverlay: {e}")
    raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(clothing_overlay.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    return clothing_overlay.process_image(file)

@app.route('/change_outfit', methods=['POST'])
def change_outfit():
    outfit_data = request.json
    clothing_overlay.set_current_outfit(outfit_data)
    return jsonify({'success': True})

@app.route('/process_frame', methods=['POST'])
def process_frame():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']

    try:
        # Read the uploaded image
        img_str = file.read()
        nparr = np.frombuffer(img_str, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Process the image for both face and body detection
        processed_image = clothing_overlay.apply_overlay(image, None, None, image.shape[1], image.shape[0])

        # Encode the processed image to return as response
        _, img_encoded = cv2.imencode('.jpg', processed_image)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        return jsonify({'image': img_base64})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
