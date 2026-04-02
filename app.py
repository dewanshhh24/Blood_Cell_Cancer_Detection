from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__, template_folder='templates', static_folder='static')

# ====================== MODEL LOADING ======================
print("Loading models... (this may take 10-20 seconds)")

cell_model = tf.keras.models.load_model('models/cell_model.keras')
cancer_model = tf.keras.models.load_model('models/cancer_model.keras')

# Class names (tera Colab se liya)
CELL_CLASSES = ['basophil', 'eosinophil', 'erythroblast', 'ig', 
                'lymphocyte', 'monocyte', 'neutrophil', 'platelet']

CANCER_CLASSES = ['benign', 'early_pre_b', 'pre_b', 'pro_b']

print("Models loaded successfully!")

def preprocess_image(file):
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    try:
        img_array = preprocess_image(file)
        
        # Cell Model Prediction
        cell_pred = cell_model.predict(img_array, verbose=0)
        cell_idx = np.argmax(cell_pred[0])
        cell_conf = float(cell_pred[0][cell_idx] * 100)
        
        # Cancer Model Prediction
        cancer_pred = cancer_model.predict(img_array, verbose=0)
        cancer_idx = np.argmax(cancer_pred[0])
        cancer_conf = float(cancer_pred[0][cancer_idx] * 100)
        
        return jsonify({
            "cell": CELL_CLASSES[cell_idx],
            "cell_conf": round(cell_conf, 2),
            "cancer": CANCER_CLASSES[cancer_idx],
            "cancer_conf": round(cancer_conf, 2)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)