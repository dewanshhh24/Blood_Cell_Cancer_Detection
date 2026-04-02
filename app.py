from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import io
import tensorflow as tf

app = Flask(__name__, template_folder='templates', static_folder='static')

print("Loading TensorFlow Lite models...")

# Load TFLite models
interpreter_cell = tf.lite.Interpreter(model_path='models/cell_model.tflite')
interpreter_cell.allocate_tensors()

interpreter_cancer = tf.lite.Interpreter(model_path='models/cancer_model.tflite')
interpreter_cancer.allocate_tensors()

# Get input and output details
input_details_cell = interpreter_cell.get_input_details()
output_details_cell = interpreter_cell.get_output_details()

input_details_cancer = interpreter_cancer.get_input_details()
output_details_cancer = interpreter_cancer.get_output_details()

CELL_CLASSES = ['basophil', 'eosinophil', 'erythroblast', 'ig', 
                'lymphocyte', 'monocyte', 'neutrophil', 'platelet']

CANCER_CLASSES = ['benign', 'early_pre_b', 'pre_b', 'pro_b']

def preprocess_image(file):
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
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
        
        # Cell Prediction
        interpreter_cell.set_tensor(input_details_cell[0]['index'], img_array)
        interpreter_cell.invoke()
        cell_pred = interpreter_cell.get_tensor(output_details_cell[0]['index'])
        cell_idx = np.argmax(cell_pred[0])
        cell_conf = float(cell_pred[0][cell_idx] * 100)
        
        # Cancer Prediction
        interpreter_cancer.set_tensor(input_details_cancer[0]['index'], img_array)
        interpreter_cancer.invoke()
        cancer_pred = interpreter_cancer.get_tensor(output_details_cancer[0]['index'])
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