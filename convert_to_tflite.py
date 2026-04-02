import tensorflow as tf

# Convert Cell Model
cell_model = tf.keras.models.load_model('models/cell_model.keras')
converter = tf.lite.TFLiteConverter.from_keras_model(cell_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_cell_model = converter.convert()

with open('models/cell_model.tflite', 'wb') as f:
    f.write(tflite_cell_model)

print("Cell model converted to .tflite")

# Convert Cancer Model
cancer_model = tf.keras.models.load_model('models/cancer_model.keras')
converter = tf.lite.TFLiteConverter.from_keras_model(cancer_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_cancer_model = converter.convert()

with open('models/cancer_model.tflite', 'wb') as f:
    f.write(tflite_cancer_model)

print("Cancer model converted to .tflite")
print("Conversion completed successfully!")