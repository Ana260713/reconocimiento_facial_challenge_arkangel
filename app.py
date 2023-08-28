from flask import Flask, render_template, request, jsonify
import numpy as np
from keras.models import load_model
from keras.applications.vgg19 import preprocess_input
from PIL import Image


app = Flask(__name__)

# Cargar el modelo previamente entrenado
model = load_model('best_model_vgg16.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Received POST request for prediction")
        image_file = request.files['image']
        image = Image.open(image_file).resize((48, 48))
        image_data = np.array(image).astype(np.float32)
        image_data = image_data.reshape(1, 48, 48, 3)
        image_data = preprocess_input(image_data)
        
        print("Performing prediction")
        prediction = model.predict(image_data)
        predicted_class = np.argmax(prediction)
        
        response = {'predicted_class': int(predicted_class)}
        
        print("Prediction successful")
    except Exception as e:
        print("An error occurred:", str(e))
        response = {'error': str(e)}
        
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)