from flask import Flask, render_template, request
import pickle
import numpy as np
from PIL import Image
import io

app = Flask(__name__)


with open('python/model.pkl', 'rb') as file:
    model = pickle.load(file)

CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', message='No selected file')

    if file:
        # Read the uploaded image
        img = Image.open(io.BytesIO(file.read()))
        img = img.resize((256, 256))  # Resize to match your model's input size
        

        
        img_array = np.array(img)  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)     

        predictions = model.predict(img_array)

        predicted_class_index = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_class_index]       
        
        return render_template('index.html', data=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
