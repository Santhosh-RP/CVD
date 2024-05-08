# app/app.py
import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
model = load_model('app/model/your_trained_model.h5')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path, target_size=(128, 128)):
    image = Image.open(image_path)
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        if 'image_file' not in request.files:
            return render_template('index.html', prediction="No file uploaded")
        
        file = request.files['image_file']
        if file.filename == '':
            return render_template('index.html', prediction="No file selected")

        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Preprocess the uploaded image and make prediction
            preprocessed_image = preprocess_image(file_path)
            prediction_score = model.predict(preprocessed_image)
            prediction = "Diabetic Retinopathy (DR)" if prediction_score >= 0.5 else "No Diabetic Retinopathy"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
