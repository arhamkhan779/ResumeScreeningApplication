from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
import PyPDF2
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

# Load the model and preprocessor
model = keras.models.load_model("D:\\NLP\\ResumeScreeningApplication\\artifacts\\training\\trained_model.h5")
text_preprocessor = joblib.load("D:\\NLP\\ResumeScreeningApplication\\artifacts\\data_preprocess\\text_preprocessor.pkl")

# Categories
categories = ['Arts', 'Mechanical Engineer', 'DevOps Engineer', 'Hadoop', 'ETL Developer', 'Blockchain', 'Civil Engineer', 
              'Electrical Engineering', 'PMO', 'SAP Developer', 'HR', 'DotNet Developer', 'Python Developer', 'Operations Manager',
              'Data Science', 'Database', 'Business Analyst', 'Web Designing', 'Testing', 'Health and fitness', 
              'Network Security Engineer', 'Automation Testing', 'Sales', 'Java Developer', 'Advocate']

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check if file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "No file part"})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"success": False, "message": "No selected file"})
    
    if file and allowed_file(file.filename):
        # Ensure the upload folder exists
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the file
        file.save(file_path)

        # Extract text from the PDF
        text = extract_text_from_pdf(file_path)

        # Apply text preprocessing
        processed_text = text_preprocessor.transform([text])
        
        # Prediction
        prediction = model.predict(processed_text)
        predicted_category = categories[np.argmax(prediction)]

        return jsonify({"success": True, "category": predicted_category})
    else:
        return jsonify({"success": False, "message": "Invalid file type. Only PDF is allowed."})

if __name__ == '__main__':
    app.run(debug=True)
