import os
from flask import Flask, render_template, request, jsonify
from inference import predict_disease
from remedy import generate_remedy

app = Flask(__name__)

# Configure Uploads
basedir = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(basedir, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5MB limit
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
        
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected image'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only JPG and PNG are allowed.'}), 400
        
    if file:
        try:
            # Read image bytes for prediction
            image_bytes = file.read()
            
            # Predict
            predictions = predict_disease(image_bytes)
            
            # Generate remedy if a disease was found
            if 'disease' in predictions:
                remedy = generate_remedy(predictions['disease'])
                if remedy:
                    predictions['remedy'] = remedy
            
            return jsonify(predictions)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
