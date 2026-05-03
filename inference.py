import os
import json
import time
from PIL import Image
import io
import gc
import numpy as np
import onnxruntime as ort

# Configuration
ONNX_PATH = 'model.onnx'
CLASSES_PATH = 'classes.json'

# Initialize session
session = None
class_names = []

if os.path.exists(CLASSES_PATH):
    try:
        with open(CLASSES_PATH, 'r') as f:
            class_names = json.load(f)
        print(f"Successfully loaded {len(class_names)} classes.")
    except Exception as e:
        print(f"Error loading classes: {e}")

if os.path.exists(ONNX_PATH):
    try:
        print(f"DEBUG: Loading ONNX model from {ONNX_PATH}...")
        # Force CPU usage and limit threads to save RAM
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        
        session = ort.InferenceSession(ONNX_PATH, sess_options, providers=['CPUExecutionProvider'])
        print(f"SUCCESS: ONNX model loaded successfully.")
    except Exception as e:
        print(f"CRITICAL: Failed to load ONNX model: {e}")

def format_disease_name(raw_name):
    # e.g., "Tomato___Tomato_Yellow_Leaf_Curl_Virus" -> "Tomato Yellow Leaf Curl Virus"
    words = raw_name.replace('___', '_').split('_')
    # Remove duplicates while preserving order
    seen = set()
    unique_words = []
    for w in words:
        if w and w.lower() not in seen:
            unique_words.append(w.capitalize())
            seen.add(w.lower())
    return " ".join(unique_words)

def predict_disease(image_bytes):
    """
    Predicts the disease from an image using ONNX Runtime.
    """
    if session is not None and class_names:
        try:
            # 1. Preprocess the image
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((224, 224))
            
            # Convert to numpy (NO manual normalization here, 
            # because the model has a Rescaling(1/255) layer built-in!)
            img_array = np.array(img).astype(np.float32)
            img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
            
            # 2. Predict
            input_name = session.get_inputs()[0].name
            predictions = session.run(None, {input_name: img_array})[0][0]
            
            # 3. Get top 2 indices
            top_indices = np.argsort(predictions)[-2:][::-1]
            
            top_1_idx = top_indices[0]
            top_1_conf = float(predictions[top_1_idx])
            
            # Format results
            if top_1_conf < 0.5:
                result = {
                    'disease': 'Unknown Disease / Not a Leaf',
                    'confidence': round(top_1_conf * 100, 2),
                    'is_unknown': True
                }
            else:
                top_1_name = format_disease_name(class_names[top_1_idx])
                result = {
                    'disease': top_1_name,
                    'confidence': round(top_1_conf * 100, 2),
                    'is_unknown': False
                }
            
            if len(top_indices) > 1 and not result['is_unknown']:
                top_2_idx = top_indices[1]
                top_2_conf = float(predictions[top_2_idx])
                top_2_name = format_disease_name(class_names[top_2_idx])
                result['top_2'] = {
                    'disease': top_2_name,
                    'confidence': round(top_2_conf * 100, 2)
                }
            
            # Cleanup
            del img
            del img_array
            gc.collect()
                
            return result

        except Exception as e:
            print(f"Prediction error: {e}")
            gc.collect()

    # MOCK PREDICTION (Fallback)
    return {
        'disease': 'Potato Early Blight (Demo)',
        'confidence': 98.2,
        'is_unknown': False
    }
