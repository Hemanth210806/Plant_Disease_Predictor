import os
import json
import time
from PIL import Image
import io

try:
    import tensorflow as tf
    import numpy as np
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

MODEL_PATH = 'model.keras'
CLASSES_PATH = 'classes.json'

model = None
class_names = []

# Load Classes first
if TF_AVAILABLE:
    if os.path.exists(CLASSES_PATH):
        try:
            with open(CLASSES_PATH, 'r') as f:
                class_names = json.load(f)
            print(f"Successfully loaded {len(class_names)} classes.")
        except Exception as e:
            print(f"Error loading classes: {e}")

    # Then load Model
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"Successfully loaded model from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model directly: {e}")
            print("Attempting to rebuild architecture and load weights...")
            try:
                img_height = 224
                img_width = 224
                base_model = tf.keras.applications.MobileNetV2(
                    input_shape=(img_height, img_width, 3),
                    include_top=False,
                    weights=None
                )
                model = tf.keras.Sequential([
                    tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
                    base_model,
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dense(len(class_names), activation='softmax')
                ])
                model.load_weights(MODEL_PATH)
                print(f"Successfully loaded model weights from {MODEL_PATH}")
            except Exception as e2:
                print(f"Failed to load weights: {e2}")

def format_disease_name(raw_name):
    # e.g., "apple_black_rot" -> "Apple Black Rot"
    return " ".join(word.capitalize() for word in raw_name.split('_') if word)

def predict_disease(image_bytes):
    """
    Predicts the disease from an image.
    Returns the top 2 predictions.
    If the top confidence is < 0.5, returns "Unknown".
    """
    if model is not None and class_names:
        try:
            # 1. Preprocess the image
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((224, 224))
            
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            # 2. Predict (Rescaling is handled by the model's first layer)
            predictions = model.predict(img_array, verbose=0)[0]
            
            # 3. Get top 2 indices
            top_indices = np.argsort(predictions)[-2:][::-1]
            
            top_1_idx = top_indices[0]
            top_1_conf = float(predictions[top_1_idx])
            
            # Check for unknown
            if top_1_conf < 0.5:
                return {
                    'disease': 'Unknown Disease / Not a Leaf',
                    'confidence': round(top_1_conf * 100, 2),
                    'is_unknown': True
                }
            
            # Format successful prediction
            top_1_name = format_disease_name(class_names[top_1_idx])
            result = {
                'disease': top_1_name,
                'confidence': round(top_1_conf * 100, 2),
                'is_unknown': False
            }
            
            # Add top 2 if available
            if len(top_indices) > 1:
                top_2_idx = top_indices[1]
                top_2_conf = float(predictions[top_2_idx])
                top_2_name = format_disease_name(class_names[top_2_idx])
                result['top_2'] = {
                    'disease': top_2_name,
                    'confidence': round(top_2_conf * 100, 2)
                }
                
            return result

        except Exception as e:
            print(f"Prediction error: {e}")
            pass

    # MOCK PREDICTION (Fallback if no model)
    time.sleep(1.5)  # Simulate processing time
    
    # Simulate a fake prediction
    return {
        'disease': 'Apple Black Rot (Mock)',
        'confidence': 92.4,
        'is_unknown': False,
        'top_2': {
            'disease': 'Apple Scab (Mock)',
            'confidence': 5.1
        }
    }
