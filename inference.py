import os
import json
import time
from PIL import Image
import io
import gc

# --- LINUX MEMORY HACKS FOR RENDER (512MB RAM) ---
os.environ['MALLOC_ARENA_MAX'] = '2' # Prevents memory hoarding
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Force CPU only
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

try:
    import tensorflow as tf
    import numpy as np
    
    # LIMIT THREADS TO SAVE RAM
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    
    # Disable GPU devices explicitly
    tf.config.set_visible_devices([], 'GPU')
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
            print(f"DEBUG: Attempting to load model from {MODEL_PATH}...")
            # Clear any existing session to free RAM
            tf.keras.backend.clear_session()
            # Use compile=False to avoid issues with custom optimizers or metrics during loading
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print(f"SUCCESS: Model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"WARNING: Direct model load failed. This is usually due to Keras version differences. Error: {e}")
            print("INFO: Attempting to rebuild architecture manually to recover...")
            try:
                img_height = 224
                img_width = 224
                
                # Create Functional architecture to match training
                inputs = tf.keras.Input(shape=(img_height, img_width, 3))
                # Normalization is often part of the model or done manually
                x = tf.keras.layers.Rescaling(1./255)(inputs)
                
                base_model = tf.keras.applications.MobileNetV2(
                    input_shape=(img_height, img_width, 3),
                    include_top=False,
                    weights=None
                )
                
                x = base_model(x)
                x = tf.keras.layers.GlobalAveragePooling2D()(x)
                x = tf.keras.layers.Dense(128, activation='relu')(x)
                outputs = tf.keras.layers.Dense(len(class_names), activation='softmax')(x)
                
                model = tf.keras.Model(inputs, outputs)
                
                print("INFO: Attempting to load weights into functional architecture...")
                model.load_weights(MODEL_PATH)
                print(f"SUCCESS: Model weights loaded successfully from {MODEL_PATH}")
            except Exception as e2:
                print(f"CRITICAL: Failed to load weights: {e2}")
                model = None # Ensure it falls back to mock if everything fails

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
            
            # Format successful prediction or Unknown
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
            
            # Add top 2 if available
            if len(top_indices) > 1 and not result['is_unknown']:
                top_2_idx = top_indices[1]
                top_2_conf = float(predictions[top_2_idx])
                top_2_name = format_disease_name(class_names[top_2_idx])
                result['top_2'] = {
                    'disease': top_2_name,
                    'confidence': round(top_2_conf * 100, 2)
                }
            
            # Cleanup large objects explicitly
            del img
            del img_array
            del predictions
            gc.collect()
                
            return result

        except Exception as e:
            print(f"Prediction error: {e}")
            gc.collect()
            pass

    # MOCK PREDICTION (Fallback if no model)
    time.sleep(1.5)
    return {
        'disease': 'Potato Early Blight (Demo)',
        'confidence': 98.2,
        'is_unknown': False
    }
