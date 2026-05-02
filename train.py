import os
import shutil
import json
import subprocess
import zipfile
import tensorflow as tf
import tensorflow as tf
import keras

# Configuration
DATASET_NAME = "emmarex/plantdisease"
DATASET_ZIP = "plantdisease.zip"
BASE_DIR = "dataset_raw"
FILTERED_DIR = "dataset"
MODEL_PATH = "model.keras"
CLASSES_PATH = "classes.json"

# Specific crops to include
TARGET_CROPS = ["Tomato", "Potato", "Corn_(maize)", "Apple", "Grape"]

def download_and_extract():
    print("Downloading dataset from Kaggle...")
    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(DATASET_NAME, path=BASE_DIR, unzip=True)
        print("Download and extraction complete.")
    except Exception as e:
        print("Error downloading or extracting dataset. Please ensure your Kaggle API key is setup in ~/.kaggle/kaggle.json")
        print(f"Details: {e}")
        return False
    return True

def filter_dataset():
    print("Filtering dataset for selected crops...")
    if os.path.exists(FILTERED_DIR):
        shutil.rmtree(FILTERED_DIR)
    os.makedirs(FILTERED_DIR)

    # The kaggle dataset extracts into a structure like: dataset_raw/PlantVillage/
    # We will search for the PlantVillage directory
    plantvillage_dir = None
    for root, dirs, files in os.walk(BASE_DIR):
        if 'PlantVillage' in dirs:
            plantvillage_dir = os.path.join(root, 'PlantVillage')
            break
            
    if not plantvillage_dir:
        # Sometimes it extracts directly
        plantvillage_dir = os.path.join(BASE_DIR, 'plantvillage')
        if not os.path.exists(plantvillage_dir):
            plantvillage_dir = BASE_DIR # Fallback
            
    classes_found = 0
    for class_name in os.listdir(plantvillage_dir):
        class_path = os.path.join(plantvillage_dir, class_name)
        if os.path.isdir(class_path):
            # Check if this class belongs to our target crops
            if any(crop in class_name for crop in TARGET_CROPS):
                # Clean up the name e.g., "Apple___Black_rot" -> "apple_black_rot"
                clean_name = class_name.lower().replace('___', '_').replace('__', '_').replace(' ', '_').replace('(', '').replace(')', '')
                dest_path = os.path.join(FILTERED_DIR, clean_name)
                shutil.copytree(class_path, dest_path)
                classes_found += 1
                
    print(f"Filtered {classes_found} classes belonging to {TARGET_CROPS}.")

def train_model():
    print("Loading dataset for training...")
    batch_size = 32
    img_height = 224
    img_width = 224

    train_ds = keras.utils.image_dataset_from_directory(
        FILTERED_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = keras.utils.image_dataset_from_directory(
        FILTERED_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    print("Classes:", class_names)
    
    # Save classes to JSON
    with open(CLASSES_PATH, 'w') as f:
        json.dump(class_names, f)

    # Setup the model architecture
    base_model = keras.applications.MobileNetV2(
        input_shape=(img_height, img_width, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze base layers

    model = keras.Sequential([
        keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Starting training...")
    epochs = 10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    print("Saving model to", MODEL_PATH)
    model.save(MODEL_PATH)
    print("Training complete!")

if __name__ == "__main__":
    if not os.path.exists(FILTERED_DIR):
        if not os.path.exists(BASE_DIR):
            success = download_and_extract()
            if not success:
                exit(1)
        filter_dataset()
    else:
        print("Filtered dataset already exists. Skipping download and filtering.")
        
    train_model()
