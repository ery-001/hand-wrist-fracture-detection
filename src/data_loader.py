import os
import cv2
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# CRITICAL FIX: Importing DenseNet's specific mathematical normalizer
from tensorflow.keras.applications.densenet import preprocess_input

# =====================================================================
# CUSTOM PREPROCESSING (CLAHE + IMAGENET NORMALIZATION)
# =====================================================================
def apply_clahe_and_preprocess(img):
    """
    Step 1: Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
            to aggressively highlight micro-fractures and bone structures.
    Step 2: Normalizes the tensor using DenseNet121's expected ImageNet statistics
            (Mean subtraction and standard deviation scaling) instead of naive 1/255 scaling.
    """
    # Keras passes images as float32; OpenCV CLAHE expects uint8 (0-255)
    img_uint8 = img.astype('uint8')
    
    # X-rays are loaded as RGB by Keras. Convert to grayscale for CLAHE.
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    
    # DenseNet expects 3 channels (RGB). Convert back.
    enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
    
    # Convert back to float32 for TensorFlow mathematical operations
    img_float32 = enhanced_rgb.astype('float32')
    
    # Apply the native DenseNet normalization
    # This replaces the fatal 'rescale=1./255' logic
    final_img = preprocess_input(img_float32)
    
    return final_img

# =====================================================================
# DATAFRAME CONSTRUCTION & BALANCING
# =====================================================================
def prep_dataframe(csv_path):
    """
    Loads the MURA CSV, isolates Hand/Wrist cases, extracts binary labels,
    and shuffles the data to prevent chronological or sequential bias.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find dataset CSV at: {csv_path}")

    # Read MURA's pathlist
    df = pd.read_csv(csv_path, header=None, names=['image_path'])
    
    # SEMINAR DEFENSE: We filter strictly for Hand and Wrist to narrow 
    # the anatomical scope and improve the model's specialized accuracy.
    df = df[df['image_path'].str.contains('XR_HAND|XR_WRIST')]
    
    # Extract the binary label from the folder naming convention
    # 1 = Fracture/Abnormal, 0 = Normal
    df['label'] = df['image_path'].apply(lambda x: '1' if 'positive' in x else '0')
    
    # VITAL: Shuffle the entire dataframe so classes are evenly distributed
    # throughout the epochs, preventing erratic gradient jumps.
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

# =====================================================================
# DATA GENERATOR PIPELINE
# =====================================================================
def get_data_generators(base_dir='.', batch_size=32, target_size=(224, 224)):
    """
    Constructs the Training, Validation, and Testing generators.
    Implements a 70/15/15 split using MURA's train/valid directories.
    """
    train_csv = os.path.join(base_dir, 'MURA-v1.1', 'train_image_paths.csv')
    valid_csv = os.path.join(base_dir, 'MURA-v1.1', 'valid_image_paths.csv')
    
    # Prepare the dataframes
    train_df = prep_dataframe(train_csv)
    test_df = prep_dataframe(valid_csv) # Using MURA's 'valid' set as our absolute Test set
    
    print(f"\n[DATA] Total Training/Validation Images: {len(train_df)}")
    print(f"[DATA] Total Testing Images: {len(test_df)}")

    # 1. Training & Validation Augmentation Strategy
    # NOTE: rescale=1./255 has been strictly removed here.
    train_datagen = ImageDataGenerator(
        rotation_range=15,          # Mild rotation to simulate imperfect patient positioning
        horizontal_flip=True,       # Left/Right hand variations
        zoom_range=0.2,             # Simulate different camera distances
        preprocessing_function=apply_clahe_and_preprocess, # Our custom logic
        validation_split=0.176      # Extracts ~15% of total data for validation
    )

    # 2. Testing Generator Strategy
    # Zero augmentation allowed here. Only scaling and CLAHE processing.
    test_datagen = ImageDataGenerator(
        preprocessing_function=apply_clahe_and_preprocess
    )

    # --- Initialize Flow Streams ---
    
    print("\n[DATA] Building Training Generator (Sub-sampled)...")
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=base_dir,  # Paths in CSV are relative to base_dir
        x_col='image_path',
        y_col='label',
        target_size=target_size,
        class_mode='binary',
        batch_size=batch_size,
        subset='training',
        shuffle=True
    )

    print("[DATA] Building Validation Generator (Sub-sampled)...")
    valid_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=base_dir,
        x_col='image_path',
        y_col='label',
        target_size=target_size,
        class_mode='binary',
        batch_size=batch_size,
        subset='validation',
        shuffle=False # Strict rule: Never shuffle validation data
    )

    print("[DATA] Building Testing Generator (From valid_csv)...")
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=base_dir,
        x_col='image_path',
        y_col='label',
        target_size=target_size,
        class_mode='binary',
        batch_size=batch_size,
        shuffle=False # Strict rule: Maintain order for accurate Confusion Matrix mapping
    )

    return train_generator, valid_generator, test_generator

# =====================================================================
# STANDALONE TESTING BLOCK
# =====================================================================
if __name__ == "__main__":
    print("Testing data loader pipeline...")
    try:
        # Defaults to looking for 'MURA-v1.1' in the current directory
        train_gen, val_gen, test_gen = get_data_generators()
        print("\nSUCCESS: Data loading pipeline is fully functional!")
        print(f"Batch shape: {train_gen[0][0].shape}") 
    except Exception as e:
        print(f"\nERROR: Could not run pipeline. Make sure MURA-v1.1 folder is in the root directory.\nDetails: {e}")