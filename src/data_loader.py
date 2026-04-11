import pandas as pd
import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def apply_clahe(img):
    """
    Custom preprocessing function to apply Contrast Limited Adaptive Histogram Equalization.
    Enhances bone structures in X-rays before feeding them to the model.
    """
    # Keras passes images as float32; OpenCV CLAHE expects uint8
    img = img.astype('uint8')
    
    # X-rays are grayscale, but loaded as RGB by default. 
    # Convert to grayscale, apply CLAHE, and convert back to 3 channels for ResNet/DenseNet
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    
    # Convert back to RGB format
    enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
    
    # Return as float32 for TensorFlow
    return enhanced_rgb.astype('float32')

def prep_dataframe(csv_path):
    """
    Loads the MURA CSV, filters for Hand/Wrist, and extracts the 0/1 labels.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find dataset CSV at: {csv_path}")

    df = pd.read_csv(csv_path, header=None, names=['image_path'])
    
    # Filter to keep ONLY Hand and Wrist images
    df = df[df['image_path'].str.contains('XR_HAND|XR_WRIST')]
    
    # Extract the label (1 for positive/fracture, 0 for negative/normal)
    df['label'] = df['image_path'].apply(lambda x: '1' if 'positive' in x else '0')
    
    # NEW LINE: Shuffle the entire dataframe so positive/negative cases are evenly distributed!
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def get_data_generators(base_dir='.', batch_size=32, target_size=(224, 224)):
    """
    Creates and returns the Training, Validation, and Testing generators.
    Implements the 70/15/15 split by partitioning the MURA train set and 
    reserving the MURA valid set strictly for testing.
    """
    train_csv = os.path.join(base_dir, 'MURA-v1.1', 'train_image_paths.csv')
    valid_csv = os.path.join(base_dir, 'MURA-v1.1', 'valid_image_paths.csv')
    
    # Prepare the dataframes
    train_df = prep_dataframe(train_csv)
    test_df = prep_dataframe(valid_csv) # We use MURA's 'valid' set as our 15% Test set
    
    print(f"Total Training/Validation Images: {len(train_df)}")
    print(f"Total Testing Images: {len(test_df)}")

    # 1. Training & Validation Generator (with augmentation)
    # To get a 15% validation set out of the remaining 85% training data, the split is ~0.176
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        horizontal_flip=True,
        zoom_range=0.2,
        preprocessing_function=apply_clahe,
        validation_split=0.176 
    )

    # 2. Testing Generator (NO augmentation, only scaling and CLAHE)
    test_datagen = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=apply_clahe
    )

    # --- Create the actual flows ---
    
    # Training Flow (70%)
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=base_dir,  # Paths in MURA CSVs are relative to the root (e.g., 'MURA-v1.1/train/...')
        x_col='image_path',
        y_col='label',
        target_size=target_size,
        class_mode='binary',
        batch_size=batch_size,
        subset='training',
        shuffle=True
    )

    # Validation Flow (15%)
    valid_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=base_dir,
        x_col='image_path',
        y_col='label',
        target_size=target_size,
        class_mode='binary',
        batch_size=batch_size,
        subset='validation',
        shuffle=False # Do not shuffle validation data
    )

    # Testing Flow (15%)
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=base_dir,
        x_col='image_path',
        y_col='label',
        target_size=target_size,
        class_mode='binary',
        batch_size=batch_size,
        shuffle=False # Do not shuffle test data to maintain order for confusion matrix
    )

    return train_generator, valid_generator, test_generator

# Quick test block to ensure everything works if you run this script directly
if __name__ == "__main__":
    print("Testing data loader pipeline...")
    train_gen, val_gen, test_gen = get_data_generators()
    print("Data loading pipeline is functional!")