import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Import our existing custom tools
from evaluate import get_latest_run_dir
from data_loader import apply_clahe
from grad_cam import make_gradcam_heatmap, display_gradcam

def load_and_preprocess_image(img_path):
    """Prepares a raw image exactly how the training generator did."""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Could not find image at: {img_path}")

    # 1. Read the image using OpenCV
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Invalid image file.")

    # OpenCV reads in BGR format, but our model was trained on RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Resize to the DenseNet121 standard input size (224x224)
    img = cv2.resize(img, (224, 224))

    # 3. Apply your custom CLAHE enhancement
    img = apply_clahe(img)

    # 4. Rescale pixels to 0-1 (Mimicking rescale=1./255 in ImageDataGenerator)
    img = img / 255.0

    # 5. Expand dimensions so it looks like a "batch" of 1 image: shape becomes (1, 224, 224, 3)
    img_array = np.expand_dims(img, axis=0)
    return img_array

def main(image_path):
    # 1. Find and load the fully trained model
    latest_run_dir = get_latest_run_dir()
    if 'finetune' in latest_run_dir:
        model_path = os.path.join(latest_run_dir, 'best_finetuned_model.keras')
    else:
        model_path = os.path.join(latest_run_dir, 'best_model.keras')

    print(f"Loading model from: {model_path}")
    model = load_model(model_path)

    # 2. Process the raw user image
    print(f"Processing image: {image_path}")
    img_array = load_and_preprocess_image(image_path)

    # 3. Make the Clinical Prediction
    # prediction returns an array like [[0.85]], so we grab the exact number
    prediction_score = model.predict(img_array, verbose=0)[0][0]

    if prediction_score > 0.30:
        diagnosis = "FRACTURE DETECTED"
        confidence = prediction_score * 100
    else:
        diagnosis = "NORMAL (NO FRACTURE)"
        confidence = (1 - prediction_score) * 100

    # Print a clean clinical report to the terminal
    print("\n" + "="*40)
    print(f" CLINICAL DIAGNOSIS : {diagnosis}")
    print(f" AI CONFIDENCE      : {confidence:.2f}%")
    print("="*40 + "\n")

    # 4. Generate the Explainable AI visual
    print("Generating XAI Heatmap...")
    heatmap = make_gradcam_heatmap(img_array, model, 'relu')
    overlaid_img = display_gradcam(img_array, heatmap, prediction_score)

    # 5. Display the final result
    plt.figure(figsize=(6, 6))
    plt.imshow(overlaid_img)
    plt.title(f"{diagnosis}\nConfidence: {confidence:.2f}%")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Ensure the user provided an image path when running the script
    if len(sys.argv) < 2:
        print("Usage: python3 predict.py <path_to_your_xray_image.png>")
        sys.exit(1)
    
    # Pass the image path from the terminal into the main function
    main(sys.argv[1])