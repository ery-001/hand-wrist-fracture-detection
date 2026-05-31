import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model

# Import our existing custom tools
from evaluate import get_latest_run_dir
from data_loader import apply_clahe
from grad_cam import make_gradcam_heatmap, display_gradcam

def load_and_preprocess_image(img_path):
    """Prepares a raw image exactly how the training generator did."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read the image file at: {img_path}")

    # OpenCV reads in BGR format, convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize, apply CLAHE, and normalize
    img = cv2.resize(img, (224, 224))
    img = apply_clahe(img)
    img = img / 255.0

    # Expand dimensions for the model (1, 224, 224, 3)
    img_array = np.expand_dims(img, axis=0)
    return img_array

def select_image_gui():
    """Opens a Mac file browser UI to let the user select an X-ray."""
    # Create a hidden root window for tkinter
    root = tk.Tk()
    root.withdraw() 
    
    # Force the window to appear on top of other applications
    root.call('wm', 'attributes', '.', '-topmost', True)

    print("Opening file browser window...")
    file_path = filedialog.askopenfilename(
        title="Upload Hand/Wrist X-Ray for AI Analysis",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
    )
    
    return file_path

def main():
    # 1. Trigger the GUI File Upload
    image_path = select_image_gui()
    
    # If the user clicks "Cancel" in the file browser, exit gracefully
    if not image_path:
        print("Upload canceled by user. Exiting...")
        return

    # 2. Find and load the fully trained model
    latest_run_dir = get_latest_run_dir()
    if 'finetune' in latest_run_dir:
        model_path = os.path.join(latest_run_dir, 'best_finetuned_model.keras')
    else:
        model_path = os.path.join(latest_run_dir, 'best_model.keras')

    print(f"\nLoading AI Model from: {model_path}")
    model = load_model(model_path)

    # 3. Process the uploaded image
    print(f"Processing uploaded X-Ray: {os.path.basename(image_path)}")
    img_array = load_and_preprocess_image(image_path)

    # 4. Make the Clinical Prediction
    prediction_score = model.predict(img_array, verbose=0)[0][0]

    # Using the balanced 0.5 threshold
    if prediction_score > 0.5:
        diagnosis = "FRACTURE DETECTED"
        confidence = prediction_score * 100
        text_color = 'red'
    else:
        diagnosis = "NORMAL (NO FRACTURE)"
        confidence = (1 - prediction_score) * 100
        text_color = 'green'

    # Print a clean report to the terminal
    print("\n" + "="*45)
    print(f" CLINICAL DIAGNOSIS : {diagnosis}")
    print(f" AI CONFIDENCE      : {confidence:.2f}%")
    print("="*45 + "\n")

    # 5. Generate the Explainable AI visual
    print("Generating XAI Heatmap...")
    heatmap = make_gradcam_heatmap(img_array, model)
    overlaid_img = display_gradcam(img_array, heatmap)

    # 6. Display the final GUI Result Window
    plt.figure(figsize=(8, 8))
    plt.imshow(overlaid_img)
    plt.title(f"AI Diagnosis: {diagnosis}\nConfidence: {confidence:.2f}%", 
              fontsize=16, fontweight='bold', color=text_color, pad=20)
    plt.axis('off')
    
    # Add a clean footer for the presentation
    plt.figtext(0.5, 0.05, f"File analyzed: {os.path.basename(image_path)}", 
                ha="center", fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()