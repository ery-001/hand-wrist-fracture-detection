import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

from data_loader import get_data_generators
from evaluate import get_latest_run_dir

BASE_DIR = '../'
CLINICAL_THRESHOLD = 0.30  # Optimized ER Triage threshold

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="relu"):
    """
    Generates a Grad-CAM heatmap for a given image array and model.
    'relu' is the standard final convolutional activation layer in DenseNet121.
    """
    # FIX: Removed the list brackets around model.input to clear the UserWarning.
    grad_model = tf.keras.models.Model(
        model.input, 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img_array, heatmap, prediction_score, alpha=0.4):
    """
    Overlays the heatmap onto the original image and draws a Target Lock 
    bounding box if a fracture is detected.
    """
    # 1. Un-normalize the image for display purposes
    # DenseNet preprocessing alters pixel values mathematically. We scale it back to 0-255.
    img = img_array[0]
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-5)
    img = np.uint8(255 * img)
    
    # 2. Resize and format the heatmap
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # 3. Create the JET thermal overlay
    jet = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB) # Keras/Matplotlib uses RGB

    # 4. Superimpose the heatmap onto the X-ray
    superimposed_img = cv2.addWeighted(img, 1 - alpha, jet, alpha, 0)

    # 5. THE TARGET LOCK FEATURE
    if prediction_score >= CLINICAL_THRESHOLD:
        # Find the hottest 20% of pixels in the heatmap
        _, thresh = cv2.threshold(heatmap_uint8, int(255 * 0.8), 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest cluster of heat
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Draw the clinical bounding box (Red in RGB)
            cv2.rectangle(superimposed_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(superimposed_img, "TARGET", (x, max(y-5, 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return superimposed_img

def main():
    latest_run_dir = get_latest_run_dir()
    
    if 'finetune' in latest_run_dir:
        model_path = os.path.join(latest_run_dir, 'best_finetuned_model.keras')
    else:
        model_path = os.path.join(latest_run_dir, 'best_model.keras')
        
    print(f"\n[SYSTEM] Loading model for Grad-CAM: {model_path}")
    model = load_model(model_path)

    print("[DATA] Loading a batch of test images...")
    _, _, test_gen = get_data_generators(base_dir=BASE_DIR, batch_size=5)
    
    # Grab one batch of images and labels
    images, labels = next(test_gen)
    predictions = model.predict(images)

    # Setup the presentation slide
    plt.figure(figsize=(20, 8))
    
    for i in range(5):
        img_array = np.expand_dims(images[i], axis=0)
        raw_score = predictions[i][0]
        
        # Generate mathematical heatmap
        heatmap = make_gradcam_heatmap(img_array, model)
        
        # Pass the raw score so the function knows whether to draw the Target Lock
        overlaid_img = display_gradcam(img_array, heatmap, raw_score)

        # Formatting the titles based on the ER Triage threshold
        true_label = "Fracture" if labels[i] == 1 else "Normal"
        pred_label = "Fracture" if raw_score >= CLINICAL_THRESHOLD else "Normal"

        plt.subplot(1, 5, i + 1)
        plt.imshow(overlaid_img)
        
        # Color code the text for the presentation: Green if correct, Red if wrong
        title_color = 'green' if true_label == pred_label else 'red'
        plt.title(f"True: {true_label}\nPred: {pred_label} ({raw_score:.2f})", color=title_color, fontweight='bold')
        plt.axis('off')

    # Save and display
    save_path = os.path.join(latest_run_dir, 'gradcam_visualizations.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[SUCCESS] Grad-CAM visual saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()