import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

from data_loader import get_data_generators
from evaluate import get_latest_run_dir

BASE_DIR = '../'

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="relu"):
    """
    Generates a Grad-CAM heatmap for a given image array and model.
    'relu' is the standard final convolutional activation layer in DenseNet121.
    """
    # Create a model that maps the input image to the activations of the last conv layer,
    # as well as the final prediction layer
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        # For binary classification, preds is a single probability
        class_channel = preds[:, 0]

    # This is the gradient of the output neuron with respect to the output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Vector where each entry is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map array by "how important this channel is"
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to the heatmap to only keep features that have a positive influence
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img_array, heatmap, alpha=0.5):
    """Overlays the heatmap onto the original image with thick, beautiful coloring."""
    # Convert original image from [0, 1] float to [0, 255] uint8
    img = np.uint8(255 * img_array[0])
    
    # Resize the tiny heatmap to match the original image dimensions
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # NEW: Apply Gaussian blur to smooth out the blocky upscaling artifacts
    # This creates thick, organic, smooth patches of color
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
    
    # NEW: Apply a slight power transformation to intensify the "hot" spots
    # and suppress the low-confidence background noise
    heatmap = np.power(heatmap, 1.5)
    
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # NEW: Use COLORMAP_TURBO (or COLORMAP_INFERNO) for highly vibrant, modern, deep colors
    color_map = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)
    
    # Keras loads images as RGB, cv2 uses BGR. Convert the colormap to RGB
    color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)

    # NEW: Safely blend the images using cv2.addWeighted for a clean, saturated overlay
    superimposed_img = cv2.addWeighted(color_map, alpha, img, 1 - alpha, 0)
    
    return tf.keras.utils.array_to_img(superimposed_img)

def main():
    latest_run_dir = get_latest_run_dir()
    
    # Ensure we load the fine-tuned model if available
    if 'finetune' in latest_run_dir:
        model_path = os.path.join(latest_run_dir, 'best_finetuned_model.keras')
    else:
        model_path = os.path.join(latest_run_dir, 'best_model.keras')
        
    print(f"Loading model for Grad-CAM: {model_path}")
    model = load_model(model_path)

    print("Loading a batch of test images...")
    _, _, test_gen = get_data_generators(base_dir=BASE_DIR, batch_size=5)
    
    # Grab one batch of images and labels
    images, labels = next(test_gen)
    predictions = model.predict(images)

    # Plot the first 5 images in the batch
    plt.figure(figsize=(20, 8))
    for i in range(5):
        img_array = np.expand_dims(images[i], axis=0)
        
        # Generate the heatmap
        heatmap = make_gradcam_heatmap(img_array, model)
        overlaid_img = display_gradcam(img_array, heatmap)

        # Formatting the titles
        true_label = "Fracture" if labels[i] == 1 else "Normal"
        pred_label = "Fracture" if predictions[i][0] > 0.5 else "Normal"
        confidence = predictions[i][0] if pred_label == "Fracture" else 1 - predictions[i][0]

        plt.subplot(1, 5, i + 1)
        plt.imshow(overlaid_img)
        plt.title(f"True: {true_label}\nPred: {pred_label} ({confidence:.2f})")
        plt.axis('off')

    # Save the output visualization
    save_path = os.path.join(latest_run_dir, 'gradcam_visualizations.png')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Grad-CAM visual saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()