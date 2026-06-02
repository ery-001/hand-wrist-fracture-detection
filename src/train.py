import os
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# TensorFlow & Keras Core Libraries
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

# Custom Module Imports (Your Pipeline)
from data_loader import get_data_generators
from model import build_model

# =====================================================================
# HYPERPARAMETER CONFIGURATION
# =====================================================================
# BATCH_SIZE: Set to 32 to prevent Out-Of-Memory (OOM) errors on the GPU 
# while still providing enough gradient variation per step.
BATCH_SIZE = 32

# EPOCHS: Maximum number of times the AI will see the entire dataset. 
# Capped at 30, but EarlyStopping will likely halt it sooner to prevent overfitting.
EPOCHS = 30

# LEARNING_RATE: 1e-4 (0.0001). We use a slightly larger learning rate here 
# than in fine-tuning because we need to aggressively train the newly added 
# custom classification head before we touch the deep DenseNet layers.
LEARNING_RATE = 1e-3 

# BASE_DIR: Relative path pointing to the root where the MURA dataset is located.
BASE_DIR = '../' 


# =====================================================================
# SYSTEM FUNCTIONS
# =====================================================================
def setup_run_folder():
    """
    Creates an isolated, timestamped directory for each training execution.
    
    Why this matters for the seminar: 
    In ML engineering, version control is critical. This ensures previous 
    models, weights, and metrics are never accidentally overwritten, 
    creating a perfect scientific paper trail of our experiments.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(BASE_DIR, 'runs', f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def plot_training_history(history, run_dir):
    """
    Generates high-quality mathematical visualizations of the training process.
    
    Why this matters for the seminar:
    Visualizing the divergence between the 'Train' and 'Val' curves is the 
    exact method used to mathematically prove if a model is 'Overfitting' 
    (memorizing) or successfully generalizing to unseen medical data.
    """
    plt.figure(figsize=(14, 5))
    
    # 1. Plot Accuracy Trajectory
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)
    plt.title('Model Accuracy Progress')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')

    # 2. Plot Loss (Binary Crossentropy) Trajectory
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
    plt.title('Binary Crossentropy Loss')
    plt.ylabel('Loss (Error Rate)')
    plt.xlabel('Epoch')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')

    # Save the artifact cleanly
    chart_path = os.path.join(run_dir, 'training_history.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()


# =====================================================================
# MAIN EXECUTION PIPELINE
# =====================================================================
def main():
    # 1. Initialize Logging Environment
    run_dir = setup_run_folder()
    print(f"\n[SYSTEM] Initializing Training Pipeline...")
    print(f"[SYSTEM] Saving artifacts to: {run_dir}\n")

    # 2. Data Ingestion & Preprocessing
    # Invokes data_loader.py to resize, apply CLAHE contrast enhancement, 
    # and split the data into Training, Validation, and Testing generators.
    print("[DATA] Loading and preprocessing MURA dataset...")
    train_gen, valid_gen, test_gen = get_data_generators(base_dir=BASE_DIR, batch_size=BATCH_SIZE)

    # 3. Class Weight Calculation (Crucial for Medical Imbalance)
    # Medical datasets inherently contain more 'Normal' cases than 'Fracture' cases.
    # This computes a mathematical penalty forcing the optimizer to penalize 
    # false negatives (missed fractures) more heavily than false positives.
    print("\n[MATH] Computing balanced class weights...")
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"[MATH] Applied Weights -> Normal (0): {class_weights[0]:.2f} | Fracture (1): {class_weights[1]:.2f}")

    # 4. Neural Network Architecture Integration
    # Invokes model.py to pull the pre-trained DenseNet121 ImageNet weights 
    # and attach our custom Binary Classification (Sigmoid) head.
    print("\n[MODEL] Constructing DenseNet121 Architecture...")
    model = build_model(learning_rate=LEARNING_RATE)

    # 5. Define Safety Nets & Optimization Callbacks
    callbacks = [
        # ModelCheckpoint: Only saves the file if the validation accuracy beats the previous best.
        ModelCheckpoint(
            filepath=os.path.join(run_dir, 'best_model.keras'), 
            monitor='val_accuracy', 
            save_best_only=True, 
            verbose=1
        ),
        # EarlyStopping: Kills training if loss doesn't improve for 8 epochs, 
        # and automatically rewinds the model weights to the best performing epoch.
        EarlyStopping(
            monitor='val_loss', 
            patience=8, 
            restore_best_weights=True,
            verbose=1
        ),
        # ReduceLROnPlateau: The "Smart Learning" mechanism. If validation loss plateaus, 
        # it microscopically shrinks the learning rate to find a deeper mathematical minimum.
        ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=3, 
            min_lr=1e-8,
            verbose=1
        ),
        # CSVLogger: Maintains a perfect spreadsheet of all metrics for defense validation.
        CSVLogger(os.path.join(run_dir, 'training_log.csv'))
    ]

    # 6. Execute Hardware-Accelerated Training Loop
    print("\n[EXECUTION] Commencing base training loop...")
    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=EPOCHS,
        class_weight=class_weight_dict, # Enforces the strict fracture focus
        callbacks=callbacks
    )

    # 7. Finalize and Export Artifacts
    print("\n[SYSTEM] Training concluded. Generating visual metrics...")
    plot_training_history(history, run_dir)
    print(f"[SYSTEM] Pipeline successfully terminated. All assets secured in {run_dir}")

if __name__ == "__main__":
    main()