import os
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Import your custom modules from the other files!
from data_loader import get_data_generators
from model import build_model

# --- Configuration ---
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
BASE_DIR = '../' # Points to the root where MURA-v1.1 is located

def setup_run_folder():
    """Creates a unique timestamped folder inside runs/ to save this specific training attempt."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(BASE_DIR, 'runs', f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def plot_training_history(history, run_dir):
    """Saves a graph of the training accuracy and loss."""
    plt.figure(figsize=(12, 4))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.legend()

    plt.savefig(os.path.join(run_dir, 'training_history.png'))
    plt.close()

def main():
    # 1. Setup the safe directory
    run_dir = setup_run_folder()
    print(f"Starting training. All outputs will be saved to: {run_dir}")

    # 2. Bring in the Data
    print("Loading datasets...")
    train_gen, valid_gen, test_gen = get_data_generators(base_dir=BASE_DIR, batch_size=BATCH_SIZE)

    # NEW: 3. Calculate Class Weights to fix the imbalance
    print("Calculating class weights...")
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights applied: {class_weight_dict}")

    # 4. Bring in the Architecture
    print("Building model...")
    model = build_model(learning_rate=LEARNING_RATE)

    # 5. Set up Callbacks
    callbacks = [
        ModelCheckpoint(filepath=os.path.join(run_dir, 'best_model.keras'), monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        CSVLogger(os.path.join(run_dir, 'training_log.csv'))
    ]

    # 6. START TRAINING (Now passing the class_weight_dict!)
    print("Commencing training loop...")
    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=EPOCHS,
        class_weight=class_weight_dict, # This line forces it to focus on fractures!
        callbacks=callbacks
    )

    # 7. Save final charts
    plot_training_history(history, run_dir)
    print("Training complete! Model and charts are saved.")


if __name__ == "__main__":
    main()