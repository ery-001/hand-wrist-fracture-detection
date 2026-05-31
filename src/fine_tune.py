import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

# Import from your existing files
from data_loader import get_data_generators

# --- Configuration ---
BASE_DIR = '../'
BATCH_SIZE = 32
EPOCHS = 40 # Increased because we have smarter stopping mechanisms now
# Starting with a very gentle learning rate
FINE_TUNE_LEARNING_RATE = 1e-5 

def setup_finetune_folder():
    """Creates a new folder specifically for the fine-tuned model outputs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(BASE_DIR, 'runs', f'finetune_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def get_latest_base_run():
    """Specifically hunts for the latest base training run, ignoring finetune folders."""
    runs_dir = os.path.join(BASE_DIR, 'runs')
    base_runs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) 
                 if os.path.isdir(os.path.join(runs_dir, d)) and os.path.basename(d).startswith('run_')]
    
    if not base_runs:
        raise FileNotFoundError("Could not find any base 'run_' directories.")
        
    return max(base_runs, key=os.path.getmtime)

def main():
    run_dir = setup_finetune_folder()
    print(f"Starting Deep Fine-Tuning. Outputs will save to: {run_dir}")

    # Load your best saved base model
    latest_run_dir = get_latest_base_run()
    model_path = os.path.join(latest_run_dir, 'best_model.keras')
    print(f"Loading base model from: {model_path}")
    model = load_model(model_path)

    # --- THE BUFF: DEEP UNFREEZING ---
    print("Unfreezing the top 150 layers of DenseNet121...")
    model.trainable = True 
    
    # We are unlocking the entire top block of the neural network
    for layer in model.layers[:-150]:
        layer.trainable = False

    # Recompile
    model.compile(
        optimizer=Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )

    print("Loading datasets...")
    train_gen, valid_gen, _ = get_data_generators(base_dir=BASE_DIR, batch_size=BATCH_SIZE)
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weight_dict = dict(enumerate(class_weights))

    # --- THE BUFF: SMART LEARNING RATE ---
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2,   # If stuck, drop learning rate to 20% of its current value
        patience=2,   # Wait 2 epochs before dropping
        min_lr=1e-7,  # Never drop below this microscopic limit
        verbose=1     # Print a message to the terminal when it drops
    )

    callbacks = [
        ModelCheckpoint(filepath=os.path.join(run_dir, 'best_finetuned_model.keras'), monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True), # Increased patience so it doesn't quit too early
        CSVLogger(os.path.join(run_dir, 'finetuning_log.csv')),
        reduce_lr # Added the new smart callback here
    ]

    print("Commencing deep fine-tuning loop on M2 GPU...")
    model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=EPOCHS,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )
    print("Deep Fine-tuning complete!")

if __name__ == "__main__":
    main()