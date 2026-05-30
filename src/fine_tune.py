import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.utils.class_weight import compute_class_weight

# Import from your existing files
from data_loader import get_data_generators

# --- Configuration ---
BASE_DIR = '../'
BATCH_SIZE = 32
EPOCHS = 25
# Notice how tiny this learning rate is (0.00001). 
# We want to gently tweak the weights, not destroy what it already learned!
FINE_TUNE_LEARNING_RATE = 1e-5 

def setup_finetune_folder():
    """Creates a new folder specifically for the fine-tuned model outputs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(BASE_DIR, 'runs', f'finetune_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def get_latest_base_run():
    runs_dir = os.path.abspath(os.path.join(BASE_DIR, 'runs')) # Use Absolute Path
    if not os.path.exists(runs_dir):
        raise FileNotFoundError(f"Directory not found: {runs_dir}")
        
    base_runs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) 
                 if os.path.basename(d).startswith('run_')]
    
    if not base_runs:
        raise FileNotFoundError("Could not find any base 'run_' directories.")
        
    return max(base_runs, key=os.path.getmtime)

def main():
    # 1. Setup the new save directory
    run_dir = setup_finetune_folder()
    print(f"Starting Fine-Tuning. Outputs will save to: {run_dir}")

    # 2. Find and Load your best saved base model
    latest_run_dir = get_latest_base_run()
    model_path = os.path.join(latest_run_dir, 'best_model.keras')
    print(f"Loading base model from: {model_path}")
    model = load_model(model_path)

    # 3. UNFREEZE THE BACKBONE
    print("Unfreezing the top 50 layers of DenseNet121...")
    model.trainable = True # Unlock the whole model first
    
    # Freeze all layers EXCEPT the last 50
    # This keeps the basic edge-detection intact, but allows the complex 
    # pattern-recognition layers to learn "bone fractures"
    for layer in model.layers[:-50]:
        layer.trainable = False

    # 4. Recompile the model with the microscopic learning rate
    model.compile(
        optimizer=Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )

    # 5. Load Datasets and compute Class Weights
    print("Loading datasets...")
    train_gen, valid_gen, _ = get_data_generators(base_dir=BASE_DIR, batch_size=BATCH_SIZE)
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weight_dict = dict(enumerate(class_weights))

    # 6. Callbacks
    callbacks = [
        ModelCheckpoint(filepath=os.path.join(run_dir, 'best_finetuned_model.keras'), monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
        CSVLogger(os.path.join(run_dir, 'finetuning_log.csv'))
    ]

    # 7. START FINE-TUNING!
    print("Commencing fine-tuning loop on M2 GPU...")
    model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=EPOCHS,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )
    print("Fine-tuning complete!")

if __name__ == "__main__":
    main()