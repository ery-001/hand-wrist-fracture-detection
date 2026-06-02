import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

# Custom Module Imports
from data_loader import get_data_generators

# =====================================================================
# HYPERPARAMETER CONFIGURATION
# =====================================================================
BASE_DIR = '../'
BATCH_SIZE = 32

# EPOCHS: Increased to 40 because fine-tuning requires smaller mathematical 
# steps. The EarlyStopping callback will safely kill it before 40 if it plateaus.
EPOCHS = 40 

# FINE_TUNE_LEARNING_RATE: 1e-5 (0.00001). 
# SEMINAR DEFENSE: "We used a microscopic learning rate here to prevent 
# 'Catastrophic Forgetting.' We want to gently nudge the pre-trained weights 
# to recognize fractures without destroying the foundational edge-detection 
# knowledge Google originally built into the architecture."
FINE_TUNE_LEARNING_RATE = 1e-5 


# =====================================================================
# SYSTEM FUNCTIONS
# =====================================================================
def setup_finetune_folder():
    """Creates an isolated, timestamped directory for fine-tuning outputs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(BASE_DIR, 'runs', f'finetune_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def get_latest_base_run():
    """
    Hunts for the most recent base training run to use as the foundation.
    Uses absolute paths to guarantee cross-platform compatibility (Mac/Windows).
    """
    runs_dir = os.path.abspath(os.path.join(BASE_DIR, 'runs'))
    base_runs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) 
                 if os.path.isdir(os.path.join(runs_dir, d)) and os.path.basename(d).startswith('run_')]
    
    if not base_runs:
        raise FileNotFoundError("Could not find any base 'run_' directories.")
        
    return max(base_runs, key=os.path.getmtime)


# =====================================================================
# MAIN DEEP FINE-TUNING PIPELINE
# =====================================================================
def main():
    # 1. Initialize Logging Environment
    run_dir = setup_finetune_folder()
    print(f"\n[SYSTEM] Starting Deep Fine-Tuning...")
    print(f"[SYSTEM] Outputs will save to: {run_dir}\n")

    # 2. Load the Foundation Model
    latest_run_dir = get_latest_base_run()
    model_path = os.path.join(latest_run_dir, 'best_model.keras')
    print(f"[MODEL] Loading base model from: {model_path}")
    model = load_model(model_path)

    # 3. DEEP UNFREEZING (The Core of Transfer Learning)
    # SEMINAR DEFENSE: "By unfreezing the deepest 150 layers, we are allowing 
    # the network's highest-level feature extractors to specifically adapt 
    # to the visual textures of bone splinters and micro-fractures, while 
    # keeping the lower-level geometric vision frozen and stable."
    print("\n[ARCHITECTURE] Unfreezing the top 150 layers of DenseNet121...")
    model.trainable = True 
    for layer in model.layers[:-150]:
        layer.trainable = False

    # 4. Recompile with Micro-Learning Rate
    model.compile(
        optimizer=Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )

    # 5. Data Ingestion & Class Weights
    print("\n[DATA] Loading datasets and computing clinical class weights...")
    train_gen, valid_gen, _ = get_data_generators(base_dir=BASE_DIR, batch_size=BATCH_SIZE)
    
    # SEMINAR DEFENSE: "This mathematically forces the AI to prioritize the minority 
    # class (Fractures), ensuring it doesn't default to guessing 'Normal' just 
    # to artificially inflate its overall accuracy."
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weight_dict = dict(enumerate(class_weights))

    # 6. Safety Nets & Dynamic Optimization
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(run_dir, 'best_finetuned_model.keras'), 
            monitor='val_accuracy', 
            save_best_only=True, 
            verbose=1
        ),
        # Patience increased to 6 because high-res fine-tuning is slow and needs time to adjust
        EarlyStopping(
            monitor='val_loss', 
            patience=8, 
            restore_best_weights=True
        ),
        CSVLogger(os.path.join(run_dir, 'finetuning_log.csv')),
        
        # THE BUFF: Smart Learning Rate Decay
        # SEMINAR DEFENSE: "This callback acts as our dynamic navigator. If the model 
        # stops improving for 2 epochs, it slashes the learning rate by 80% (factor=0.2), 
        # forcing the optimizer to look for a much narrower, deeper mathematical minimum."
        ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2,   
            patience=2,   
            min_lr=1e-8,  
            verbose=1     
        )
    ]

    # 7. Execute Hardware-Accelerated Training Loop
    # Utilizes the M2 unified memory architecture via TensorFlow Metal plugin.
    print("\n[EXECUTION] Commencing deep fine-tuning loop on M2 GPU...")
    model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=EPOCHS,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )
    print("\n[SYSTEM] Deep Fine-tuning successfully complete! Final model saved.")

if __name__ == "__main__":
    main()