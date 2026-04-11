import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

# Import the data loader to get the test set
from data_loader import get_data_generators

# --- Configuration ---
BASE_DIR = '../'
RUNS_DIR = os.path.join(BASE_DIR, 'runs')

def get_latest_run_dir():
    """Finds the most recently created run folder based on actual computer time."""
    if not os.path.exists(RUNS_DIR):
        raise FileNotFoundError("The 'runs' directory does not exist.")
    
    all_runs = [os.path.join(RUNS_DIR, d) for d in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, d))]
    
    # NEW FIX: Only consider folders that actually contain a .keras model file!
    valid_runs = [r for r in all_runs if any(f.endswith('.keras') for f in os.listdir(r))]
    
    if not valid_runs:
        raise FileNotFoundError("No run directories containing a .keras model were found.")
        
    latest_run = max(valid_runs, key=os.path.getmtime)
    return latest_run

def plot_confusion_matrix(y_true, y_pred, run_dir):
    """Generates and saves a clean, presentation-ready confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal (0)', 'Fracture (1)'], 
                yticklabels=['Normal (0)', 'Fracture (1)'])
    
    plt.title('Confusion Matrix: Hand Fracture Detection')
    plt.ylabel('Actual Diagnosis')
    plt.xlabel('Predicted Diagnosis')
    
    # Save it to the exact same folder as the model that generated it
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'confusion_matrix.png'))
    plt.close()
    print("Confusion Matrix saved.")

def main():
    # 1. Find the latest model folder
    latest_run_dir = get_latest_run_dir()
    
    # Check if we are evaluating a fine-tuned model or a base model
    if 'finetune' in latest_run_dir:
        model_path = os.path.join(latest_run_dir, 'best_finetuned_model.keras')
    else:
        model_path = os.path.join(latest_run_dir, 'best_model.keras')
        
    print(f"Loading best model from: {model_path}")
    
    model = load_model(model_path)

    # 2. Load ONLY the Testing Data
    print("Loading test dataset...")
    _, _, test_gen = get_data_generators(base_dir=BASE_DIR, batch_size=32)

    # 3. Generate Predictions
    print("Evaluating model against the test set...")
    predictions = model.predict(test_gen, verbose=1)
    
    # Lowering the threshold to 0.3 for clinical safety (Strategy 2)
    y_pred = (predictions > 0.3).astype(int).flatten()
    y_true = test_gen.classes

    # 4. Generate the Classification Report
    print("\n--- Evaluation Metrics ---")
    report = classification_report(y_true, y_pred, target_names=['Normal (0)', 'Fracture (1)'])
    print(report)

    with open(os.path.join(latest_run_dir, 'evaluation_metrics.txt'), 'w') as f:
        f.write("Classification Report\n=====================\n")
        f.write(report)
        print("Metrics report saved.")

    # 5. Generate and Save the Confusion Matrix
    plot_confusion_matrix(y_true, y_pred, latest_run_dir)

if __name__ == "__main__":
    main()