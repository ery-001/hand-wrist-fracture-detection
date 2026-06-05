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

# SEMINAR DEFENSE: We define multiple clinical thresholds to prove 
# how the AI can be tuned for different hospital environments.
CLINICAL_THRESHOLDS = [0.3, 0.5, 0.7]

def get_latest_run_dir():
    """Finds the most recently created run folder containing a saved model."""
    if not os.path.exists(RUNS_DIR):
        raise FileNotFoundError("The 'runs' directory does not exist.")
    
    all_runs = [os.path.join(RUNS_DIR, d) for d in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, d))]
    valid_runs = [r for r in all_runs if any(f.endswith('.keras') for f in os.listdir(r))]
    
    if not valid_runs:
        raise FileNotFoundError("No run directories containing a .keras model were found.")
        
    latest_run = max(valid_runs, key=os.path.getmtime)
    return latest_run

def plot_confusion_matrix(y_true, y_pred, run_dir, threshold):
    """Generates and saves a threshold-specific presentation matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal (0)', 'Fracture (1)'], 
                yticklabels=['Normal (0)', 'Fracture (1)'])
    
    # Dynamically update the title to show the current threshold
    plt.title(f'Hand Fracture Detection (Threshold: {threshold})')
    plt.ylabel('Actual Diagnosis')
    plt.xlabel('Predicted Diagnosis')
    
    # Save the matrix with the threshold in the filename so it doesn't overwrite
    filename = f'confusion_matrix_t{threshold}.png'
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, filename))
    plt.close()
    print(f"-> Saved: {filename}")

def main():
    # 1. Find the latest model folder
    latest_run_dir = get_latest_run_dir()
    
    if 'finetune' in latest_run_dir:
        model_path = os.path.join(latest_run_dir, 'best_finetuned_model.keras')
    else:
        model_path = os.path.join(latest_run_dir, 'best_model.keras')
        
    print(f"\n[SYSTEM] Loading best model from: {model_path}")
    model = load_model(model_path)

    # 2. Load ONLY the Testing Data
    print("[DATA] Loading test dataset...")
    _, _, test_gen = get_data_generators(base_dir=BASE_DIR, batch_size=32)

    # 3. Generate Raw Probabilities ONCE
    # Predicting takes time. We only want to run the images through the GPU once.
    print("\n[EXECUTION] Extracting raw probability scores from the test set...")
    raw_predictions = model.predict(test_gen, verbose=1).flatten()
    y_true = test_gen.classes

    # 4. Loop through the Clinical Thresholds
    print(f"\n{'='*50}")
    print("[ANALYSIS] Commencing Multi-Threshold Evaluation")
    print(f"{'='*50}")
    
    for threshold in CLINICAL_THRESHOLDS:
        print(f"\n--- Evaluating Threshold: {threshold} ---")
        
        # Apply the mathematical cutoff
        y_pred = (raw_predictions >= threshold).astype(int)
        
        # Generate the Classification Report
        report = classification_report(y_true, y_pred, target_names=['Normal (0)', 'Fracture (1)'])
        print(report)

        # Save the Metrics Report
        report_filename = f'evaluation_metrics_t{threshold}.txt'
        with open(os.path.join(latest_run_dir, report_filename), 'w') as f:
            f.write(f"Classification Report (Threshold: {threshold})\n")
            f.write("========================================\n")
            f.write(report)
        print(f"-> Saved: {report_filename}")

        # Generate and Save the Confusion Matrix
        plot_confusion_matrix(y_true, y_pred, latest_run_dir, threshold)

    print(f"\n[SUCCESS] All evaluations complete. Artifacts saved to {latest_run_dir}")

if __name__ == "__main__":
    main()