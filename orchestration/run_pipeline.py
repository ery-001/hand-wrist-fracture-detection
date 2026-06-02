import sys
import subprocess
import time

# =====================================================================
# END-TO-END PIPELINE ORCHESTRATOR
# =====================================================================
# SEMINAR DEFENSE: "To ensure reproducibility and modularity, we isolated 
# our code into specific modules (data, model, training, fine-tuning) 
# and use this orchestrator script to execute the pipeline sequentially. 
# This mirrors industry-standard MLOps practices."

def run_phase(script_name, phase_description):
    """
    Executes a Python script as a subprocess.
    If the script fails or crashes, the pipeline automatically halts
    to prevent cascading errors.
    """
    print(f"\n{'='*70}")
    print(f"[PIPELINE STAGE] {phase_description}")
    print(f"[SYSTEM] Executing: {script_name}")
    print(f"{'='*70}\n")
    
    # sys.executable ensures it uses your exact Python/Conda environment
    result = subprocess.run([sys.executable, script_name])
    
    # Check if the script crashed or was manually interrupted (e.g., Ctrl+C)
    if result.returncode != 0:
        print(f"\n[CRITICAL ERROR] Pipeline halted!")
        print(f"-> {script_name} failed with exit code {result.returncode}.")
        sys.exit(1) # Kill the whole pipeline

def main():
    print("\n[SYSTEM] Initializing End-to-End MURA Fracture Detection Pipeline...")
    start_time = time.time()
    
    # ---------------------------------------------------------
    # PHASE 1: Base Training (Trains the new classification head)
    # Ensure LEARNING_RATE is 1e-3 in train.py
    # ---------------------------------------------------------
    run_phase(
        script_name="../src/train.py", 
        phase_description="PHASE 1: Base Training (Custom Head Optimization)"
    )
    
    # ---------------------------------------------------------
    # PHASE 2: Deep Fine-Tuning (Unfreezes DenseNet layers)
    # Ensure FINE_TUNE_LEARNING_RATE is 1e-5 in finetune.py
    # ---------------------------------------------------------
    run_phase(
        script_name="../src/fine_tune.py", 
        phase_description="PHASE 2: Deep Fine-Tuning (Top 150 Layers Unfrozen)"
    )
    
     
    # ---------------------------------------------------------
    # PHASE 3: Evaluation with unseen 1119
    # ---------------------------------------------------------
    run_phase(
        script_name="../src/evaluate.py", 
        phase_description="PHASE 3: Evaluation"
    )
    


    # ---------------------------------------------------------
    # PIPELINE COMPLETION
    # ---------------------------------------------------------
    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    
    print(f"\n{'='*70}")
    print(f"[SUCCESS] Entire Machine Learning Pipeline Completed!")
    print(f"[METRIC] Total Execution Time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"[SYSTEM] All models and metrics have been safely saved to the /runs directory.")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()