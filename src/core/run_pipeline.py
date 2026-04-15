"""
run_pipeline.py  —  Master Controller
=====================================
Automates the entire "Computational Poetics" pipeline:
1. Data Cleaning
2. Ultra Feature Engineering (Transformers + Stylo + POS)
3. XGBoost Model Training

Usage: python run_pipeline.py
"""

import subprocess
import sys
import time
import os
import argparse

def get_latest_mtime(directory, extensions=('.csv', '.npz', '.pkl')):
    """Returns the latest modification time of any file in a directory with given extensions."""
    latest = 0
    if not os.path.exists(directory):
        return 0
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith(extensions):
                latest = max(latest, os.path.getmtime(os.path.join(root, f)))
    return latest

def run_step(name, cmd):
    print(f"\n\n\033[1;34m[STEP] {name}\033[0m")
    print(f"Running: {cmd}")
    start = time.time()
    
    # Run and stream output
    process = subprocess.Popen(cmd, shell=True)
    process.wait()
    
    elapsed = time.time() - start
    if process.returncode == 0:
        print(f"\n\033[1;32m {name} Completed in {elapsed:.1f}s\033[0m")
    else:
        print(f"\n\033[1;31m {name} Failed with exit code {process.returncode}\033[0m")
        sys.exit(process.returncode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computational Poetics Master Pipeline")
    parser.add_argument("--force", action="store_true", help="Force re-run all steps")
    args = parser.parse_args()

    print("\033[1;35m" + "="*60)
    print("\tCOMPUTATIONAL POETICS — PIPELINE")
    print("\t(Lazy Mode: Active)" + (" [FORCE]" if args.force else ""))
    print("="*60 + "\033[0m")
    
    # Root Path
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(root, 'data')
    raw_dir = os.path.join(data_dir, 'raw')
    proc_dir = os.path.join(data_dir, 'processed')
    model_dir = os.path.join(root, 'saved_models')

    # Check dependencies
    print("Verifying environment...")
    try:
        import nltk
        import torch
        print(f"\t- PyTorch CUDA: {'Available' if torch.cuda.is_available() else 'Not Available'}")
    except ImportError as e:
        print(f"\t- Error: Missing dependency: {e}")
        sys.exit(1)

    # Master Skip Check
    if not args.force:
        raw_mtime = get_latest_mtime(raw_dir)
        proc_mtime = get_latest_mtime(proc_dir)
        model_mtime = get_latest_mtime(model_dir)

        # If everything is processed and models are newer than data, we are done
        if proc_mtime > raw_mtime and model_mtime >= proc_mtime:
            print("\n\033[1;32m ALL STEPS ARE UP TO DATE. SKIPPING PIPELINE.\033[0m")
            print("To force re-run, use: python src/core/run_pipeline.py --force")
            sys.exit(0)

    # Sequence
    force_flag = " --force" if args.force else ""
    run_step("CLEANING DATA", f'"{sys.executable}" src/core/preprocessing/clean_data.py{force_flag}')
    run_step("LEXICON GENERATION (LLM)", f'"{sys.executable}" src/core/preprocessing/lexicon_generator.py')
    run_step("FEATURE ENGINEERING", f'"{sys.executable}" src/core/preprocessing/build_features.py{force_flag}')
    run_step("MODEL TRAINING", f'"{sys.executable}" src/core/models/train_models.py{force_flag}')

    print("\n\n\033[1;32m" + "="*60)
    print("PIPELINE COMPLETE — APP READY")
    print("="*60 + "\033[0m")
    print("Run frontend: streamlit run src/frontend/app.py")
