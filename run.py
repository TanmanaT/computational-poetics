"""
run.py  —  Unified Entry Point for Poetics Analyzer
===================================================
Usage:
  python run.py --pipeline    # Run the processing pipeline (Step 1-3)
  python run.py --app         # Launch the Streamlit frontend
  python run.py --api         # Launch the FastAPI backend
"""

import sys, os, subprocess, argparse

def main():
    parser = argparse.ArgumentParser(description="Computational Poetics Analyzer — Unified Runner")
    parser.add_argument("--pipeline", action="store_true", help="Run the full data/model pipeline")
    parser.add_argument("--app", action="store_true", help="Launch the Streamlit frontend")
    parser.add_argument("--api", action="store_true", help="Launch the FastAPI backend")
    parser.add_argument("--force", action="store_true", help="Force pipeline re-run")
    
    args = parser.parse_args()
    
    if args.pipeline:
        print("\n\033[1;35mStarting Pipeline...\033[0m")
        cmd = [sys.executable, "src/core/run_pipeline.py"]
        if args.force: cmd.append("--force")
        subprocess.run(cmd)
    
    elif args.app:
        print("\n\033[1;34mLaunching Streamlit App...\033[0m")
        # Ensure we run from the project root
        subprocess.run(["streamlit", "run", "src/frontend/app.py"])
        
    elif args.api:
        print("\n\033[1;32mLaunching FastAPI Backend...\033[0m")
        # uvicorn uses dot notation for path
        subprocess.run(["uvicorn", "src.backend.api:app", "--reload", "--port", "8000"])
        
    else:
        print("\033[1;33mWelcome to Poetics Analyzer Overhaul!\033[0m")
        print("-" * 40)
        parser.print_help()

if __name__ == "__main__":
    main()
