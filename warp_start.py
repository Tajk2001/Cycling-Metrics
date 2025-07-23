#!/usr/bin/env python3
"""
Warp-optimized startup script for the Enhanced Cycling Analysis Dashboard.
This script automatically handles everything for Warp terminal users.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_in_warp():
    """Run the dashboard optimized for Warp terminal."""
    print("🚴 Enhanced Cycling Analysis Dashboard")
    print("=" * 50)
    
    # Check if we can run directly
    try:
        import streamlit
        import pandas
        import numpy
        import matplotlib
        print("✅ All dependencies available - launching dashboard...")
        launch_dashboard()
        return True
    except ImportError:
        print("📦 Setting up environment...")
    
    # Set up virtual environment if needed
    venv_path = Path("venv")
    
    # Create venv if it doesn't exist
    if not venv_path.exists():
        print("🔄 Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
    
    # Use the virtual environment's Python
    venv_python = venv_path / "bin" / "python"
    venv_pip = venv_path / "bin" / "pip"
    
    # Install dependencies
    print("📦 Installing dependencies...")
    subprocess.run([str(venv_pip), "install", "-r", "requirements.txt"], check=True)
    
    # Launch dashboard
    print("🚀 Launching dashboard...")
    subprocess.run([
        str(venv_python), "-m", "streamlit", "run", 
        "enhanced_dashboard.py",
        "--server.port", "8501",
        "--server.address", "localhost"
    ])

def launch_dashboard():
    """Launch the enhanced dashboard."""
    print("📊 Dashboard will open in your browser at: http://localhost:8501")
    print("🔄 Press Ctrl+C to stop the dashboard")
    print("-" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "enhanced_dashboard.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped. Goodbye!")

if __name__ == "__main__":
    run_in_warp() 