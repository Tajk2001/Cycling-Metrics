#!/usr/bin/env python3
"""
Simple startup script for the Enhanced Cycling Analysis Dashboard.
This script handles virtual environment activation and launches the dashboard.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_venv():
    """Check if virtual environment is activated."""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def activate_venv_and_run():
    """Activate virtual environment and run the dashboard."""
    venv_path = Path("venv")
    
    if not venv_path.exists():
        print("❌ Virtual environment not found. Please run:")
        print("   python -m venv venv")
        print("   source venv/bin/activate  # On macOS/Linux")
        print("   venv\\Scripts\\activate     # On Windows")
        print("   pip install -r requirements.txt")
        return False
    
    # Check if we're already in a virtual environment
    if check_venv():
        print("✅ Virtual environment is active")
        return True
    else:
        print("⚠️  Virtual environment not active. Please run:")
        print("   source venv/bin/activate  # On macOS/Linux")
        print("   venv\\Scripts\\activate     # On Windows")
        print("   Then run: python start.py")
        return False

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import streamlit
        import pandas
        import numpy
        import matplotlib
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def launch_dashboard():
    """Launch the enhanced dashboard."""
    print("🚴 Launching Enhanced Cycling Analysis Dashboard...")
    print("📊 Dashboard will open in your browser at: http://localhost:8501")
    print("🔄 Press Ctrl+C to stop the dashboard")
    print("-" * 50)
    
    try:
        # Launch the dashboard
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "enhanced_dashboard.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped. Goodbye!")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

def main():
    """Main startup function."""
    print("🚴 Enhanced Cycling Analysis Dashboard")
    print("=" * 50)
    
    # Check virtual environment
    if not activate_venv_and_run():
        return
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Launch dashboard
    launch_dashboard()

if __name__ == "__main__":
    main() 