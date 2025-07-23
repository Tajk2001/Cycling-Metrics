#!/usr/bin/env python3
"""
Simple startup script for the Enhanced Cycling Analysis Dashboard.
Optimized for Warp terminal with automatic virtual environment handling.
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
        print("❌ Virtual environment not found. Creating one...")
        try:
            subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
            print("✅ Virtual environment created successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            return False
    
    # Check if we're already in a virtual environment
    if check_venv():
        print("✅ Virtual environment is active")
        return True
    else:
        print("🔄 Activating virtual environment...")
        try:
            # For Warp/macOS, use source command
            if os.name == 'posix':  # macOS/Linux
                activate_script = venv_path / "bin" / "activate"
                if activate_script.exists():
                    # Use subprocess to activate and run
                    env = os.environ.copy()
                    env['VIRTUAL_ENV'] = str(venv_path)
                    env['PATH'] = f"{venv_path}/bin:{env.get('PATH', '')}"
                    
                    # Check if dependencies are installed
                    try:
                        result = subprocess.run([
                            f"{venv_path}/bin/python", "-c", 
                            "import streamlit, pandas, numpy, matplotlib"
                        ], capture_output=True, text=True)
                        
                        if result.returncode == 0:
                            print("✅ Dependencies are installed")
                            return True
                        else:
                            print("📦 Installing dependencies...")
                            subprocess.run([
                                f"{venv_path}/bin/pip", "install", "-r", "requirements.txt"
                            ], check=True)
                            print("✅ Dependencies installed successfully")
                            return True
                            
                    except subprocess.CalledProcessError as e:
                        print(f"❌ Failed to install dependencies: {e}")
                        return False
                else:
                    print("❌ Virtual environment activation script not found")
                    return False
            else:
                print("❌ Unsupported operating system")
                return False
                
        except Exception as e:
            print(f"❌ Failed to activate virtual environment: {e}")
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
        print("Installing dependencies...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
            print("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
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
    
    # For Warp, try to run directly first
    try:
        # Check if dependencies are available in current environment
        import streamlit
        import pandas
        import numpy
        import matplotlib
        print("✅ Dependencies found in current environment")
        launch_dashboard()
        return
    except ImportError:
        print("📦 Dependencies not found, checking virtual environment...")
    
    # Check virtual environment
    if not activate_venv_and_run():
        print("\n💡 Manual setup required:")
        print("1. Run: source venv/bin/activate")
        print("2. Run: pip install -r requirements.txt")
        print("3. Run: streamlit run enhanced_dashboard.py")
        return
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Launch dashboard
    launch_dashboard()

if __name__ == "__main__":
    main() 