#!/usr/bin/env python3
"""
Cycling Analysis Startup Script
Simple launcher for the cycling analysis application.

Usage:
    python start.py          # Launch web dashboard
    python start.py --cli    # Launch command line interface
    python start.py --help   # Show help
"""

import sys
import subprocess
import argparse
from pathlib import Path

def main():
    """Main startup function"""
    parser = argparse.ArgumentParser(
        description="Cycling Analysis Startup Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python start.py              # Launch web dashboard
    python start.py --cli        # Launch CLI interface
    python start.py --cli --help # Show CLI help
        """
    )
    
    parser.add_argument('--cli', action='store_true',
                       help='Launch command line interface instead of web dashboard')
    parser.add_argument('--help-cli', action='store_true',
                       help='Show CLI help and exit')
    
    args = parser.parse_args()
    
    # Check if required files exist
    if not Path('dashboard.py').exists():
        print("❌ Error: dashboard.py not found")
        print("Please ensure you're in the cycling_analysis directory")
        sys.exit(1)
    
    if not Path('cli.py').exists():
        print("❌ Error: cli.py not found")
        print("Please ensure you're in the cycling_analysis directory")
        sys.exit(1)
    
    if args.help_cli:
        # Show CLI help
        print("🚴 Cycling Analysis CLI Help")
        print("=" * 50)
        subprocess.run([sys.executable, 'cli.py', '--help'])
        return
    
    if args.cli:
        # Launch CLI interface
        print("🚴 Launching Cycling Analysis CLI...")
        print("Use 'python start.py --help-cli' to see CLI options")
        print("=" * 50)
        subprocess.run([sys.executable, 'cli.py'])
    else:
        # Launch web dashboard
        print("🚴 Launching Cycling Analysis Dashboard...")
        print("Dashboard will open in your browser at http://localhost:8501")
        print("Press Ctrl+C to stop the dashboard")
        print("=" * 50)
        
        try:
            subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'dashboard.py'])
        except KeyboardInterrupt:
            print("\n✅ Dashboard stopped")
        except FileNotFoundError:
            print("❌ Error: Streamlit not found")
            print("Please install streamlit: pip install streamlit")
            sys.exit(1)

if __name__ == "__main__":
    main() 