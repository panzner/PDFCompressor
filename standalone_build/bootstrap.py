#!/usr/bin/env python3
"""
Bootstrap script for PDF Compressor standalone app.
This script handles the application launch and initialization.
"""

import sys
import os

# Add the current directory to Python path for imports
if hasattr(sys, '_MEIPASS'):
    # When running as PyInstaller bundle
    sys.path.insert(0, sys._MEIPASS)
else:
    # When running in development
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    sys.path.insert(0, parent_dir)

# Import and run the main application
try:
    from pdf_compressor import main
    if __name__ == "__main__":
        main()
except ImportError as e:
    print(f"Error importing PDF Compressor: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error running PDF Compressor: {e}", file=sys.stderr)
    sys.exit(1)
