"""
LEGO Classifier - Main Entry Point
Run with: streamlit run main.py
"""

import sys
from pathlib import Path

# Add src to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent))

from src.app import main

# Call main directly (not inside if __name__ == "__main__")
main()