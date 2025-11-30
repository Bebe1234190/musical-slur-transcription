#!/usr/bin/env python3
"""
MIDI Piano Roll ML System - Main Entry Point
Wrapper script that calls the main ML workflow

Usage:
    python main.py                    # Run complete ML pipeline
    python main.py --step data        # Generate initial data only
    python main.py --step train       # Train transformer only
    python main.py --help             # Show all options

Author: MIDI Piano Roll ML System v2.0 - Transformer Edition
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the main ML script
if __name__ == "__main__":
    from main_ml import main
    main()
