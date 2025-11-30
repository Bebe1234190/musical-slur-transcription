# Migration Setup Guide - New Machine Deployment

## 📋 **Complete Dependency Information**

Yes, **everything needed is documented** and can be recreated on your new machine! The project is fully self-contained with complete dependency specifications.

## 🛠️ **Required Dependencies (From Source Code Analysis)**

### **Core Python Packages**
```bash
# Essential dependencies (used in source code)
numpy>=1.21.0           # Core numerical computing
pandas>=1.3.0           # Data manipulation  
torch>=1.9.0            # PyTorch for transformer model
mido>=1.2.0            # MIDI file processing
```

### **Python Standard Library** (Included with Python)
```python
import os              # File system operations
import sys             # System-specific parameters
import math            # Mathematical functions  
import argparse        # Command-line argument parsing
from pathlib import Path  # Modern path handling
```

### **PyTorch Components** (Included with PyTorch)
```python
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
```

### **Optional Dependencies**
```bash
# For enhanced functionality (not required for core operation)
music21>=8.0.0         # Advanced MIDI analysis (optional)
matplotlib>=3.5.0      # Plotting (optional)
seaborn>=0.11.0        # Statistical plotting (optional)  
scikit-learn>=1.0.0    # ML utilities (optional)
jupyter>=1.0.0         # Notebook environment (optional)
```

## 🚀 **Migration Steps for New Machine**

### **Step 1: Setup Python Environment**
```bash
# Ensure Python 3.8+ is installed
python3 --version  # Should be 3.8 or higher

# Create virtual environment (recommended)
python3 -m venv midi_ml_env
source midi_ml_env/bin/activate  # Linux/Mac
# or
# midi_ml_env\Scripts\activate  # Windows
```

### **Step 2: Install Dependencies**
```bash
# Navigate to project directory
cd MIDI_Piano_Roll_ML_System

# Install all dependencies from requirements.txt
pip install -r requirements.txt

# Verify core dependencies are installed
python3 -c "import numpy, pandas, torch, mido; print('✅ Core dependencies installed')"
```

### **Step 3: Verify Installation**
```bash
# Test the complete pipeline
python3 main.py --step test-pipeline

# Expected output: "❌ Error: No slur annotations found in CSV"
# This confirms the pipeline is working correctly
```

### **Step 4: Test Model Creation**
```bash
# Test transformer model
python3 src/ml_transformer_model.py

# Expected output: "🎵 Model ready for training!"
```

## 📦 **Minimal Installation (Core Only)**

If you want just the essential dependencies for the transformer:

```bash
# Absolute minimum for transformer functionality
pip install numpy pandas torch mido

# Test minimal installation
python3 -c "
import numpy as np
import pandas as pd  
import torch
import mido
print('✅ Minimal dependencies working')
"
```

## 🔍 **Dependencies Source Verification**

### **From Source Code Analysis**
✅ **numpy** - Used in all 6 source files  
✅ **pandas** - Used in 4/6 source files  
✅ **torch** - Used in 3/6 ML files  
✅ **mido** - Used in 2/6 MIDI processing files  
✅ **Standard library** - os, sys, math, argparse, pathlib  

### **From requirements.txt**
✅ **All core deps included**  
✅ **Version constraints specified**  
✅ **Optional deps documented**  

### **From Documentation**
✅ **Installation steps in README.md**  
✅ **Usage examples provided**  
✅ **Troubleshooting guidance available**  

## 🎯 **Migration Success Checklist**

```bash
# 1. Python version check
[ ] Python 3.8+ installed

# 2. Dependencies installed  
[ ] pip install -r requirements.txt completed successfully
[ ] Core imports working: numpy, pandas, torch, mido

# 3. Project structure verified
[ ] All directories present: src/, output/, data/, docs/
[ ] Main entry point works: python3 main.py --help

# 4. Pipeline functionality
[ ] Data pipeline test passes: python3 main.py --step test-pipeline  
[ ] Model creation works: python3 src/ml_transformer_model.py

# 5. Data files present
[ ] MIDI file exists: data/*.mid
[ ] Annotated CSV exists: data/*slur_annotated*.csv (if doing training)
[ ] Generated data exists: output/*.npy files (if recreating)
```

## ⚡ **Quick Start on New Machine**

```bash
# 1. Clone/copy project to new machine
# (You'll need to transfer the entire MIDI_Piano_Roll_ML_System directory)

# 2. Setup
cd MIDI_Piano_Roll_ML_System
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Test
python3 main.py --step test-pipeline

# 4. Run training (if you have annotated data)
python3 main.py --step train --epochs 500
```

## 📋 **What You Need to Transfer**

### **Essential Files**
```
MIDI_Piano_Roll_ML_System/
├── main.py                    # Entry point
├── requirements.txt           # Dependencies  
├── README.md                  # User guide
├── src/                      # All source code (6 files)
├── data/                     # Your MIDI files + annotations
└── docs/                     # Documentation (3 essential files)
```

### **Optional Files** 
```
├── output/                   # Generated data (can be recreated)
└── archive_matrix_approach/  # Historical files (not needed for running)
```

## 🔧 **Troubleshooting**

### **Common Issues**

#### **"Module not found" errors**
```bash
# Solution: Install missing dependency
pip install [package_name]
```

#### **PyTorch installation issues**
```bash
# For CPU-only (most compatible)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# For GPU (if available)
pip install torch  # Will auto-detect CUDA
```

#### **mido/MIDI issues**
```bash
# If MIDI files don't load
pip install mido
# Ensure MIDI files are in data/ directory
```

### **Platform-Specific Notes**

#### **macOS**
- Python usually pre-installed or via Homebrew
- PyTorch supports Apple Silicon (M1/M2) for performance

#### **Windows**  
- Install Python from python.org
- Use `python` instead of `python3` in commands
- Use `\` instead of `/` in paths if needed

#### **Linux**
- Use system package manager for Python if needed: `sudo apt install python3 python3-pip`
- Everything else should work identically

## ✅ **Success Confirmation**

When migration is successful, you should be able to run:

```bash
python3 main.py --step train --epochs 10
```

And see the transformer training start with output like:
```
🎵 MIDI SLUR TRANSFORMER - OVERFITTING TEST
============================================================
📊 STEP 1: DATA PREPARATION  
✓ Data loaded: 2640 notes, 5 features
🧠 STEP 2: MODEL CREATION
✓ Model created: 794,372 parameters
🔥 STEP 4: OVERFITTING TEST
Epoch   1/10 | Loss: 0.693 | Acc: 0.XXX | ...
```

**Yes, everything needed for migration is documented and can be recreated from the existing files!** 🚀

---

*Migration guide prepared: September 13, 2025*  
*Compatible with: macOS, Windows, Linux*  
*Python requirement: 3.8+*
