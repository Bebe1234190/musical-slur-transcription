# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Setup
```bash
# Navigate to project directory
cd MIDI_Piano_Roll_ML_System

# Install dependencies
pip install numpy pandas mido music21
```

### Step 2: Process Your First MIDI File
```python
# Place your MIDI file in data/ directory
# Then run:

from src.complete_midi_processor import process_midi_file

result = process_midi_file("data/your_song.mid", output_dir="output/")
```

### Step 3: Check Results
```bash
# Your output/ directory now contains:
ls output/
# - your_song_notes.npy (note velocities)
# - your_song_pedal.npy (pedal states)  
# - your_song_slur_annotation.csv (for manual annotation)
# - your_song_metadata.txt (documentation)
```

### Step 4: Load for ML Training
```python
import numpy as np

# Load matrices (perfectly aligned!)
notes = np.load('output/your_song_notes.npy')  # (88, time_steps)
pedal = np.load('output/your_song_pedal.npy')  # (3, time_steps)

print(f"Ready for ML training: {notes.shape}")
```

## 🎯 For Slur Annotations

### Manual Annotation Process
1. **Open CSV**: `output/your_song_slur_annotation.csv` in Excel
2. **Fill Categories**: Add values 0-4 in `Slur_Category` column:
   - 0: Empty/No annotation  
   - 1: Slur beginning
   - 2: Slur middle
   - 3: Slur end  
   - 4: No slur
3. **Save File**

### Generate Slur Matrix
```python
from src.slur_annotation_tool import create_slur_matrix_from_partial_csv

slur_file = create_slur_matrix_from_partial_csv(
    "output/your_song_slur_annotation.csv",
    "data/your_song.mid"
)

# Now you have all three matrices!
slurs = np.load('output/your_song_slur_matrix.npy')  # (88, time_steps)
```

## ✅ Verify Everything Works

```python
# Check perfect alignment
assert notes.shape == pedal.shape[1] == slurs.shape
print("🎉 All matrices perfectly aligned!")

# Check some stats
print(f"Total notes: {np.count_nonzero(notes)}")
print(f"Total pedal events: {np.count_nonzero(pedal)}")
print(f"Annotated regions: {np.count_nonzero(slurs)}")
```

**That's it! You now have production-ready ML data.** 🎵✨

## 📁 Recommended Workflow

```
1. Place MIDI files in data/
2. Run complete_midi_processor.py
3. Manually annotate CSV files  
4. Run slur_annotation_tool.py
5. Load matrices for ML training
```

See `docs/README.md` for complete documentation.
