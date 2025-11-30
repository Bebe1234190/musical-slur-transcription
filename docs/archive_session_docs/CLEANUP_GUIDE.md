# Codebase Cleanup Guide - Transition to Transformer Approach

## Files to Archive or Remove

### Matrix-Based Files (No Longer Used)
These files were part of the matrix-based CNN approach and are no longer needed for the transformer:

#### Remove from `src/`:
- ~~`slur_annotation_tool.py`~~ - **KEEP** (still needed for initial annotation CSV generation)
- ~~`validation_tools.py`~~ - **ARCHIVE** (matrix validation no longer needed)

#### Remove from output (optional):
- `*_slur_matrix.npy` - Large matrix files no longer used
- `*_slur_matrix.csv` - CSV matrix files no longer used  
- `*_notes.csv` - Large CSV files (keep .npy for reference)
- `*_pedal.csv` - Large CSV files (keep .npy for pedal data)
- `*_slur_matrix_condensed.csv` - Condensed matrix files
- `*_slur_metadata.txt` - Matrix-specific metadata

### Main Scripts
- ~~`main.py`~~ - **ARCHIVE** (replace with `main_ml.py`)

## Files to Keep

### Core Processing (Still Needed)
- `src/complete_midi_processor.py` - **KEEP** (generates initial data)
- `src/slur_annotation_tool.py` - **KEEP** (creates annotation CSV)

### New ML Components
- `src/ml_data_pipeline.py` - **NEW** (transformer data preparation)
- `src/ml_transformer_model.py` - **NEW** (transformer architecture)
- `src/ml_train.py` - **NEW** (training script)
- `main_ml.py` - **NEW** (streamlined ML workflow)

### Essential Data Files
- `output/*_notes.npy` - **KEEP** (reference data)
- `output/*_pedal.npy` - **KEEP** (needed for transformer)
- `output/*_slur_annotation.csv` - **KEEP** (core annotation data)
- `output/*_metadata.txt` - **KEEP** (processing documentation)

### Documentation
- `docs/` - **KEEP ALL** (historical record and reference)
- `requirements.txt` - **UPDATE** (add PyTorch)
- `README_ML_APPROACH.md` - **NEW** (transformer guide)

## Cleanup Actions

### 1. Archive Matrix Processing
```bash
mkdir archive_matrix_approach
mv src/validation_tools.py archive_matrix_approach/
mv main.py archive_matrix_approach/
```

### 2. Clean Large Output Files (Optional)
```bash
# Remove large matrix files to save space
rm output/*_slur_matrix.csv
rm output/*_notes.csv  
rm output/*_pedal.csv
rm output/*_slur_matrix.npy
```

### 3. Update Requirements
Add to `requirements.txt`:
```
torch>=2.0.0
```

### 4. Update Main Documentation
- Update main README to point to transformer approach
- Archive old session documentation

## Size Savings

### Before Cleanup:
- Matrix CSV files: ~150MB each
- Matrix NPY files: ~25MB each  
- Total matrix storage: ~500MB per piece

### After Cleanup:
- Processed tensor data: ~1MB per piece
- Model files: ~2MB per trained model
- Total storage: ~5MB per piece

**Storage reduction: 99% smaller**

## File Structure After Cleanup

```
MIDI_Piano_Roll_ML_System/
├── main_ml.py                     # NEW: Main ML workflow
├── README_ML_APPROACH.md          # NEW: Transformer guide
├── requirements.txt               # UPDATED: Add PyTorch
├── data/                          # Input files
├── src/                           
│   ├── complete_midi_processor.py # KEEP: Initial data generation
│   ├── slur_annotation_tool.py    # KEEP: Annotation CSV creation
│   ├── ml_data_pipeline.py        # NEW: Transformer data prep
│   ├── ml_transformer_model.py    # NEW: Model architecture
│   └── ml_train.py                # NEW: Training script
├── output/                        
│   ├── *_notes.npy                # KEEP: Reference
│   ├── *_pedal.npy                # KEEP: Pedal data
│   ├── *_slur_annotation.csv      # KEEP: Core data
│   ├── *_processed_for_ml.pt      # NEW: ML tensors
│   └── *_overfitted_model.pt      # NEW: Trained models
├── docs/                          # KEEP: Documentation
└── archive_matrix_approach/       # NEW: Archived files
    ├── main.py                    
    └── validation_tools.py        
```

## Benefits of Cleanup

1. **Clarity**: Focus on transformer approach
2. **Performance**: 99% storage reduction
3. **Maintainability**: Simpler codebase
4. **Speed**: Faster data loading/processing

## Migration Path

1. **Test new pipeline**: Verify transformer training works
2. **Archive old files**: Move to `archive_matrix_approach/`  
3. **Clean output**: Remove large matrix files
4. **Update docs**: Point to new workflow
5. **Commit changes**: Clean git history
