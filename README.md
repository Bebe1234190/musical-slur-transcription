# MIDI Piano Roll ML System - Transformer Edition

A machine learning system for predicting musical slur annotations from MIDI files using transformer-based sequence modeling.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline
```bash
python main.py
```

This will:
1. Generate note and pedal data from MIDI files
2. Create annotation CSV template
3. Prompt you to annotate slur categories
4. Train transformer model

### 3. Individual Steps
```bash
python main.py --step data        # Generate initial data only
python main.py --step prepare     # Prepare ML data only  
python main.py --step train       # Train transformer only
python main.py --step test-pipeline  # Test data pipeline
```

## Project Structure

```
MIDI_Piano_Roll_ML_System/
├── main.py                     # Main entry point
├── requirements.txt            # Dependencies
├── data/                       # Input MIDI files
├── output/                     # Generated data files
├── src/                        # Source code
│   ├── main_ml.py             # ML workflow orchestration
│   ├── complete_midi_processor.py  # MIDI → data conversion
│   ├── slur_annotation_tool.py     # Annotation CSV creation
│   ├── ml_data_pipeline.py         # Transformer data prep
│   ├── ml_transformer_model.py     # PyTorch transformer
│   └── ml_train.py                 # Training script
├── docs/                       # Documentation
│   ├── README_ML_APPROACH.md      # Transformer approach guide
│   ├── MIGRATION_SUMMARY.md       # Migration details
│   ├── CLEANUP_GUIDE.md           # Cleanup documentation
│   └── [session docs...]          # Development history
└── archive_matrix_approach/    # Old matrix-based files
```

## Approach

### Transformer-Based Sequence Modeling
- **Input**: Musical note sequences [start_time, duration, pitch, velocity, sustain_start, sustain_end]
- **Output**: 5-class predictions [slur_start, slur_middle, slur_end, no_slur, slur_start_and_end]
- **Architecture**: Self-attention transformer encoder (794K parameters)
- **Loss Function**: CrossEntropyLoss (multi-class classification)

### Key Advantages
- **99.95% data reduction**: 2,640 notes vs 26M matrix elements
- **Musical focus**: Models temporal relationships directly
- **Efficient training**: Fast convergence on musical patterns
- **Scalable**: Easy to add more pieces and composers

## Workflow

1. **MIDI Processing**: Extract notes, timing, pedal information
2. **Annotation**: Manually label slur categories (1=start, 2=middle, 3=end, 4=none)
3. **Training**: Transformer learns slur patterns from annotated sequences
4. **Prediction**: Apply trained model to new pieces

## Model Details

- **Architecture**: Transformer encoder with 4 layers, 8 attention heads
- **Input features**: 6 (start_time, duration, pitch, velocity, sustain_start, sustain_end)
- **Output classes**: 5 (slur_start, slur_middle, slur_end, no_slur, slur_start_and_end)
- **Loss Function**: CrossEntropyLoss (softmax applied internally)
- **Training**: Multi-combination evaluation (2 train, 1 val, 1 test) with multiple trials

## Documentation

See `docs/` directory for detailed documentation:
- `README_ML_APPROACH.md` - Transformer approach details
- `MIGRATION_SUMMARY.md` - Migration from matrix approach
- Session documentation - Development history and insights

## Next Steps

1. **Annotation**: Complete slur category annotation in CSV files (5 categories: 0-4, plus 5 for single-note slurs)
2. **Training**: Multi-combination evaluation across all piece combinations
3. **Expansion**: Currently 4 annotated pieces (~13,000 notes)
4. **Evaluation**: Comprehensive per-class analysis and research reports
5. **Improvement**: Address high variance through better initialization and regularization

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy, Pandas
- mido, music21 (MIDI processing)

See `requirements.txt` for complete list.

---

**MIDI Piano Roll ML System v2.0** - Focusing on musical intelligence through transformer-based sequence modeling.
