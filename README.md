# Musical Slur Transcription Project

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
Slur Transcription Project/
├── main.py                     # Main entry point
├── requirements.txt            # Dependencies
├── output/                      # Generated data files (MIDI, CSVs, models)
├── src/                         # Source code
│   ├── main_ml.py              # ML workflow orchestration
│   ├── complete_midi_processor.py  # MIDI → data conversion
│   ├── slur_annotation_tool.py      # Annotation CSV creation
│   ├── ml_data_pipeline.py         # Transformer data prep
│   ├── ml_transformer_model.py     # PyTorch transformer
│   ├── ml_chunked_train.py         # Chunked training logic
│   ├── ml_chunked_pipeline.py      # Chunking utilities
│   ├── run_multi_trial_training.py # Multi-trial training script
│   └── ml_train.py                 # Training script
└── docs/                        # Documentation
    ├── PROJECT_COMPREHENSIVE_DOCUMENTATION_DECEMBER_2025.md
    ├── MODEL_ARCHITECTURE_SECTION.md
    ├── PROJECT_ORGANIZATION.md
    └── [other documentation files...]
```

## Approach

### Transformer-Based Sequence Modeling
- **Input**: Musical note sequences [start_time, duration, pitch, velocity, sustain_start, sustain_end]
- **Output**: 5-class predictions [slur_start, slur_middle, slur_end, no_slur, slur_start_and_end]
- **Architecture**: Self-attention transformer encoder (794K parameters)
- **Loss Function**: CrossEntropyLoss (multi-class classification)

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
- `PROJECT_COMPREHENSIVE_DOCUMENTATION_DECEMBER_2025.md` - Complete project documentation
- `MODEL_ARCHITECTURE_SECTION.md` - Detailed model architecture for research papers
- `PROJECT_ORGANIZATION.md` - Project structure and organization
- Other session and migration documentation

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy, Pandas
- mido, music21 (MIDI processing)

See `requirements.txt` for complete list.

---

**Musical Slur Transcription Project** - Transformer-based sequence modeling for musical slur prediction.
