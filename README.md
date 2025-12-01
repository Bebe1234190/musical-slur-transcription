# Musical Slur Transcription

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

### 4. Multi-Trial Training (Using Pre-annotated Data)

The repository includes 4 fully annotated pieces. To run multi-trial training with all combinations:

```bash
# Run all 12 combinations with 5 trials each
python src/run_multi_trial_training.py \
    --chunk-size 200 \
    --chunk-overlap 100 \
    --trials-per-combination 5 \
    --epochs 200

# Run a specific combination multiple times
python src/run_multi_trial_training.py \
    --combination 1 \
    --repeat-combination 5 \
    --chunk-size 200 \
    --chunk-overlap 100 \
    --epochs 200
```

**Available annotated pieces:**
- `Beethoven_Piano_Sonata_No_10_Op_14_No_2_fQqNsTUvqCY_cut_mov_1`
- `midis_for_evaluation_ground_truth_beethoven_sonata_no_16_hisamori_cut_mov_1`
- `midis_for_evaluation_ground_truth_beethoven_rondo_a_capriccio_op_129_smythe`
- `midis_for_evaluation_ground_truth_chopin_etude_op_10_no_12`

The script automatically finds all files ending with `_slur_annotation_completed.csv` in the `output/` directory.

**Training Outputs:**
- Results are printed to the terminal in real-time
- Per-trial metrics (test accuracy, validation accuracy, training accuracy, epochs, time)
- Summary statistics across all trials (mean ± std, ranges)
- Per-class analysis (precision, recall, F1-score for each slur category)
- Detailed results table showing all combination trials

**Note:** Model checkpoints (`.pt` files) and large data files (`.npy`) are excluded from the repository via `.gitignore` but are generated locally during training.

## Structure

```
Musical Slur Transcription/
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
2. **Annotation**: Manually label slur categories (see Annotation Format below)
3. **Training**: Transformer learns slur patterns from annotated sequences
4. **Prediction**: Apply trained model to new pieces

## Annotation Format

Slur categories are labeled as integers in the annotation CSV files:
- **0**: Background (unused in current implementation)
- **1**: Slur start
- **2**: Slur middle
- **3**: Slur end
- **4**: No slur
- **5**: Slur start and end (single-note slur)

The model maps these to 5 output classes: `[slur_start, slur_middle, slur_end, no_slur, slur_start_and_end]` (category 0 is mapped to class 3: no_slur).

## Model Details

- **Architecture**: Transformer encoder with 4 layers, 8 attention heads
- **Input features**: 6 (start_time, duration, pitch, velocity, sustain_start, sustain_end)
- **Output classes**: 5 (slur_start, slur_middle, slur_end, no_slur, slur_start_and_end)
- **Loss Function**: CrossEntropyLoss (softmax applied internally)
- **Training**: Multi-combination evaluation (2 train, 1 val, 1 test) with multiple trials

## Documentation

See `docs/` directory for detailed documentation:
- `PROJECT_COMPREHENSIVE_DOCUMENTATION_DECEMBER_2025.md` - Complete documentation
- `MODEL_ARCHITECTURE_SECTION.md` - Detailed model architecture for research papers
- `PROJECT_ORGANIZATION.md` - Structure and organization
- Other session and migration documentation

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy, Pandas
- mido, music21 (MIDI processing)

See `requirements.txt` for complete list.

---

**Musical Slur Transcription** - Transformer-based sequence modeling for musical slur prediction.
