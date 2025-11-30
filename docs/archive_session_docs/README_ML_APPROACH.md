# MIDI Piano Roll ML System - Transformer Edition

## Quick Start

### 1. Generate Initial Data
```bash
python main_ml.py --step data
```
This creates the annotation CSV template from your MIDI file.

### 2. Annotate Slur Categories
Open the generated `*_slur_annotation.csv` file and fill in the `Slur_Category` column:
- `1` = Slur start
- `2` = Slur middle  
- `3` = Slur end
- `4` = No slur

### 3. Train Transformer (Overfitting Test)
```bash
python main_ml.py --step train --epochs 1000
```

### 4. Or Run Complete Pipeline
```bash
python main_ml.py
```

## New ML Architecture

### Transformer-Based Approach
- **Input**: Note sequences [start_time, duration, pitch, velocity, sustain]
- **Output**: Binary predictions [slur_start, slur_middle, slur_end, no_slur]
- **Size**: ~2,640 notes vs 26M matrix elements (10,000x reduction)

### Key Files

#### Core ML Components
- `src/ml_data_pipeline.py` - Data preprocessing for transformer
- `src/ml_transformer_model.py` - Transformer architecture  
- `src/ml_train.py` - Training script with overfitting test
- `main_ml.py` - Streamlined ML workflow

#### Data Files Needed
- `output/*_pedal.npy` - Sustain pedal information
- `output/*_slur_annotation.csv` - Annotated note sequences

#### Generated Files
- `output/*_processed_for_ml.pt` - Preprocessed tensor data
- `output/*_overfitted_model.pt` - Trained model (if successful)
- `output/*_training_history.pt` - Training metrics and curves

## Architecture Details

### Model Configuration
```python
d_model = 128          # Hidden dimension
n_layers = 4           # Transformer layers  
n_heads = 8            # Attention heads
sequence_length = 2640 # Notes in piece
```

### Training Strategy
1. **Overfitting test**: Memorize single piece to verify capability
2. **Multi-piece training**: Generalize across multiple sonatas (future)
3. **Evaluation**: Musical meaningfulness + classification metrics

## What's Different

### Old Matrix Approach
- Sparse 88×295593 matrices (96% empty)
- CNN-based processing
- Memory intensive

### New Sequence Approach  
- Dense note sequences (2640 events)
- Transformer-based processing
- 87.5% memory reduction
- Focus on temporal patterns

## Testing the Approach

The overfitting test verifies if the transformer can learn musical slur patterns:

**Success criteria:**
- 95%+ accuracy on single piece
- Clean convergence curves
- Meaningful slur boundary predictions

**If successful:** Ready for multi-piece training
**If unsuccessful:** Debug task formulation or architecture

## Next Steps

1. **Verify single-piece memorization**
2. **Collect more annotated Beethoven sonatas**  
3. **Train generalized model**
4. **Evaluate musical quality**
5. **Extend to other composers**
