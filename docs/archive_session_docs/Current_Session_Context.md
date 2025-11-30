# Current Session Context

**Last Updated**: December 2025

## Current System Status

### Model Architecture
- **Type**: Transformer encoder (4 layers, 8 attention heads, 128 hidden dims)
- **Input**: 6 features (start_time, duration, pitch, velocity, sustain_start, sustain_end)
- **Output**: 5 classes (slur_start, slur_middle, slur_end, no_slur, slur_start_and_end)
- **Loss Function**: CrossEntropyLoss (softmax applied internally)
- **Activation**: None (raw logits output)

### Dataset
- **4 Annotated Pieces**:
  1. Beethoven Sonata No. 10 Op. 14 No. 2
  2. Beethoven Sonata No. 16 Op. 31 No. 1
  3. Beethoven Rondo a Capriccio Op. 129 (5,592 notes, includes category 5 examples)
  4. Chopin Etude Op. 10 No. 12 (2,088 notes, multi-track MIDI)
- **Total**: ~13,000 notes
- **Annotation System**: 6 categories (0-5), mapped to 5 classes

### Training Configuration
- **Chunk Size**: 200 notes
- **Chunk Overlap**: 100 notes
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Epochs**: 200 (with early stopping, patience=50)
- **Early Stopping**: Enabled

### Recent Changes (December 2025)

1. **Loss Function**: Migrated from BCELoss to CrossEntropyLoss
   - Removed sigmoid activation from model output
   - Changed targets from one-hot to class indices
   - Updated accuracy calculation to use argmax

2. **5th Category**: Added support for `slur_start_and_end` (category 5)
   - Updated model output_dim from 4 to 5
   - Updated all training scripts and annotation tools

3. **Multi-Combination Training**: Implemented comprehensive evaluation
   - All 12 combinations of 4 pieces (2 train, 1 val, 1 test)
   - Support for multiple trials per combination
   - Automatic research summary report generation

4. **Per-Class Analysis**: Added detailed class performance analysis
   - Detects mode collapse
   - Identifies majority class over-prediction
   - Per-class precision, recall, F1 scores

5. **MIDI Processing**: Fixed multi-track support
   - Now processes all tracks, not just first track
   - Maintains proper temporal ordering

### Current Performance
- **Test Accuracy**: 37.36% ± 15.92% (high variance)
- **Best Case**: 88.22% test accuracy
- **Worst Case**: 5.87% (mode collapse)
- **Key Issue**: High variance indicates sensitivity to initialization

### Key Files
- `src/run_multi_trial_training.py` - Main multi-combination training script
- `src/ml_data_pipeline.py` - Data loading and target creation (5 classes)
- `src/ml_transformer_model.py` - Transformer model (5 output classes)
- `src/ml_chunked_train.py` - Chunked training with CrossEntropyLoss
- `output/research_summary_report.txt` - Comprehensive evaluation report

### Next Steps
1. Address high variance (random seed, class weighting)
2. Improve generalization (more data, regularization)
3. Hyperparameter tuning
4. Architecture improvements
