# Files to Include in GitHub Repository

## ✅ Essential Files (Must Include)

### Core Source Code
- `main.py` - Main entry point
- `src/complete_midi_processor.py` - MIDI file processing
- `src/slur_annotation_tool.py` - Annotation CSV creation
- `src/ml_data_pipeline.py` - Data preprocessing for ML
- `src/ml_transformer_model.py` - Transformer model architecture
- `src/ml_chunked_pipeline.py` - Chunked data processing
- `src/ml_chunked_train.py` - Chunked training implementation
- `src/ml_train.py` - Basic training script
- `src/main_ml.py` - ML workflow orchestration
- `src/run_multi_trial_training.py` - Multi-trial training system
- `src/run_training_experiments.py` - Training experiments
- `src/train_with_stagnation.py` - Advanced training
- `src/test_overfitting_same_piece.py` - Overfitting test script
- `src/test_same_piece_overfitting.py` - Alternative overfitting test

### Configuration Files
- `requirements.txt` - Full dependencies
- `requirements_minimal.txt` - Minimal dependencies
- `.gitignore` - Git ignore rules
- `README.md` - Project documentation

### Documentation
- `docs/MODEL_ARCHITECTURE_SECTION.md` - Model architecture (for paper)
- `docs/PROJECT_COMPREHENSIVE_DOCUMENTATION_DECEMBER_2025.md` - Complete project history
- `docs/PROJECT_ORGANIZATION.md` - Project structure
- `docs/SESSION_UPDATE_DECEMBER_2025.md` - Recent session updates
- `docs/README.md` (if exists) - Additional docs

### Dataset Files

#### MIDI Files (4 pieces):
1. `output/Beethoven_Piano_Sonata_No_10_Op_14_No_2_fQqNsTUvqCY_cut_mov_1.mid` 
   - OR from `Slur Training Dataset/Beethoven_Piano_Sonata_No_10_Op_14_No_2_fQqNsTUvqCY_cut_mov_1.mid`
2. `output/midis_for_evaluation_ground_truth_beethoven_sonata_no_16_hisamori_cut_mov_1.mid`
3. `output/midis_for_evaluation_ground_truth_beethoven_rondo_a_capriccio_op_129_smythe.mid`
4. `output/midis_for_evaluation_ground_truth_chopin_etude_op_10_no_12.mid`

#### Completed Annotation CSV Files (4 pieces):
1. `output/Beethoven_Piano_Sonata_No_10_Op_14_No_2_fQqNsTUvqCY_cut_mov_1_slur_annotation_completed.csv`
2. `output/midis_for_evaluation_ground_truth_beethoven_sonata_no_16_hisamori_cut_mov_1_slur_annotation_completed.csv`
3. `output/midis_for_evaluation_ground_truth_beethoven_rondo_a_capriccio_op_129_smythe_slur_annotation_completed.csv`
4. `output/midis_for_evaluation_ground_truth_chopin_etude_op_10_no_12_slur_annotation_completed.csv`

#### Supporting Data Files (Optional but helpful):
- `output/*_metadata.txt` - Processing metadata for each piece
- `output/*_pedal.csv` - Pedal data for each piece

## ❌ Files to Exclude (via .gitignore)

- `__pycache__/` - Python cache directories
- `*.pyc`, `*.pyo` - Compiled Python files
- `*.pt`, `*.pth` - Model checkpoint files (too large)
- `*.npy`, `*.npz` - Large numpy arrays
- `*.log` - Log files
- `output/multi_trial_combination_results.txt` - Large result files
- `output/research_summary_report.txt` - Can be regenerated
- `.DS_Store` - macOS system files
- `venv/`, `env/` - Virtual environments

## 📁 Recommended Repository Structure

```
musical-slur-transcription/
├── .gitignore
├── README.md
├── LICENSE (optional - add later)
├── main.py
├── requirements.txt
├── requirements_minimal.txt
├── GITHUB_SETUP_GUIDE.md (this guide)
├── src/
│   ├── complete_midi_processor.py
│   ├── slur_annotation_tool.py
│   ├── ml_data_pipeline.py
│   ├── ml_transformer_model.py
│   ├── ml_chunked_pipeline.py
│   ├── ml_chunked_train.py
│   ├── ml_train.py
│   ├── main_ml.py
│   ├── run_multi_trial_training.py
│   ├── run_training_experiments.py
│   ├── train_with_stagnation.py
│   └── [other source files]
├── docs/
│   ├── MODEL_ARCHITECTURE_SECTION.md
│   ├── PROJECT_COMPREHENSIVE_DOCUMENTATION_DECEMBER_2025.md
│   ├── PROJECT_ORGANIZATION.md
│   └── [other documentation]
├── data/ (or dataset/)
│   ├── midi/
│   │   ├── beethoven_sonata_10.mid
│   │   ├── beethoven_sonata_16.mid
│   │   ├── beethoven_rondo.mid
│   │   └── chopin_etude.mid
│   └── annotations/
│       ├── beethoven_sonata_10_slur_annotation_completed.csv
│       ├── beethoven_sonata_16_slur_annotation_completed.csv
│       ├── beethoven_rondo_slur_annotation_completed.csv
│       └── chopin_etude_slur_annotation_completed.csv
└── output/ (optional - for example outputs)
    └── [metadata and pedal CSV files]
```

## 🎯 Quick Checklist Before Pushing

- [ ] All source code files are included
- [ ] `.gitignore` is set up correctly
- [ ] `README.md` is up to date
- [ ] All 4 MIDI files are included
- [ ] All 4 completed annotation CSV files are included
- [ ] No large model files (*.pt) are included
- [ ] No log files are included
- [ ] No __pycache__ directories are included
- [ ] Requirements files are included

## 📝 Notes

- **Model files (*.pt)**: These are large and can be regenerated. Exclude them from the repository. Users can train their own models.
- **Annotation files**: Only include `*_completed.csv` files, not the template files.
- **Documentation**: Include key documentation files, especially the comprehensive documentation and model architecture section.

