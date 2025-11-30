#!/usr/bin/env python3
"""
MIDI Piano Roll ML System - Transformer-Based Main Script
Streamlined workflow for transformer-based slur prediction

This script orchestrates the transformer-based pipeline:
1. Generate initial data matrices (notes, pedal) from MIDI files
2. Create annotation CSV for slur labeling  
3. Train transformer model for slur prediction
4. Evaluate model performance

Author: MIDI Piano Roll ML System v2.0 - Transformer Edition
Date: 2024
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from complete_midi_processor import process_midi_file
from ml_data_pipeline import load_and_prepare_data
from ml_train import run_overfitting_test
from ml_chunked_train import run_chunked_overfitting_test

# Configuration variables - modify these for your files
MIDI_FILE = "Slur Training Dataset/Beethoven_Piano_Sonata_No_10_Op_14_No_2_fQqNsTUvqCY_cut_mov_1.mid"
ANNOTATED_CSV = "Piano Slur Annotations/Beethoven_Piano_Sonata_No_10_Op_14_No_2_slur_annotation_completed - Beethoven_Piano_Sonata_No_10_Op_14_No_2_fQqNsTUvqCY_cut_mov_1_slur_annotation.csv"
OUTPUT_DIR = "output"

# Chunked training configuration
CHUNKED_CONFIG = {
    'use_chunking': False,  # Set to True to enable chunked training
    'chunk_size': 264,  # Number of notes per chunk (optimal from testing: 264 notes = 10 chunks for 2640-note piece)
    'chunk_overlap': 0,  # Number of notes to overlap between consecutive chunks (0 = no overlap)
    'train_ratio': 0.6,
    'val_ratio': 0.2,
    'test_ratio': 0.2,
    'shuffle_chunks': True,
    'stratified_splitting': True,  # Use stratified splitting for balanced distributions
    'ignore_boundary_notes': 0,  # Start with 0, test with 2
    'learning_rate': 0.005,  # Increased from 0.001 to help escape local minima
    'epochs': 500,
    'early_stopping': True,  # Enable early stopping based on validation
    'patience': 50  # Stop if no improvement for 50 epochs
}

def generate_initial_data(midi_file=MIDI_FILE, output_dir=OUTPUT_DIR):
    """
    Step 1: Generate initial data (notes, pedal matrices and annotation CSV) from MIDI file
    
    Args:
        midi_file (str): Path to MIDI file
        output_dir (str): Output directory for generated files
        
    Returns:
        dict: Processing results including file paths and metadata
    """
    print("🎵 STEP 1: GENERATING INITIAL DATA FROM MIDI")
    print("=" * 60)
    
    if not os.path.exists(midi_file):
        print(f"❌ Error: MIDI file not found: {midi_file}")
        return None
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process MIDI file to generate matrices and annotation CSV
    results = process_midi_file(
        midi_file_path=midi_file,
        output_dir=output_dir,
        note_range=(21, 108),
        preserve_velocity=True
    )
    
    if results:

        print(f"✓ Pedal matrix: {results['metadata']['pedal_matrix_shape']}")
        print(f"✓ Total notes extracted: {results['metadata']['total_notes']}")
        print(f"✓ Total pedal events: {results['metadata']['total_pedal_events']}")
        print(f"✓ Files saved to: {output_dir}/")
        
        # List generated files needed for ML
        base_name = results['base_filename']
        print(f"\nGenerated files for ML pipeline:")
        print(f"  - {base_name}_notes.npy (for reference, not used in transformer)")
        print(f"  - {base_name}_pedal.npy (sustain pedal data)")
        print(f"  - {base_name}_slur_annotation.csv (raw note data + annotation template)")
        print(f"  - {base_name}_metadata.txt")
        print(f"\n📝 Next step: Manually annotate slur categories in the CSV file")
    else:
        print("❌ Failed to process MIDI file")
    
    return results

def prepare_ml_data(base_filename=None, output_dir=OUTPUT_DIR):
    """
    Step 2: Prepare data for ML training (requires annotated CSV)
    
    Args:
        base_filename (str): Base filename (auto-detected if None)
        output_dir (str): Output directory
        
    Returns:
        tuple: (inputs, targets, norm_params, stats) or None if failed
    """
    print("📊 STEP 2: PREPARING DATA FOR ML TRAINING")
    print("=" * 60)
    
    if base_filename is None:
        base_filename = os.path.splitext(os.path.basename(MIDI_FILE))[0]
    
    # Check if annotated CSV exists
    annotation_csv = os.path.join(output_dir, f"{base_filename}_slur_annotation.csv")
    if not os.path.exists(annotation_csv):
        print(f"❌ Error: Annotation CSV not found: {annotation_csv}")
        print("   Please annotate the slur categories first using the CSV file")
        return None
    
    # Check if annotations are complete
    df = pd.read_csv(annotation_csv)
    annotated_count = df['Slur_Category'].notna().sum()
    total_count = len(df)
    
    if annotated_count == 0:
        print(f"❌ Error: No slur annotations found in CSV")
        print("   Please fill in the 'Slur_Category' column with values 1, 2, 3, or 4")
        return None
    
    print(f"✓ Found {annotated_count}/{total_count} annotated notes ({annotated_count/total_count:.1%})")
    
    if annotated_count < total_count:
        print(f"⚠️  Partial annotation detected - will use available data")
    
    try:
        # Prepare data for ML
        inputs, targets, norm_params, stats = load_and_prepare_data(base_filename, output_dir)
        
        # Save processed data for later use
        from ml_data_pipeline import save_processed_data
        processed_data_path = os.path.join(output_dir, f"{base_filename}_processed_for_ml.pt")
        save_processed_data(inputs, targets, norm_params, stats, processed_data_path)
        
        print(f"✅ Data prepared for transformer training:")
        print(f"   Input shape: {inputs.shape} (sequence_length, features)")
        print(f"   Target shape: {targets.shape} (sequence_length, binary_outputs)")
        print(f"   Features: start_time, duration, midi_pitch, velocity, sustain_start, sustain_end")
        print(f"   Outputs: slur_start, slur_middle, slur_end, no_slur, slur_start_and_end")
        print(f"   Saved to: {processed_data_path}")
        
        return inputs, targets, norm_params, stats
        
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        return None

def train_transformer(base_filename=None, output_dir=OUTPUT_DIR, epochs=1000, learning_rate=1e-3, use_chunking=False):
    """
    Step 3: Train transformer model on the annotated data
    
    Args:
        base_filename (str): Base filename (auto-detected if None)
        output_dir (str): Output directory
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for training
        use_chunking (bool): Whether to use chunked training approach
        
    Returns:
        dict: Training results
    """
    print("🧠 STEP 3: TRAINING TRANSFORMER MODEL")
    print("=" * 60)
    
    if base_filename is None:
        base_filename = os.path.splitext(os.path.basename(MIDI_FILE))[0]
    
    try:
        if use_chunking:
            print("🎵 Using CHUNKED training approach")
            print(f"   Chunk size: {CHUNKED_CONFIG['chunk_size']} notes per chunk")
            if CHUNKED_CONFIG.get('chunk_overlap', 0) > 0:
                print(f"   Chunk overlap: {CHUNKED_CONFIG['chunk_overlap']} notes")
            print(f"   Train/Val/Test: {CHUNKED_CONFIG['train_ratio']:.1%}/{CHUNKED_CONFIG['val_ratio']:.1%}/{CHUNKED_CONFIG['test_ratio']:.1%}")
            print(f"   Ignore boundary notes: {CHUNKED_CONFIG['ignore_boundary_notes']}")
            
            # Update config with provided parameters
            config = CHUNKED_CONFIG.copy()
            config['epochs'] = epochs
            config['learning_rate'] = learning_rate
            
            # Run chunked overfitting test
            results = run_chunked_overfitting_test(
                midi_file=MIDI_FILE,
                annotation_file=ANNOTATED_CSV,
                config=config
            )
        else:
            print("🎵 Using ORIGINAL training approach")
            
            # Run original overfitting test
            results = run_overfitting_test(
                base_filename=base_filename,
                output_dir=output_dir,
                epochs=epochs,
                learning_rate=learning_rate,
                device="cpu"  # Change to "cuda" if you have GPU
            )
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_complete_ml_pipeline(use_chunking=False):
    """
    Execute the complete ML pipeline
    
    Args:
        use_chunking (bool): Whether to use chunked training approach
    """
    approach_name = "CHUNKED" if use_chunking else "ORIGINAL"
    print(f"🎹 MIDI PIANO ROLL ML SYSTEM v2.0 - TRANSFORMER EDITION ({approach_name})")
    print("=" * 70)
    print("Complete ML workflow: MIDI → Data → Annotation → Transformer Training")
    print("=" * 70)
    
    # Step 1: Generate initial data
    results = generate_initial_data()
    if not results:
        print("❌ Pipeline failed at data generation")
        return False
    
    print("\n" + "="*70)
    
    # Step 2: Prepare ML data (check if annotation is ready)
    ml_data = prepare_ml_data()
    if not ml_data:
        print("❌ Pipeline failed at ML data preparation")
        print("   Complete the annotation step and try again")
        return False
    
    print("\n" + "="*70)
    
    # Step 3: Train transformer
    training_results = train_transformer(use_chunking=use_chunking)
    if not training_results:
        print("❌ Pipeline failed at training")
        return False
    
    print("\n" + "="*70)
    print("🎉 ML PIPELINE COMPLETE!")
    print("=" * 70)
    
    if training_results.get('success', False) or training_results.get('overfitting_success', False):
        print("✅ SUCCESS: Transformer successfully learned the slur patterns!")
        print("   → Model can memorize musical phrasing from note sequences")
        print("   → Ready for multi-piece training and generalization")
        if 'final_metrics' in training_results:
            print(f"   → Final accuracy: {training_results['final_metrics']['accuracy']:.1%}")
        elif 'test_accuracy' in training_results:
            print(f"   → Final accuracy: {training_results['test_accuracy']:.1%}")
    else:
        print("❌ TRAINING INCOMPLETE: Model didn't fully memorize the patterns")
        print("   → Consider longer training or architecture adjustments")
        if 'final_metrics' in training_results:
            print(f"   → Achieved accuracy: {training_results['final_metrics']['accuracy']:.1%}")
        elif 'test_accuracy' in training_results:
            print(f"   → Achieved accuracy: {training_results['test_accuracy']:.1%}")
    
    print(f"📁 All results saved in: {OUTPUT_DIR}/")
    print("📖 See docs/ for detailed documentation")
    
    return training_results.get('success', False) or training_results.get('overfitting_success', False)

def test_data_pipeline_only():
    """Test only the data pipeline without training"""
    print("🧪 TESTING DATA PIPELINE ONLY")
    print("=" * 50)
    
    # Test data preparation
    ml_data = prepare_ml_data()
    if ml_data:
        inputs, targets, norm_params, stats = ml_data
        print(f"✅ Data pipeline test successful!")
        print(f"   Ready for transformer training")
        return True
    else:
        print(f"❌ Data pipeline test failed")
        return False

def main():
    """
    Main entry point with command line argument handling
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='MIDI Piano Roll ML System - Transformer Edition')
    parser.add_argument('--step', choices=['data', 'prepare', 'train', 'test-pipeline', 'all'],
                        default='all', help='Which step to run')
    parser.add_argument('--midi', default=MIDI_FILE, help='MIDI file path')
    parser.add_argument('--output', default=OUTPUT_DIR, help='Output directory')
    parser.add_argument('--epochs', type=int, default=1000, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--use-chunking', action='store_true', 
                       help='Use chunked training approach')
    parser.add_argument('--chunk-size', type=int, default=264,
                       help='Number of notes per chunk (chunked mode only)')
    parser.add_argument('--chunk-overlap', type=int, default=0,
                       help='Number of notes to overlap between consecutive chunks (chunked mode only)')
    parser.add_argument('--ignore-boundary', type=int, default=0,
                       help='Number of boundary notes to ignore (chunked mode only)')
    
    args = parser.parse_args()
    
    # Update configuration
    midi_file = args.midi
    output_dir = args.output
    
    # Update chunked config if chunking is enabled
    if args.use_chunking:
        CHUNKED_CONFIG['chunk_size'] = args.chunk_size
        CHUNKED_CONFIG['chunk_overlap'] = args.chunk_overlap
        CHUNKED_CONFIG['ignore_boundary_notes'] = args.ignore_boundary
    
    if args.step == 'data':
        generate_initial_data(midi_file, output_dir)
    elif args.step == 'prepare':
        base_filename = os.path.splitext(os.path.basename(midi_file))[0]
        prepare_ml_data(base_filename, output_dir)
    elif args.step == 'train':
        base_filename = os.path.splitext(os.path.basename(midi_file))[0]
        train_transformer(base_filename, output_dir, args.epochs, args.lr, args.use_chunking)
    elif args.step == 'test-pipeline':
        base_filename = os.path.splitext(os.path.basename(midi_file))[0]
        prepare_ml_data(base_filename, output_dir)
    else:  # 'all'
        run_complete_ml_pipeline(args.use_chunking)

if __name__ == "__main__":
    main()
