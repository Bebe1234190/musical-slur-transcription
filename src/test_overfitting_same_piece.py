#!/usr/bin/env python3
"""
Overfitting Test: Train on same piece for train/val/test
Uses the existing run_single_trial function from run_multi_trial_training.py
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import functions from run_multi_trial_training
from run_multi_trial_training import (
    find_annotated_pieces,
    load_and_chunk_all_pieces,
    group_chunks_by_piece,
    run_single_trial
)

def test_overfitting_all_pieces(chunk_size=200, overlap=100, max_epochs=2000, learning_rate=0.001):
    """
    Test overfitting by training on the same piece for train/val/test.
    Uses the same run_single_trial function but with modified splits.
    """
    print("="*80)
    print("OVERFITTING TEST: Same Piece for Train/Val/Test")
    print("="*80)
    print("This test verifies the implementation by training on the same data")
    print("for train/val/test. If the implementation is correct, we should")
    print("achieve very high accuracy (close to 100%).")
    print("="*80)
    
    # Find all annotated pieces
    print("\n📂 Finding annotated pieces...")
    pieces = find_annotated_pieces("output")
    
    if len(pieces) == 0:
        print("❌ No annotated pieces found!")
        return
    
    print(f"✓ Found {len(pieces)} annotated piece(s)")
    
    # Load and chunk all pieces
    print("\n📊 Loading and chunking all pieces...")
    all_chunks = load_and_chunk_all_pieces(pieces, chunk_size, overlap)
    
    if len(all_chunks) == 0:
        print("❌ No chunks created!")
        return
    
    # Group chunks by piece
    print("\n📊 Grouping chunks by piece...")
    chunks_by_piece = group_chunks_by_piece(all_chunks)
    
    print(f"\n  📊 Pieces found ({len(chunks_by_piece)}):")
    for piece_id, chunks in chunks_by_piece.items():
        print(f"    {piece_id}: {len(chunks)} chunks")
    
    # Training configuration
    config = {
        'chunk_size': chunk_size,
        'chunk_overlap': overlap,
        'train_ratio': 0.6,  # Not used, but required
        'val_ratio': 0.2,    # Not used, but required
        'test_ratio': 0.2,   # Not used, but required
        'learning_rate': learning_rate,
        'epochs': max_epochs,
        'early_stopping': True,
        'patience': 50,  # Early stopping patience
        'ignore_boundary_notes': 0,
        'shuffle_chunks': True,
        'stratified_splitting': False,
        'num_trials': 1
    }
    
    # Test each piece separately
    results = {}
    
    for piece_id, chunks in chunks_by_piece.items():
        print(f"\n{'='*80}")
        print(f"TESTING: {piece_id}")
        print(f"{'='*80}")
        
        # Use the SAME chunks for train, val, and test (overfitting test)
        splits = {
            'train': chunks,
            'val': chunks,
            'test': chunks
        }
        
        print(f"  Train chunks: {len(splits['train'])}")
        print(f"  Val chunks: {len(splits['val'])}")
        print(f"  Test chunks: {len(splits['test'])}")
        print(f"  (All using the same chunks - overfitting test)")
        
        try:
            # Use the existing run_single_trial function
            trial_results = run_single_trial(splits, config, trial_num=1)
            
            test_acc = trial_results['test_acc']
            results[piece_id] = test_acc
            
            print(f"\n{'='*80}")
            print(f"RESULTS for {piece_id}:")
            print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
            print(f"  Final Val Accuracy: {trial_results['final_val_acc']:.4f} ({trial_results['final_val_acc']*100:.2f}%)")
            print(f"  Max Val Accuracy: {trial_results['max_val_acc']:.4f} ({trial_results['max_val_acc']*100:.2f}%)")
            print(f"  Final Train Accuracy: {trial_results['final_train_acc']:.4f} ({trial_results['final_train_acc']*100:.2f}%)")
            print(f"  Epochs: {trial_results['epoch_stopped']}")
            print(f"{'='*80}\n")
            
        except Exception as e:
            print(f"\n❌ Error testing {piece_id}: {e}")
            import traceback
            traceback.print_exc()
            results[piece_id] = None
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: Test Accuracy for Each Piece")
    print("="*80)
    for piece_id, acc in results.items():
        piece_short = os.path.basename(piece_id) if piece_id else "Unknown"
        if acc is not None:
            print(f"  {piece_short:<60}: {acc:.4f} ({acc*100:.2f}%)")
        else:
            print(f"  {piece_short:<60}: FAILED")
    print("="*80)
    print("\nNote: If implementation is correct, all accuracies should be > 0.95 (95%)")
    print("="*80)


if __name__ == "__main__":
    test_overfitting_all_pieces(
        chunk_size=200,
        overlap=100,
        max_epochs=2000,
        learning_rate=0.001
    )

