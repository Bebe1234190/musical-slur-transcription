#!/usr/bin/env python3
"""
Test script to verify implementation by training on the same piece for train/val/test.
This should achieve near-perfect accuracy if the implementation is correct.
"""

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_chunked_pipeline import prepare_chunked_data, chunk_sequence
from ml_transformer_model import create_model
from ml_chunked_train import train_chunked_epoch, validate_chunked_epoch
from ml_data_pipeline import load_and_prepare_data


def test_single_piece(piece_name: str, chunk_size: int = 200, overlap: int = 100, 
                      max_epochs: int = 2000, learning_rate: float = 0.001):
    """
    Test training on a single piece where train/val/test all use the same piece.
    
    Args:
        piece_name: Base filename of the piece (without extension)
        chunk_size: Size of chunks
        overlap: Overlap between chunks
        max_epochs: Maximum number of epochs
        learning_rate: Learning rate
    
    Returns:
        Final test accuracy
    """
    print(f"\n{'='*80}")
    print(f"TESTING: {piece_name}")
    print(f"{'='*80}")
    
    output_dir = "output"
    
    # Load data
    print(f"\n📊 Loading data...")
    inputs, targets, norm_params, stats = load_and_prepare_data(piece_name, output_dir)
    
    # Convert to dictionary format
    raw_data = {
        'features': inputs,
        'targets': targets,
        'piece_id': piece_name
    }
    
    total_notes = raw_data['features'].shape[0]
    print(f"✓ Loaded {total_notes} notes")
    
    # Create chunks
    print(f"\n🔪 Creating chunks (size={chunk_size}, overlap={overlap})...")
    chunks = chunk_sequence(raw_data, chunk_size, overlap)
    print(f"✓ Created {len(chunks)} chunks")
    
    # Use ALL chunks for train, val, and test (same data everywhere)
    print(f"\n📊 Using same chunks for train/val/test (overfitting test)...")
    splits = {
        'train': chunks,
        'val': chunks,
        'test': chunks
    }
    
    print(f"  Train chunks: {len(splits['train'])}")
    print(f"  Val chunks: {len(splits['val'])}")
    print(f"  Test chunks: {len(splits['test'])}")
    
    # Create model
    print(f"\n🧠 Creating model...")
    input_dim = chunks[0]['features'].shape[1]
    model = create_model(input_dim=input_dim)
    print(f"✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n🚀 Training (max_epochs={max_epochs})...")
    print("-" * 80)
    
    best_val_acc = 0.0
    patience = 50
    patience_counter = 0
    
    config = {
        'chunk_size': chunk_size,
        'chunk_overlap': overlap,
        'ignore_boundary_notes': 0
    }
    
    for epoch in range(1, max_epochs + 1):
        # Train
        train_loss, train_acc = train_chunked_epoch(
            model, splits['train'], optimizer, criterion, config
        )
        
        # Validate
        val_loss, val_acc = validate_chunked_epoch(
            model, splits['val'], criterion, config
        )
        
        # Print progress every 50 epochs
        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} ({train_acc*100:.2f}%) | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} ({val_acc*100:.2f}%)")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch > 100:
                print(f"\n🛑 Early stopping at epoch {epoch} (best val acc: {best_val_acc:.4f})")
                break
    
    # Final test evaluation
    print(f"\n📊 Final evaluation on test set...")
    test_loss, test_acc = validate_chunked_epoch(
        model, splits['test'], criterion, config
    )
    
    print(f"\n{'='*80}")
    print(f"RESULTS for {piece_name}:")
    print(f"  Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Final Test Loss: {test_loss:.4f}")
    print(f"  Best Val Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"{'='*80}\n")
    
    return test_acc


def main():
    """Test all 4 pieces."""
    pieces = [
        "Beethoven_Piano_Sonata_No_10_Op_14_No_2_fQqNsTUvqCY_cut_mov_1",
        "midis_for_evaluation_ground_truth_beethoven_sonata_no_16_hisamori_cut_mov_1",
        "midis_for_evaluation_ground_truth_beethoven_rondo_a_capriccio_op_129_smythe",
        "midis_for_evaluation_ground_truth_chopin_etude_op_10_no_12"
    ]
    
    print("="*80)
    print("OVERFITTING TEST: Same Piece for Train/Val/Test")
    print("="*80)
    print("This test verifies the implementation by training on the same data")
    print("for train/val/test. If the implementation is correct, we should")
    print("achieve very high accuracy (close to 100%).")
    print("="*80)
    
    results = {}
    
    for piece in pieces:
        try:
            test_acc = test_single_piece(
                piece_name=piece,
                chunk_size=200,
                overlap=100,
                max_epochs=2000,
                learning_rate=0.001
            )
            results[piece] = test_acc
        except Exception as e:
            print(f"\n❌ Error testing {piece}: {e}")
            import traceback
            traceback.print_exc()
            results[piece] = None
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: Test Accuracy for Each Piece")
    print("="*80)
    for piece, acc in results.items():
        piece_short = piece.split('_')[-1] if '_' in piece else piece
        if acc is not None:
            print(f"  {piece_short:<50}: {acc:.4f} ({acc*100:.2f}%)")
        else:
            print(f"  {piece_short:<50}: FAILED")
    print("="*80)
    print("\nNote: If implementation is correct, all accuracies should be > 0.95 (95%)")
    print("="*80)


if __name__ == "__main__":
    main()

