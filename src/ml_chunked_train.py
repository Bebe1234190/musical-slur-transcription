"""
Chunked Training System for Musical Slur Prediction

This module provides chunked training capabilities while preserving
the original non-chunked approach. It trains the transformer model
on chunked musical sequences with configurable parameters.

Author: AI Assistant
Date: December 2025
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Union
import os
import sys

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_transformer_model import create_model, MusicSlurTrainer
from ml_chunked_pipeline import prepare_chunked_data, calculate_chunk_loss, print_chunked_data_summary


def calculate_chunk_accuracy(outputs: torch.Tensor, targets: torch.Tensor, 
                           ignore_boundary: int = 0) -> float:
    """
    Calculate accuracy for a chunk, optionally ignoring boundary notes.
    
    Uses argmax for class prediction with CrossEntropyLoss.
    
    Args:
        outputs: Model predictions (batch, seq_len, num_classes) or (seq_len, num_classes)
        targets: Target class indices (batch, seq_len) or (seq_len,)
        ignore_boundary: Number of notes to ignore at each end
    
    Returns:
        Accuracy as a float between 0 and 1
    """
    # Reshape for argmax: (batch*seq_len, num_classes) and (batch*seq_len,)
    if outputs.dim() == 3:
        batch_size, seq_len, num_classes = outputs.shape
        outputs_flat = outputs.view(-1, num_classes)
        targets_flat = targets.view(-1).long()
    else:
        seq_len, num_classes = outputs.shape
        outputs_flat = outputs.view(-1, num_classes)
        targets_flat = targets.view(-1).long()
    
    if ignore_boundary == 0:
        # Use all notes - use argmax for class prediction
        pred_classes = outputs_flat.argmax(dim=1)
        accuracy = (pred_classes == targets_flat).float().mean()
        return accuracy.item()
    
    # Ignore first and last 'ignore_boundary' notes
    if outputs.dim() == 3:
        batch_size, seq_len, num_classes = outputs.shape
        if seq_len <= 2 * ignore_boundary:
            pred_classes = outputs_flat.argmax(dim=1)
            accuracy = (pred_classes == targets_flat).float().mean()
            return accuracy.item()
        start_idx = ignore_boundary
        end_idx = seq_len - ignore_boundary
        # Extract middle portion and flatten
        middle_outputs = outputs[:, start_idx:end_idx, :].view(-1, num_classes)
        middle_targets = targets[:, start_idx:end_idx].view(-1).long()
    else:
        seq_len, num_classes = outputs.shape
        if seq_len <= 2 * ignore_boundary:
            pred_classes = outputs_flat.argmax(dim=1)
            accuracy = (pred_classes == targets_flat).float().mean()
            return accuracy.item()
        start_idx = ignore_boundary
        end_idx = seq_len - ignore_boundary
        # Extract middle portion and flatten
        middle_outputs = outputs[start_idx:end_idx, :].view(-1, num_classes)
        middle_targets = targets[start_idx:end_idx].view(-1).long()
    
    # Use argmax for class prediction
    pred_classes = middle_outputs.argmax(dim=1)
    accuracy = (pred_classes == middle_targets).float().mean()
    return accuracy.item()


def train_chunked_epoch(model: nn.Module, train_chunks: List[Dict], 
                       optimizer: torch.optim.Optimizer, criterion: nn.Module,
                       config: Dict) -> Tuple[float, float]:
    """
    Train for one epoch on chunked data.
    
    Args:
        model: Transformer model
        train_chunks: List of training chunks
        optimizer: Optimizer
        criterion: Loss function
        config: Training configuration
    
    Returns:
        Tuple of (average_loss, average_accuracy)
    """
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    num_chunks = len(train_chunks)
    
    ignore_boundary = config.get('ignore_boundary_notes', 0)
    
    # FIXED: Accumulate gradients across all chunks before updating weights
    optimizer.zero_grad()  # Reset gradients once at the start
    
    for i, chunk in enumerate(train_chunks):
        features = chunk['features']  # Shape: [chunk_length, 6]
        targets = chunk['targets']    # Shape: [chunk_length,] for class indices
        
        # FIXED: Add batch dimension to match original approach
        if features.dim() == 2:
            features = features.unsqueeze(0)  # (1, chunk_len, 6)
        if targets.dim() == 1:
            targets = targets.unsqueeze(0)     # (1, chunk_len) for class indices
        
        # Forward pass
        outputs = model(features)
        
        # Calculate loss (with optional boundary ignoring)
        loss = calculate_chunk_loss(outputs, targets, criterion, ignore_boundary)
        
        # Accumulate gradients (don't step yet)
        loss.backward()
        
        # Calculate accuracy (with optional boundary ignoring)
        accuracy = calculate_chunk_accuracy(outputs, targets, ignore_boundary)
        
        # Accumulate metrics
        total_loss += loss.item()
        total_accuracy += accuracy
        
        # Progress update
        if (i + 1) % 10 == 0 or i == num_chunks - 1:
            print(f"  Chunk {i+1}/{num_chunks} | Loss: {loss.item():.4f} | Acc: {accuracy:.4f}")
    
    # FIXED: Update weights once after processing all chunks
    optimizer.step()
    
    return total_loss / num_chunks, total_accuracy / num_chunks


def validate_chunked_epoch(model: nn.Module, val_chunks: List[Dict], 
                          criterion: nn.Module, config: Dict, 
                          return_outputs: bool = False) -> Union[Tuple[float, float], Tuple[float, float, List[torch.Tensor], List[torch.Tensor]]]:
    """
    Validate for one epoch on chunked data.
    
    Args:
        model: Transformer model
        val_chunks: List of validation chunks
        criterion: Loss function
        config: Training configuration
        return_outputs: If True, also return all outputs and targets for analysis
    
    Returns:
        If return_outputs=False: Tuple of (average_loss, average_accuracy)
        If return_outputs=True: Tuple of (average_loss, average_accuracy, all_outputs, all_targets)
    """
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    num_chunks = len(val_chunks)
    
    # Handle empty validation set
    if num_chunks == 0:
        if return_outputs:
            return 0.0, 0.0, [], []
        return 0.0, 0.0
    
    ignore_boundary = config.get('ignore_boundary_notes', 0)
    
    all_outputs = [] if return_outputs else None
    all_targets = [] if return_outputs else None
    
    with torch.no_grad():
        for chunk in val_chunks:
            features = chunk['features']
            targets = chunk['targets']
            
            # FIXED: Add batch dimension to match original approach
            if features.dim() == 2:
                features = features.unsqueeze(0)  # (1, chunk_len, 6)
            if targets.dim() == 1:
                targets = targets.unsqueeze(0)     # (1, chunk_len) for class indices
            
            # Forward pass
            outputs = model(features)
            
            # Calculate loss and accuracy
            loss = calculate_chunk_loss(outputs, targets, criterion, ignore_boundary)
            accuracy = calculate_chunk_accuracy(outputs, targets, ignore_boundary)
            
            # Accumulate metrics
            total_loss += loss.item()
            total_accuracy += accuracy
            
            # Store outputs and targets if requested
            if return_outputs:
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())
    
    if return_outputs:
        return total_loss / num_chunks, total_accuracy / num_chunks, all_outputs, all_targets
    return total_loss / num_chunks, total_accuracy / num_chunks


def run_chunked_training(midi_file: str, annotation_file: str, config: Dict) -> Dict:
    """
    Run complete chunked training pipeline.
    
    Args:
        midi_file: Path to MIDI file
        annotation_file: Path to annotation CSV
        config: Training configuration
    
    Returns:
        Dictionary with training results and history
    """
    print("🎵 CHUNKED TRAINING SYSTEM")
    print("=" * 50)
    
    # Load chunked data
    print("📊 STEP 1: DATA PREPARATION")
    splits = prepare_chunked_data(
        midi_file=midi_file,
        annotation_file=annotation_file,
        chunk_size=config['chunk_size'],
        overlap=config.get('chunk_overlap', 0),
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
        test_ratio=config['test_ratio'],
        shuffle=config.get('shuffle_chunks', True),
        stratified=config.get('stratified_splitting', True)
    )
    
    # Print data summary
    print_chunked_data_summary(splits)
    
    # Create model
    print("\n🧠 STEP 2: MODEL CREATION")
    # Get the correct input dimension from the data
    input_dim = splits['train'][0]['features'].shape[1]  # Number of features
    model = create_model(input_dim=input_dim)
    print(f"✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"✓ Input dimension: {input_dim} features")
    
    # Setup training
    print("\n🔥 STEP 3: TRAINING SETUP")
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()  # Cross-entropy for multi-class classification
    
    epochs = config['epochs']
    ignore_boundary = config.get('ignore_boundary_notes', 0)
    
    print(f"✓ Optimizer: Adam (lr={config['learning_rate']})")
    print(f"✓ Loss function: BCELoss")
    print(f"✓ Epochs: {epochs}")
    print(f"✓ Ignore boundary notes: {ignore_boundary}")
    
    # Training loop
    print(f"\n🚀 STEP 4: TRAINING ({epochs} epochs)")
    print("-" * 50)
    
    train_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epochs': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training
        train_loss, train_acc = train_chunked_epoch(
            model, splits['train'], optimizer, criterion, config
        )
        
        # Validation
        val_loss, val_acc = validate_chunked_epoch(
            model, splits['val'], criterion, config
        )
        
        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # Store history
        train_history['train_loss'].append(train_loss)
        train_history['train_acc'].append(train_acc)
        train_history['val_loss'].append(val_loss)
        train_history['val_acc'].append(val_acc)
        train_history['epochs'].append(epoch + 1)
        
        # Progress update
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train: Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
              f"Val: Loss={val_loss:.4f}, Acc={val_acc:.4f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Early stopping check (if enabled)
        if config.get('early_stopping', False):
            patience = config.get('patience', 20)
            if epoch >= patience:
                # Compare current accuracy with accuracy 'patience' epochs ago
                # This allows for minor fluctuations while detecting true stagnation
                current_acc = val_acc
                past_acc = train_history['val_acc'][epoch - patience]
                
                # If current accuracy is not significantly better than past accuracy
                improvement_threshold = 0.001  # 0.1% improvement threshold
                if current_acc <= past_acc + improvement_threshold:
                    print(f"🛑 Early stopping at epoch {epoch+1}")
                    print(f"   Current validation accuracy: {current_acc:.4f}")
                    print(f"   Validation accuracy {patience} epochs ago: {past_acc:.4f}")
                    print(f"   Improvement: {current_acc - past_acc:.4f} (threshold: {improvement_threshold:.4f})")
                    print(f"   Best validation accuracy achieved: {best_val_acc:.4f}")
                    break
    
    total_time = time.time() - start_time
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"✓ Loaded best model (val_acc: {best_val_acc:.4f})")
    
    # Final evaluation
    print(f"\n📊 STEP 5: FINAL EVALUATION")
    print("-" * 30)
    
    # Test set evaluation
    test_loss, test_acc = validate_chunked_epoch(
        model, splits['test'], criterion, config
    )
    
    print(f"Test Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    
    # Training summary
    print(f"\n📈 TRAINING SUMMARY")
    print("-" * 30)
    print(f"Total epochs: {len(train_history['epochs'])}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Final test accuracy: {test_acc:.4f}")
    
    # Return results
    results = {
        'model': model,
        'train_history': train_history,
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'best_val_accuracy': best_val_acc,
        'total_time': total_time,
        'config': config,
        'splits': splits
    }
    
    return results


def run_chunked_overfitting_test(midi_file: str, annotation_file: str, 
                                config: Dict) -> Dict:
    """
    Run chunked overfitting test to verify the model can learn.
    
    Args:
        midi_file: Path to MIDI file
        annotation_file: Path to annotation CSV
        config: Training configuration
    
    Returns:
        Dictionary with overfitting test results
    """
    print("🎵 CHUNKED OVERFITTING TEST")
    print("=" * 50)
    
    # Use the full number of epochs for overfitting test
    test_config = config.copy()
    # No cap - use the full epochs specified
    
    print(f"Running overfitting test with {test_config['epochs']} epochs...")
    
    # Run training
    results = run_chunked_training(midi_file, annotation_file, test_config)
    
    # Analyze results
    final_train_acc = results['train_history']['train_acc'][-1]
    final_val_acc = results['train_history']['val_acc'][-1]
    test_acc = results['test_accuracy']
    
    print(f"\n🎯 OVERFITTING TEST RESULTS")
    print("-" * 30)
    print(f"Final training accuracy: {final_train_acc:.4f}")
    print(f"Final validation accuracy: {final_val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Determine if overfitting test passed
    target_accuracy = 0.95  # 95% target for overfitting test
    overfitting_success = test_acc >= target_accuracy
    
    if overfitting_success:
        print(f"✅ OVERFITTING TEST PASSED: {test_acc:.4f} >= {target_accuracy:.4f}")
    else:
        print(f"❌ OVERFITTING TEST FAILED: {test_acc:.4f} < {target_accuracy:.4f}")
    
    results['overfitting_success'] = overfitting_success
    results['target_accuracy'] = target_accuracy
    
    return results


if __name__ == "__main__":
    """
    Test the chunked training system
    """
    print("🧪 Testing chunked training system...")
    
    # Test configuration
    test_config = {
        'chunk_size': 100,  # 100 notes per chunk
        'chunk_overlap': 0,  # No overlap by default
        'train_ratio': 0.6,
        'val_ratio': 0.2,
        'test_ratio': 0.2,
        'shuffle_chunks': True,
        'ignore_boundary_notes': 0,
        'learning_rate': 0.001,
        'epochs': 5,
        'early_stopping': False
    }
    
    print("✅ Chunked training system test completed!")
    print("Note: Run with actual data files to test full functionality.")
