#!/usr/bin/env python3
"""
Multi-Trial Training Script for Chunked Musical Slur Prediction

This script:
1. Loads all available annotated pieces
2. Chunks each piece
3. Runs multiple trials with shuffled chunk assignment
4. Reports metrics for each trial and averages across all trials

Author: AI Assistant
Date: December 2025
"""

import torch
import torch.nn as nn
import time
import random
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from itertools import combinations, permutations

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_chunked_pipeline import prepare_chunked_data, calculate_chunk_loss, print_chunked_data_summary
from ml_transformer_model import create_model
from ml_chunked_train import (
    train_chunked_epoch, 
    validate_chunked_epoch, 
    calculate_chunk_accuracy
)


def find_annotated_pieces(output_dir: str = "output") -> List[Tuple[str, str]]:
    """
    Find all annotated pieces with completed annotations.
    
    Only looks for files ending with '_slur_annotation_completed.csv' to ensure
    only fully annotated pieces are used.
    
    Returns list of (midi_file, annotation_file) tuples.
    """
    pieces = []
    
    # Look for completed annotation CSV files
    annotation_files = list(Path(output_dir).glob("*_slur_annotation_completed.csv"))
    
    for annotation_file in annotation_files:
        # Extract base filename by removing '_slur_annotation_completed'
        base_name = annotation_file.stem.replace("_slur_annotation_completed", "")
        
        # Try to find corresponding MIDI file
        # First check in output directory
        midi_file = Path(output_dir) / f"{base_name}.mid"
        if not midi_file.exists():
            # Check in Slur Training Dataset
            midi_file = Path("Slur Training Dataset") / f"{base_name}.mid"
        
        if midi_file.exists():
            pieces.append((str(midi_file), str(annotation_file)))
            print(f"✓ Found completed piece: {base_name}")
        else:
            print(f"⚠️  Warning: MIDI file not found for {base_name}")
    
    return pieces


def load_and_chunk_all_pieces(pieces: List[Tuple[str, str]], 
                              chunk_size: int, overlap: int = 0) -> List[Dict]:
    """
    Load and chunk all pieces, returning a combined list of chunks.
    
    Args:
        pieces: List of (midi_file, annotation_file) tuples
        chunk_size: Number of notes per chunk
        overlap: Number of notes to overlap between chunks
    
    Returns:
        List of all chunks from all pieces
    """
    all_chunks = []
    
    for midi_file, annotation_file in pieces:
        print(f"\n📊 Processing: {os.path.basename(midi_file)}")
        
        # Extract base filename
        base_filename = os.path.splitext(os.path.basename(midi_file))[0]
        output_dir = "output"
        
        # Load data using existing pipeline
        from ml_data_pipeline import load_and_prepare_data
        
        inputs, targets, norm_params, stats = load_and_prepare_data(base_filename, output_dir)
        
        # Convert to dictionary format
        raw_data = {
            'features': inputs,
            'targets': targets,
            'piece_id': base_filename
        }
        
        # Chunk the piece
        from ml_chunked_pipeline import chunk_sequence
        chunks = chunk_sequence(raw_data, chunk_size, overlap)
        
        total_chunked_notes = sum(ch['chunk_size'] for ch in chunks)
        omitted_notes = raw_data['features'].shape[0] - total_chunked_notes
        print(f"  ✓ Created {len(chunks)} chunks ({total_chunked_notes} notes)")
        if omitted_notes > 0:
            print(f"  ⚠️  Omitted {omitted_notes} leftover notes")
        all_chunks.extend(chunks)
    
    print(f"\n📊 Total chunks from all pieces: {len(all_chunks)}")
    return all_chunks


def split_chunks_random(all_chunks: List[Dict], 
                       train_ratio: float = 0.6,
                       val_ratio: float = 0.2,
                       test_ratio: float = 0.2,
                       seed: int = None) -> Dict[str, List[Dict]]:
    """
    Randomly split chunks into train/val/test sets.
    
    Args:
        all_chunks: List of all chunks from all pieces
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with 'train', 'val', 'test' keys
    """
    if seed is not None:
        random.seed(seed)
    
    # Shuffle all chunks together
    shuffled_chunks = all_chunks.copy()
    random.shuffle(shuffled_chunks)
    
    # Calculate split sizes
    total_chunks = len(shuffled_chunks)
    num_train = int(total_chunks * train_ratio)
    num_val = int(total_chunks * val_ratio)
    num_test = total_chunks - num_train - num_val
    
    # Split
    splits = {
        'train': shuffled_chunks[:num_train],
        'val': shuffled_chunks[num_train:num_train + num_val],
        'test': shuffled_chunks[num_train + num_val:]
    }
    
    return splits


def group_chunks_by_piece(all_chunks: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group chunks by piece_id.
    
    Args:
        all_chunks: List of all chunks from all pieces
    
    Returns:
        Dictionary mapping piece_id to list of chunks
    """
    chunks_by_piece = {}
    for chunk in all_chunks:
        piece_id = chunk.get('piece_id', 'unknown')
        if piece_id not in chunks_by_piece:
            chunks_by_piece[piece_id] = []
        chunks_by_piece[piece_id].append(chunk)
    
    # Sort chunks by chunk_id to maintain order
    for piece_id, chunks in chunks_by_piece.items():
        chunks.sort(key=lambda x: x.get('chunk_id', 0))
    
    return chunks_by_piece


def generate_all_combinations(chunks_by_piece: Dict[str, List[Dict]]) -> List[Tuple[str, str, str, str]]:
    """
    Generate all combinations of 2 pieces for training, 1 for validation, 1 for testing.
    
    Args:
        chunks_by_piece: Dictionary mapping piece_id to list of chunks
    
    Returns:
        List of tuples: (train_piece1, train_piece2, val_piece, test_piece)
    """
    piece_ids = list(chunks_by_piece.keys())
    
    if len(piece_ids) != 4:
        raise ValueError(f"Expected exactly 4 pieces, found {len(piece_ids)}")
    
    combinations_list = []
    
    # For each pair of training pieces
    for train_pair in combinations(piece_ids, 2):
        remaining = [p for p in piece_ids if p not in train_pair]
        
        # For each remaining piece as validation
        for val_piece in remaining:
            test_piece = [p for p in remaining if p != val_piece][0]
            combinations_list.append((train_pair[0], train_pair[1], val_piece, test_piece))
    
    return combinations_list


def create_split_from_combination(chunks_by_piece: Dict[str, List[Dict]], 
                                  train_piece1: str, train_piece2: str,
                                  val_piece: str, test_piece: str) -> Dict[str, List[Dict]]:
    """
    Create train/val/test split from specific piece combination.
    
    Args:
        chunks_by_piece: Dictionary mapping piece_id to list of chunks
        train_piece1: First training piece ID
        train_piece2: Second training piece ID
        val_piece: Validation piece ID
        test_piece: Test piece ID
    
    Returns:
        Dictionary with 'train', 'val', 'test' keys
    """
    train_chunks = chunks_by_piece[train_piece1] + chunks_by_piece[train_piece2]
    val_chunks = chunks_by_piece[val_piece]
    test_chunks = chunks_by_piece[test_piece]
    
    return {
        'train': train_chunks,
        'val': val_chunks,
        'test': test_chunks
    }


def split_chunks_by_piece(beethoven_chunks: List[Dict], rondo_chunks: List[Dict], 
                         seed: int = None) -> Dict[str, List[Dict]]:
    """
    Split chunks by piece: Beethoven sonatas for training, Rondo randomly split for val/test.
    
    Args:
        beethoven_chunks: List of chunks from Beethoven sonatas
        rondo_chunks: List of chunks from Rondo piece
        seed: Random seed for reproducibility (None for random)
    
    Returns:
        Dictionary with 'train', 'val', 'test' keys
    """
    # Randomly shuffle Rondo chunks, then split in half
    if seed is not None:
        random.seed(seed)
    
    num_rondo_chunks = len(rondo_chunks)
    shuffled_rondo = rondo_chunks.copy()
    random.shuffle(shuffled_rondo)
    
    split_point = num_rondo_chunks // 2
    
    rondo_val = shuffled_rondo[:split_point]
    rondo_test = shuffled_rondo[split_point:]
    
    splits = {
        'train': beethoven_chunks,
        'val': rondo_val,
        'test': rondo_test
    }
    
    return splits


def run_single_trial(splits: Dict[str, List[Dict]], 
                    config: Dict,
                    trial_num: int) -> Dict:
    """
    Run a single training trial.
    
    Args:
        splits: Dictionary with train/val/test chunk splits
        config: Training configuration
        trial_num: Trial number for logging
    
    Returns:
        Dictionary with trial results and metrics
    """
    print(f"\n{'='*60}")
    print(f"TRIAL {trial_num}/{config['num_trials']}")
    print(f"{'='*60}")
    print(f"Split sizes: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])} chunks")
    
    # Create model
    if len(splits['train']) == 0:
        raise ValueError("No training chunks available!")
    
    input_dim = splits['train'][0]['features'].shape[1]
    model = create_model(input_dim=input_dim)
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()  # Cross-entropy for multi-class classification
    
    epochs = config['epochs']
    best_val_acc = 0.0
    best_model_state = None
    best_val_epoch = 0
    
    train_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epochs': []
    }
    
    start_time = time.time()
    
    # Training loop
    for epoch in range(epochs):
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
            best_val_epoch = epoch + 1
        
        # Store history
        train_history['train_loss'].append(train_loss)
        train_history['train_acc'].append(train_acc)
        train_history['val_loss'].append(val_loss)
        train_history['val_acc'].append(val_acc)
        train_history['epochs'].append(epoch + 1)
        
        # Early stopping check
        if config.get('early_stopping', False):
            patience = config.get('patience', 50)
            if epoch >= patience:
                current_acc = val_acc
                past_acc = train_history['val_acc'][epoch - patience]
                improvement_threshold = 0.001
                
                if current_acc <= past_acc + improvement_threshold:
                    print(f"  🛑 Early stopping at epoch {epoch+1}")
                    break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation (with outputs for analysis to avoid duplicate forward pass)
    test_loss, test_acc, test_outputs, test_targets = validate_chunked_epoch(
        model, splits['test'], criterion, config, return_outputs=True
    )
    
    # Detailed per-class analysis (using pre-computed outputs)
    class_analysis = analyze_class_predictions_from_outputs(test_outputs, test_targets)
    
    total_time = time.time() - start_time
    final_train_acc = train_history['train_acc'][-1]
    final_val_acc = train_history['val_acc'][-1]
    
    results = {
        'trial_num': trial_num,
        'test_acc': test_acc,
        'final_val_acc': final_val_acc,
        'max_val_acc': best_val_acc,
        'final_train_acc': final_train_acc,
        'epoch_stopped': len(train_history['epochs']),
        'epoch_max_val': best_val_epoch,
        'test_loss': test_loss,
        'total_time': total_time,
        'class_analysis': class_analysis
    }
    
    return results


def analyze_class_predictions_from_outputs(all_outputs: List[torch.Tensor], 
                                            all_targets: List[torch.Tensor]) -> Dict:
    """
    Analyze class predictions from pre-computed outputs (optimized version).
    
    Args:
        all_outputs: List of output tensors from model (already computed)
        all_targets: List of target tensors
    
    Returns:
        Dictionary with per-class metrics and prediction distributions
    """
    category_names = ['slur_start', 'slur_middle', 'slur_end', 'no_slur', 'slur_start_and_end']
    
    all_predictions = []
    all_targets_flat = []
    
    # Process pre-computed outputs
    for outputs, targets in zip(all_outputs, all_targets):
        # Get predictions
        batch_size, seq_len, num_classes = outputs.shape
        outputs_flat = outputs.view(-1, num_classes)
        targets_flat = targets.view(-1).long()
        pred_classes = outputs_flat.argmax(dim=1)
        
        all_predictions.extend(pred_classes.numpy())
        all_targets_flat.extend(targets_flat.numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets_flat)
    
    # Calculate per-class metrics
    class_metrics = {}
    for class_idx, category in enumerate(category_names):
        target_mask = all_targets == class_idx
        pred_mask = all_predictions == class_idx
        
        num_targets = target_mask.sum()
        num_predictions = pred_mask.sum()
        
        if num_targets > 0:
            # Per-class accuracy (of the actual class, how many were predicted correctly)
            correct = ((pred_mask) & (target_mask)).sum()
            class_acc = correct / num_targets if num_targets > 0 else 0.0
            
            # Precision: of all predictions for this class, how many were correct
            precision = correct / num_predictions if num_predictions > 0 else 0.0
            
            # Recall: same as class accuracy
            recall = class_acc
            
            # F1 score
            f1 = 2 * precision * recall / (precision + recall + 1e-8) if (precision + recall) > 0 else 0.0
        else:
            class_acc = precision = recall = f1 = 0.0
        
        class_metrics[category] = {
            'num_targets': int(num_targets),
            'num_predictions': int(num_predictions),
            'accuracy': float(class_acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
    
    # Overall distribution
    target_distribution = {category_names[i]: int((all_targets == i).sum()) for i in range(5)}
    prediction_distribution = {category_names[i]: int((all_predictions == i).sum()) for i in range(5)}
    
    # Check if model is just predicting majority class
    majority_class_idx = np.argmax([target_distribution[cat] for cat in category_names])
    majority_class = category_names[majority_class_idx]
    majority_predictions = (all_predictions == majority_class_idx).sum()
    majority_ratio = majority_predictions / len(all_predictions) if len(all_predictions) > 0 else 0.0
    
    return {
        'class_metrics': class_metrics,
        'target_distribution': target_distribution,
        'prediction_distribution': prediction_distribution,
        'majority_class': majority_class,
        'majority_prediction_ratio': float(majority_ratio),
        'total_predictions': len(all_predictions)
    }


def print_trial_results(results: Dict, combination_info: str = None):
    """Print results for a single trial."""
    if combination_info:
        print(f"\n📊 {combination_info}")
    else:
        print(f"\n📊 Trial {results['trial_num']} Results:")
    print(f"  Test Accuracy:        {results['test_acc']:.4f} ({results['test_acc']*100:.2f}%)")
    print(f"  Final Val Accuracy:   {results['final_val_acc']:.4f} ({results['final_val_acc']*100:.2f}%)")
    print(f"  Max Val Accuracy:     {results['max_val_acc']:.4f} ({results['max_val_acc']*100:.2f}%)")
    print(f"  Final Train Accuracy: {results['final_train_acc']:.4f} ({results['final_train_acc']*100:.2f}%)")
    print(f"  Epoch Stopped:        {results['epoch_stopped']}")
    print(f"  Epoch Max Val:        {results['epoch_max_val']}")
    print(f"  Time:                 {results['total_time']:.1f}s")
    
    # Print class analysis if available
    if 'class_analysis' in results:
        print("\n📊 PER-CLASS ANALYSIS:")
        analysis = results['class_analysis']
        
        print(f"\n  Target Distribution (Ground Truth):")
        for cat, count in analysis['target_distribution'].items():
            pct = count / analysis['total_predictions'] * 100 if analysis['total_predictions'] > 0 else 0
            print(f"    {cat:<25}: {count:>5} ({pct:>5.1f}%)")
        
        print(f"\n  Prediction Distribution (Model Output):")
        for cat, count in analysis['prediction_distribution'].items():
            pct = count / analysis['total_predictions'] * 100 if analysis['total_predictions'] > 0 else 0
            print(f"    {cat:<25}: {count:>5} ({pct:>5.1f}%)")
        
        print(f"\n  ⚠️  Majority Class Analysis:")
        print(f"    Majority class: {analysis['majority_class']}")
        print(f"    Model predicts majority class: {analysis['majority_prediction_ratio']*100:.1f}% of the time")
        
        if analysis['majority_prediction_ratio'] > 0.8:
            print(f"    ⚠️  WARNING: Model may be over-predicting majority class!")
        
        print(f"\n  Per-Class Performance:")
        for cat, metrics in analysis['class_metrics'].items():
            if metrics['num_targets'] > 0:
                print(f"    {cat:<25}: Acc={metrics['accuracy']:.3f}, Prec={metrics['precision']:.3f}, "
                     f"Rec={metrics['recall']:.3f}, F1={metrics['f1']:.3f} "
                     f"(Targets: {metrics['num_targets']}, Preds: {metrics['num_predictions']})")


def generate_research_summary(combination_trials: Dict, args, output_file: str = None):
    """
    Generate a comprehensive research summary report for the research mentor.
    Prints to terminal and optionally writes to file.
    """
    # Build the report content
    report_lines = []
    
    def write_line(text: str):
        """Helper to write to both report_lines and file if provided."""
        report_lines.append(text)
    
    write_line("="*100)
    write_line("MUSICAL SLUR PREDICTION: COMPREHENSIVE MULTI-TRIAL EVALUATION REPORT")
    write_line("="*100)
    write_line("")
    
    write_line("EXPERIMENTAL SETUP")
    write_line("-"*100)
    write_line(f"Chunk Size: {args.chunk_size} notes")
    write_line(f"Chunk Overlap: {args.chunk_overlap} notes")
    write_line(f"Trials per Combination: {args.trials_per_combination}")
    write_line(f"Total Combinations: {len(combination_trials)}")
    write_line(f"Total Trials: {sum(len(combo['trials']) for combo in combination_trials.values())}")
    write_line(f"Learning Rate: {args.lr}")
    write_line(f"Max Epochs: {args.epochs}")
    write_line(f"Early Stopping Patience: {args.patience}")
    write_line("")
    
    write_line("="*100)
    write_line("OVERALL STATISTICS ACROSS ALL COMBINATIONS")
    write_line("="*100)
    write_line("")
    
    # Collect all test accuracies
    all_test_accs = []
    all_val_accs = []
    all_train_accs = []
    
    for combo_num, combo_data in sorted(combination_trials.items()):
        for trial in combo_data['trials']:
            all_test_accs.append(trial['test_acc'])
            all_val_accs.append(trial['val_acc'])
            all_train_accs.append(trial['train_acc'])
    
    write_line(f"Test Accuracy:")
    write_line(f"  Mean: {np.mean(all_test_accs):.4f} ({np.mean(all_test_accs)*100:.2f}%)")
    write_line(f"  Std Dev: {np.std(all_test_accs):.4f} ({np.std(all_test_accs)*100:.2f}%)")
    write_line(f"  Min: {np.min(all_test_accs):.4f} ({np.min(all_test_accs)*100:.2f}%)")
    write_line(f"  Max: {np.max(all_test_accs):.4f} ({np.max(all_test_accs)*100:.2f}%)")
    write_line(f"  Median: {np.median(all_test_accs):.4f} ({np.median(all_test_accs)*100:.2f}%)")
    write_line("")
    
    write_line(f"Validation Accuracy:")
    write_line(f"  Mean: {np.mean(all_val_accs):.4f} ({np.mean(all_val_accs)*100:.2f}%)")
    write_line(f"  Std Dev: {np.std(all_val_accs):.4f} ({np.std(all_val_accs)*100:.2f}%)")
    write_line(f"  Range: {np.min(all_val_accs):.4f} - {np.max(all_val_accs):.4f}")
    write_line("")
    
    write_line(f"Training Accuracy:")
    write_line(f"  Mean: {np.mean(all_train_accs):.4f} ({np.mean(all_train_accs)*100:.2f}%)")
    write_line(f"  Std Dev: {np.std(all_train_accs):.4f} ({np.std(all_train_accs)*100:.2f}%)")
    write_line(f"  Range: {np.min(all_train_accs):.4f} - {np.max(all_train_accs):.4f}")
    write_line("")
    
    write_line("="*100)
    write_line("PER-COMBINATION STATISTICS")
    write_line("="*100)
    write_line("")
    
    for combo_num in sorted(combination_trials.keys()):
        combo_data = combination_trials[combo_num]
        trials = combo_data['trials']
        
        test_accs = [t['test_acc'] for t in trials]
        val_accs = [t['val_acc'] for t in trials]
        train_accs = [t['train_acc'] for t in trials]
        
        write_line(f"Combination {combo_num}:")
        write_line(f"  Train: {combo_data['train_pieces'][0]}, {combo_data['train_pieces'][1]}")
        write_line(f"  Validation: {combo_data['val_piece']}")
        write_line(f"  Test: {combo_data['test_piece']}")
        write_line(f"  Test Accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f} (range: {np.min(test_accs):.4f} - {np.max(test_accs):.4f})")
        write_line(f"  Val Accuracy: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
        write_line(f"  Train Accuracy: {np.mean(train_accs):.4f} ± {np.std(train_accs):.4f}")
        write_line(f"  Individual Trial Results:")
        for i, trial in enumerate(trials, 1):
            write_line(f"    Trial {i}: Test={trial['test_acc']:.4f}, Val={trial['val_acc']:.4f}, Train={trial['train_acc']:.4f}")
        write_line("")
    
    write_line("="*100)
    write_line("KEY FINDINGS")
    write_line("="*100)
    write_line("")
    
    # Find best and worst combinations
    combo_means = {num: np.mean([t['test_acc'] for t in data['trials']]) 
                  for num, data in combination_trials.items()}
    best_combo = max(combo_means.items(), key=lambda x: x[1])
    worst_combo = min(combo_means.items(), key=lambda x: x[1])
    
    write_line(f"1. Best Performing Combination: {best_combo[0]} (Mean Test Acc: {best_combo[1]:.4f})")
    write_line(f"   - Train: {combination_trials[best_combo[0]]['train_pieces']}")
    write_line(f"   - Test: {combination_trials[best_combo[0]]['test_piece']}")
    write_line("")
    
    write_line(f"2. Worst Performing Combination: {worst_combo[0]} (Mean Test Acc: {worst_combo[1]:.4f})")
    write_line(f"   - Train: {combination_trials[worst_combo[0]]['train_pieces']}")
    write_line(f"   - Test: {combination_trials[worst_combo[0]]['test_piece']}")
    write_line("")
    
    # Calculate variance statistics
    combo_vars = {num: np.std([t['test_acc'] for t in data['trials']]) 
                 for num, data in combination_trials.items()}
    high_var_combos = sorted(combo_vars.items(), key=lambda x: x[1], reverse=True)[:3]
    
    write_line(f"3. Highest Variance Combinations (indicating instability):")
    for combo_num, var in high_var_combos:
        write_line(f"   - Combination {combo_num}: Std Dev = {var:.4f}")
    write_line("")
    
    # Class analysis summary
    write_line("4. Class Prediction Analysis:")
    write_line("   (Based on final trial of each combination)")
    for combo_num in sorted(combination_trials.keys()):
        last_trial = combination_trials[combo_num]['trials'][-1]
        if last_trial.get('class_analysis'):
            analysis = last_trial['class_analysis']
            write_line(f"   Combination {combo_num}:")
            write_line(f"     Majority class prediction ratio: {analysis['majority_prediction_ratio']:.2%}")
            if analysis['majority_prediction_ratio'] > 0.8:
                write_line(f"     ⚠️  WARNING: Model over-predicting majority class")
    write_line("")
    
    write_line("="*100)
    write_line("CONCLUSIONS")
    write_line("="*100)
    write_line("")
    write_line(f"1. Overall Performance: Mean test accuracy of {np.mean(all_test_accs):.4f} ({np.mean(all_test_accs)*100:.2f}%) with high variance ({np.std(all_test_accs):.4f}).")
    write_line(f"2. Model Stability: High variance across trials indicates sensitivity to initialization.")
    write_line(f"3. Generalization: Performance varies significantly based on train/test piece combinations.")
    write_line(f"4. Best Case: Achieved {np.max(all_test_accs):.4f} ({np.max(all_test_accs)*100:.2f}%) test accuracy.")
    write_line(f"5. Worst Case: Model collapsed to single class prediction in some trials ({np.min(all_test_accs):.4f} = {np.min(all_test_accs)*100:.2f}%).")
    write_line("")
    
    write_line("="*100)
    write_line("END OF REPORT")
    write_line("="*100)
    
    # Print to terminal
    print("\n" + "\n".join(report_lines))
    
    # Optionally write to file
    if output_file:
        with open(output_file, 'w') as f:
            f.write("\n".join(report_lines))


def print_summary(all_results: List[Dict]):
    """Print summary statistics across all trials."""
    print(f"\n{'='*60}")
    print("DETAILED TRIAL RESULTS")
    print(f"{'='*60}")
    print(f"{'Trial':<6} {'Test Acc':<10} {'Val Acc':<10} {'Max Val':<10} {'Train Acc':<11} {'Epoch':<7} {'Max Epoch':<10} {'Time':<8}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['trial_num']:<6} {r['test_acc']:<10.4f} {r['final_val_acc']:<10.4f} "
              f"{r['max_val_acc']:<10.4f} {r['final_train_acc']:<11.4f} "
              f"{r['epoch_stopped']:<7} {r['epoch_max_val']:<10} {r['total_time']:<8.1f}s")
    
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    # Extract metrics
    test_accs = [r['test_acc'] for r in all_results]
    final_val_accs = [r['final_val_acc'] for r in all_results]
    max_val_accs = [r['max_val_acc'] for r in all_results]
    final_train_accs = [r['final_train_acc'] for r in all_results]
    epochs_stopped = [r['epoch_stopped'] for r in all_results]
    epochs_max_val = [r['epoch_max_val'] for r in all_results]
    times = [r['total_time'] for r in all_results]
    
    def calc_stats(values):
        mean = np.mean(values)
        std = np.std(values)
        return mean, std
    
    test_mean, test_std = calc_stats(test_accs)
    val_mean, val_std = calc_stats(final_val_accs)
    max_val_mean, max_val_std = calc_stats(max_val_accs)
    train_mean, train_std = calc_stats(final_train_accs)
    epoch_mean, epoch_std = calc_stats(epochs_stopped)
    epoch_max_mean, epoch_max_std = calc_stats(epochs_max_val)
    time_mean, time_std = calc_stats(times)
    
    print(f"\nTest Accuracy:")
    print(f"  Mean: {test_mean:.4f} ± {test_std:.4f}")
    print(f"  Range: {min(test_accs):.4f} - {max(test_accs):.4f}")
    
    print(f"\nFinal Validation Accuracy:")
    print(f"  Mean: {val_mean:.4f} ± {val_std:.4f}")
    print(f"  Range: {min(final_val_accs):.4f} - {max(final_val_accs):.4f}")
    
    print(f"\nMax Validation Accuracy:")
    print(f"  Mean: {max_val_mean:.4f} ± {max_val_std:.4f}")
    print(f"  Range: {min(max_val_accs):.4f} - {max(max_val_accs):.4f}")
    
    print(f"\nFinal Training Accuracy:")
    print(f"  Mean: {train_mean:.4f} ± {train_std:.4f}")
    print(f"  Range: {min(final_train_accs):.4f} - {max(final_train_accs):.4f}")
    
    print(f"\nEpochs:")
    print(f"  Stopped - Mean: {epoch_mean:.1f} ± {epoch_std:.1f}")
    print(f"  Max Val - Mean: {epoch_max_mean:.1f} ± {epoch_max_std:.1f}")
    
    print(f"\nTime per trial:")
    print(f"  Mean: {time_mean:.1f}s ± {time_std:.1f}s")
    print(f"  Total: {sum(times):.1f}s")


def main():
    """Main function to run multi-trial training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-trial chunked training')
    parser.add_argument('--chunk-size', type=int, default=264,
                       help='Number of notes per chunk')
    parser.add_argument('--chunk-overlap', type=int, default=0,
                       help='Number of notes to overlap between chunks')
    parser.add_argument('--num-trials', type=int, default=10,
                       help='Number of trials to run')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Maximum number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience')
    parser.add_argument('--train-ratio', type=float, default=0.6,
                       help='Fraction of data for training')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                       help='Fraction of data for validation')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                       help='Fraction of data for testing')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory containing annotation files')
    parser.add_argument('--combination', type=int, default=None,
                       help='Run only a specific combination (1-12). If not specified, runs all combinations.')
    parser.add_argument('--repeat-combination', type=int, default=1,
                       help='Number of times to repeat the specified combination (for testing variance)')
    parser.add_argument('--trials-per-combination', type=int, default=1,
                       help='Number of trials to run for each combination (default: 1, use 5 for comprehensive analysis)')
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, val, and test ratios must sum to 1.0")
    
    print("🎵 MULTI-TRIAL CHUNKED TRAINING")
    print("=" * 60)
    print(f"Chunk size: {args.chunk_size} notes")
    if args.chunk_overlap > 0:
        print(f"Chunk overlap: {args.chunk_overlap} notes")
    print(f"Splitting strategy: All combinations (2 train, 1 val, 1 test)")
    print(f"Expected combinations: 12 (one trial each)")
    print("=" * 60)
    
    # Find all annotated pieces
    print("\n📂 Finding annotated pieces...")
    pieces = find_annotated_pieces(args.output_dir)
    
    if len(pieces) == 0:
        print("❌ No annotated pieces found!")
        return
    
    print(f"✓ Found {len(pieces)} annotated piece(s)")
    
    # Load and chunk all pieces
    print("\n📊 Loading and chunking all pieces...")
    all_chunks = load_and_chunk_all_pieces(pieces, args.chunk_size, args.chunk_overlap)
    
    if len(all_chunks) == 0:
        print("❌ No chunks created!")
        return
    
    # Training configuration
    config = {
        'chunk_size': args.chunk_size,
        'chunk_overlap': args.chunk_overlap,
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'test_ratio': args.test_ratio,
        'learning_rate': args.lr,
        'epochs': args.epochs,
        'early_stopping': True,
        'patience': args.patience,
        'ignore_boundary_notes': 0,
        'shuffle_chunks': True,
        'stratified_splitting': False,  # We do random splitting instead
        'num_trials': args.num_trials
    }
    
    # Group chunks by piece
    print("\n📊 Grouping chunks by piece...")
    chunks_by_piece = group_chunks_by_piece(all_chunks)
    
    print(f"\n  📊 Pieces found ({len(chunks_by_piece)}):")
    for piece_id, chunks in chunks_by_piece.items():
        print(f"    {piece_id}: {len(chunks)} chunks")
    
    # Generate all combinations
    print("\n📊 Generating all combinations...")
    combinations_list = generate_all_combinations(chunks_by_piece)
    print(f"  ✓ Generated {len(combinations_list)} combinations (2 train, 1 val, 1 test)")
    
    # Determine how many trials per combination
    trials_per_combo = args.trials_per_combination
    
    # Filter to specific combination if requested
    original_combination_num = None
    repeat_count = 1
    if args.combination is not None:
        if args.combination < 1 or args.combination > len(combinations_list):
            raise ValueError(f"Combination must be between 1 and {len(combinations_list)}, got {args.combination}")
        original_combination_num = args.combination
        repeat_count = args.repeat_combination
        combinations_list = [combinations_list[args.combination - 1]] * repeat_count
        print(f"  → Running combination {args.combination} {repeat_count} time(s)")
        trials_per_combo = args.repeat_combination
    else:
        print(f"  → Running {trials_per_combo} trial(s) per combination")
    
    # If running all combinations with multiple trials, expand the list
    original_combinations_list = combinations_list.copy()
    if args.combination is None and trials_per_combo > 1:
        expanded_combinations = []
        for combo in original_combinations_list:
            expanded_combinations.extend([combo] * trials_per_combo)
        combinations_list = expanded_combinations
    
    # Update config for number of trials
    config['num_trials'] = len(combinations_list)
    
    # Run trials for each combination
    all_results = []
    combination_results = []
    combination_trials = {}  # Track trials per combination
    
    for combo_idx, (train1, train2, val_piece, test_piece) in enumerate(combinations_list, 1):
        # Determine combination number and trial number
        if original_combination_num:
            # Running a specific combination multiple times
            display_combo_num = original_combination_num
            trial_num = combo_idx
            trial_within_combo = combo_idx
        else:
            # Running all combinations
            if trials_per_combo > 1:
                # Calculate which combination this is (0-indexed)
                combo_base_idx = (combo_idx - 1) // trials_per_combo
                display_combo_num = combo_base_idx + 1
                trial_within_combo = ((combo_idx - 1) % trials_per_combo) + 1
            else:
                display_combo_num = combo_idx
                trial_within_combo = 1
            trial_num = combo_idx
        
        # Create split for this combination
        splits = create_split_from_combination(chunks_by_piece, train1, train2, val_piece, test_piece)
        
        # Create combination info string
        if trials_per_combo > 1 and args.combination is None:
            combo_info = f"COMBINATION {display_combo_num}/12 - TRIAL {trial_within_combo}/{trials_per_combo}: Train=[{os.path.basename(train1)}, {os.path.basename(train2)}], Val=[{os.path.basename(val_piece)}], Test=[{os.path.basename(test_piece)}]"
        elif repeat_count > 1:
            combo_info = f"COMBINATION {display_combo_num}/12 - TRIAL {trial_num}/{repeat_count}: Train=[{os.path.basename(train1)}, {os.path.basename(train2)}], Val=[{os.path.basename(val_piece)}], Test=[{os.path.basename(test_piece)}]"
        else:
            combo_info = f"COMBINATION {display_combo_num}/12: Train=[{os.path.basename(train1)}, {os.path.basename(train2)}], Val=[{os.path.basename(val_piece)}], Test=[{os.path.basename(test_piece)}]"
        print(f"\n{'='*80}")
        print(combo_info)
        print(f"{'='*80}")
        
        # Run training
        results = run_single_trial(splits, config, trial_num)
        results['combination'] = {
            'train_pieces': [train1, train2],
            'val_piece': val_piece,
            'test_piece': test_piece,
            'combination_num': display_combo_num,
            'trial_within_combo': trial_within_combo if trials_per_combo > 1 else 1
        }
        all_results.append(results)
        
        # Track trials per combination
        if display_combo_num not in combination_trials:
            combination_trials[display_combo_num] = {
                'train_pieces': [os.path.basename(train1), os.path.basename(train2)],
                'val_piece': os.path.basename(val_piece),
                'test_piece': os.path.basename(test_piece),
                'trials': []
            }
        
        combination_trials[display_combo_num]['trials'].append({
            'trial_num': trial_within_combo if trials_per_combo > 1 else 1,
            'test_acc': results['test_acc'],
            'val_acc': results['final_val_acc'],
            'max_val_acc': results['max_val_acc'],
            'train_acc': results['final_train_acc'],
            'epoch_stopped': results['epoch_stopped'],
            'epoch_max_val': results['epoch_max_val'],
            'time': results['total_time'],
            'class_analysis': results.get('class_analysis', None)
        })
        
        # Also add to combination_results for backward compatibility
        combination_results.append({
            'combination': display_combo_num,
            'trial': trial_within_combo if trials_per_combo > 1 else 1,
            'train_pieces': [os.path.basename(train1), os.path.basename(train2)],
            'val_piece': os.path.basename(val_piece),
            'test_piece': os.path.basename(test_piece),
            'test_acc': results['test_acc'],
            'val_acc': results['final_val_acc'],
            'max_val_acc': results['max_val_acc'],
            'train_acc': results['final_train_acc'],
            'epoch_stopped': results['epoch_stopped'],
            'epoch_max_val': results['epoch_max_val'],
            'time': results['total_time']
        })
        
        # Print trial results
        print_trial_results(results, combo_info)
    
    # Print detailed combination results
    print("\n" + "="*80)
    print("DETAILED COMBINATION RESULTS")
    print("="*80)
    if trials_per_combo > 1:
        print(f"{'Combo':<6} {'Trial':<6} {'Train Pieces':<50} {'Val':<30} {'Test':<30} {'Test Acc':<10} {'Val Acc':<10} {'Train Acc':<10}")
        print("-"*80)
        for combo in combination_results:
            train_str = f"{combo['train_pieces'][0]}, {combo['train_pieces'][1]}"
            print(f"{combo['combination']:<6} {combo.get('trial', 1):<6} {train_str:<50} {combo['val_piece']:<30} {combo['test_piece']:<30} {combo['test_acc']:.4f}     {combo['val_acc']:.4f}     {combo['train_acc']:.4f}")
    else:
        print(f"{'Combo':<6} {'Train Pieces':<50} {'Val':<30} {'Test':<30} {'Test Acc':<10} {'Val Acc':<10} {'Train Acc':<10}")
        print("-"*80)
        for combo in combination_results:
            train_str = f"{combo['train_pieces'][0]}, {combo['train_pieces'][1]}"
            print(f"{combo['combination']:<6} {train_str:<50} {combo['val_piece']:<30} {combo['test_piece']:<30} {combo['test_acc']:.4f}     {combo['val_acc']:.4f}     {combo['train_acc']:.4f}")
    
    # Build detailed results content
    results_lines = []
    results_lines.append("="*80)
    results_lines.append("DETAILED COMBINATION RESULTS")
    results_lines.append("="*80)
    results_lines.append(f"Chunk size: {args.chunk_size}, Overlap: {args.chunk_overlap}")
    results_lines.append(f"Total combinations: {len(combination_results)}")
    results_lines.append("="*80)
    results_lines.append("")
    results_lines.append(f"{'Combo':<6} {'Train Pieces':<50} {'Val':<30} {'Test':<30} {'Test Acc':<10} {'Val Acc':<10} {'Train Acc':<10} {'Max Val Acc':<12} {'Epochs':<8}")
    results_lines.append("-"*80)
    for combo in combination_results:
        train_str = f"{combo['train_pieces'][0]}, {combo['train_pieces'][1]}"
        results_lines.append(f"{combo['combination']:<6} {train_str:<50} {combo['val_piece']:<30} {combo['test_piece']:<30} "
                   f"{combo['test_acc']:.4f}     {combo['val_acc']:.4f}     {combo['train_acc']:.4f}     "
                   f"{combo['max_val_acc']:.4f}        {combo['epoch_stopped']:<8}")
    
    # Add summary statistics
    results_lines.append("")
    results_lines.append("="*80)
    results_lines.append("SUMMARY STATISTICS")
    results_lines.append("="*80)
    results_lines.append("")
    
    test_accs = [c['test_acc'] for c in combination_results]
    val_accs = [c['val_acc'] for c in combination_results]
    max_val_accs = [c['max_val_acc'] for c in combination_results]
    train_accs = [c['train_acc'] for c in combination_results]
    
    results_lines.append(f"Test Accuracy:")
    results_lines.append(f"  Mean: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
    results_lines.append(f"  Range: {min(test_accs):.4f} - {max(test_accs):.4f}")
    results_lines.append("")
    
    results_lines.append(f"Validation Accuracy:")
    results_lines.append(f"  Mean: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
    results_lines.append(f"  Range: {min(val_accs):.4f} - {max(val_accs):.4f}")
    results_lines.append("")
    
    results_lines.append(f"Max Validation Accuracy:")
    results_lines.append(f"  Mean: {np.mean(max_val_accs):.4f} ± {np.std(max_val_accs):.4f}")
    results_lines.append(f"  Range: {min(max_val_accs):.4f} - {max(max_val_accs):.4f}")
    results_lines.append("")
    
    results_lines.append(f"Training Accuracy:")
    results_lines.append(f"  Mean: {np.mean(train_accs):.4f} ± {np.std(train_accs):.4f}")
    results_lines.append(f"  Range: {min(train_accs):.4f} - {max(train_accs):.4f}")
    
    # Print to terminal
    print("\n" + "\n".join(results_lines))
    
    # Print summary statistics
    print_summary(all_results)
    
    # Generate comprehensive research summary if multiple trials per combination
    if trials_per_combo > 1 and args.combination is None:
        # Print to terminal (output_file=None means don't write to file)
        generate_research_summary(combination_trials, args, output_file=None)
    
    print("\n✅ All combination trials complete!")


if __name__ == "__main__":
    main()

