"""
Chunked Data Pipeline for Musical Slur Prediction

This module provides chunked data processing capabilities while preserving
the original non-chunked approach. It splits musical sequences into chunks
for training/validation/testing with configurable parameters.

Author: AI Assistant
Date: December 2025
"""

import torch
import numpy as np
import pandas as pd
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os
import sys

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_data_pipeline import load_and_prepare_data


def chunk_sequence(data: Dict, chunk_size: int, overlap: int = 0) -> List[Dict]:
    """
    Split a sequence into chunks of specified size with optional overlap.
    Omits the last chunk if it would be smaller than the target chunk size.
    
    Args:
        data: Dictionary with 'features' and 'targets' tensors
        chunk_size: Number of notes per chunk
        overlap: Number of notes to overlap between consecutive chunks (default: 0)
    
    Returns:
        List of chunk dictionaries with features, targets, and metadata
    """
    if overlap >= chunk_size:
        raise ValueError(f"Overlap ({overlap}) must be less than chunk_size ({chunk_size})")
    
    total_notes = data['features'].shape[0]
    
    chunks = []
    start_idx = 0
    chunk_id = 0
    step_size = chunk_size - overlap  # How much to advance for next chunk
    
    while start_idx < total_notes:
        # Calculate end index
        end_idx = start_idx + chunk_size
        
        # Skip the last chunk if it would be smaller than target size
        if end_idx > total_notes:
            break
        
        current_chunk_size = end_idx - start_idx
        
        chunk = {
            'features': data['features'][start_idx:end_idx],
            'targets': data['targets'][start_idx:end_idx],
            'chunk_id': chunk_id,
            'piece_id': data.get('piece_id', 'unknown'),
            'chunk_size': current_chunk_size,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'overlap': overlap if chunk_id > 0 else 0  # First chunk has no overlap
        }
        chunks.append(chunk)
        
        # Advance by step_size (chunk_size - overlap) for next chunk
        start_idx += step_size
        chunk_id += 1
    
    return chunks


def split_chunks(chunks: List[Dict], train_ratio: float = 0.6, 
                val_ratio: float = 0.2, test_ratio: float = 0.2,
                shuffle: bool = True, stratified: bool = True) -> Dict[str, List[Dict]]:
    """
    Split chunks into train/validation/test sets with optional stratification.
    
    Args:
        chunks: List of chunk dictionaries
        train_ratio: Fraction for training
        val_ratio: Fraction for validation  
        test_ratio: Fraction for testing
        shuffle: Whether to shuffle chunks before splitting
        stratified: Whether to use stratified splitting based on class distribution
    
    Returns:
        Dictionary with 'train', 'val', 'test' keys containing chunk lists
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    if stratified:
        return _stratified_split_chunks(chunks, train_ratio, val_ratio, test_ratio, shuffle)
    else:
        return _simple_split_chunks(chunks, train_ratio, val_ratio, test_ratio, shuffle)


def _simple_split_chunks(chunks: List[Dict], train_ratio: float, 
                        val_ratio: float, test_ratio: float, shuffle: bool) -> Dict[str, List[Dict]]:
    """Simple sequential splitting of chunks."""
    num_chunks = len(chunks)
    num_train = int(num_chunks * train_ratio)
    num_val = int(num_chunks * val_ratio)
    
    # Shuffle chunks for random splitting
    if shuffle:
        shuffled_chunks = chunks.copy()
        random.shuffle(shuffled_chunks)
    else:
        shuffled_chunks = chunks
    
    return {
        'train': shuffled_chunks[:num_train],
        'val': shuffled_chunks[num_train:num_train + num_val],
        'test': shuffled_chunks[num_train + num_val:]
    }


def _stratified_split_chunks(chunks: List[Dict], train_ratio: float, 
                           val_ratio: float, test_ratio: float, shuffle: bool) -> Dict[str, List[Dict]]:
    """
    Stratified splitting to ensure balanced class distributions across splits.
    
    This method analyzes the class distribution in each chunk and tries to balance
    the overall class distribution across train/val/test splits.
    """
    # Calculate class distribution for each chunk
    chunk_stats = []
    for i, chunk in enumerate(chunks):
        targets = chunk['targets']
        # Get class distribution (assuming one-hot encoded targets)
        class_counts = torch.sum(targets, dim=0)
        total_notes = targets.shape[0]
        class_proportions = class_counts.float() / total_notes
        
        chunk_stats.append({
            'chunk_idx': i,
            'class_proportions': class_proportions,
            'total_notes': total_notes,
            'dominant_class': torch.argmax(class_proportions).item()
        })
    
    # Sort chunks by dominant class to help with balanced distribution
    chunk_stats.sort(key=lambda x: x['dominant_class'])
    
    # Add randomness to the assignment while maintaining stratification
    if shuffle:
        # Shuffle within each dominant class group to add randomness
        import random
        grouped_stats = {}
        for stat in chunk_stats:
            dominant_class = stat['dominant_class']
            if dominant_class not in grouped_stats:
                grouped_stats[dominant_class] = []
            grouped_stats[dominant_class].append(stat)
        
        # Shuffle each group and flatten
        shuffled_stats = []
        for dominant_class in sorted(grouped_stats.keys()):
            random.shuffle(grouped_stats[dominant_class])
            shuffled_stats.extend(grouped_stats[dominant_class])
        chunk_stats = shuffled_stats
    
    # Simple stratified approach: distribute chunks to maintain balance
    splits = {'train': [], 'val': [], 'test': []}
    
    # Round-robin assignment to splits
    for i, chunk_stat in enumerate(chunk_stats):
        chunk_idx = chunk_stat['chunk_idx']
        chunk = chunks[chunk_idx]
        
        if i % 3 == 0:
            splits['train'].append(chunk)
        elif i % 3 == 1:
            splits['val'].append(chunk)
        else:
            splits['test'].append(chunk)
    
    # Adjust sizes to match desired ratios
    total_chunks = len(chunks)
    target_train = int(total_chunks * train_ratio)
    target_val = int(total_chunks * val_ratio)
    target_test = total_chunks - target_train - target_val
    
    # Rebalance if needed
    current_train = len(splits['train'])
    current_val = len(splits['val'])
    current_test = len(splits['test'])
    
    # Move chunks between splits to match target sizes
    while current_train < target_train and current_val > target_val:
        splits['train'].append(splits['val'].pop())
        current_train += 1
        current_val -= 1
    
    while current_train < target_train and current_test > target_test:
        splits['train'].append(splits['test'].pop())
        current_train += 1
        current_test -= 1
    
    while current_val < target_val and current_test > target_test:
        splits['val'].append(splits['test'].pop())
        current_val += 1
        current_test -= 1
    
    # Shuffle within each split if requested
    if shuffle:
        for split_name in splits:
            random.shuffle(splits[split_name])
    
    return splits


def calculate_chunk_loss(outputs: torch.Tensor, targets: torch.Tensor, 
                        criterion, ignore_boundary: int = 0) -> torch.Tensor:
    """
    Calculate loss while optionally ignoring boundary notes.
    
    Args:
        outputs: Model predictions (batch, seq_len, num_classes) or (seq_len, num_classes)
        targets: Target class indices (batch, seq_len) or (seq_len,)
        criterion: Loss function (CrossEntropyLoss)
        ignore_boundary: Number of notes to ignore at each end
    
    Returns:
        Loss value
    """
    # Reshape for CrossEntropyLoss: (batch*seq_len, num_classes) and (batch*seq_len,)
    if outputs.dim() == 3:
        batch_size, seq_len, num_classes = outputs.shape
        outputs_flat = outputs.view(-1, num_classes)
        targets_flat = targets.view(-1).long()
    else:
        seq_len, num_classes = outputs.shape
        outputs_flat = outputs.view(-1, num_classes)
        targets_flat = targets.view(-1).long()
    
    if ignore_boundary == 0:
        # Use all notes
        return criterion(outputs_flat, targets_flat)
    
    # Ignore first and last 'ignore_boundary' notes
    if outputs.dim() == 3:
        batch_size, seq_len, num_classes = outputs.shape
        if seq_len <= 2 * ignore_boundary:
            return criterion(outputs_flat, targets_flat)
        start_idx = ignore_boundary
        end_idx = seq_len - ignore_boundary
        # Extract middle portion and flatten
        middle_outputs = outputs[:, start_idx:end_idx, :].view(-1, num_classes)
        middle_targets = targets[:, start_idx:end_idx].view(-1).long()
    else:
        seq_len, num_classes = outputs.shape
        if seq_len <= 2 * ignore_boundary:
            return criterion(outputs_flat, targets_flat)
        start_idx = ignore_boundary
        end_idx = seq_len - ignore_boundary
        # Extract middle portion and flatten
        middle_outputs = outputs[start_idx:end_idx, :].view(-1, num_classes)
        middle_targets = targets[start_idx:end_idx].view(-1).long()
    
    return criterion(middle_outputs, middle_targets)


def prepare_chunked_data(midi_file: str, annotation_file: str, 
                        chunk_size: int = 100, overlap: int = 0,
                        train_ratio: float = 0.6, val_ratio: float = 0.2, 
                        test_ratio: float = 0.2, shuffle: bool = True, 
                        stratified: bool = True) -> Dict[str, List[Dict]]:
    """
    Load data and split into chunks for training/validation/testing.
    
    Args:
        midi_file: Path to MIDI file
        annotation_file: Path to annotation CSV (not used, kept for compatibility)
        chunk_size: Number of notes per chunk
        overlap: Number of notes to overlap between consecutive chunks (default: 0)
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        shuffle: Whether to shuffle chunks before splitting
        stratified: Whether to use stratified splitting
    
    Returns:
        Dictionary with train/val/test chunk splits
    """
    print(f"📊 Loading data for chunked training...")
    
    # Extract base filename from MIDI file
    base_filename = os.path.splitext(os.path.basename(midi_file))[0]
    output_dir = "output"
    
    # Load original data using existing pipeline (same as original system)
    inputs, targets, norm_params, stats = load_and_prepare_data(base_filename, output_dir)
    
    # Convert to dictionary format for chunking
    raw_data = {
        'features': inputs,
        'targets': targets,
        'piece_id': base_filename
    }
    
    total_notes = raw_data['features'].shape[0]
    print(f"✓ Data loaded: {total_notes} notes")
    
    # Split into chunks
    if overlap > 0:
        print(f"🔪 Splitting into chunks of size {chunk_size} notes with {overlap} note overlap...")
    else:
        print(f"🔪 Splitting into chunks of size {chunk_size} notes...")
    chunks = chunk_sequence(raw_data, chunk_size, overlap)
    
    # Calculate chunk size statistics
    chunk_sizes = [chunk['chunk_size'] for chunk in chunks]
    total_chunked_notes = sum(chunk_sizes)
    omitted_notes = total_notes - total_chunked_notes
    
    if len(chunks) > 0:
        min_size = min(chunk_sizes)
        max_size = max(chunk_sizes)
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        
        print(f"✓ Chunks created: {len(chunks)} chunks")
        print(f"  - Size range: {min_size}-{max_size} notes")
        print(f"  - Average size: {avg_size:.1f} notes")
        if overlap > 0:
            step_size = chunk_size - overlap
            print(f"  - Overlap: {overlap} notes between consecutive chunks (step size: {step_size} notes)")
            print(f"  - Data augmentation: {len(chunks)} chunks from {total_notes} notes ({len(chunks) * chunk_size / total_notes:.1f}x more data)")
        if omitted_notes > 0:
            print(f"  - Note: Omitted {omitted_notes} leftover notes at the end ({omitted_notes/total_notes*100:.1f}% of total)")
    else:
        print(f"⚠️  Warning: No chunks created (piece too short for chunk size {chunk_size})")
    
    # Split into train/val/test
    print(f"📊 Splitting chunks: {train_ratio:.1%} train, {val_ratio:.1%} val, {test_ratio:.1%} test")
    if stratified:
        print(f"  Using stratified splitting for balanced class distributions")
    splits = split_chunks(chunks, train_ratio, val_ratio, test_ratio, shuffle, stratified)
    
    # Print split statistics
    for split_name, split_chunk_list in splits.items():
        total_notes = sum(chunk['chunk_size'] for chunk in split_chunk_list)
        print(f"  - {split_name}: {len(split_chunk_list)} chunks, {total_notes} notes")
    
    return splits


def get_chunk_statistics(chunks: List[Dict]) -> Dict:
    """
    Calculate statistics for a list of chunks.
    
    Args:
        chunks: List of chunk dictionaries
    
    Returns:
        Dictionary with chunk statistics
    """
    if not chunks:
        return {}
    
    chunk_sizes = [chunk['chunk_size'] for chunk in chunks]
    total_notes = sum(chunk_sizes)
    
    # Calculate annotation distribution across all chunks
    all_targets = torch.cat([chunk['targets'] for chunk in chunks], dim=0)
    annotation_counts = all_targets.sum(dim=0)
    
    return {
        'num_chunks': len(chunks),
        'total_notes': total_notes,
        'min_chunk_size': min(chunk_sizes),
        'max_chunk_size': max(chunk_sizes),
        'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
        'annotation_counts': annotation_counts.tolist(),
        'annotation_percentages': (annotation_counts / total_notes * 100).tolist()
    }


def print_chunked_data_summary(splits: Dict[str, List[Dict]]):
    """
    Print a summary of the chunked data splits.
    
    Args:
        splits: Dictionary with train/val/test chunk splits
    """
    print("\n📊 CHUNKED DATA SUMMARY")
    print("=" * 50)
    
    for split_name, chunks in splits.items():
        stats = get_chunk_statistics(chunks)
        if stats:
            print(f"\n{split_name.upper()} SET:")
            print(f"  Chunks: {stats['num_chunks']}")
            print(f"  Total notes: {stats['total_notes']}")
            print(f"  Chunk size: {stats['min_chunk_size']}-{stats['max_chunk_size']} "
                  f"(avg: {stats['avg_chunk_size']:.1f})")
            
            print(f"  Annotation distribution:")
            categories = ['slur_start', 'slur_middle', 'slur_end', 'no_slur', 'slur_start_and_end']
            for i, (count, pct) in enumerate(zip(stats['annotation_counts'], 
                                                stats['annotation_percentages'])):
                print(f"    {categories[i]}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    """
    Test the chunked data pipeline
    """
    print("🧪 Testing chunked data pipeline...")
    
    # Test with sample data
    test_data = {
        'features': torch.randn(100, 5),
        'targets': torch.randint(0, 2, (100, 4)).float(),
        'piece_id': 'test_piece'
    }
    
    # Test chunking (10 notes per chunk)
    chunks = chunk_sequence(test_data, chunk_size=10)
    print(f"✓ Created {len(chunks)} chunks from 100 notes (10 notes per chunk)")
    
    # Test splitting
    splits = split_chunks(chunks, 0.6, 0.2, 0.2)
    print(f"✓ Split into {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")
    
    # Test statistics
    stats = get_chunk_statistics(chunks)
    print(f"✓ Statistics calculated: {stats['num_chunks']} chunks, {stats['total_notes']} total notes")
    
    print("✅ Chunked pipeline test completed!")
