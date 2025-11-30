#!/usr/bin/env python3
"""
ML Data Pipeline for MIDI Piano Roll Transformer
Converts MIDI annotation data to transformer-ready format

Part of the MIDI Piano Roll ML System v2.0
"""

import numpy as np
import pandas as pd
import torch
import os
from pathlib import Path

def midi_name_to_number(pitch_name):
    """Convert pitch name to MIDI number (e.g., 'C4' -> 60)"""
    if pd.isna(pitch_name) or pitch_name == '':
        return 0
        
    # Parse note name and octave
    if '#' in pitch_name:
        note = pitch_name[:2]
        octave = int(pitch_name[2:])
    else:
        note = pitch_name[0]
        octave = int(pitch_name[1:])
    
    # Note to semitone mapping
    note_map = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 
                'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
    
    midi_number = note_map[note] + (octave + 1) * 12
    return midi_number

def extract_pedal_events(pedal_data, time_step=0.00208333):
    """
    Extract pedal on/off events from pedal matrix
    
    Args:
        pedal_data (np.ndarray): Pedal matrix (3, time_steps) with sustain in row 0
        time_step (float): Time per tick in seconds
        
    Returns:
        list: List of pedal events [(time, event_type), ...] where event_type is 'on' or 'off'
    """
    pedal_events = []
    sustain_row = pedal_data[0, :]  # Row 0 = sustain
    
    # Find transitions from 0 to 127 (pedal on) and 127 to 0 (pedal off)
    for i in range(1, len(sustain_row)):
        if sustain_row[i-1] == 0 and sustain_row[i] == 127:
            pedal_events.append((i * time_step, 'on'))
        elif sustain_row[i-1] == 127 and sustain_row[i] == 0:
            pedal_events.append((i * time_step, 'off'))
    
    return pedal_events

def get_pedal_state_simple(note_start_time, note_end_time, pedal_events, note_index, pedal_start_indices, pedal_end_indices):
    """
    Determine pedal state for a note using simple first-note-after-pedal approach
    
    Args:
        note_start_time (float): Note start time in seconds
        note_end_time (float): Note end time in seconds
        pedal_events (list): List of (time, event_type) tuples
        note_index (int): Index of current note in sequence
        pedal_start_indices (set): Set of note indices that are first notes after pedal on
        pedal_end_indices (set): Set of note indices that are last notes before pedal off
        
    Returns:
        int: Pedal state (0=no_pedal, 1=beginning, 2=middle, 3=end)
    """
    # Check if this note is the first note after a pedal on event
    if note_index in pedal_start_indices:
        return 1  # Beginning of pedal press
    
    # Check if this note is the last note before a pedal off event
    if note_index in pedal_end_indices:
        return 3  # End of pedal press
    
    # Find the current pedal state (on/off) at note start
    current_pedal_on = False
    for time, event_type in pedal_events:
        if time <= note_start_time:
            current_pedal_on = (event_type == 'on')
        else:
            break
    
    if not current_pedal_on:
        return 0  # No pedal
    
    # If pedal is on but not at beginning or end
    return 2  # Middle of pedal press

def get_pedal_state(note_start_time, note_end_time, pedal_events, window_ms=100):
    """
    Determine pedal state for a note based on pedal events (WINDOW-BASED APPROACH)
    
    Args:
        note_start_time (float): Note start time in seconds
        note_end_time (float): Note end time in seconds
        pedal_events (list): List of (time, event_type) tuples
        window_ms (float): Temporal window in milliseconds for pedal start/end detection
        
    Returns:
        int: Pedal state (0=no_pedal, 1=beginning, 2=middle, 3=end)
    """
    window_sec = window_ms / 1000.0
    
    # Find the current pedal state (on/off) at note start
    current_pedal_on = False
    for time, event_type in pedal_events:
        if time <= note_start_time:
            current_pedal_on = (event_type == 'on')
        else:
            break
    
    if not current_pedal_on:
        return 0  # No pedal
    
    # Check if note is at beginning of pedal press
    for time, event_type in pedal_events:
        if event_type == 'on' and abs(time - note_start_time) <= window_sec:
            return 1  # Beginning of pedal press
    
    # Check if note is at end of pedal press (last note before pedal off)
    for time, event_type in pedal_events:
        if event_type == 'off' and abs(time - note_end_time) <= window_sec:
            return 3  # End of pedal press
    
    # If pedal is on but not at beginning or end
    return 2  # Middle of pedal press

def create_model_input_simple_pedal(notes_df, pedal_data):
    """
    Create model input features using simple first-note-after-pedal approach
    
    Args:
        notes_df (pd.DataFrame): Annotated notes with columns [Start_Time, Duration, Pitch, Velocity, Start_Ticks, Slur_Category]
        pedal_data (np.ndarray): Pedal matrix (3, time_steps) with sustain in row 0
        
    Returns:
        np.ndarray: Model inputs (num_notes, 5) [start_time, duration, midi_pitch, velocity, pedal_state]
    """
    model_inputs = []
    
    print(f"Processing {len(notes_df)} notes with simple pedal approach...")
    
    # Extract pedal events
    pedal_events = extract_pedal_events(pedal_data)
    print(f"Found {len(pedal_events)} pedal events")
    
    # Find pedal start and end note indices
    pedal_start_indices = set()
    pedal_end_indices = set()
    
    # Get all note start times for comparison
    note_times = notes_df['Start_Time'].values
    
    for pedal_time, event_type in pedal_events:
        if event_type == 'on':
            # Find the first note after this pedal on event
            for i, note_time in enumerate(note_times):
                if note_time > pedal_time:
                    pedal_start_indices.add(i)
                    break
        elif event_type == 'off':
            # Find the last note before this pedal off event
            for i in range(len(note_times) - 1, -1, -1):
                if note_times[i] < pedal_time:
                    pedal_end_indices.add(i)
                    break
    
    print(f"Identified {len(pedal_start_indices)} pedal start notes")
    print(f"Identified {len(pedal_end_indices)} pedal end notes")
    
    for idx, row in notes_df.iterrows():
        # Extract basic note features
        start_time = float(row['Start_Time'])
        duration = float(row['Duration'])
        end_time = start_time + duration
        
        # Convert pitch name to MIDI number
        pitch_name = row['Pitch']
        midi_pitch = midi_name_to_number(pitch_name)
        
        velocity = int(row['Velocity'])
        
        # Get pedal state using simple approach
        pedal_state = get_pedal_state_simple(start_time, end_time, pedal_events, idx, pedal_start_indices, pedal_end_indices)
        
        model_inputs.append([start_time, duration, midi_pitch, velocity, pedal_state])
    
    return np.array(model_inputs, dtype=np.float32)

def create_model_input_new_pedal(notes_df, pedal_data, window_ms=100):
    """
    Create model input features using new pedal state approach (WINDOW-BASED)
    
    Args:
        notes_df (pd.DataFrame): Annotated notes with columns [Start_Time, Duration, Pitch, Velocity, Start_Ticks, Slur_Category]
        pedal_data (np.ndarray): Pedal matrix (3, time_steps) with sustain in row 0
        window_ms (float): Temporal window in milliseconds for pedal start/end detection
        
    Returns:
        np.ndarray: Model inputs (num_notes, 5) [start_time, duration, midi_pitch, velocity, pedal_state]
    """
    model_inputs = []
    
    print(f"Processing {len(notes_df)} notes with window-based pedal approach (window: {window_ms}ms)...")
    
    # Extract pedal events
    pedal_events = extract_pedal_events(pedal_data)
    print(f"Found {len(pedal_events)} pedal events")
    
    for idx, row in notes_df.iterrows():
        # Extract basic note features
        start_time = float(row['Start_Time'])
        duration = float(row['Duration'])
        end_time = start_time + duration
        
        # Convert pitch name to MIDI number
        pitch_name = row['Pitch']
        midi_pitch = midi_name_to_number(pitch_name)
        
        velocity = int(row['Velocity'])
        
        # Get pedal state using window-based approach
        pedal_state = get_pedal_state(start_time, end_time, pedal_events, window_ms)
        
        model_inputs.append([start_time, duration, midi_pitch, velocity, pedal_state])
    
    return np.array(model_inputs, dtype=np.float32)

def create_model_input(notes_df, pedal_data):
    """
    Create model input features from notes dataframe and pedal data (ORIGINAL APPROACH)
    
    Args:
        notes_df (pd.DataFrame): Annotated notes with columns [Start_Time, Duration, Pitch, Velocity, Start_Ticks, Slur_Category]
        pedal_data (np.ndarray): Pedal matrix (3, time_steps) with sustain in row 0
        
    Returns:
        np.ndarray: Model inputs (num_notes, 6) [start_time, duration, midi_pitch, velocity, sustain_start, sustain_end]
    """
    model_inputs = []
    
    print(f"Processing {len(notes_df)} notes...")
    
    for idx, row in notes_df.iterrows():
        # Extract basic note features
        start_time = float(row['Start_Time'])
        duration = float(row['Duration'])
        
        # Convert pitch name to MIDI number
        pitch_name = row['Pitch']
        midi_pitch = midi_name_to_number(pitch_name)
        
        velocity = int(row['Velocity'])
        
        # Look up sustain pedal state at note start time
        start_tick = int(row['Start_Ticks'])
        if start_tick < pedal_data.shape[1]:
            sustain_start = pedal_data[0, start_tick]  # Row 0 = sustain
        else:
            sustain_start = 0  # Default to no sustain if out of bounds
        
        # Look up sustain pedal state at note end time
        end_tick = start_tick + int(row['Duration_Ticks']) if 'Duration_Ticks' in row else start_tick + int(duration * 480)  # Approximate end tick
        if end_tick < pedal_data.shape[1]:
            sustain_end = pedal_data[0, end_tick]  # Row 0 = sustain
        else:
            sustain_end = 0  # Default to no sustain if out of bounds
        
        model_inputs.append([start_time, duration, midi_pitch, velocity, sustain_start, sustain_end])
    
    return np.array(model_inputs, dtype=np.float32)

def normalize_features_new_pedal(inputs):
    """
    Normalize input features for new pedal approach (5 features)
    
    Args:
        inputs (np.ndarray): Raw inputs (num_notes, 5)
        
    Returns:
        np.ndarray: Normalized inputs (num_notes, 5)
        dict: Normalization parameters for later use
    """
    normalized = inputs.copy()
    norm_params = {}
    
    # Normalize start times to 0-100
    start_times = inputs[:, 0]
    start_min, start_max = start_times.min(), start_times.max()
    if start_max > start_min:
        normalized[:, 0] = (start_times - start_min) / (start_max - start_min) * 100
    norm_params['start_time'] = {'min': start_min, 'max': start_max}
    
    # Normalize durations to 0-100
    durations = inputs[:, 1]
    dur_min, dur_max = durations.min(), durations.max()
    if dur_max > dur_min:
        normalized[:, 1] = (durations - dur_min) / (dur_max - dur_min) * 100
    norm_params['duration'] = {'min': dur_min, 'max': dur_max}
    
    # Convert MIDI pitch to relative (0-87) - no normalization needed
    normalized[:, 2] = inputs[:, 2] - 21  # MIDI 21-108 → 0-87
    norm_params['pitch'] = {'offset': 21}
    
    # Normalize velocity to 0-100
    normalized[:, 3] = inputs[:, 3] / 127 * 100
    norm_params['velocity'] = {'max': 127}
    
    # Pedal state is already categorical (0-3), normalize to 0-100 for consistency
    normalized[:, 4] = inputs[:, 4] / 3 * 100
    norm_params['pedal_state'] = {'max': 3}
    
    print(f"✓ Normalized features (new pedal approach):")
    print(f"  Start time: {start_min:.3f}-{start_max:.3f} → 0-100")
    print(f"  Duration: {dur_min:.3f}-{dur_max:.3f} → 0-100")
    print(f"  MIDI pitch: {inputs[:, 2].min():.0f}-{inputs[:, 2].max():.0f} → {normalized[:, 2].min():.0f}-{normalized[:, 2].max():.0f}")
    print(f"  Velocity: {inputs[:, 3].min():.0f}-{inputs[:, 3].max():.0f} → 0-100")
    print(f"  Pedal state: {inputs[:, 4].min():.0f}-{inputs[:, 4].max():.0f} → 0-100")
    
    return normalized, norm_params

def normalize_features(inputs):
    """
    Normalize input features to similar scales (ORIGINAL APPROACH - 6 features)
    
    Args:
        inputs (np.ndarray): Raw inputs (num_notes, 6)
        
    Returns:
        np.ndarray: Normalized inputs (num_notes, 6)
        dict: Normalization parameters for later use
    """
    normalized = inputs.copy()
    norm_params = {}
    
    # Normalize start times to 0-100
    start_times = inputs[:, 0]
    start_min, start_max = start_times.min(), start_times.max()
    if start_max > start_min:
        normalized[:, 0] = (start_times - start_min) / (start_max - start_min) * 100
    norm_params['start_time'] = {'min': start_min, 'max': start_max}
    
    # Normalize durations to 0-100
    durations = inputs[:, 1]
    dur_min, dur_max = durations.min(), durations.max()
    if dur_max > dur_min:
        normalized[:, 1] = (durations - dur_min) / (dur_max - dur_min) * 100
    norm_params['duration'] = {'min': dur_min, 'max': dur_max}
    
    # Convert MIDI pitch to relative (0-87) - no normalization needed
    normalized[:, 2] = inputs[:, 2] - 21  # MIDI 21-108 → 0-87
    norm_params['pitch'] = {'offset': 21}
    
    # Normalize velocity to 0-100
    normalized[:, 3] = inputs[:, 3] / 127 * 100
    norm_params['velocity'] = {'max': 127}
    
    # Normalize sustain_start to 0-100 (consistent with other features)
    normalized[:, 4] = inputs[:, 4] / 127 * 100
    norm_params['sustain_start'] = {'max': 127}
    
    # Normalize sustain_end to 0-100 (consistent with other features)
    normalized[:, 5] = inputs[:, 5] / 127 * 100
    norm_params['sustain_end'] = {'max': 127}
    
    print(f"✓ Normalized features (original approach):")
    print(f"  Start time: {start_min:.3f}-{start_max:.3f} → 0-100")
    print(f"  Duration: {dur_min:.3f}-{dur_max:.3f} → 0-100")
    print(f"  MIDI pitch: {inputs[:, 2].min():.0f}-{inputs[:, 2].max():.0f} → {normalized[:, 2].min():.0f}-{normalized[:, 2].max():.0f}")
    print(f"  Velocity: {inputs[:, 3].min():.0f}-{inputs[:, 3].max():.0f} → 0-100")
    print(f"  Sustain start: {inputs[:, 4].min():.0f}-{inputs[:, 4].max():.0f} → 0-100")
    print(f"  Sustain end: {inputs[:, 5].min():.0f}-{inputs[:, 5].max():.0f} → 0-100")
    
    return normalized, norm_params

def create_targets(notes_df):
    """
    Create target labels from slur annotations (class indices for CrossEntropyLoss)
    
    Args:
        notes_df (pd.DataFrame): Annotated notes with Slur_Category column
        
    Returns:
        np.ndarray: Class indices (num_notes,) where:
            0 = slur_start, 1 = slur_middle, 2 = slur_end, 3 = no_slur, 4 = slur_start_and_end
            Background (category 0) is mapped to no_slur (3)
    """
    targets = []
    category_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    class_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 0: 3}  # Map to class indices, background -> no_slur
    
    for _, row in notes_df.iterrows():
        slur_category = row['Slur_Category']
        
        # Handle missing or invalid categories
        if pd.isna(slur_category) or slur_category == '':
            slur_category = 0
        else:
            slur_category = int(slur_category)
            
        category_counts[slur_category] += 1
        
        # Map to class index (0-4)
        class_idx = class_mapping.get(slur_category, 3)  # Default to no_slur for unknown categories
        targets.append(class_idx)
    
    print(f"✓ Target distribution:")
    print(f"  Category 0 (Background): {category_counts[0]} notes -> mapped to 'no_slur'")
    print(f"  Category 1 (Slur start): {category_counts[1]} notes -> class 0")
    print(f"  Category 2 (Slur middle): {category_counts[2]} notes -> class 1")
    print(f"  Category 3 (Slur end): {category_counts[3]} notes -> class 2")
    print(f"  Category 4 (No slur): {category_counts[4]} notes -> class 3")
    print(f"  Category 5 (Slur start and end): {category_counts[5]} notes -> class 4")
    
    return np.array(targets, dtype=np.int64)

def prepare_for_training(inputs, targets):
    """
    Convert to PyTorch tensors for training
    
    Args:
        inputs (np.ndarray): Normalized inputs (num_notes, features)
        targets (np.ndarray): Class indices (num_notes,) for CrossEntropyLoss
        
    Returns:
        torch.Tensor: Input tensor for model
        torch.Tensor: Target tensor for training (class indices)
    """
    input_tensor = torch.FloatTensor(inputs)    # Shape: (sequence_length, features)
    target_tensor = torch.LongTensor(targets)   # Shape: (sequence_length,) for class indices
    
    print(f"✓ Tensors created:")
    print(f"  Input shape: {input_tensor.shape}")
    print(f"  Target shape: {target_tensor.shape}")
    
    return input_tensor, target_tensor

def load_and_prepare_data_simple_pedal(base_filename, output_dir="output"):
    """
    Complete data pipeline using simple pedal approach: load, process, and prepare data for transformer training
    
    Args:
        base_filename (str): Base filename without extensions
        output_dir (str): Directory containing the data files
        
    Returns:
        torch.Tensor: Input tensor (sequence_length, 5)
        torch.Tensor: Target tensor (sequence_length, 4)
        dict: Normalization parameters
        dict: Dataset statistics
    """
    print("🔄 LOADING AND PREPARING DATA FOR TRANSFORMER TRAINING (SIMPLE PEDAL APPROACH)")
    print("=" * 80)
    
    # Construct file paths
    # Prefer completed annotation files, fall back to regular annotation files
    notes_csv_completed = os.path.join(output_dir, f"{base_filename}_slur_annotation_completed.csv")
    notes_csv = os.path.join(output_dir, f"{base_filename}_slur_annotation.csv")
    if os.path.exists(notes_csv_completed):
        notes_csv = notes_csv_completed
    pedal_npy = os.path.join(output_dir, f"{base_filename}_pedal.npy")
    
    # Check if files exist
    if not os.path.exists(notes_csv):
        raise FileNotFoundError(f"Notes CSV not found: {notes_csv}")
    if not os.path.exists(pedal_npy):
        raise FileNotFoundError(f"Pedal matrix not found: {pedal_npy}")
    
    # Load data
    print(f"📂 Loading data...")
    print(f"  Notes CSV: {notes_csv}")
    print(f"  Pedal matrix: {pedal_npy}")
    
    notes_df = pd.read_csv(notes_csv)
    pedal_data = np.load(pedal_npy)
    
    print(f"✓ Data loaded:")
    print(f"  Notes: {len(notes_df)} entries")
    print(f"  Pedal matrix: {pedal_data.shape}")
    
    # Create features and targets using simple pedal approach
    print(f"\n🔧 Processing features (simple pedal approach)...")
    inputs = create_model_input_simple_pedal(notes_df, pedal_data)
    targets = create_targets(notes_df)
    
    # Normalize features
    print(f"\n📐 Normalizing features...")
    normalized_inputs, norm_params = normalize_features_new_pedal(inputs)
    
    # Convert to tensors
    print(f"\n🎯 Creating tensors...")
    input_tensor, target_tensor = prepare_for_training(normalized_inputs, targets)
    
    # Compile statistics
    stats = {
        'num_notes': len(notes_df),
        'sequence_length': input_tensor.shape[0],
        'input_features': input_tensor.shape[1],
        'output_features': 5,  # 5 classes: slur_start, slur_middle, slur_end, no_slur, slur_start_and_end
        'pedal_matrix_shape': pedal_data.shape,
        'input_shape': input_tensor.shape,
        'target_shape': target_tensor.shape,
        'pedal_approach': 'simple'
    }
    
    print(f"\n✅ DATA PREPARATION COMPLETE!")
    print(f"  Ready for transformer training: {stats['sequence_length']} notes")
    print(f"  Input features: {stats['input_features']} (start_time, duration, pitch, velocity, pedal_state)")
    print(f"  Output classes: {stats['output_features']} (slur_start, slur_middle, slur_end, no_slur, slur_start_and_end)")
    print(f"  Pedal approach: Simple (first note after pedal on)")
    
    return input_tensor, target_tensor, norm_params, stats

def load_and_prepare_data_new_pedal(base_filename, output_dir="output", window_ms=100):
    """
    Complete data pipeline using new pedal approach: load, process, and prepare data for transformer training
    
    Args:
        base_filename (str): Base filename without extensions
        output_dir (str): Directory containing the data files
        window_ms (float): Temporal window in milliseconds for pedal start/end detection
        
    Returns:
        torch.Tensor: Input tensor (sequence_length, 5)
        torch.Tensor: Target tensor (sequence_length, 4)
        dict: Normalization parameters
        dict: Dataset statistics
    """
    print("🔄 LOADING AND PREPARING DATA FOR TRANSFORMER TRAINING (NEW PEDAL APPROACH)")
    print("=" * 80)
    
    # Construct file paths
    # Prefer completed annotation files, fall back to regular annotation files
    notes_csv_completed = os.path.join(output_dir, f"{base_filename}_slur_annotation_completed.csv")
    notes_csv = os.path.join(output_dir, f"{base_filename}_slur_annotation.csv")
    if os.path.exists(notes_csv_completed):
        notes_csv = notes_csv_completed
    pedal_npy = os.path.join(output_dir, f"{base_filename}_pedal.npy")
    
    # Check if files exist
    if not os.path.exists(notes_csv):
        raise FileNotFoundError(f"Notes CSV not found: {notes_csv}")
    if not os.path.exists(pedal_npy):
        raise FileNotFoundError(f"Pedal matrix not found: {pedal_npy}")
    
    # Load data
    print(f"📂 Loading data...")
    print(f"  Notes CSV: {notes_csv}")
    print(f"  Pedal matrix: {pedal_npy}")
    
    notes_df = pd.read_csv(notes_csv)
    pedal_data = np.load(pedal_npy)
    
    print(f"✓ Data loaded:")
    print(f"  Notes: {len(notes_df)} entries")
    print(f"  Pedal matrix: {pedal_data.shape}")
    
    # Create features and targets using new pedal approach
    print(f"\n🔧 Processing features (new pedal approach)...")
    inputs = create_model_input_new_pedal(notes_df, pedal_data, window_ms)
    targets = create_targets(notes_df)
    
    # Normalize features
    print(f"\n📐 Normalizing features...")
    normalized_inputs, norm_params = normalize_features_new_pedal(inputs)
    
    # Convert to tensors
    print(f"\n🎯 Creating tensors...")
    input_tensor, target_tensor = prepare_for_training(normalized_inputs, targets)
    
    # Compile statistics
    stats = {
        'num_notes': len(notes_df),
        'sequence_length': input_tensor.shape[0],
        'input_features': input_tensor.shape[1],
        'output_features': 5,  # 5 classes: slur_start, slur_middle, slur_end, no_slur, slur_start_and_end
        'pedal_matrix_shape': pedal_data.shape,
        'input_shape': input_tensor.shape,
        'target_shape': target_tensor.shape,
        'pedal_window_ms': window_ms
    }
    
    print(f"\n✅ DATA PREPARATION COMPLETE!")
    print(f"  Ready for transformer training: {stats['sequence_length']} notes")
    print(f"  Input features: {stats['input_features']} (start_time, duration, pitch, velocity, pedal_state)")
    print(f"  Output classes: {stats['output_features']} (slur_start, slur_middle, slur_end, no_slur, slur_start_and_end)")
    print(f"  Pedal window: {window_ms}ms")
    
    return input_tensor, target_tensor, norm_params, stats

def load_and_prepare_data(base_filename, output_dir="output"):
    """
    Complete data pipeline: load, process, and prepare data for transformer training (ORIGINAL APPROACH)
    
    Args:
        base_filename (str): Base filename without extensions
        output_dir (str): Directory containing the data files
        
    Returns:
        torch.Tensor: Input tensor (sequence_length, 6)
        torch.Tensor: Target tensor (sequence_length, 4)
        dict: Normalization parameters
        dict: Dataset statistics
    """
    print("🔄 LOADING AND PREPARING DATA FOR TRANSFORMER TRAINING (ORIGINAL APPROACH)")
    print("=" * 80)
    
    # Construct file paths
    # Prefer completed annotation files, fall back to regular annotation files
    notes_csv_completed = os.path.join(output_dir, f"{base_filename}_slur_annotation_completed.csv")
    notes_csv = os.path.join(output_dir, f"{base_filename}_slur_annotation.csv")
    if os.path.exists(notes_csv_completed):
        notes_csv = notes_csv_completed
    pedal_npy = os.path.join(output_dir, f"{base_filename}_pedal.npy")
    
    # Check if files exist
    if not os.path.exists(notes_csv):
        raise FileNotFoundError(f"Notes CSV not found: {notes_csv}")
    if not os.path.exists(pedal_npy):
        raise FileNotFoundError(f"Pedal matrix not found: {pedal_npy}")
    
    # Load data
    print(f"📂 Loading data...")
    print(f"  Notes CSV: {notes_csv}")
    print(f"  Pedal matrix: {pedal_npy}")
    
    notes_df = pd.read_csv(notes_csv)
    pedal_data = np.load(pedal_npy)
    
    print(f"✓ Data loaded:")
    print(f"  Notes: {len(notes_df)} entries")
    print(f"  Pedal matrix: {pedal_data.shape}")
    
    # Create features and targets
    print(f"\n🔧 Processing features (original approach)...")
    inputs = create_model_input(notes_df, pedal_data)
    targets = create_targets(notes_df)
    
    # Normalize features
    print(f"\n📐 Normalizing features...")
    normalized_inputs, norm_params = normalize_features(inputs)
    
    # Convert to tensors
    print(f"\n🎯 Creating tensors...")
    input_tensor, target_tensor = prepare_for_training(normalized_inputs, targets)
    
    # Compile statistics
    stats = {
        'num_notes': len(notes_df),
        'sequence_length': input_tensor.shape[0],
        'input_features': input_tensor.shape[1],
        'output_features': 5,  # 5 classes: slur_start, slur_middle, slur_end, no_slur, slur_start_and_end
        'pedal_matrix_shape': pedal_data.shape,
        'input_shape': input_tensor.shape,
        'target_shape': target_tensor.shape
    }
    
    print(f"\n✅ DATA PREPARATION COMPLETE!")
    print(f"  Ready for transformer training: {stats['sequence_length']} notes")
    print(f"  Input features: {stats['input_features']} (start_time, duration, pitch, velocity, sustain_start, sustain_end)")
    print(f"  Output classes: {stats['output_features']} (slur_start, slur_middle, slur_end, no_slur, slur_start_and_end)")
    
    return input_tensor, target_tensor, norm_params, stats

def save_processed_data(input_tensor, target_tensor, norm_params, stats, output_path):
    """
    Save processed data for later use
    
    Args:
        input_tensor (torch.Tensor): Processed input data
        target_tensor (torch.Tensor): Target labels
        norm_params (dict): Normalization parameters
        stats (dict): Dataset statistics
        output_path (str): Path to save processed data
    """
    torch.save({
        'inputs': input_tensor,
        'targets': target_tensor,
        'norm_params': norm_params,
        'stats': stats
    }, output_path)
    
    print(f"✓ Processed data saved: {output_path}")

def load_processed_data(input_path):
    """
    Load previously processed data
    
    Args:
        input_path (str): Path to processed data file
        
    Returns:
        torch.Tensor: Input tensor
        torch.Tensor: Target tensor
        dict: Normalization parameters
        dict: Dataset statistics
    """
    data = torch.load(input_path)
    return data['inputs'], data['targets'], data['norm_params'], data['stats']

if __name__ == "__main__":
    # Test the pipeline with the Beethoven sonata
    base_filename = "Beethoven_Piano_Sonata_No_10_Op_14_No_2_fQqNsTUvqCY_cut_mov_1"
    
    try:
        inputs, targets, norm_params, stats = load_and_prepare_data(base_filename)
        
        # Save processed data
        output_path = f"output/{base_filename}_processed_for_ml.pt"
        save_processed_data(inputs, targets, norm_params, stats, output_path)
        
        print(f"\n🎵 Pipeline test successful!")
        print(f"   Data ready for transformer training")
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
