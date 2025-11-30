#!/usr/bin/env python3
"""
Complete MIDI to Piano Roll Processor
Generates notes, pedal matrices, and annotation CSV with consistent raw MIDI timing
Production-ready system for ML data preparation

Author: Developed for MIDI Piano Roll ML Research
Version: 2.0 (Post-timing-fix)
"""

import numpy as np
import pandas as pd
import mido
import os
from pathlib import Path

def midi_to_note_name(midi_number):
    """Convert MIDI note number to musical note name (e.g., 60 -> C4)"""
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_number // 12) - 1
    note = notes[midi_number % 12]
    return f"{note}{octave}"

def extract_notes_raw_midi(midi_file_path):
    """
    Extract all notes using raw MIDI timing (ensures consistency across all matrices)
    
    This function uses direct MIDI event parsing to avoid Music21's quantization,
    ensuring perfect temporal alignment with slur annotations.
    
    Returns:
        tuple: (notes_list, time_step)
    """
    print(f"Extracting notes with raw MIDI timing from: {os.path.basename(midi_file_path)}")
    
    # Get MIDI resolution
    mid = mido.MidiFile(midi_file_path)
    time_step = 1.0 / mid.ticks_per_beat
    print(f"MIDI resolution: {time_step:.8f} quarter notes per tick ({mid.ticks_per_beat} ticks per beat)")
    
    # Extract notes directly from raw MIDI events
    # Process all tracks together with unified timing
    notes_list = []
    active_notes = {}  # Track note_on events waiting for note_off
    
    # Collect all messages from all tracks with their absolute times
    all_messages = []
    for track in mid.tracks:
        cumulative_time = 0
        for msg in track:
            cumulative_time += msg.time
            all_messages.append((cumulative_time, msg))
    
    # Sort by absolute time to process in order
    all_messages.sort(key=lambda x: x[0])
    
    cumulative_time = 0
    for msg_time, msg in all_messages:
        cumulative_time = msg_time
        
        if msg.type == 'note_on':
            if msg.velocity > 0:
                # Note start - handle overlapping notes properly
                if msg.note in active_notes:
                    # Previous note is still active, end it first
                    note_start = active_notes.pop(msg.note)
                    duration_ticks = cumulative_time - note_start['start_ticks']
                    duration_time = duration_ticks * time_step
                    
                    notes_list.append({
                        'start_time': note_start['start_time'],
                        'start_ticks': note_start['start_ticks'],
                        'pitch_midi': note_start['pitch_midi'],
                        'pitch_name': note_start['pitch_name'],
                        'velocity': note_start['velocity'],
                        'duration': duration_time,
                        'duration_ticks': duration_ticks,
                        'source': 'raw_midi_overlapped'
                    })
                
                # Now store the new note
                active_notes[msg.note] = {
                    'start_ticks': cumulative_time,
                    'start_time': cumulative_time * time_step,
                    'velocity': msg.velocity,
                    'pitch_midi': msg.note,
                    'pitch_name': midi_to_note_name(msg.note)
                }
            else:
                # Note end (velocity 0 = note_off)
                if msg.note in active_notes:
                    note_start = active_notes.pop(msg.note)
                    duration_ticks = cumulative_time - note_start['start_ticks']
                    duration_time = duration_ticks * time_step
                    
                    notes_list.append({
                        'start_time': note_start['start_time'],
                        'start_ticks': note_start['start_ticks'],
                        'pitch_midi': note_start['pitch_midi'],
                        'pitch_name': note_start['pitch_name'],
                        'velocity': note_start['velocity'],
                        'duration': duration_time,
                        'duration_ticks': duration_ticks,
                        'source': 'raw_midi'
                    })
        
        elif msg.type == 'note_off':
            # Explicit note_off event
            if msg.note in active_notes:
                note_start = active_notes.pop(msg.note)
                duration_ticks = cumulative_time - note_start['start_ticks']
                duration_time = duration_ticks * time_step
                
                notes_list.append({
                    'start_time': note_start['start_time'],
                    'start_ticks': note_start['start_ticks'],
                    'pitch_midi': note_start['pitch_midi'],
                    'pitch_name': note_start['pitch_name'],
                    'velocity': note_start['velocity'],
                    'duration': duration_time,
                    'duration_ticks': duration_ticks,
                    'source': 'raw_midi'
                })
    
    # Handle remaining active notes at end of file
    for note_num, note_start in active_notes.items():
        duration_ticks = cumulative_time - note_start['start_ticks']
        duration_time = duration_ticks * time_step
        
        notes_list.append({
            'start_time': note_start['start_time'],
            'start_ticks': note_start['start_ticks'],
            'pitch_midi': note_start['pitch_midi'],
            'pitch_name': note_start['pitch_name'],
            'velocity': note_start['velocity'],
            'duration': duration_time,
            'duration_ticks': duration_ticks,
            'source': 'raw_midi_end'
        })
    
    # Sort notes by start time, then by pitch (consistent with slur annotation)
    notes_list.sort(key=lambda x: (x['start_time'], x['pitch_midi']))
    
    # Add index for reference
    for i, note in enumerate(notes_list):
        note['index'] = i
    
    print(f"✓ Extracted {len(notes_list)} notes using raw MIDI timing")
    overlapped = len([n for n in notes_list if n['source'] == 'raw_midi_overlapped'])
    if overlapped > 0:
        print(f"✓ Handled {overlapped} overlapping notes properly")
    
    return notes_list, time_step

def extract_pedal_raw_midi(midi_file_path):
    """
    Extract pedal events using raw MIDI timing
    
    Returns:
        tuple: (pedal_events, time_step)
    """
    print(f"Extracting pedal events with raw MIDI timing...")
    
    # Get MIDI resolution
    mid = mido.MidiFile(midi_file_path)
    time_step = 1.0 / mid.ticks_per_beat
    
    # Extract pedal information using mido
    pedal_events = []
    
    for track in mid.tracks:
        cumulative_time = 0
        for msg in track:
            cumulative_time += msg.time
            if msg.type == 'control_change':
                if msg.control == 64:  # Sustain pedal
                    pedal_events.append({
                        'type': 'sustain',
                        'time_ticks': cumulative_time,
                        'time_quarters': cumulative_time * time_step,
                        'value': msg.value,
                        'state': msg.value >= 64
                    })
                elif msg.control == 66:  # Sostenuto pedal
                    pedal_events.append({
                        'type': 'sostenuto', 
                        'time_ticks': cumulative_time,
                        'time_quarters': cumulative_time * time_step,
                        'value': msg.value,
                        'state': msg.value >= 64
                    })
                elif msg.control == 67:  # Soft pedal
                    pedal_events.append({
                        'type': 'soft',
                        'time_ticks': cumulative_time,
                        'time_quarters': cumulative_time * time_step,
                        'value': msg.value,
                        'state': msg.value >= 64
                    })
    
    print(f"✓ Extracted {len(pedal_events)} pedal events")
    
    return pedal_events, time_step

def calculate_time_steps(notes_list):
    """
    Calculate total time steps needed for matrices using raw MIDI timing
    
    Returns:
        int: Total time steps (max end tick)
    """
    if not notes_list:
        return 0
    
    # Calculate dimensions using EXACT tick timing
    max_end_tick = max(note['start_ticks'] + note['duration_ticks'] for note in notes_list)
    return max_end_tick

def create_pedal_matrix(pedal_events, time_steps, preserve_velocity=True):
    """
    Create pedal matrix using raw MIDI timing
    
    Returns:
        np.ndarray: Pedal matrix (3 x time_steps)
    """
    print(f"Creating pedal matrix...")
    
    # Create pedal matrix (3 rows: sustain, sostenuto, soft)
    # Create pedal matrix with appropriate data type (3 channels: sustain, sostenuto, soft)
    pedal_matrix = np.zeros((3, time_steps), dtype=np.uint8)
    pedal_type_map = {'sustain': 0, 'sostenuto': 1, 'soft': 2}
    
    # Fill pedal matrix with state changes using EXACT tick timing
    current_pedal_states = {'sustain': 0, 'sostenuto': 0, 'soft': 0}
    
    for event in pedal_events:
        pedal_idx = pedal_type_map[event['type']]
        time_tick = event['time_ticks']
        
        if time_tick < time_steps:
            # Update state
            if preserve_velocity:
                current_pedal_states[event['type']] = event['value']
            else:
                current_pedal_states[event['type']] = 1 if event['state'] else 0
            
            # Fill from this point forward until next change
            pedal_matrix[pedal_idx, time_tick:] = current_pedal_states[event['type']]
    
    print(f"✓ Pedal matrix dimensions: {pedal_matrix.shape}")
    
    return pedal_matrix

def create_annotation_csv(notes_list, base_filename):
    """
    Create CSV file for manual slur annotation
    
    Returns:
        str: Path to created CSV file
    """
    print(f"Creating annotation CSV...")
    
    # Create DataFrame with all note information
    csv_data = []
    for note in notes_list:
        csv_data.append({
            'Index': note['index'],
            'Start_Ticks': note['start_ticks'],
            'Start_Time': round(note['start_time'], 12),  # 12 decimal places for precision
            'Pitch': note['pitch_name'],
            'Pitch_MIDI': note['pitch_midi'],
            'Velocity': note['velocity'],
            'Duration_Ticks': note['duration_ticks'],
            'Duration': round(note['duration'], 12),
            'Source': note['source'],
            'Slur_Category': '',  # Empty for manual annotation (use 0=empty, 1=begin, 2=middle, 3=end, 4=no_slur)
            'Notes': ''  # For annotator notes
        })
    
    df = pd.DataFrame(csv_data)
    
    # Save CSV
    csv_filename = f"{base_filename}_slur_annotation.csv"
    df.to_csv(csv_filename, index=False)
    
    print(f"✓ Annotation CSV created: {csv_filename}")
    print(f"✓ Ready for manual annotation of {len(notes_list)} notes")
    
    return csv_filename

def save_matrices_and_metadata(pedal_matrix, notes_list, pedal_events, 
                               time_step, midi_file_path, base_filename):
    """
    Save pedal matrix and comprehensive metadata (notes matrix removed - not used by transformer)
    """
    print(f"Saving pedal matrix and metadata...")
    
    # Save pedal matrix only
    np.save(f"{base_filename}_pedal.npy", pedal_matrix)
    
    # Save as CSV for inspection
    np.savetxt(f"{base_filename}_pedal.csv", pedal_matrix, delimiter=',', fmt='%d')
    
    # Calculate statistics
    velocities = [note['velocity'] for note in notes_list]
    overlapped_notes = len([n for n in notes_list if n['source'] == 'raw_midi_overlapped'])
    
    # Save comprehensive metadata
    with open(f"{base_filename}_metadata.txt", 'w') as f:
        f.write("=== TRANSFORMER-OPTIMIZED MIDI PROCESSING ===\n")
        f.write("Generated by: MIDI Piano Roll ML System v2.0 (Transformer Edition)\n")
        f.write("⚠️  TIMING METHOD: Raw MIDI ticks (ensures perfect alignment)\n")
        f.write(f"MIDI File: {midi_file_path}\n")
        f.write(f"Time Resolution: {time_step:.8f} quarter notes per tick\n")
        f.write(f"MIDI Resolution: {int(1/time_step)} ticks per beat\n")
        f.write(f"Total Duration: {pedal_matrix.shape[1] * time_step:.3f} quarter notes\n\n")
        
        f.write("=== NOTE DATA ===\n")
        f.write(f"Total Notes: {len(notes_list)}\n")
        f.write(f"Overlapped Notes: {overlapped_notes} (handled properly)\n")
        f.write(f"Velocity Range: {min(velocities)}-{max(velocities)}\n")
        f.write(f"Average Velocity: {sum(velocities)/len(velocities):.1f}\n")
        f.write("Note data stored in CSV for transformer training (no matrix needed)\n\n")
        
        f.write("=== PEDAL MATRIX ===\n")
        f.write(f"Shape: {pedal_matrix.shape}\n")
        f.write(f"Rows: [sustain, sostenuto, soft]\n")
        f.write(f"Values: 0=off, 1-127=pedal value\n")
        f.write(f"Total Pedal Events: {len(pedal_events)}\n")
        sustain_events = len([e for e in pedal_events if e['type'] == 'sustain'])
        sostenuto_events = len([e for e in pedal_events if e['type'] == 'sostenuto'])
        soft_events = len([e for e in pedal_events if e['type'] == 'soft'])
        f.write(f"  - Sustain: {sustain_events}\n")
        f.write(f"  - Sostenuto: {sostenuto_events}\n")
        f.write(f"  - Soft: {soft_events}\n\n")
        
        f.write("=== SLUR ANNOTATION CATEGORIES ===\n")
        f.write("0: Empty/No annotation (default background)\n")
        f.write("1: Slur beginning (first note of slur)\n")
        f.write("2: Slur middle (continuation within slur)\n")
        f.write("3: Slur end (final note of slur)\n")
        f.write("4: No slur (explicitly isolated note)\n\n")
        
        f.write("=== TRANSFORMER TRAINING USAGE ===\n")
        f.write("import numpy as np\n")
        f.write("import pandas as pd\n")
        f.write(f"pedal = np.load('{os.path.basename(base_filename)}_pedal.npy')  # Shape: {pedal_matrix.shape}\n")
        f.write(f"notes = pd.read_csv('{os.path.basename(base_filename)}_slur_annotation.csv')\n")
        f.write("# Transformer uses note sequences + pedal context for slur prediction\n")
        f.write("# No large note matrix needed - optimized for sequence-based learning!\n")
    
    print(f"✓ Pedal matrix saved: {base_filename}_pedal.npy") 
    print(f"✓ CSV files saved for inspection")
    print(f"✓ Comprehensive metadata saved: {base_filename}_metadata.txt")

def process_midi_file(midi_file_path, output_dir=None, note_range=(21, 108), preserve_velocity=True):
    """
    Complete MIDI processing workflow: generates pedal matrix and annotation CSV (optimized for transformer)
    
    Args:
        midi_file_path (str): Path to MIDI file
        output_dir (str): Output directory (if None, uses same directory as MIDI file)
        note_range (tuple): MIDI note range (kept for compatibility, not used)
        preserve_velocity (bool): Whether to preserve velocity information
        
    Returns:
        dict: {
            'pedal_matrix': np.ndarray,
            'annotation_csv': str (path),
            'base_filename': str,
            'metadata': dict
        }
    """
    
    print("=" * 70)
    print("🎵 MIDI PIANO ROLL ML SYSTEM v2.0 (TRANSFORMER EDITION) 🎵")
    print("=" * 70)
    print(f"Processing: {os.path.basename(midi_file_path)}")
    print(f"Using raw MIDI timing for perfect alignment")
    print(f"Optimized for transformer training (no unused note matrix)")
    print("-" * 70)
    
    # Validate input
    if not os.path.exists(midi_file_path):
        raise FileNotFoundError(f"MIDI file not found: {midi_file_path}")
    
    # Set up output directory and filename
    if output_dir is None:
        output_dir = os.path.dirname(midi_file_path)
    
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(midi_file_path))[0]
    base_path = os.path.join(output_dir, base_filename)
    
    try:
        # Step 1: Extract notes and pedal events
        notes_list, time_step = extract_notes_raw_midi(midi_file_path)
        pedal_events, _ = extract_pedal_raw_midi(midi_file_path)
        
        # Step 2: Calculate time steps and create pedal matrix only
        time_steps = calculate_time_steps(notes_list)
        pedal_matrix = create_pedal_matrix(pedal_events, time_steps, preserve_velocity)
        
        # Step 3: Create annotation CSV
        annotation_csv = create_annotation_csv(notes_list, base_path)
        
        # Step 4: Save everything
        save_matrices_and_metadata(pedal_matrix, notes_list, pedal_events, 
                                   time_step, midi_file_path, base_path)
        
        # Compile results
        result = {
            'pedal_matrix': pedal_matrix,
            'annotation_csv': annotation_csv,
            'base_filename': base_filename,
            'metadata': {
                'midi_file': midi_file_path,
                'output_dir': output_dir,
                'pedal_matrix_shape': pedal_matrix.shape,
                'total_notes': len(notes_list),
                'overlapped_notes': len([n for n in notes_list if n['source'] == 'raw_midi_overlapped']),
                'total_pedal_events': len(pedal_events),
                'time_step': time_step,
                'midi_ticks_per_beat': int(1/time_step),
                'preserve_velocity': preserve_velocity
            }
        }
        
        print("-" * 70)
        print("✅ PROCESSING COMPLETE!")
        print(f"✓ Pedal matrix: {pedal_matrix.shape}")
        print(f"✓ Annotation CSV: {len(notes_list)} notes ready for manual annotation")
        print(f"✓ All files saved with base name: {base_filename}")
        print(f"✓ Optimized for transformer training (no unused note matrix)")
        print("=" * 70)
        
        return result
        
    except Exception as e:
        print(f"❌ Error processing MIDI file: {e}")
        import traceback
        traceback.print_exc()
        return None

def batch_process_midi_files(midi_files, output_base_dir="processed_midi"):
    """
    Process multiple MIDI files in batch
    
    Args:
        midi_files (list): List of MIDI file paths
        output_base_dir (str): Base directory for outputs
        
    Returns:
        dict: Results for each file
    """
    
    results = {}
    
    print(f"🎵 BATCH PROCESSING {len(midi_files)} MIDI FILES 🎵")
    print("=" * 70)
    
    for i, midi_file in enumerate(midi_files, 1):
        print(f"\n[{i}/{len(midi_files)}] Processing {os.path.basename(midi_file)}...")
        
        # Create output directory for this file
        file_output_dir = os.path.join(output_base_dir, 
                                       os.path.splitext(os.path.basename(midi_file))[0])
        
        result = process_midi_file(midi_file, output_dir=file_output_dir)
        results[midi_file] = result
        
        if result:
            print(f"✅ {os.path.basename(midi_file)} processed successfully")
        else:
            print(f"❌ {os.path.basename(midi_file)} failed to process")
    
    print(f"\n🎉 BATCH PROCESSING COMPLETE!")
    success_count = sum(1 for r in results.values() if r is not None)
    print(f"Successfully processed: {success_count}/{len(midi_files)} files")
    
    return results

# Example usage
if __name__ == "__main__":
    print("🎵 MIDI PIANO ROLL ML SYSTEM v2.0 🎵")
    print("Production-ready MIDI processing for machine learning")
    print("\nUsage examples:")
    print("1. Single file: process_midi_file('song.mid', output_dir='output/')")
    print("2. Batch processing: batch_process_midi_files(['song1.mid', 'song2.mid'])")
    print("\nPlace your MIDI files in the 'data/' directory")
    print("Processed files will be saved to 'output/' directory")
    print("Use slur_annotation_tool.py for slur matrix generation")
    
    # Check for sample MIDI files
    data_dir = "../data"
    if os.path.exists(data_dir):
        midi_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.mid')]
        if midi_files:
            print(f"\nFound {len(midi_files)} MIDI files in data/ directory:")
            for f in midi_files[:5]:  # Show first 5
                print(f"  - {f}")
            if len(midi_files) > 5:
                print(f"  ... and {len(midi_files) - 5} more")
        else:
            print(f"\nNo MIDI files found in {data_dir}")
    else:
        print(f"\nData directory not found: {data_dir}")
        print("Create it and add your MIDI files!")
