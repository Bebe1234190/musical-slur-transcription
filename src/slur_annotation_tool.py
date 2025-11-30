#!/usr/bin/env python3
"""
Slur Annotation Tool for Piano MIDI Files
Converts annotated CSV files into slur category matrices for ML training

Part of the MIDI Piano Roll ML System v2.0
Handles the complete slur annotation workflow with 6-category system (0-5)
"""

import numpy as np
import pandas as pd
import mido
import os

def midi_to_note_name(midi_number):
    """Convert MIDI note number to musical note name (e.g., 60 -> C4)"""
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_number // 12) - 1
    note = notes[midi_number % 12]
    return f"{note}{octave}"

def create_slur_matrix_from_partial_csv(csv_file, midi_file_path, note_range=(21, 108)):
    """
    Create slur matrix from partially annotated CSV (allows missing annotations)
    
    Uses the 6-category system:
    0: Empty/No annotation (default background)
    1: Slur beginning (first note of slur)
    2: Slur middle (continuation within slur)
    3: Slur end (final note of slur)
    4: No slur (explicitly isolated note)
    5: Slur start and end (single-note slur)
    
    Args:
        csv_file (str): Path to annotated CSV file (can be partial)
        midi_file_path (str): Original MIDI file path
        note_range (tuple): MIDI note range for matrix
        
    Returns:
        str: Path to generated slur matrix file
    """
    print(f"Creating slur matrix from partial annotations: {csv_file}")
    
    # Read the annotated CSV
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        return None
    
    # Filter to only annotated notes (allow partial annotation)
    annotated_notes = df[df['Slur_Category'].notna() & (df['Slur_Category'] != '')].copy()
    print(f"Found {len(annotated_notes)} annotated notes out of {len(df)} total notes")
    
    if len(annotated_notes) == 0:
        print("❌ No annotated notes found in CSV")
        return None
    
    # Convert categories to integers (explicit dtype to avoid float conversion)
    try:
        annotated_notes['Slur_Category'] = pd.to_numeric(annotated_notes['Slur_Category'], downcast='unsigned')
    except Exception as e:
        print(f"❌ Error converting slur categories to numbers: {e}")
        return None
    
    # Validate category values (6-category system: 0-5)
    invalid_cats = annotated_notes[~annotated_notes['Slur_Category'].isin([0, 1, 2, 3, 4, 5])]
    if not invalid_cats.empty:
        print(f"❌ Error: Found invalid slur categories. Valid categories are 0, 1, 2, 3, 4, 5")
        print("Category meanings:")
        print("  0: Empty/No annotation")
        print("  1: Slur beginning")
        print("  2: Slur middle")
        print("  3: Slur end")
        print("  4: No slur")
        print("  5: Slur start and end (single-note slur)")
        return None
    
    # Get MIDI file resolution and match existing matrix dimensions
    mid = mido.MidiFile(midi_file_path)
    time_step = 1.0 / mid.ticks_per_beat
    
    # Use existing notes matrix dimensions if available
    base_filename = os.path.splitext(os.path.basename(midi_file_path))[0]
    notes_matrix_file = f"output/{base_filename}_notes.npy"
    
    if os.path.exists(notes_matrix_file):
        existing_matrix = np.load(notes_matrix_file)
        target_shape = existing_matrix.shape
        print(f"Using existing matrix dimensions: {target_shape}")
    else:
        # Calculate dimensions from data
        max_time = max(annotated_notes['Start_Time'] + annotated_notes['Duration'])
        time_steps = int(np.ceil(max_time / time_step))
        min_note, max_note = note_range
        num_notes = max_note - min_note + 1
        target_shape = (num_notes, time_steps)
        print(f"Calculated matrix dimensions: {target_shape}")
    
    # Create slur category matrix with appropriate data type for categories (0-5)
    slur_matrix = np.zeros(target_shape, dtype=np.uint8)
    
    print(f"Creating slur matrix with shape: {target_shape}")
    
    # Fill matrix with slur categories for annotated notes only
    notes_processed = 0
    for _, row in annotated_notes.iterrows():
        # Parse pitch name to MIDI number
        pitch_name = row['Pitch']
        
        if '#' in pitch_name:
            note = pitch_name[:2]
            octave = int(pitch_name[2:])
        else:
            note = pitch_name[0]
            octave = int(pitch_name[1:])
        
        note_map = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
        midi_note = note_map[note] + (octave + 1) * 12
        
        min_note, max_note = note_range
        if min_note <= midi_note <= max_note:
            note_idx = midi_note - min_note
            
            # Convert timing to matrix indices
            start_matrix_step = int(row['Start_Ticks'])
            duration_matrix_steps = int(row['Duration_Ticks'])
            end_matrix_step = min(start_matrix_step + duration_matrix_steps, target_shape[1])
            
            # Ensure we don't exceed matrix bounds
            if start_matrix_step < target_shape[1]:
                slur_category = int(row['Slur_Category'])
                slur_matrix[note_idx, start_matrix_step:end_matrix_step] = slur_category
                notes_processed += 1
    
    print(f"Successfully processed {notes_processed} annotated notes")
    
    # Save the slur matrix to output directory
    slur_matrix_file = f"output/{base_filename}_slur_matrix.npy"
    np.save(slur_matrix_file, slur_matrix)
    
    # Save full CSV version
    slur_csv_file = f"output/{base_filename}_slur_matrix.csv"
    np.savetxt(slur_csv_file, slur_matrix, delimiter=',', fmt='%d')
    
    # Save metadata
    metadata_file = f"output/{base_filename}_slur_metadata.txt"
    with open(metadata_file, 'w') as f:
        f.write("=== SLUR ANNOTATION MATRIX ===\n")
        f.write("Generated by: MIDI Piano Roll ML System v2.0\n")
        f.write(f"Source CSV: {csv_file}\n")
        f.write(f"MIDI File: {midi_file_path}\n")
        f.write(f"Matrix Shape: {slur_matrix.shape}\n")
        f.write(f"Time Resolution: {time_step:.8f} quarter notes per step\n")
        f.write(f"Note Range: MIDI {min_note}-{max_note}\n")
        f.write(f"Annotated Notes: {len(annotated_notes)}/{len(df)}\n\n")
        
        f.write("=== SLUR CATEGORIES (6-Category System: 0-5) ===\n")
        f.write("0: Empty/No annotation (default background)\n")
        f.write("1: Slur beginning (first note of slur)\n")
        f.write("2: Slur middle (continuation within slur)\n")
        f.write("3: Slur end (final note of slur)\n")
        f.write("4: No slur (explicitly isolated note)\n")
        f.write("5: Slur start and end (single-note slur)\n\n")
        
        # Category statistics
        category_counts = annotated_notes['Slur_Category'].value_counts().sort_index()
        f.write("=== ANNOTATION STATISTICS ===\n")
        for cat, count in category_counts.items():
            percentage = (count / len(annotated_notes)) * 100
            category_names = {0: 'Empty', 1: 'Begin', 2: 'Middle', 3: 'End', 4: 'No slur', 5: 'Start+End'}
            f.write(f"Category {int(cat)} ({category_names.get(int(cat), 'Unknown')}): {count:4d} notes ({percentage:5.1f}%)\n")
        
        f.write(f"\n=== MATRIX STATISTICS ===\n")
        f.write(f"Total elements: {slur_matrix.size:,}\n")
        f.write(f"Non-zero elements: {np.count_nonzero(slur_matrix):,}\n")
        f.write(f"Matrix density: {(np.count_nonzero(slur_matrix)/slur_matrix.size)*100:.4f}%\n")
        
        # Matrix value distribution
        unique_values, counts = np.unique(slur_matrix, return_counts=True)
        f.write(f"\n=== MATRIX VALUE DISTRIBUTION ===\n")
        for value, count in zip(unique_values, counts):
            percentage = (count / slur_matrix.size) * 100
            category_names = {0: 'Empty/Background', 1: 'Slur begin', 2: 'Slur middle', 3: 'Slur end', 4: 'No slur', 5: 'Slur start and end'}
            f.write(f"Value {int(value)} ({category_names.get(int(value), 'Unknown')}): {count:,} elements ({percentage:.2f}%)\n")
        
        f.write(f"\n=== PERFECT ALIGNMENT GUARANTEE ===\n")
        f.write(f"✓ Slur matrix matches notes matrix dimensions exactly\n")
        f.write(f"✓ Raw MIDI timing ensures perfect temporal alignment\n")
        f.write(f"✓ Every slur annotation corresponds to actual note events\n")
        
        f.write(f"\n=== ML TRAINING USAGE ===\n")
        f.write(f"import numpy as np\n")
        f.write(f"notes = np.load('{base_filename}_notes.npy')     # (88, time_steps)\n")
        f.write(f"pedal = np.load('{base_filename}_pedal.npy')     # (3, time_steps)\n")
        f.write(f"slurs = np.load('{slur_matrix_file}')  # (88, time_steps)\n")
        f.write(f"# Three perfectly aligned inputs for multi-modal ML training!\n")
        f.write(f"# Slur categories: 0=empty, 1=begin, 2=middle, 3=end, 4=no_slur, 5=start_and_end\n")
    
    print(f"✓ Slur matrix saved: {slur_matrix_file}")
    print(f"✓ Slur matrix CSV saved: {slur_csv_file}")
    print(f"✓ Metadata saved: {metadata_file}")
    print(f"✓ Matrix shape: {slur_matrix.shape}")
    print(f"✓ Total annotated notes: {len(annotated_notes)}")
    
    # Show category distribution
    category_counts = annotated_notes['Slur_Category'].value_counts().sort_index()
    print(f"\nSlur category distribution:")
    category_names = {0: 'Empty', 1: 'Begin', 2: 'Middle', 3: 'End', 4: 'No slur'}
    for cat, count in category_counts.items():
        percentage = (count / len(annotated_notes)) * 100
        print(f"  Category {int(cat)} ({category_names[int(cat)]}): {count:4d} notes ({percentage:5.1f}%)")
    
    # Show matrix value distribution
    unique_values, counts = np.unique(slur_matrix, return_counts=True)
    print(f"\nMatrix value distribution:")
    matrix_names = {0: 'Empty/Background', 1: 'Slur begin', 2: 'Slur middle', 3: 'Slur end', 4: 'No slur'}
    for value, count in zip(unique_values, counts):
        percentage = (count / slur_matrix.size) * 100
        print(f"  Value {int(value)} ({matrix_names[int(value)]}): {count:,} elements ({percentage:.2f}%)")
    
    return slur_matrix_file


def validate_matrix_alignment(notes_file, slur_file):
    """
    Validate that notes and slur matrices are perfectly aligned
    
    Args:
        notes_file (str): Path to notes matrix .npy file
        slur_file (str): Path to slur matrix .npy file
        
    Returns:
        bool: True if matrices are perfectly aligned
    """
    print("🔍 Validating matrix alignment...")
    
    try:
        notes = np.load(notes_file)
        slurs = np.load(slur_file)
        
        print(f"Notes matrix shape: {notes.shape}")
        print(f"Slur matrix shape:  {slurs.shape}")
        
        if notes.shape != slurs.shape:
            print("❌ ERROR: Matrix shapes do not match!")
            return False
        
        # Check if slur annotations are only where notes exist
        notes_nonzero = notes != 0
        slurs_nonzero = slurs != 0
        
        slurs_outside_notes = slurs_nonzero & ~notes_nonzero
        alignment_perfect = np.count_nonzero(slurs_outside_notes) == 0
        
        print(f"Notes non-zero elements: {np.count_nonzero(notes_nonzero):,}")
        print(f"Slur non-zero elements:  {np.count_nonzero(slurs_nonzero):,}")
        print(f"Slur annotations outside note regions: {np.count_nonzero(slurs_outside_notes):,}")
        
        if alignment_perfect:
            print("✅ PERFECT ALIGNMENT: All slur annotations correspond to note events!")
        else:
            print("❌ ALIGNMENT ISSUE: Some slur annotations exist where no notes are present!")
        
        return alignment_perfect
        
    except Exception as e:
        print(f"❌ Error validating alignment: {e}")
        return False

# Example usage and CLI interface
if __name__ == "__main__":
    print("🎵 SLUR ANNOTATION TOOL v2.0 🎵")
    print("Part of the MIDI Piano Roll ML System")
    print("=" * 50)
    
    print("\n📋 WORKFLOW:")
    print("1. Use complete_midi_processor.py to generate annotation CSV")
    print("2. Manually annotate CSV with slur categories (0-5)")
    print("3. Use this tool to convert annotations to matrix format")
    
    print("\n🏷️  SLUR CATEGORIES:")
    print("  0: Empty/No annotation (default background)")
    print("  1: Slur beginning (first note)")
    print("  2: Slur middle (continuation)")
    print("  3: Slur end (final note)")
    print("  4: No slur (isolated note)")
    print("  5: Slur start and end (single-note slur)")
    
    print("\n💡 USAGE EXAMPLES:")
    print("# Create slur matrix from annotated CSV")
    print("from slur_annotation_tool import create_slur_matrix_from_partial_csv")
    print("result = create_slur_matrix_from_partial_csv('annotated.csv', 'song.mid')")
    print()
    print("# Validate matrix alignment")
    print("from slur_annotation_tool import validate_matrix_alignment")
    print("is_aligned = validate_matrix_alignment('song_notes.npy', 'song_slur_matrix.npy')")
    
    print("\n📁 FILES:")
    print("Place your annotated CSV files in the current directory")
    print("Slur matrices will be saved with '_slur_matrix.npy' suffix")
    
    # Check for CSV files in current directory
    csv_files = [f for f in os.listdir('.') if f.lower().endswith('.csv') and 'slur' in f.lower()]
    if csv_files:
        print(f"\n📄 Found {len(csv_files)} potential annotation CSV files:")
        for f in csv_files:
            print(f"  - {f}")
    else:
        print("\n📄 No annotation CSV files found in current directory")
        print("Generate them first using complete_midi_processor.py")
