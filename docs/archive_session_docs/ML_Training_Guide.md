# Machine Learning Training Guide

## 🎯 Overview

This guide demonstrates how to use the generated matrices for machine learning training. The system produces three perfectly aligned matrices that can be used as multi-modal inputs for various ML tasks.

## 📊 Data Structure

### Matrix Specifications
```python
# Load your processed data
import numpy as np

notes = np.load('output/song_notes.npy')      # Shape: (88, time_steps)
pedal = np.load('output/song_pedal.npy')      # Shape: (3, time_steps)  
slurs = np.load('output/song_slur_matrix.npy') # Shape: (88, time_steps)

# Perfect temporal alignment guaranteed
assert notes.shape[1] == pedal.shape[1] == slurs.shape[1]
```

### Data Interpretation
- **Notes Matrix**: MIDI velocity values (0-127) for 88 piano keys
- **Pedal Matrix**: Pedal states (0-127) for [sustain, sostenuto, soft]
- **Slur Matrix**: Annotation categories (0-4) for musical phrasing

## 🧠 ML Task Examples

### 1. Multi-Modal Sequence Prediction
```python
import torch
import torch.nn as nn

class MultiModalPianoModel(nn.Module):
    def __init__(self, note_dim=88, pedal_dim=3, slur_dim=88, hidden_dim=256):
        super().__init__()
        
        # Separate encoders for each modality
        self.note_encoder = nn.LSTM(note_dim, hidden_dim, batch_first=True)
        self.pedal_encoder = nn.LSTM(pedal_dim, hidden_dim, batch_first=True)
        self.slur_encoder = nn.LSTM(slur_dim, hidden_dim, batch_first=True)
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_dim * 3, hidden_dim)
        self.output = nn.Linear(hidden_dim, note_dim)
        
    def forward(self, notes, pedal, slurs):
        # Encode each modality
        note_out, _ = self.note_encoder(notes.transpose(1, 2))
        pedal_out, _ = self.pedal_encoder(pedal.transpose(1, 2))
        slur_out, _ = self.slur_encoder(slurs.transpose(1, 2))
        
        # Combine features
        combined = torch.cat([note_out, pedal_out, slur_out], dim=-1)
        fused = self.fusion(combined)
        
        return self.output(fused)

# Usage
model = MultiModalPianoModel()
prediction = model(notes_tensor, pedal_tensor, slurs_tensor)
```

### 2. Slur Detection and Classification
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def prepare_slur_features(notes, pedal, slurs, window_size=10):
    """
    Create features for slur classification
    """
    features = []
    labels = []
    
    for i in range(window_size, notes.shape[1] - window_size):
        # Context window around current time
        note_context = notes[:, i-window_size:i+window_size+1].flatten()
        pedal_context = pedal[:, i-window_size:i+window_size+1].flatten()
        
        # Current note activity
        current_notes = notes[:, i]
        active_notes = np.where(current_notes > 0)[0]
        
        for note_idx in active_notes:
            # Features: note context + pedal context + note characteristics
            feature_vector = np.concatenate([
                note_context,
                pedal_context,
                [current_notes[note_idx], note_idx, i]  # velocity, pitch, time
            ])
            
            features.append(feature_vector)
            labels.append(slurs[note_idx, i])
    
    return np.array(features), np.array(labels)

# Train classifier
features, labels = prepare_slur_features(notes, pedal, slurs)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train, y_train)

accuracy = classifier.score(X_test, y_test)
print(f"Slur classification accuracy: {accuracy:.3f}")
```

### 3. Expression Transfer Learning
```python
import tensorflow as tf

def create_expression_model():
    """
    Model for learning expressive performance from pedal + slur data
    """
    # Input layers
    notes_input = tf.keras.layers.Input(shape=(None, 88), name='notes')
    pedal_input = tf.keras.layers.Input(shape=(None, 3), name='pedal')
    slurs_input = tf.keras.layers.Input(shape=(None, 88), name='slurs')
    
    # Process each input stream
    note_lstm = tf.keras.layers.LSTM(128, return_sequences=True)(notes_input)
    pedal_lstm = tf.keras.layers.LSTM(64, return_sequences=True)(pedal_input)
    slur_lstm = tf.keras.layers.LSTM(128, return_sequences=True)(slurs_input)
    
    # Attention mechanism for expression
    expression_context = tf.keras.layers.Concatenate()([pedal_lstm, slur_lstm])
    attention = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=64)
    attended_expression = attention(note_lstm, expression_context)
    
    # Output expressive notes
    output = tf.keras.layers.Dense(88, activation='sigmoid')(attended_expression)
    
    model = tf.keras.Model(
        inputs=[notes_input, pedal_input, slurs_input],
        outputs=output
    )
    
    return model

model = create_expression_model()
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```

## 🔄 Data Preprocessing

### Sequence Segmentation
```python
def create_sequences(notes, pedal, slurs, sequence_length=128, hop_size=64):
    """
    Create training sequences from full matrices
    """
    sequences = []
    
    for start in range(0, notes.shape[1] - sequence_length, hop_size):
        end = start + sequence_length
        
        sequence = {
            'notes': notes[:, start:end],
            'pedal': pedal[:, start:end], 
            'slurs': slurs[:, start:end],
            'start_time': start,
            'end_time': end
        }
        sequences.append(sequence)
    
    return sequences

sequences = create_sequences(notes, pedal, slurs)
print(f"Created {len(sequences)} training sequences")
```

### Normalization and Scaling
```python
def normalize_data(notes, pedal, slurs):
    """
    Normalize data for training
    """
    # Notes: scale velocity to [0, 1]
    notes_norm = notes / 127.0
    
    # Pedal: scale to [0, 1]  
    pedal_norm = pedal / 127.0
    
    # Slurs: one-hot encode categories
    slurs_onehot = tf.keras.utils.to_categorical(slurs, num_classes=5)
    
    return notes_norm, pedal_norm, slurs_onehot
```

## 📈 Evaluation Metrics

### Musical Metrics
```python
def evaluate_musical_accuracy(predicted_notes, target_notes, threshold=0.5):
    """
    Musical evaluation metrics
    """
    # Convert to binary
    pred_binary = (predicted_notes > threshold).astype(int)
    target_binary = (target_notes > threshold).astype(int)
    
    # Note-level accuracy
    note_accuracy = np.mean(pred_binary == target_binary)
    
    # Pitch-wise F1 scores
    from sklearn.metrics import f1_score
    pitch_f1_scores = []
    
    for pitch in range(88):
        if np.sum(target_binary[pitch]) > 0:  # Only evaluate active pitches
            f1 = f1_score(target_binary[pitch], pred_binary[pitch])
            pitch_f1_scores.append(f1)
    
    avg_pitch_f1 = np.mean(pitch_f1_scores)
    
    return {
        'note_accuracy': note_accuracy,
        'average_pitch_f1': avg_pitch_f1,
        'active_pitches': len(pitch_f1_scores)
    }
```

### Temporal Coherence
```python
def evaluate_temporal_coherence(predicted_slurs, target_slurs):
    """
    Evaluate slur prediction temporal coherence
    """
    # Check for valid slur sequences (1 -> 2 -> 3)
    valid_sequences = 0
    total_sequences = 0
    
    for pitch in range(88):
        pitch_slurs = predicted_slurs[pitch]
        
        # Find slur beginnings
        slur_starts = np.where(pitch_slurs == 1)[0]
        
        for start in slur_starts:
            total_sequences += 1
            
            # Check if followed by valid sequence
            current_pos = start + 1
            expecting = 2  # middle or end
            
            while current_pos < len(pitch_slurs):
                current_val = pitch_slurs[current_pos]
                
                if current_val == 0 or current_val == 4:  # End of slur
                    break
                elif current_val == 3:  # Proper end
                    valid_sequences += 1
                    break
                elif current_val != expecting and current_val != 3:
                    break  # Invalid sequence
                
                current_pos += 1
    
    return valid_sequences / total_sequences if total_sequences > 0 else 0
```

## 🎼 Musical Applications

### 1. Automatic Slur Detection
Train models to automatically detect musical phrasing in piano performances.

### 2. Expression Synthesis
Learn to add pedaling and phrasing to mechanical MIDI playback.

### 3. Performance Analysis
Study differences in interpretation between different performers.

### 4. Music Generation
Generate coherent piano performances with proper phrasing and expression.

### 5. Style Transfer
Transfer expressive characteristics between different musical pieces.

## 💡 Training Tips

### Data Augmentation
```python
def augment_data(notes, pedal, slurs):
    """
    Simple data augmentation techniques
    """
    # Transpose (shift pitch)
    shift = np.random.randint(-6, 7)  # ±6 semitones
    if shift != 0:
        notes_aug = np.roll(notes, shift, axis=0)
        slurs_aug = np.roll(slurs, shift, axis=0)
    else:
        notes_aug, slurs_aug = notes, slurs
    
    # Time stretching (simple decimation/interpolation)
    stretch_factor = np.random.uniform(0.9, 1.1)
    if stretch_factor != 1.0:
        from scipy import signal
        new_length = int(notes.shape[1] * stretch_factor)
        notes_aug = signal.resample(notes_aug, new_length, axis=1)
        pedal_aug = signal.resample(pedal, new_length, axis=1)
        slurs_aug = signal.resample(slurs_aug, new_length, axis=1)
    
    return notes_aug, pedal_aug, slurs_aug
```

### Handling Class Imbalance
```python
from sklearn.utils.class_weight import compute_class_weight

# For slur classification
slur_labels = slurs.flatten()
classes = np.unique(slur_labels)
class_weights = compute_class_weight('balanced', classes=classes, y=slur_labels)
class_weight_dict = dict(zip(classes, class_weights))

print(f"Class weights for balanced training: {class_weight_dict}")
```

## 🔗 Integration with Existing Libraries

### Music21 Integration
```python
from music21 import stream, note, meter, tempo

def convert_prediction_to_music21(predicted_notes, time_step, tempo_bpm=120):
    """
    Convert model predictions back to Music21 format
    """
    score = stream.Score()
    score.append(tempo.TempoIndication(number=tempo_bpm))
    score.append(meter.TimeSignature('4/4'))
    
    part = stream.Part()
    
    # Convert to notes
    for t in range(predicted_notes.shape[1]):
        active_pitches = np.where(predicted_notes[:, t] > 0.5)[0]
        
        if len(active_pitches) > 0:
            offset_quarters = t * time_step
            
            if len(active_pitches) == 1:
                # Single note
                midi_note = active_pitches[0] + 21
                n = note.Note(midi=midi_note)
                n.offset = offset_quarters
                n.quarterLength = time_step
                part.append(n)
            else:
                # Chord
                chord_notes = [note.Note(midi=p + 21) for p in active_pitches]
                c = chord.Chord(chord_notes)
                c.offset = offset_quarters
                c.quarterLength = time_step
                part.append(c)
    
    score.append(part)
    return score
```

This guide provides a comprehensive starting point for ML research using the generated matrices. The perfect temporal alignment ensures that multi-modal learning approaches will work correctly without timing-related artifacts.

