# ML Architecture Discussion Session - MIDI Piano Roll ML System

**Date**: December 2024  
**System Version**: v2.0  
**Session Focus**: Machine Learning Model Architecture Design & Implementation Strategy

## Session Overview

This session transitioned from data optimization to ML model architecture design. The focus was on developing a transformer-based approach for predicting slur annotations from musical note sequences, moving away from the original CNN matrix-based approach.

## Key Architectural Decisions

### 1. Model Type Selection: Transformer vs CNN

**Initial Consideration**: CNN for matrix-based input
- Strengths: Temporal pattern recognition, multi-scale features, translation invariance
- Concerns: Class imbalance (96% background), computational overhead

**Final Decision**: Transformer for sequence-based input
- **User Insight**: "The main indicator will be that the ending of the previous note happens after the beginning of the next note, so the notes are musically connected"
- Better suited for capturing note-to-note relationships
- More efficient data representation

### 2. Data Representation Shift: Matrix → Sequence

**From**: Piano roll matrices (88 × 295,593 elements)
- Notes matrix: Binary/velocity values per piano key per time step
- Pedal matrix: 3 pedal types per time step
- Slur matrix: Category values per piano key per time step

**To**: Note sequence representation
- Input: 4 features per note [Start_Time, Duration, MIDI_Pitch, Velocity]
- Output: 4 binary columns per note [Slur_Start, Slur_Middle, Slur_End, No_Slur]
- Massive size reduction: ~1M matrix elements → ~2,640 note sequences

### 3. Problem Reframing: Note Classification → Temporal Segmentation

**Original Understanding**: Which specific notes belong to which slurs
**Actual Goal**: When do slurs start/middle/end in time
- **User Clarification**: "My goal is not an end-to-end model. I just want the slur data relative to time. I don't need to know which slur falls on which note."
- Simplified from polyphonic voice assignment to temporal pattern recognition
- Focus on musical phrase boundaries rather than individual note labeling

## Technical Architecture Evolution

### Input Representation Design

**Version 1**: Raw note features only
```
[Start_Time, Duration, MIDI_Pitch, Velocity]
```

**Version 2**: Including pedal information as mixed tokens
```
Note token:  [Start_Time, Duration, MIDI_Pitch, Velocity, 1, 0, 0]  # Is_Note=1
Pedal token: [Start_Time, 0, 0, Pedal_Value, 0, 1, Pedal_Type]     # Is_Pedal=1
```

**Version 3**: Explicit token type encoding (current direction)
```
[Start_Time, Duration, MIDI_Pitch, Velocity, Is_Note, Is_Sustain_Event, Sustain_Value]
```

### Key Design Challenges Addressed

#### 1. Temporal Precision Concerns
**Initial Concern**: Loss of exact timing relationships in sequence representation
**Resolution**: Start_Time and Duration preserve all temporal information needed for overlap calculations

#### 2. Polyphonic Complexity (Simultaneity)
**Challenge**: How to represent simultaneous notes in a sequence
**User Solution**: 
- Most "simultaneous" notes differ by milliseconds in real performance
- Order by time, then by pitch for true simultaneity
- Transformer learns that rapid sequences represent intended chords

#### 3. Pedal Information Integration
**Options Considered**:
1. Pedal state features (simple, loses timing)
2. Mixed token sequences (complex, breaks 1:1 input/output)
3. Explicit token encoding (compromise solution)

**Current Issue**: Input/output length mismatch with mixed tokens
- Input: Notes + Pedal events
- Output: Only note predictions needed
- Need to resolve 1:1 relationship

## Musical Insights Informing Architecture

### Note Overlap as Primary Feature
**Core Musical Principle**: Connected notes overlap in time (legato playing)
- Slurred notes: Previous note ending > next note beginning
- Non-slurred notes: Gap between note ending and next beginning
- This physical/temporal relationship is the primary slur indicator

### Performance Reality vs. Musical Notation
**Micro-timing Recognition**: Real piano performance has subtle timing variations
- "Simultaneous" chords often have 1-5ms timing differences
- These variations carry musical meaning (voicing, emphasis)
- Sequence representation captures performance nuance better than quantized matrices

### Beethoven Sonata Domain Constraints
**Training Data Requirements**: For Beethoven-specific model
- Current: 1 annotated sonata movement (~1M note events)
- Minimum viable: 8-12 movements for training + validation + test
- Robust dataset: 20-30 movements total
- Domain constraints reduce data requirements vs. general music model

## Data Efficiency Analysis

### Current Annotated Dataset
- **Scale**: 1,025,164 non-zero note events
- **Distribution**: 
  - Slur Begin (1): 27.7% of notes
  - Slur Middle (2): 31.7% of notes  
  - Slur End (3): 17.7% of notes
  - No Slur (4): 23.0% of notes
- **Quality**: Dense annotation (every note labeled), human expert annotation

### Size Reduction Benefits
**Matrix approach**: 26M+ elements (96% sparse)
**Sequence approach**: ~2,640 note events
- 10,000x size reduction
- Eliminates sparse background data
- Focuses model on musical events only

## Remaining Technical Decisions

### 1. Pedal Information Integration (Unresolved)
**Current dilemma**: How to include sustain pedal without breaking sequence structure
**Options under consideration**:
- Mixed tokens with filtering (complex training)
- Pedal state encoding per note (loses exact timing)
- Separate pedal context features

### 2. Feature Engineering Strategy
**Decision**: Let transformer learn relationships from raw features
- No explicit overlap/gap calculations
- No inter-onset interval features
- Trust self-attention to discover temporal patterns
- Simpler, more flexible approach

### 3. Model Training Considerations
**Sequence preprocessing**:
- Chronological ordering by start time
- Pitch-based ordering for true simultaneity
- Normalization of time scales (start times vs. durations)

**Loss function design**:
- Handle class distribution (not 96% sparse like matrices)
- Ensure valid slur sequences (begin→middle→end logic)
- Focus on transition boundaries

## Session Workflow

1. **Architecture Selection**: Evaluated CNN vs. Transformer approaches
2. **Data Representation**: Shifted from matrices to sequences
3. **Problem Reframing**: Clarified temporal segmentation vs. note classification
4. **Feature Design**: Evolved from raw notes to explicit token encoding
5. **Pedal Integration**: Explored multiple approaches for pedal information
6. **Simultaneity Handling**: Addressed polyphonic representation challenges
7. **Efficiency Analysis**: Quantified benefits of sequence approach

## Next Steps & Open Questions

### Implementation Priorities
1. **Resolve pedal integration approach** (blocking issue)
2. **Implement sequence preprocessing pipeline**
3. **Design transformer architecture** (attention heads, layers, embedding)
4. **Create training pipeline** with proper validation split
5. **Implement evaluation metrics** for musical meaningfulness

### Research Questions
- **Pedal timing importance**: How crucial is exact pedal change timing vs. state?
- **Context window size**: How much musical history needed for slur decisions?
- **Transfer learning**: Can model generalize from Beethoven to other composers?
- **Evaluation metrics**: How to measure musical quality beyond classification accuracy?

## Key Insights from Discussion

### Musical Domain Knowledge
- Note overlap is the primary physical indicator of slurs
- Performance timing contains more information than notation timing
- Sustain pedal significantly affects slur perception
- Classical repertoire has predictable phrase structures

### Technical Architecture Benefits
- Sequence representation matches musical event structure
- Transformer self-attention naturally models note relationships
- Dramatic size reduction enables more efficient training
- Problem simplification (temporal vs. polyphonic) reduces complexity

### Pragmatic Design Decisions
- Focus on commonly used features (sustain pedal only)
- Accept rare edge cases (true simultaneity) with simple solutions
- Prioritize musical realism over mathematical perfection
- Trust model to learn complex relationships from simple features

## Conclusion

This session successfully evolved the approach from a matrix-based CNN to a sequence-based transformer architecture. The key breakthrough was reframing the problem from individual note classification to temporal slur boundary detection, which dramatically simplifies the task while maintaining musical validity.

The architecture discussion revealed deep musical insights about performance timing and note relationships that inform both model design and feature selection. The focus on Beethoven sonatas provides a constrained domain that should enable effective learning with moderate amounts of training data.

**Current Status**: Architecture conceptually designed, pending resolution of pedal integration approach before implementation begins.

---

**Files Referenced This Session**:
- Previous optimization work: `Matrix_Optimization_Session.md`
- Current annotated data: 1,025,164 perfectly aligned note/slur pairs
- Target implementation: Transformer-based sequence-to-sequence model
