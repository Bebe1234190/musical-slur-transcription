#!/usr/bin/env python3
"""
Test script to compare original vs new pedal approaches
"""

import os
import sys
import torch
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.dirname(__file__))

from ml_data_pipeline import load_and_prepare_data, load_and_prepare_data_new_pedal, save_processed_data
from ml_transformer_model import MusicSlurTransformer
from train_with_stagnation import train_with_stagnation_monitoring

def test_pedal_approaches(base_filename="Beethoven_Piano_Sonata_No_10_Op_14_No_2_fQqNsTUvqCY_cut_mov_1", 
                         output_dir="output", 
                         window_ms=100):
    """
    Test both pedal approaches and compare results
    """
    print("🎵 PEDAL APPROACH COMPARISON TEST")
    print("=" * 60)
    print(f"📂 Base filename: {base_filename}")
    print(f"📁 Output directory: {output_dir}")
    print(f"⏱️  Pedal window: {window_ms}ms")
    print("=" * 60)
    
    # Test 1: Original Approach (6 features)
    print("\n🔵 TEST 1: ORIGINAL APPROACH (6 features)")
    print("-" * 50)
    
    try:
        inputs_orig, targets_orig, norm_params_orig, stats_orig = load_and_prepare_data(base_filename, output_dir)
        
        # Save original approach data
        orig_data_path = os.path.join(output_dir, f"{base_filename}_original_approach.pt")
        save_processed_data(inputs_orig, targets_orig, norm_params_orig, stats_orig, orig_data_path)
        
        print(f"✅ Original approach data prepared:")
        print(f"   Input shape: {inputs_orig.shape}")
        print(f"   Features: sustain_start, sustain_end")
        print(f"   Saved to: {orig_data_path}")
        
    except Exception as e:
        print(f"❌ Original approach failed: {e}")
        return
    
    # Test 2: New Approach (5 features)
    print("\n🟢 TEST 2: NEW APPROACH (5 features)")
    print("-" * 50)
    
    try:
        inputs_new, targets_new, norm_params_new, stats_new = load_and_prepare_data_new_pedal(base_filename, output_dir, window_ms)
        
        # Save new approach data
        new_data_path = os.path.join(output_dir, f"{base_filename}_new_approach.pt")
        save_processed_data(inputs_new, targets_new, norm_params_new, stats_new, new_data_path)
        
        print(f"✅ New approach data prepared:")
        print(f"   Input shape: {inputs_new.shape}")
        print(f"   Features: pedal_state (0=no_pedal, 1=beginning, 2=middle, 3=end)")
        print(f"   Saved to: {new_data_path}")
        
    except Exception as e:
        print(f"❌ New approach failed: {e}")
        return
    
    # Compare feature distributions
    print("\n📊 FEATURE COMPARISON")
    print("-" * 50)
    
    # Analyze pedal state distribution
    pedal_states = inputs_new[:, 4]
    unique_states, counts = np.unique(pedal_states, return_counts=True)
    
    print(f"Pedal state distribution (new approach):")
    state_names = {0.0: "No pedal", 33.333335876464844: "Beginning", 66.66667175292969: "Middle", 100.0: "End"}
    for state, count in zip(unique_states, counts):
        percentage = (count / len(pedal_states)) * 100
        state_name = state_names.get(state, f"Unknown ({state})")
        print(f"  {state:.1f} ({state_name}): {count} notes ({percentage:.1f}%)")
    
    # Analyze sustain values (original approach)
    sustain_start = inputs_orig[:, 4]
    sustain_end = inputs_orig[:, 5]
    
    print(f"\nSustain value distribution (original approach):")
    print(f"  Sustain start: {sustain_start.min():.0f}-{sustain_start.max():.0f}")
    print(f"  Sustain end: {sustain_end.min():.0f}-{sustain_end.max():.0f}")
    
    # Test training with both approaches
    print("\n🚀 TRAINING COMPARISON")
    print("-" * 50)
    
    # Train original approach
    print("\n🔵 Training Original Approach...")
    try:
        # Copy the original approach data to the expected filename
        import shutil
        expected_orig_path = os.path.join(output_dir, f"{base_filename}_processed_for_ml.pt")
        shutil.copy(orig_data_path, expected_orig_path)
        
        results_orig = train_with_stagnation_monitoring(
            base_filename=base_filename,
            output_dir=output_dir,
            epochs=500,  # Shorter for comparison
            learning_rate=0.001,
            stagnation_epochs=20,
            min_loss_change=1e-15,
            print_interval=50,
            device="cpu"
        )
        
        print(f"✅ Original approach training completed:")
        print(f"   Final accuracy: {results_orig['final_accuracy']:.4f}")
        print(f"   Final loss: {results_orig['final_loss']:.6f}")
        print(f"   Epochs: {results_orig['total_epochs']}")
        
    except Exception as e:
        print(f"❌ Original approach training failed: {e}")
        results_orig = None
    
    # Train new approach
    print("\n🟢 Training New Approach...")
    try:
        # Copy the new approach data to the expected filename
        expected_new_path = os.path.join(output_dir, f"{base_filename}_processed_for_ml.pt")
        shutil.copy(new_data_path, expected_new_path)
        
        results_new = train_with_stagnation_monitoring(
            base_filename=base_filename,
            output_dir=output_dir,
            epochs=500,  # Shorter for comparison
            learning_rate=0.001,
            stagnation_epochs=20,
            min_loss_change=1e-15,
            print_interval=50,
            device="cpu"
        )
        
        print(f"✅ New approach training completed:")
        print(f"   Final accuracy: {results_new['final_accuracy']:.4f}")
        print(f"   Final loss: {results_new['final_loss']:.6f}")
        print(f"   Epochs: {results_new['total_epochs']}")
        
    except Exception as e:
        print(f"❌ New approach training failed: {e}")
        results_new = None
    
    # Final comparison
    print("\n📈 FINAL COMPARISON")
    print("=" * 60)
    
    if results_orig and results_new:
        accuracy_diff = results_new['final_accuracy'] - results_orig['final_accuracy']
        loss_diff = results_new['final_loss'] - results_orig['final_loss']
        
        print(f"Original Approach (6 features):")
        print(f"  Accuracy: {results_orig['final_accuracy']:.4f}")
        print(f"  Loss: {results_orig['final_loss']:.6f}")
        
        print(f"\nNew Approach (5 features):")
        print(f"  Accuracy: {results_new['final_accuracy']:.4f}")
        print(f"  Loss: {results_new['final_loss']:.6f}")
        
        print(f"\nDifference (New - Original):")
        print(f"  Accuracy: {accuracy_diff:+.4f}")
        print(f"  Loss: {loss_diff:+.6f}")
        
        if accuracy_diff > 0:
            print(f"\n🎉 New approach is BETTER by {accuracy_diff:.4f} accuracy!")
        elif accuracy_diff < 0:
            print(f"\n📉 Original approach is better by {abs(accuracy_diff):.4f} accuracy")
        else:
            print(f"\n🤝 Both approaches perform equally!")
            
    else:
        print("❌ Could not complete comparison due to training failures")
    
    print("\n✅ Comparison test completed!")

if __name__ == "__main__":
    test_pedal_approaches()
