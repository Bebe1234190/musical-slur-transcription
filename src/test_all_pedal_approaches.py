#!/usr/bin/env python3
"""
Test script to compare all pedal approaches: Original (6 features), Window-based (5 features), and Simple (5 features)
"""

import os
import sys
import torch
import numpy as np
import shutil

# Add src directory to path
sys.path.insert(0, os.path.dirname(__file__))

from ml_data_pipeline import (
    load_and_prepare_data, 
    load_and_prepare_data_new_pedal, 
    load_and_prepare_data_simple_pedal,
    save_processed_data
)
from ml_transformer_model import MusicSlurTransformer
from train_with_stagnation import train_with_stagnation_monitoring

def test_all_pedal_approaches(base_filename="Beethoven_Piano_Sonata_No_10_Op_14_No_2_fQqNsTUvqCY_cut_mov_1", 
                             output_dir="output", 
                             window_ms=100):
    """
    Test all three pedal approaches and compare results
    """
    print("🎵 COMPREHENSIVE PEDAL APPROACH COMPARISON")
    print("=" * 70)
    print(f"📂 Base filename: {base_filename}")
    print(f"📁 Output directory: {output_dir}")
    print(f"⏱️  Window-based window: {window_ms}ms")
    print("=" * 70)
    
    results = {}
    
    # Test 1: Original Approach (6 features)
    print("\n🔵 TEST 1: ORIGINAL APPROACH (6 features)")
    print("-" * 60)
    
    try:
        inputs_orig, targets_orig, norm_params_orig, stats_orig = load_and_prepare_data(base_filename, output_dir)
        
        # Save original approach data
        orig_data_path = os.path.join(output_dir, f"{base_filename}_original_approach.pt")
        save_processed_data(inputs_orig, targets_orig, norm_params_orig, stats_orig, orig_data_path)
        
        print(f"✅ Original approach data prepared:")
        print(f"   Input shape: {inputs_orig.shape}")
        print(f"   Features: sustain_start, sustain_end")
        print(f"   Saved to: {orig_data_path}")
        
        results['original'] = {
            'inputs': inputs_orig,
            'targets': targets_orig,
            'stats': stats_orig,
            'data_path': orig_data_path
        }
        
    except Exception as e:
        print(f"❌ Original approach failed: {e}")
        results['original'] = None
    
    # Test 2: Window-based Approach (5 features)
    print("\n🟡 TEST 2: WINDOW-BASED APPROACH (5 features)")
    print("-" * 60)
    
    try:
        inputs_window, targets_window, norm_params_window, stats_window = load_and_prepare_data_new_pedal(base_filename, output_dir, window_ms)
        
        # Save window-based approach data
        window_data_path = os.path.join(output_dir, f"{base_filename}_window_approach.pt")
        save_processed_data(inputs_window, targets_window, norm_params_window, stats_window, window_data_path)
        
        print(f"✅ Window-based approach data prepared:")
        print(f"   Input shape: {inputs_window.shape}")
        print(f"   Features: pedal_state (window: {window_ms}ms)")
        print(f"   Saved to: {window_data_path}")
        
        results['window'] = {
            'inputs': inputs_window,
            'targets': targets_window,
            'stats': stats_window,
            'data_path': window_data_path
        }
        
    except Exception as e:
        print(f"❌ Window-based approach failed: {e}")
        results['window'] = None
    
    # Test 3: Simple Approach (5 features)
    print("\n🟢 TEST 3: SIMPLE APPROACH (5 features)")
    print("-" * 60)
    
    try:
        inputs_simple, targets_simple, norm_params_simple, stats_simple = load_and_prepare_data_simple_pedal(base_filename, output_dir)
        
        # Save simple approach data
        simple_data_path = os.path.join(output_dir, f"{base_filename}_simple_approach.pt")
        save_processed_data(inputs_simple, targets_simple, norm_params_simple, stats_simple, simple_data_path)
        
        print(f"✅ Simple approach data prepared:")
        print(f"   Input shape: {inputs_simple.shape}")
        print(f"   Features: pedal_state (first note after pedal on)")
        print(f"   Saved to: {simple_data_path}")
        
        results['simple'] = {
            'inputs': inputs_simple,
            'targets': targets_simple,
            'stats': stats_simple,
            'data_path': simple_data_path
        }
        
    except Exception as e:
        print(f"❌ Simple approach failed: {e}")
        results['simple'] = None
    
    # Compare feature distributions
    print("\n📊 FEATURE COMPARISON")
    print("-" * 60)
    
    for approach_name, approach_data in results.items():
        if approach_data is None:
            continue
            
        inputs = approach_data['inputs']
        print(f"\n{approach_name.upper()} APPROACH:")
        
        if approach_name == 'original':
            # Analyze sustain values (original approach)
            sustain_start = inputs[:, 4]
            sustain_end = inputs[:, 5]
            print(f"  Sustain start: {sustain_start.min():.0f}-{sustain_start.max():.0f}")
            print(f"  Sustain end: {sustain_end.min():.0f}-{sustain_end.max():.0f}")
        else:
            # Analyze pedal state distribution (new approaches)
            pedal_states = inputs[:, 4]
            unique_states, counts = np.unique(pedal_states, return_counts=True)
            
            state_names = {0.0: "No pedal", 33.333335876464844: "Beginning", 66.66667175292969: "Middle", 100.0: "End"}
            for state, count in zip(unique_states, counts):
                percentage = (count / len(pedal_states)) * 100
                state_name = state_names.get(state, f"Unknown ({state})")
                print(f"  {state:.1f} ({state_name}): {count} notes ({percentage:.1f}%)")
    
    # Test training with all approaches
    print("\n🚀 TRAINING COMPARISON")
    print("-" * 60)
    
    training_results = {}
    
    for approach_name, approach_data in results.items():
        if approach_data is None:
            continue
            
        print(f"\n{'🔵' if approach_name == 'original' else '🟡' if approach_name == 'window' else '🟢'} Training {approach_name.upper()} Approach...")
        
        try:
            # Copy the approach data to the expected filename
            expected_path = os.path.join(output_dir, f"{base_filename}_processed_for_ml.pt")
            shutil.copy(approach_data['data_path'], expected_path)
            
            train_results = train_with_stagnation_monitoring(
                base_filename=base_filename,
                output_dir=output_dir,
                epochs=500,  # Shorter for comparison
                learning_rate=0.001,
                stagnation_epochs=20,
                min_loss_change=1e-15,
                print_interval=50,
                device="cpu"
            )
            
            print(f"✅ {approach_name} approach training completed:")
            print(f"   Final accuracy: {train_results['final_accuracy']:.4f}")
            print(f"   Final loss: {train_results['final_loss']:.6f}")
            print(f"   Epochs: {train_results['total_epochs']}")
            
            training_results[approach_name] = train_results
            
        except Exception as e:
            print(f"❌ {approach_name} approach training failed: {e}")
            training_results[approach_name] = None
    
    # Final comparison
    print("\n📈 FINAL COMPARISON")
    print("=" * 70)
    
    if len(training_results) > 1:
        print(f"{'Approach':<15} {'Accuracy':<10} {'Loss':<12} {'Epochs':<8}")
        print("-" * 50)
        
        for approach_name, train_results in training_results.items():
            if train_results is not None:
                print(f"{approach_name.capitalize():<15} {train_results['final_accuracy']:<10.4f} {train_results['final_loss']:<12.6f} {train_results['total_epochs']:<8}")
        
        # Find best approach
        valid_results = {k: v for k, v in training_results.items() if v is not None}
        if valid_results:
            best_approach = max(valid_results.keys(), key=lambda k: valid_results[k]['final_accuracy'])
            best_accuracy = valid_results[best_approach]['final_accuracy']
            
            print(f"\n🏆 BEST APPROACH: {best_approach.upper()}")
            print(f"   Accuracy: {best_accuracy:.4f}")
            
            # Compare with others
            for approach_name, train_results in valid_results.items():
                if approach_name != best_approach:
                    diff = train_results['final_accuracy'] - best_accuracy
                    print(f"   vs {approach_name}: {diff:+.4f}")
                    
    else:
        print("❌ Could not complete comparison due to training failures")
    
    print("\n✅ Comprehensive comparison test completed!")
    
    return results, training_results

if __name__ == "__main__":
    test_all_pedal_approaches()
