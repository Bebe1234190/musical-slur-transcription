#!/usr/bin/env python3
"""
Training Experiments Script
Runs multiple training configurations to find optimal settings

Part of the MIDI Piano Roll ML System v2.0
"""

import sys
import os
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from train_with_stagnation import train_with_stagnation_monitoring

def run_experiments():
    """
    Run multiple training experiments with different configurations
    """
    base_filename = "Beethoven_Piano_Sonata_No_10_Op_14_No_2_fQqNsTUvqCY_cut_mov_1"
    
    # Experiment configurations
    experiments = [
        {
            "name": "Default Settings",
            "learning_rate": 0.001,
            "stagnation_threshold": 20,
            "min_loss_change": 1e-8,
            "print_interval": 10
        },
        {
            "name": "Higher Learning Rate",
            "learning_rate": 0.01,
            "stagnation_threshold": 10,
            "min_loss_change": 1e-6,
            "print_interval": 5
        },
        {
            "name": "Very High Learning Rate",
            "learning_rate": 0.1,
            "stagnation_threshold": 5,
            "min_loss_change": 1e-5,
            "print_interval": 5
        },
        {
            "name": "Conservative Settings",
            "learning_rate": 0.0001,
            "stagnation_threshold": 50,
            "min_loss_change": 1e-10,
            "print_interval": 25
        }
    ]
    
    print("🧪 RUNNING TRAINING EXPERIMENTS")
    print("=" * 60)
    
    results = []
    
    for i, config in enumerate(experiments, 1):
        print(f"\n🔬 EXPERIMENT {i}: {config['name']}")
        print("-" * 40)
        print(f"Learning rate: {config['learning_rate']}")
        print(f"Stagnation threshold: {config['stagnation_threshold']}")
        print(f"Min loss change: {config['min_loss_change']}")
        print("-" * 40)
        
        try:
            result = train_with_stagnation_monitoring(
                base_filename=base_filename,
                output_dir="output",
                epochs=1000,
                learning_rate=config['learning_rate'],
                stagnation_threshold=config['stagnation_threshold'],
                min_loss_change=config['min_loss_change'],
                print_interval=config['print_interval'],
                device="cpu"
            )
            
            results.append({
                'experiment': config['name'],
                'final_loss': result['final_loss'],
                'final_accuracy': result['final_accuracy'],
                'total_epochs': result['total_epochs'],
                'config': config
            })
            
            print(f"✅ Experiment {i} completed successfully!")
            
        except Exception as e:
            print(f"❌ Experiment {i} failed: {e}")
            results.append({
                'experiment': config['name'],
                'error': str(e),
                'config': config
            })
    
    # Summary
    print(f"\n📊 EXPERIMENT SUMMARY")
    print("=" * 60)
    
    successful_experiments = [r for r in results if 'error' not in r]
    
    if successful_experiments:
        print(f"{'Experiment':<20} {'Final Loss':<15} {'Accuracy':<10} {'Epochs':<8}")
        print("-" * 60)
        
        for result in successful_experiments:
            print(f"{result['experiment']:<20} {result['final_loss']:<15.6f} {result['final_accuracy']:<10.4f} {result['total_epochs']:<8}")
        
        # Find best result
        best_result = min(successful_experiments, key=lambda x: x['final_loss'])
        print(f"\n🏆 BEST RESULT: {best_result['experiment']}")
        print(f"   Final loss: {best_result['final_loss']:.6f}")
        print(f"   Final accuracy: {best_result['final_accuracy']:.4f}")
        print(f"   Epochs: {best_result['total_epochs']}")
    
    else:
        print("❌ No experiments completed successfully")
    
    return results

if __name__ == "__main__":
    try:
        results = run_experiments()
        print(f"\n🎉 All experiments completed!")
        
    except Exception as e:
        print(f"❌ Experiments failed: {e}")
        import traceback
        traceback.print_exc()
