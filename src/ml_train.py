#!/usr/bin/env python3
"""
Training Script for MIDI Slur Transformer
Main entry point for training and testing the transformer model

Part of the MIDI Piano Roll ML System v2.0
"""

import torch
import sys
import os
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from ml_data_pipeline import load_and_prepare_data, save_processed_data, load_processed_data
from ml_transformer_model import create_model, MusicSlurTrainer

def run_overfitting_test(base_filename="Beethoven_Piano_Sonata_No_10_Op_14_No_2_fQqNsTUvqCY_cut_mov_1", 
                        output_dir="../output",
                        epochs=1000,
                        learning_rate=1e-3,
                        device="cpu"):
    """
    Run overfitting test on a single piece to verify model capability
    
    Args:
        base_filename (str): Base filename of the piece to train on
        output_dir (str): Directory containing the data files
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for training
        device (str): Device to train on ('cpu' or 'cuda')
        
    Returns:
        dict: Training results and metrics
    """
    print("🎵 MIDI SLUR TRANSFORMER - OVERFITTING TEST")
    print("=" * 60)
    print(f"📂 Base filename: {base_filename}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🔄 Epochs: {epochs}")
    print(f"📈 Learning rate: {learning_rate}")
    print(f"💻 Device: {device}")
    print("=" * 60)
    
    # Step 1: Load and prepare data
    print(f"\n📊 STEP 1: DATA PREPARATION")
    print("-" * 30)
    
    processed_data_path = os.path.join(output_dir, f"{base_filename}_processed_for_ml.pt")
    
    # Check if processed data exists
    if os.path.exists(processed_data_path):
        print(f"⚡ Loading pre-processed data: {processed_data_path}")
        inputs, targets, norm_params, stats = load_processed_data(processed_data_path)
    else:
        print(f"🔄 Processing raw data...")
        inputs, targets, norm_params, stats = load_and_prepare_data(base_filename, output_dir)
        save_processed_data(inputs, targets, norm_params, stats, processed_data_path)
    
    print(f"✅ Data ready: {stats['sequence_length']} notes, {stats['input_features']} features")
    
    # Step 2: Create model
    print(f"\n🧠 STEP 2: MODEL CREATION")
    print("-" * 30)
    
    model = create_model(
        input_dim=stats['input_features'],
        d_model=128,
        n_heads=8,
        n_layers=4,
        output_dim=stats['output_features']
    )
    
    model_info = model.get_model_info()
    print(f"✅ Model created:")
    print(f"   Architecture: {model_info['architecture']}")
    print(f"   Parameters: {model_info['total_parameters']:,}")
    print(f"   Input → Hidden → Output: {model_info['input_dim']} → {model_info['d_model']} → {model_info['output_dim']}")
    
    # Step 3: Initialize trainer
    print(f"\n🏃 STEP 3: TRAINING SETUP")
    print("-" * 30)
    
    # Check device availability
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available, falling back to CPU")
        device = "cpu"
    
    trainer = MusicSlurTrainer(model, device=device)
    print(f"✅ Trainer initialized on {device}")
    
    # Step 4: Run overfitting test
    print(f"\n🔥 STEP 4: OVERFITTING TEST")
    print("-" * 30)
    
    history, final_metrics = trainer.train_overfitting_test(
        inputs=inputs,
        targets=targets,
        epochs=epochs,
        lr=learning_rate,
        print_every=max(1, epochs // 20)  # Print 20 times during training
    )
    
    # Step 5: Analyze results
    print(f"\n📈 STEP 5: RESULTS ANALYSIS")
    print("-" * 30)
    
    # Check if memorization was successful
    success_threshold = 0.95  # 95% accuracy
    memorization_success = final_metrics['accuracy'] >= success_threshold
    
    print(f"🎯 Memorization Success: {'✅ YES' if memorization_success else '❌ NO'}")
    print(f"   Target accuracy: {success_threshold:.1%}")
    print(f"   Achieved accuracy: {final_metrics['accuracy']:.1%}")
    print(f"   Final loss: {final_metrics['loss']:.6f}")
    
    print(f"\n📊 Category Performance:")
    for category, metrics in final_metrics['category_metrics'].items():
        print(f"   {category:15}: Acc={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}, "
              f"Prec={metrics['precision']:.3f}, Rec={metrics['recall']:.3f}")
    
    # Convergence analysis
    if len(history['losses']) > 100:
        early_loss = sum(history['losses'][:50]) / 50
        late_loss = sum(history['losses'][-50:]) / 50
        loss_improvement = (early_loss - late_loss) / early_loss
        
        print(f"\n🔄 Training Convergence:")
        print(f"   Early loss (epochs 1-50): {early_loss:.6f}")
        print(f"   Late loss (final 50): {late_loss:.6f}")
        print(f"   Improvement: {loss_improvement:.1%}")
        
        if loss_improvement < 0.1:
            print("   ⚠️  Low convergence - consider longer training or higher learning rate")
        else:
            print("   ✅ Good convergence detected")
    
    # Step 6: Save results
    print(f"\n💾 STEP 6: SAVING RESULTS")
    print("-" * 30)
    
    # Save model if successful
    if memorization_success:
        model_path = os.path.join(output_dir, f"{base_filename}_overfitted_model.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_dim': stats['input_features'],
                'd_model': 128,
                'n_heads': 8,
                'n_layers': 4,
                'output_dim': stats['output_features']
            },
            'training_history': history,
            'final_metrics': final_metrics,
            'normalization_params': norm_params,
            'stats': stats
        }, model_path)
        print(f"✅ Overfitted model saved: {model_path}")
    
    # Save training history
    history_path = os.path.join(output_dir, f"{base_filename}_training_history.pt")
    torch.save({
        'history': history,
        'final_metrics': final_metrics,
        'config': {
            'epochs': epochs,
            'learning_rate': learning_rate,
            'device': device,
            'success': memorization_success
        }
    }, history_path)
    print(f"✅ Training history saved: {history_path}")
    
    # Summary
    print(f"\n🎉 OVERFITTING TEST COMPLETE!")
    print("=" * 60)
    if memorization_success:
        print("✅ SUCCESS: Model can memorize the musical patterns!")
        print("   → The transformer architecture is capable of learning slur patterns")
        print("   → Ready to proceed with multi-piece training")
    else:
        print("❌ MEMORIZATION FAILED: Model couldn't learn the patterns")
        print("   → Consider: longer training, higher learning rate, or architecture changes")
        print("   → May need to debug the task formulation")
    
    return {
        'success': memorization_success,
        'final_metrics': final_metrics,
        'history': history,
        'model': model if memorization_success else None,
        'trainer': trainer
    }

def main():
    """Main entry point for training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train MIDI Slur Transformer')
    parser.add_argument('--piece', default='Beethoven_Piano_Sonata_No_10_Op_14_No_2_fQqNsTUvqCY_cut_mov_1',
                        help='Base filename of piece to train on')
    parser.add_argument('--output-dir', default='../output',
                        help='Directory containing data files')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                        help='Device to train on')
    parser.add_argument('--test-only', action='store_true',
                        help='Only test data pipeline without training')
    
    args = parser.parse_args()
    
    if args.test_only:
        print("🧪 TESTING DATA PIPELINE ONLY")
        print("=" * 40)
        
        # Test data pipeline
        try:
            inputs, targets, norm_params, stats = load_and_prepare_data(args.piece, args.output_dir)
            print(f"✅ Data pipeline test successful!")
            print(f"   Input shape: {inputs.shape}")
            print(f"   Target shape: {targets.shape}")
            return True
        except Exception as e:
            print(f"❌ Data pipeline test failed: {e}")
            return False
    else:
        # Run full overfitting test
        try:
            results = run_overfitting_test(
                base_filename=args.piece,
                output_dir=args.output_dir,
                epochs=args.epochs,
                learning_rate=args.lr,
                device=args.device
            )
            return results['success']
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
