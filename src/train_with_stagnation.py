#!/usr/bin/env python3
"""
Training Script with Stagnation Monitoring
Trains the transformer model with automatic stopping when loss stagnates

Part of the MIDI Piano Roll ML System v2.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from ml_data_pipeline import load_processed_data
from ml_transformer_model import MusicSlurTransformer

def train_with_stagnation_monitoring(base_filename="Beethoven_Piano_Sonata_No_10_Op_14_No_2_fQqNsTUvqCY_cut_mov_1",
                                   output_dir="output",
                                   epochs=20000,
                                   learning_rate=0.001,
                                   stagnation_epochs=50,
                                   min_loss_change=1e-15,
                                   print_interval=10,
                                   device="cpu"):
    """
    Train transformer model with automatic stagnation detection
    
    Args:
        base_filename (str): Base filename of the processed data
        output_dir (str): Directory containing the data files
        epochs (int): Maximum number of training epochs
        learning_rate (float): Learning rate for training
        stagnation_epochs (int): Require same loss for N consecutive epochs before stopping
        min_loss_change (float): Minimum loss change to consider non-stagnant
        print_interval (int): Print progress every N epochs
        device (str): Device to train on ('cpu' or 'cuda')
        
    Returns:
        dict: Training results and final metrics
    """
    print("🎵 TRANSFORMER TRAINING WITH STAGNATION MONITORING")
    print("=" * 60)
    print(f"📂 Base filename: {base_filename}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🔄 Max epochs: {epochs}")
    print(f"📈 Learning rate: {learning_rate}")
    print(f"⏱️  Stagnation check: {stagnation_epochs} consecutive epochs")
    print(f"📊 Min loss change: {min_loss_change}")
    print(f"💻 Device: {device}")
    print("=" * 60)
    
    # Load data
    data_path = os.path.join(output_dir, f"{base_filename}_processed_for_ml.pt")
    print(f"\n📊 Loading data from: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data not found: {data_path}")
    
    inputs, targets, norm_params, stats = load_processed_data(data_path)
    print(f"✅ Data loaded: {inputs.shape}, {targets.shape}")
    
    # Create model
    print(f"\n🧠 Creating model...")
    model = MusicSlurTransformer(
        input_dim=stats['input_features'],
        d_model=128, 
        n_heads=8, 
        n_layers=4, 
        d_ff=512,
        output_dim=stats['output_features'],
        dropout=0.1
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created with {total_params:,} parameters")
    print(f"   Input features: {stats['input_features']}")
    print(f"   Output features: {stats['output_features']}")
    
    # Training setup
    criterion = nn.BCELoss()  # Use BCELoss like the original successful training
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device(device)
    model.to(device)
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    # Training loop
    print(f"\n🔥 Starting training...")
    print(f"Epoch | Loss (high precision) | Accuracy")
    print("-" * 50)
    
    loss_history = []
    accuracy_history = []
    stagnation_check_epochs = 20  # Require same loss for 20 consecutive epochs
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        with torch.no_grad():
            predictions = torch.sigmoid(outputs)
            predicted_classes = torch.argmax(predictions, dim=1)
            true_classes = torch.argmax(targets, dim=1)
            accuracy = (predicted_classes == true_classes).float().mean().item()
        
        loss_value = loss.item()
        loss_history.append(loss_value)
        accuracy_history.append(accuracy)
        
        # Print progress
        if (epoch + 1) % print_interval == 0:
            print(f"{epoch+1:4d} | {loss_value:.10f} | {accuracy:.4f}")
        
        # Check for stagnation - require same loss for 20 consecutive epochs
        if epoch >= stagnation_check_epochs - 1:  # Need at least 20 epochs to check
            # Check if the last 20 losses are all the same (within precision threshold)
            recent_losses = loss_history[-stagnation_check_epochs:]
            current_loss = recent_losses[-1]
            
            # Check if all recent losses are identical (within precision)
            all_same = True
            for i in range(len(recent_losses) - 1):
                if abs(recent_losses[i] - current_loss) >= min_loss_change:
                    all_same = False
                    break
            
            # Print stagnation check every 10 epochs for readability
            if (epoch + 1) % print_interval == 0:
                loss_variance = max(recent_losses) - min(recent_losses)
                print(f"Stagnation check at epoch {epoch+1}: Loss variance over last {stagnation_check_epochs} epochs = {loss_variance:.15f}")
            
            if all_same:
                print(f"\n🛑 Training stopped due to stagnation at epoch {epoch+1}")
                print(f"   Loss has been identical for {stagnation_check_epochs} consecutive epochs")
                print(f"   Loss value: {current_loss:.15f}")
                print(f"   Variance over last {stagnation_check_epochs} epochs: {max(recent_losses) - min(recent_losses):.15f}")
                break
    
    # Final evaluation
    print(f"\n📈 FINAL RESULTS")
    print("-" * 30)
    final_loss = loss_history[-1]
    final_accuracy = accuracy_history[-1]
    
    print(f"Final loss: {final_loss:.10f}")
    print(f"Final accuracy: {final_accuracy:.4f}")
    print(f"Total epochs: {len(loss_history)}")
    
    # Calculate improvement
    if len(loss_history) > 10:
        early_loss = sum(loss_history[:10]) / 10
        late_loss = sum(loss_history[-10:]) / 10
        improvement = ((early_loss - late_loss) / early_loss) * 100
        print(f"Loss improvement: {improvement:.1f}%")
    
    # Save results
    results = {
        'final_loss': final_loss,
        'final_accuracy': final_accuracy,
        'total_epochs': len(loss_history),
        'loss_history': loss_history,
        'accuracy_history': accuracy_history,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'stats': stats
    }
    
    output_path = os.path.join(output_dir, f"{base_filename}_stagnation_training_results.pt")
    torch.save(results, output_path)
    print(f"✅ Results saved: {output_path}")
    
    return results

if __name__ == "__main__":
    # Default training run
    base_filename = "Beethoven_Piano_Sonata_No_10_Op_14_No_2_fQqNsTUvqCY_cut_mov_1"
    
    try:
        results = train_with_stagnation_monitoring(
            base_filename=base_filename,
            output_dir="output",
            epochs=20000,
            learning_rate=0.001,
            stagnation_epochs=50,
            min_loss_change=1e-15,
            print_interval=10,
            device="cpu"
        )
        
        print(f"\n🎉 Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
