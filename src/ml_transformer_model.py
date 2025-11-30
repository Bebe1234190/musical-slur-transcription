#!/usr/bin/env python3
"""
Transformer Model for MIDI Slur Prediction
Predicts slur annotations from musical note sequences

Part of the MIDI Piano Roll ML System v2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class MusicSlurTransformer(nn.Module):
    """
    Transformer model for predicting slur annotations from musical sequences
    
    Architecture:
    - Input: (sequence_length, 6) [start_time, duration, pitch, velocity, sustain_start, sustain_end]
    - Output: (sequence_length, 5) raw logits for [slur_start, slur_middle, slur_end, no_slur, slur_start_and_end]
    - Uses CrossEntropyLoss (softmax applied internally by loss function)
    - Predicts mutually exclusive classes (one category per note)
    """
    
    def __init__(self, 
                 input_dim=6,
                 d_model=128, 
                 n_heads=8, 
                 n_layers=4, 
                 d_ff=512,
                 output_dim=5,
                 dropout=0.1):
        """
        Initialize the transformer model
        
        Args:
            input_dim (int): Number of input features (6)
            d_model (int): Hidden dimension size (128)
            n_heads (int): Number of attention heads (8)
            n_layers (int): Number of transformer layers (4)
            d_ff (int): Feed-forward dimension (512)
            output_dim (int): Number of output features (5)
            dropout (float): Dropout rate
        """
        super(MusicSlurTransformer, self).__init__()
        
        self.d_model = d_model
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Input projection layer
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Transformer encoder layers
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='relu',
            batch_first=True  # Input shape: (batch, sequence, features)
        )
        
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers
        )
        
        # Output projection layer
        self.output_projection = nn.Linear(d_model, output_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        # Xavier uniform initialization for linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x, src_key_padding_mask=None):
        """
        Forward pass through the transformer
        
        Args:
            x (torch.Tensor): Input tensor (batch, sequence_length, input_dim)
            src_key_padding_mask (torch.Tensor): Mask for padding tokens
            
        Returns:
            torch.Tensor: Output predictions (batch, sequence_length, output_dim)
        """
        # Input projection: (batch, seq_len, input_dim) -> (batch, seq_len, d_model)
        x = self.input_projection(x)
        
        # Apply transformer encoder
        # Note: We use bidirectional attention (no causal mask)
        transformer_output = self.transformer_encoder(
            x, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Output projection: (batch, seq_len, d_model) -> (batch, seq_len, output_dim)
        output = self.output_projection(transformer_output)
        
        # No activation - CrossEntropyLoss expects raw logits (applies softmax internally)
        return output
    
    def get_model_info(self):
        """Return model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_dim': self.input_dim,
            'd_model': self.d_model,
            'output_dim': self.output_dim,
            'architecture': 'TransformerEncoder'
        }

class MusicSlurTrainer:
    """
    Training class for the MusicSlurTransformer
    """
    
    def __init__(self, model, device='cpu'):
        """
        Initialize trainer
        
        Args:
            model (MusicSlurTransformer): The transformer model
            device (str): Device to train on ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.train_accuracies = []
        
    def train_step(self, inputs, targets, optimizer, criterion):
        """
        Single training step
        
        Args:
            inputs (torch.Tensor): Input sequences
            targets (torch.Tensor): Target labels
            optimizer: PyTorch optimizer
            criterion: Loss function
            
        Returns:
            float: Training loss
            dict: Training metrics
        """
        self.model.train()
        
        # Move data to device
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Add batch dimension if needed
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)  # (1, seq_len, features)
        if targets.dim() == 1:
            targets = targets.unsqueeze(0)  # (1, seq_len) for class indices
        
        # Forward pass
        optimizer.zero_grad()
        predictions = self.model(inputs)  # (batch, seq_len, num_classes)
        
        # Reshape for CrossEntropyLoss: (batch*seq_len, num_classes) and (batch*seq_len,)
        batch_size, seq_len, num_classes = predictions.shape
        predictions_flat = predictions.view(-1, num_classes)
        targets_flat = targets.view(-1).long()
        
        # Calculate loss
        loss = criterion(predictions_flat, targets_flat)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            # Get predicted classes (argmax)
            pred_classes = predictions_flat.argmax(dim=1)
            accuracy = (pred_classes == targets_flat).float().mean()
            
            # Per-category accuracy
            accuracies = {}
            category_names = ['slur_start', 'slur_middle', 'slur_end', 'no_slur', 'slur_start_and_end']
            for class_idx, category in enumerate(category_names):
                mask = targets_flat == class_idx
                if mask.sum() > 0:
                    cat_acc = (pred_classes[mask] == targets_flat[mask]).float().mean()
                    accuracies[category] = cat_acc.item()
                else:
                    accuracies[category] = 0.0
        
        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'category_accuracies': accuracies
        }
        
        return loss.item(), metrics
    
    def evaluate(self, inputs, targets, criterion):
        """
        Evaluate model on given data
        
        Args:
            inputs (torch.Tensor): Input sequences
            targets (torch.Tensor): Target labels
            criterion: Loss function
            
        Returns:
            dict: Evaluation metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            # Move data to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Add batch dimension if needed
            if inputs.dim() == 2:
                inputs = inputs.unsqueeze(0)
            if targets.dim() == 1:
                targets = targets.unsqueeze(0)
            
            # Forward pass
            predictions = self.model(inputs)  # (batch, seq_len, num_classes)
            
            # Reshape for CrossEntropyLoss
            batch_size, seq_len, num_classes = predictions.shape
            predictions_flat = predictions.view(-1, num_classes)
            targets_flat = targets.view(-1).long()
            
            # Calculate loss
            loss = criterion(predictions_flat, targets_flat)
            
            # Calculate metrics
            pred_classes = predictions_flat.argmax(dim=1)
            accuracy = (pred_classes == targets_flat).float().mean()
            
            # Per-category metrics
            category_metrics = {}
            category_names = ['slur_start', 'slur_middle', 'slur_end', 'no_slur', 'slur_start_and_end']
            for class_idx, category in enumerate(category_names):
                target_mask = targets_flat == class_idx
                pred_mask = pred_classes == class_idx
                
                if target_mask.sum() > 0:
                    # Accuracy for this category
                    cat_acc = (pred_classes[target_mask] == targets_flat[target_mask]).float().mean()
                    
                    # Precision, recall, F1
                    tp = ((pred_mask) & (target_mask)).sum().float()
                    fp = ((pred_mask) & (~target_mask)).sum().float()
                    fn = ((~pred_mask) & (target_mask)).sum().float()
                    
                    precision = tp / (tp + fp + 1e-8)
                    recall = tp / (tp + fn + 1e-8)
                    f1 = 2 * precision * recall / (precision + recall + 1e-8)
                else:
                    cat_acc = precision = recall = f1 = 0.0
                
                category_metrics[category] = {
                    'accuracy': cat_acc.item(),
                    'precision': precision.item(),
                    'recall': recall.item(),
                    'f1': f1.item()
                }
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'category_metrics': category_metrics,
            'predictions': predictions.cpu(),
            'targets': targets.cpu()
        }
    
    def train_overfitting_test(self, inputs, targets, epochs=1000, lr=1e-3, print_every=50):
        """
        Train model to overfit on single piece (memorization test)
        
        Args:
            inputs (torch.Tensor): Training inputs
            targets (torch.Tensor): Training targets
            epochs (int): Number of training epochs
            lr (float): Learning rate
            print_every (int): Print progress every N epochs
            
        Returns:
            dict: Training history
        """
        print(f"🚀 STARTING OVERFITTING TEST")
        print(f"   Model: {self.model.__class__.__name__}")
        print(f"   Epochs: {epochs}")
        print(f"   Learning rate: {lr}")
        print(f"   Sequence length: {inputs.shape[0]}")
        print("=" * 50)
        
        # Setup training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()  # Cross-entropy for multi-class classification
        
        # Training history
        history = {
            'losses': [],
            'accuracies': [],
            'category_accuracies': {cat: [] for cat in ['slur_start', 'slur_middle', 'slur_end', 'no_slur', 'slur_start_and_end']}
        }
        
        # Training loop
        for epoch in range(epochs):
            loss, metrics = self.train_step(inputs, targets, optimizer, criterion)
            
            # Record history
            history['losses'].append(loss)
            history['accuracies'].append(metrics['accuracy'])
            for cat, acc in metrics['category_accuracies'].items():
                history['category_accuracies'][cat].append(acc)
            
            # Print progress
            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1:4d}/{epochs} | "
                      f"Loss: {loss:.6f} | "
                      f"Acc: {metrics['accuracy']:.4f} | "
                      f"Start: {metrics['category_accuracies']['slur_start']:.3f} | "
                      f"Mid: {metrics['category_accuracies']['slur_middle']:.3f} | "
                      f"End: {metrics['category_accuracies']['slur_end']:.3f} | "
                      f"None: {metrics['category_accuracies']['no_slur']:.3f}")
        
        # Final evaluation
        final_metrics = self.evaluate(inputs, targets, criterion)
        
        print(f"\n✅ OVERFITTING TEST COMPLETE!")
        print(f"   Final loss: {final_metrics['loss']:.6f}")
        print(f"   Final accuracy: {final_metrics['accuracy']:.4f}")
        print(f"   Category performance:")
        for cat, metrics in final_metrics['category_metrics'].items():
            print(f"     {cat:12}: Acc={metrics['accuracy']:.3f}, "
                  f"F1={metrics['f1']:.3f}")
        
        return history, final_metrics

def create_model(input_dim=5, d_model=128, n_heads=8, n_layers=4, output_dim=5):
    """
    Create a MusicSlurTransformer with specified architecture
    
    Args:
        input_dim (int): Number of input features
        d_model (int): Hidden dimension
        n_heads (int): Number of attention heads
        n_layers (int): Number of transformer layers
        output_dim (int): Number of output features
        
    Returns:
        MusicSlurTransformer: Initialized model
    """
    model = MusicSlurTransformer(
        input_dim=input_dim,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_model * 4,  # Standard 4x multiplier
        output_dim=output_dim
    )
    
    return model

if __name__ == "__main__":
    # Test model creation
    print("🧠 TESTING TRANSFORMER MODEL")
    print("=" * 40)
    
    # Create model
    model = create_model()
    info = model.get_model_info()
    
    print(f"✓ Model created:")
    print(f"  Architecture: {info['architecture']}")
    print(f"  Total parameters: {info['total_parameters']:,}")
    print(f"  Input features: {info['input_dim']}")
    print(f"  Output features: {info['output_dim']}")
    print(f"  Hidden dimension: {info['d_model']}")
    
    # Test forward pass
    batch_size = 1
    seq_length = 100
    test_input = torch.randn(batch_size, seq_length, 5)
    
    model.eval()
    with torch.no_grad():
        test_output = model(test_input)
    
    print(f"\n✓ Forward pass test:")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {test_output.shape}")
    print(f"  Output range: {test_output.min():.3f} - {test_output.max():.3f}")
    
    print(f"\n🎵 Model ready for training!")
