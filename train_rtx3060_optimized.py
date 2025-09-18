#!/usr/bin/env python3
"""
Optimized training script for Tacotron2 on RTX 3060
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tacotron2.configs.config import Config
from tacotron2.models.tacotron2 import Tacotron2, HParams
from tacotron2.training.train import Trainer

class RTX3060OptimizedTrainer(Trainer):
    """Optimized trainer for RTX 3060"""
    
    def __init__(self, config):
        super().__init__(config)
        self.setup_rtx3060_optimizations()
    
    def setup_rtx3060_optimizations(self):
        """Apply RTX 3060 specific optimizations"""
        print("Applying RTX 3060 optimizations...")
        
        # Enable mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()
        self.use_amp = True
        
        # Optimize memory usage
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Set optimal batch size for RTX 3060
        self.config.BATCH_SIZE = 6
        self.config.VAL_BATCH_SIZE = 4
        
        # Enable gradient accumulation
        self.gradient_accumulation_steps = 2
        
        print(f"✓ Mixed precision training enabled")
        print(f"✓ Batch size optimized: {self.config.BATCH_SIZE}")
        print(f"✓ Gradient accumulation steps: {self.gradient_accumulation_steps}")
    
    def train_step(self, batch):
        """Optimized training step with mixed precision"""
        self.model.train()
        
        # Move batch to device
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch
        text_padded = text_padded.to(self.device, non_blocking=True)
        input_lengths = input_lengths.to(self.device, non_blocking=True)
        mel_padded = mel_padded.to(self.device, non_blocking=True)
        gate_padded = gate_padded.to(self.device, non_blocking=True)
        output_lengths = output_lengths.to(self.device, non_blocking=True)
        
        # Prepare inputs
        inputs = (text_padded, input_lengths, mel_padded, torch.max(input_lengths).item(), output_lengths)
        targets = (mel_padded, gate_padded)
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            model_output = self.model(inputs)
            total_loss, mel_loss, gate_loss = self.criterion(model_output, targets)
            
            # Scale loss for gradient accumulation
            total_loss = total_loss / self.gradient_accumulation_steps
        
        # Backward pass
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        
        return {
            'total_loss': total_loss.item() * self.gradient_accumulation_steps,
            'mel_loss': mel_loss.item(),
            'gate_loss': gate_loss.item()
        }
    
    def optimizer_step(self):
        """Optimized optimizer step with gradient scaling"""
        if self.use_amp:
            # Gradient clipping with scaler
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP_THRESH)
            
            # Optimizer step with scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard gradient clipping and step
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP_THRESH)
            self.optimizer.step()
        
        self.optimizer.zero_grad()
    
    def train(self):
        """Optimized training loop for RTX 3060"""
        from tacotron2.data.dataset import create_dataloaders
        
        # Create data loaders with optimized settings
        train_loader, val_loader = create_dataloaders(self.config)
        
        print(f"Starting optimized training for RTX 3060")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"Total epochs: {self.config.N_EPOCHS}")
        
        # Training metrics
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epoch, self.config.N_EPOCHS):
            self.epoch = epoch
            epoch_losses = []
            
            # Training phase
            self.model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.N_EPOCHS}")
            
            accumulated_losses = {'total_loss': 0, 'mel_loss': 0, 'gate_loss': 0}
            
            for batch_idx, batch in enumerate(pbar):
                # Training step
                losses = self.train_step(batch)
                
                # Accumulate losses
                for key in accumulated_losses:
                    accumulated_losses[key] += losses[key]
                
                # Optimizer step every gradient_accumulation_steps
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer_step()
                    self.iteration += 1
                    
                    # Average accumulated losses
                    avg_losses = {k: v / self.gradient_accumulation_steps 
                                for k, v in accumulated_losses.items()}
                    epoch_losses.append(avg_losses)
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'Loss': f"{avg_losses['total_loss']:.4f}",
                        'Mel': f"{avg_losses['mel_loss']:.4f}",
                        'Gate': f"{avg_losses['gate_loss']:.4f}",
                        'GPU': f"{torch.cuda.memory_allocated()/1e9:.1f}GB"
                    })
                    
                    # Log to tensorboard
                    if self.iteration % self.config.LOG_INTERVAL == 0 and self.writer:
                        for key, value in avg_losses.items():
                            self.writer.add_scalar(f'train/{key}', value, self.iteration)
                        
                        self.writer.add_scalar('train/learning_rate',
                                             self.optimizer.param_groups[0]['lr'], self.iteration)
                        
                        # Log GPU memory usage
                        gpu_memory = torch.cuda.memory_allocated() / 1e9
                        self.writer.add_scalar('system/gpu_memory_gb', gpu_memory, self.iteration)
                    
                    # Reset accumulated losses
                    accumulated_losses = {'total_loss': 0, 'mel_loss': 0, 'gate_loss': 0}
                    
                    # Save checkpoint
                    if self.iteration % self.config.SAVE_INTERVAL == 0:
                        self.save_checkpoint()
            
            # Calculate average epoch losses
            if epoch_losses:
                avg_epoch_losses = {}
                for key in epoch_losses[0].keys():
                    avg_epoch_losses[key] = np.mean([loss[key] for loss in epoch_losses])
            else:
                avg_epoch_losses = {'total_loss': 0, 'mel_loss': 0, 'gate_loss': 0}
            
            # Validation phase
            if epoch % 1 == 0:  # Validate every epoch
                val_losses = self.validate(val_loader)
                
                # Log validation losses
                if self.writer:
                    for key, value in val_losses.items():
                        self.writer.add_scalar(f'val/{key}', value, epoch)
                
                # Learning rate scheduling
                self.scheduler.step(val_losses['total_loss'])
                
                # Early stopping and best model saving
                if val_losses['total_loss'] < best_val_loss:
                    best_val_loss = val_losses['total_loss']
                    patience_counter = 0
                    self.save_checkpoint(is_best=True)
                    print(f"✓ New best model saved (val_loss: {best_val_loss:.4f})")
                else:
                    patience_counter += 1
                
                print(f"Epoch {epoch+1:3d} | "
                      f"Train: {avg_epoch_losses['total_loss']:.4f} | "
                      f"Val: {val_losses['total_loss']:.4f} | "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                      f"GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")
                
                # Early stopping
                if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered after {patience_counter} epochs without improvement")
                    break
            
            # Save epoch checkpoint
            self.save_checkpoint()
            
            # Clear GPU cache periodically
            if epoch % 5 == 0:
                torch.cuda.empty_cache()
        
        print("Training completed!")
        if self.writer:
            self.writer.close()

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Tacotron2 optimized for RTX 3060')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    args = parser.parse_args()
    
    # Load and update config
    config = Config()
    
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.epochs:
        config.N_EPOCHS = args.epochs
    if args.learning_rate:
        config.LEARNING_RATE = args.learning_rate
    
    # Verify GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires GPU.")
        return 1
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {gpu_memory:.1f} GB")
    
    if gpu_memory < 10:
        print("WARNING: Less than 10GB VRAM detected. Consider reducing batch size.")
    
    # Initialize optimized trainer
    trainer = RTX3060OptimizedTrainer(config)
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_checkpoint()
        print("Checkpoint saved")
    except Exception as e:
        print(f"Training failed with error: {e}")
        trainer.save_checkpoint()
        print("Emergency checkpoint saved")
        raise
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
