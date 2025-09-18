"""
Training script for Tacotron2
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Optional TensorBoard import
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    logger.warning("TensorBoard not available. Install with: pip install tensorboard")
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ..configs.config import Config
    from ..models.tacotron2 import Tacotron2, HParams
    from ..data.dataset import create_dataloaders
    from .loss import Tacotron2Loss
    from ..utils.text import sequence_to_text
except ImportError:
    from tacotron2.configs.config import Config
    from tacotron2.models.tacotron2 import Tacotron2, HParams
    from tacotron2.data.dataset import create_dataloaders
    from tacotron2.training.loss import Tacotron2Loss
    from tacotron2.utils.text import sequence_to_text


class Trainer:
    """
    Trainer class for Tacotron2 training.

    This class manages the model, optimizer, loss function, data loading,
    checkpointing, and training/validation loops.
    """

    def __init__(self, config):
        """
        Initialize the Trainer.

        Args:
            config (Config): Training configuration object.
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Create directories for saving logs and checkpoints
        config.create_dirs()

        # Initialize model and hyperparameters
        self.hparams = HParams()
        self.model = Tacotron2(self.hparams).to(self.device)

        # Loss function
        self.criterion = Tacotron2Loss(
            mel_loss_weight=config.MEL_LOSS_WEIGHT,
            gate_loss_weight=config.GATE_LOSS_WEIGHT
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # TensorBoard writer
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(config.LOG_DIR)
        else:
            self.writer = None

        # Training state
        self.epoch = 0
        self.iteration = 0
        self.best_val_loss = float('inf')

        # Load checkpoint if available
        self.load_checkpoint()

    def load_checkpoint(self, checkpoint_path=None):
        """
        Load training checkpoint if exists.

        Args:
            checkpoint_path (str, optional): Path to checkpoint file.
        """
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.config.CHECKPOINT_DIR, 'latest_checkpoint.pth')

        if os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self.iteration = checkpoint['iteration']
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

            logger.info(f"Resumed from epoch {self.epoch}, iteration {self.iteration}")
        else:
            logger.info("No checkpoint found, starting from scratch")

    def save_checkpoint(self, is_best=False):
        """
        Save a training checkpoint.

        Args:
            is_best (bool): If True, saves as the best model checkpoint.
        """
        checkpoint = {
            'epoch': self.epoch,
            'iteration': self.iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        # Save latest checkpoint
        checkpoint_path = os.path.join(self.config.CHECKPOINT_DIR, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config.CHECKPOINT_DIR, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint with validation loss: {self.best_val_loss:.4f}")

        # Save periodic epoch checkpoint
        if self.epoch % 10 == 0:
            epoch_path = os.path.join(self.config.CHECKPOINT_DIR, f'checkpoint_epoch_{self.epoch}.pth')
            torch.save(checkpoint, epoch_path)

    def train_step(self, batch):
        """
        Execute a single training step.

        Args:
            batch (tuple): A batch of training data.

        Returns:
            dict: Loss values (total, mel, gate).
        """
        self.model.train()

        # Move batch to device
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch
        text_padded = text_padded.to(self.device)
        input_lengths = input_lengths.to(self.device)
        mel_padded = mel_padded.to(self.device)
        gate_padded = gate_padded.to(self.device)
        output_lengths = output_lengths.to(self.device)

        # Forward pass
        inputs = (text_padded, input_lengths, mel_padded, torch.max(input_lengths).item(), output_lengths)
        targets = (mel_padded, gate_padded)

        model_output = self.model(inputs)

        # Calculate loss
        total_loss, mel_loss, gate_loss = self.criterion(model_output, targets)

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP_THRESH)

        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'mel_loss': mel_loss.item(),
            'gate_loss': gate_loss.item()
        }

    def validate(self, val_loader):
        """
        Perform validation.

        Args:
            val_loader (DataLoader): Validation data loader.

        Returns:
            dict: Average validation losses.
        """
        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Move batch to device
                text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch
                text_padded = text_padded.to(self.device)
                input_lengths = input_lengths.to(self.device)
                mel_padded = mel_padded.to(self.device)
                gate_padded = gate_padded.to(self.device)
                output_lengths = output_lengths.to(self.device)

                # Forward pass
                inputs = (text_padded, input_lengths, mel_padded, torch.max(input_lengths).item(), output_lengths)
                targets = (mel_padded, gate_padded)

                model_output = self.model(inputs)

                # Calculate loss
                total_loss, mel_loss, gate_loss = self.criterion(model_output, targets)

                val_losses.append({
                    'total_loss': total_loss.item(),
                    'mel_loss': mel_loss.item(),
                    'gate_loss': gate_loss.item()
                })

        # Average losses
        avg_losses = {}
        for key in val_losses[0].keys():
            avg_losses[key] = np.mean([loss[key] for loss in val_losses])

        return avg_losses

    def train(self):
        """
        Main training loop.
        """
        # Create data loaders
        train_loader, val_loader = create_dataloaders(self.config)

        logger.info(f"Starting training from epoch {self.epoch}")
        logger.info(f"Total epochs: {self.config.N_EPOCHS}")
        logger.info(f"Batch size: {self.config.BATCH_SIZE}")
        logger.info(f"Learning rate: {self.config.LEARNING_RATE}")

        for epoch in range(self.epoch, self.config.N_EPOCHS):
            self.epoch = epoch
            epoch_losses = []

            # Training loop
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.N_EPOCHS}")
            for batch_idx, batch in enumerate(pbar):
                losses = self.train_step(batch)
                epoch_losses.append(losses)
                self.iteration += 1

                # Update progress bar
                pbar.set_postfix({
                    'Loss': f"{losses['total_loss']:.4f}",
                    'Mel': f"{losses['mel_loss']:.4f}",
                    'Gate': f"{losses['gate_loss']:.4f}"
                })

                # Log to tensorboard
                if self.iteration % self.config.LOG_INTERVAL == 0 and self.writer:
                    for key, value in losses.items():
                        self.writer.add_scalar(f'train/{key}', value, self.iteration)

                    self.writer.add_scalar('train/learning_rate',
                                           self.optimizer.param_groups[0]['lr'], self.iteration)

                # Save checkpoint
                if self.iteration % self.config.SAVE_INTERVAL == 0:
                    self.save_checkpoint()

            # Average epoch losses
            avg_epoch_losses = {}
            for key in epoch_losses[0].keys():
                avg_epoch_losses[key] = np.mean([loss[key] for loss in epoch_losses])

            # Validation
            if epoch % 1 == 0:  # Validate every epoch
                val_losses = self.validate(val_loader)

                # Log validation losses
                if self.writer:
                    for key, value in val_losses.items():
                        self.writer.add_scalar(f'val/{key}', value, epoch)

                # Update LR scheduler
                self.scheduler.step(val_losses['total_loss'])

                # Save best model
                if val_losses['total_loss'] < self.best_val_loss:
                    self.best_val_loss = val_losses['total_loss']
                    self.save_checkpoint(is_best=True)

                logger.info(
                    f"Epoch {epoch+1} - Train Loss: {avg_epoch_losses['total_loss']:.4f}, "
                    f"Val Loss: {val_losses['total_loss']:.4f}"
                )

            # Save checkpoint at end of epoch
            self.save_checkpoint()

        logger.info("Training completed!")
        if self.writer:
            self.writer.close()


def main():
    """
    Main function to start Tacotron2 training.
    """
    parser = argparse.ArgumentParser(description='Train Tacotron2')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint')
    args = parser.parse_args()

    # Load config
    config = Config()

    # Initialize trainer
    trainer = Trainer(config)

    # Load specific checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
