"""
Text-to-Speech synthesis using trained Tacotron2 model

This module provides a complete pipeline for synthesizing speech from text using
a pre-trained Tacotron2 model. It includes functionality for single text synthesis,
batch processing, and interactive mode with comprehensive logging and visualization.

Features:
- Text preprocessing and tokenization
- Mel spectrogram generation using Tacotron2
- Griffin-Lim vocoder for audio reconstruction
- Attention alignment visualization
- Batch processing capabilities
- Comprehensive logging system

Example usage:
    python synthesize.py --checkpoint path/to/checkpoint.pt --text "Hello world"
    python synthesize.py --checkpoint path/to/checkpoint.pt --text_file texts.txt
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from scipy.io.wavfile import write
import argparse
import logging
import time

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config
from models.tacotron2 import Tacotron2, HParams
from utils.text import text_to_sequence, clean_text
from utils.audio import mel_spectrogram

# Configure logging system
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Output to console
        logging.FileHandler('tacotron2_synthesis.log')  # Output to log file
    ]
)
logger = logging.getLogger(__name__)


class Tacotron2Synthesizer:
    """
    A synthesizer class for text-to-speech conversion using Tacotron2 model.
    
    This class handles the complete TTS pipeline including model loading,
    text processing, mel spectrogram generation, and audio synthesis.
    
    Attributes:
        device (torch.device): Computation device (CPU or GPU)
        config (Config): Configuration object with audio parameters
        hparams (HParams): Hyperparameters for Tacotron2 model
        model (Tacotron2): Loaded Tacotron2 model instance
    """
    
    def __init__(self, checkpoint_path, config=None):
        """
        Initialize Tacotron2 synthesizer with a pre-trained model checkpoint.
        
        Args:
            checkpoint_path (str): Path to trained model checkpoint file
            config (Config, optional): Configuration object. If None, uses default config.
            
        Raises:
            FileNotFoundError: If the checkpoint file does not exist
            RuntimeError: If model loading fails
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load configuration
        if config is None:
            self.config = Config()
        else:
            self.config = config
        
        # Initialize model with hyperparameters
        self.hparams = HParams()
        self.model = Tacotron2(self.hparams).to(self.device)
        
        # Load model weights from checkpoint
        self.load_checkpoint(checkpoint_path)
        
        # Set model to evaluation mode for inference
        self.model.eval()
        
        logger.info("Tacotron2 synthesizer initialized successfully!")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model weights from a checkpoint file.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file
            
        Raises:
            FileNotFoundError: If checkpoint file does not exist
            RuntimeError: If checkpoint loading fails
        """
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state dictionary
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            logger.info("Checkpoint loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise RuntimeError(f"Checkpoint loading failed: {e}")
    
    def text_to_mel(self, text):
        """
        Convert input text to mel spectrogram using Tacotron2 model.
        
        Args:
            text (str): Input text to synthesize
            
        Returns:
            tuple: 
                mel_spectrogram (numpy.ndarray): Generated mel spectrogram of shape (n_mels, time_steps)
                alignments (numpy.ndarray): Attention alignment matrix of shape (decoder_steps, encoder_steps)
                
        Raises:
            ValueError: If text is empty or contains invalid characters
        """
        if not text or not text.strip():
            logger.error("Empty text provided for synthesis")
            raise ValueError("Text cannot be empty")
        
        # Clean and convert text to sequence
        cleaned_text = clean_text(text)
        sequence = text_to_sequence(cleaned_text)
        
        # Convert to tensor and add batch dimension
        text_tensor = torch.LongTensor(sequence).unsqueeze(0).to(self.device)
        
        # Generate mel spectrogram with model inference
        with torch.no_grad():
            mel_outputs, mel_outputs_postnet, gate_outputs, alignments = self.model.inference(text_tensor)
        
        # Extract and convert outputs to numpy arrays
        mel_spectrogram = mel_outputs_postnet[0].cpu().numpy()
        alignments = alignments[0].cpu().numpy()
        
        return mel_spectrogram, alignments
    
    def mel_to_audio(self, mel_spectrogram, output_path=None):
        """
        Convert mel spectrogram to audio waveform using Griffin-Lim algorithm.
        
        Note: This is a simplified vocoder implementation. For better quality,
        consider using WaveGlow, HiFi-GAN, or other neural vocoders.
        
        Args:
            mel_spectrogram (numpy.ndarray): Mel spectrogram to convert
            output_path (str, optional): Path to save the generated audio file
            
        Returns:
            numpy.ndarray: Generated audio waveform normalized to [-1, 1]
            
        Raises:
            ValueError: If mel spectrogram has invalid shape or values
        """
        # Validate input mel spectrogram
        if mel_spectrogram is None or mel_spectrogram.size == 0:
            logger.error("Invalid mel spectrogram provided")
            raise ValueError("Mel spectrogram cannot be empty")
        
        # Denormalize mel spectrogram (assuming log-scale input)
        mel_spectrogram = np.exp(mel_spectrogram)
        
        # Convert mel to linear spectrogram using inverse mel filter bank
        n_fft = self.config.N_FFT
        hop_length = self.config.HOP_LENGTH
        win_length = self.config.WIN_LENGTH
        
        # Create mel filter bank
        mel_basis = librosa.filters.mel(
            sr=self.config.SAMPLE_RATE,
            n_fft=n_fft,
            n_mels=self.config.N_MELS,
            fmin=0,
            fmax=self.config.SAMPLE_RATE // 2
        )
        
        # Pseudo-inverse to convert mel to linear spectrogram
        linear_spectrogram = np.dot(np.linalg.pinv(mel_basis), mel_spectrogram)
        
        # Apply Griffin-Lim algorithm for phase reconstruction
        audio = librosa.griffinlim(
            linear_spectrogram,
            hop_length=hop_length,
            win_length=win_length,
            n_iter=60  # Number of iterations for convergence
        )
        
        # Normalize audio to prevent clipping
        audio = audio / np.max(np.abs(audio))
        
        # Save audio file if output path is provided
        if output_path:
            try:
                sf.write(output_path, audio, self.config.SAMPLE_RATE)
                logger.info(f"Audio saved to: {output_path}")
            except Exception as e:
                logger.error(f"Failed to save audio file: {e}")
        
        return audio
    
    def synthesize(self, text, output_path=None, save_mel=False, save_alignment=False):
        """
        Complete text-to-speech synthesis pipeline.
        
        This method performs the entire TTS process from text to audio, including
        optional visualization of intermediate results.
        
        Args:
            text (str): Input text to synthesize
            output_path (str, optional): Path to save the generated audio file
            save_mel (bool): Whether to save mel spectrogram visualization
            save_alignment (bool): Whether to save attention alignment visualization
            
        Returns:
            tuple:
                audio (numpy.ndarray): Generated audio waveform
                mel_spectrogram (numpy.ndarray): Generated mel spectrogram
                alignments (numpy.ndarray): Attention alignment matrix
                
        Raises:
            Exception: If any step in the synthesis pipeline fails
        """
        logger.info(f"Synthesizing: '{text}'")
        
        try:
            # Generate mel spectrogram from text
            mel_spectrogram, alignments = self.text_to_mel(text)
            
            # Convert mel spectrogram to audio
            audio = self.mel_to_audio(mel_spectrogram, output_path)
            
            # Save visualizations if requested
            if save_mel or save_alignment:
                base_name = output_path.replace('.wav', '') if output_path else 'synthesis'
                
                if save_mel:
                    self.plot_mel_spectrogram(mel_spectrogram, f"{base_name}_mel.png")
                
                if save_alignment:
                    self.plot_alignment(alignments, f"{base_name}_alignment.png")
            
            return audio, mel_spectrogram, alignments
            
        except Exception as e:
            logger.error(f"Synthesis failed for text '{text}': {e}")
            raise
    
    def plot_mel_spectrogram(self, mel_spectrogram, save_path):
        """
        Plot and save mel spectrogram visualization.
        
        Args:
            mel_spectrogram (numpy.ndarray): Mel spectrogram to visualize
            save_path (str): Path to save the plot image
            
        Note:
            The plot shows time on x-axis and mel frequency bins on y-axis
        """
        try:
            plt.figure(figsize=(12, 6))
            plt.imshow(mel_spectrogram, aspect='auto', origin='lower', interpolation='none')
            plt.colorbar(label='Magnitude (dB)')
            plt.title('Mel Spectrogram')
            plt.xlabel('Time Frames')
            plt.ylabel('Mel Frequency Bins')
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Mel spectrogram saved to: {save_path}")
        except Exception as e:
            logger.error(f"Failed to save mel spectrogram: {e}")
    
    def plot_alignment(self, alignments, save_path):
        """
        Plot and save attention alignment visualization.
        
        Args:
            alignments (numpy.ndarray): Attention alignment matrix
            save_path (str): Path to save the plot image
            
        Note:
            The plot shows encoder steps on y-axis and decoder steps on x-axis
        """
        try:
            plt.figure(figsize=(12, 6))
            plt.imshow(alignments.T, aspect='auto', origin='lower', interpolation='none')
            plt.colorbar(label='Attention Weight')
            plt.title('Attention Alignment')
            plt.xlabel('Decoder Steps')
            plt.ylabel('Encoder Steps')
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Attention alignment saved to: {save_path}")
        except Exception as e:
            logger.error(f"Failed to save attention alignment: {e}")
    
    def batch_synthesize(self, texts, output_dir):
        """
        Synthesize multiple texts in batch mode.
        
        Args:
            texts (list): List of text strings to synthesize
            output_dir (str): Directory to save output audio files and visualizations
            
        Note:
            Output files are named as synthesis_000.wav, synthesis_001.wav, etc.
            Corresponding mel and alignment plots are saved with the same numbering.
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Starting batch synthesis of {len(texts)} texts to {output_dir}")
        
        success_count = 0
        for i, text in enumerate(texts):
            output_path = os.path.join(output_dir, f"synthesis_{i:03d}.wav")
            try:
                logger.info(f"Processing text {i+1}/{len(texts)}: '{text[:50]}...'")
                self.synthesize(text, output_path, save_mel=True, save_alignment=True)
                logger.info(f"Completed synthesis {i+1}/{len(texts)}")
                success_count += 1
            except Exception as e:
                logger.error(f"Error synthesizing text {i}: {e}")
        
        logger.info(f"Batch synthesis completed. Success: {success_count}/{len(texts)}")


def main():
    """
    Main function for command-line interface of Tacotron2 synthesizer.
    
    Supports three modes:
    1. Single text synthesis (--text)
    2. Batch synthesis from file (--text_file)
    3. Interactive mode (no arguments)
    
    Usage examples:
        python synthesize.py --checkpoint model.pt --text "Hello world" --output hello.wav
        python synthesize.py --checkpoint model.pt --text_file texts.txt --output_dir batch_output
        python synthesize.py --checkpoint model.pt
    """
    parser = argparse.ArgumentParser(
        description='Tacotron2 Text-to-Speech Synthesis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained Tacotron2 model checkpoint file')
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--text', type=str,
                           help='Single text string to synthesize')
    input_group.add_argument('--text_file', type=str,
                           help='File containing multiple texts to synthesize (one per line)')
    
    # Output options
    parser.add_argument('--output', type=str, default='output.wav',
                       help='Output audio file path for single synthesis')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for batch synthesis')
    
    # Visualization options
    parser.add_argument('--save_mel', action='store_true',
                       help='Save mel spectrogram visualization plot')
    parser.add_argument('--save_alignment', action='store_true',
                       help='Save attention alignment visualization plot')
    
    # Logging options
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Set the logging verbosity level')
    
    args = parser.parse_args()
    
    # Set logging level based on user input
    logger.setLevel(getattr(logging, args.log_level.upper()))
    logger.info(f"Starting Tacotron2 synthesis with log level: {args.log_level}")
    
    try:
        # Initialize synthesizer with checkpoint
        synthesizer = Tacotron2Synthesizer(args.checkpoint)
        
        if args.text:
            # Single text synthesis mode
            logger.info(f"Single synthesis mode with text: '{args.text}'")
            synthesizer.synthesize(
                args.text, 
                args.output, 
                save_mel=args.save_mel, 
                save_alignment=args.save_alignment
            )
            logger.info(f"Synthesis completed. Output: {args.output}")
            
        elif args.text_file:
            # Batch synthesis mode from text file
            logger.info(f"Batch synthesis mode with file: {args.text_file}")
            try:
                with open(args.text_file, 'r', encoding='utf-8') as f:
                    texts = [line.strip() for line in f.readlines() if line.strip()]
                
                if not texts:
                    logger.error("Text file is empty or contains no valid text")
                    sys.exit(1)
                
                logger.info(f"Found {len(texts)} texts to synthesize")
                synthesizer.batch_synthesize(texts, args.output_dir)
                
            except FileNotFoundError:
                logger.error(f"Text file not found: {args.text_file}")
                sys.exit(1)
            except Exception as e:
                logger.error(f"Error reading text file: {e}")
                sys.exit(1)
                
        else:
            # Interactive mode
            logger.info("Starting interactive synthesis mode")
            print("Interactive Tacotron2 Synthesis")
            print("Type 'quit' or 'exit' to terminate")
            print("-" * 40)
            
            while True:
                text = input("\nEnter text to synthesize: ").strip()
                
                if text.lower() in ['quit', 'exit', 'q']:
                    logger.info("Interactive mode terminated by user")
                    break
                
                if not text:
                    print("Please enter some text")
                    continue
                
                # Generate unique output filename with timestamp
                timestamp = int(time.time())
                output_path = f"interactive_synthesis_{timestamp}.wav"
                
                try:
                    synthesizer.synthesize(
                        text, 
                        output_path, 
                        save_mel=True, 
                        save_alignment=True
                    )
                    print(f"Audio saved as: {output_path}")
                    
                except Exception as e:
                    logger.error(f"Interactive synthesis failed: {e}")
                    print(f"Error: {e}. Please try again.")
    
    except Exception as e:
        logger.critical(f"Fatal error in main execution: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    """Entry point for the Tacotron2 synthesis script."""
    main()