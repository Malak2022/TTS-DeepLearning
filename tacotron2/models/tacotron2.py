"""
Tacotron2 Model

This module implements the complete Tacotron2 text-to-speech model architecture.
It combines encoder, decoder, and postnet components to convert text inputs
into mel-spectrogram outputs.
"""

import torch
import torch.nn as nn
import math
try:
    from .layers import Encoder, Postnet
    from .decoder import Decoder
except ImportError:
    from tacotron2.models.layers import Encoder, Postnet
    from tacotron2.models.decoder import Decoder

class Tacotron2(nn.Module):
    """
    Tacotron2 end-to-end text-to-speech model.
    
    This model takes text inputs and generates mel-spectrogram outputs through
    a sequence-to-sequence architecture with attention mechanism.
    
    Args:
        hparams: Hyperparameters object containing model configuration
    """
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        
        # Character embedding layer
        self.embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim)
        
        # Initialize embedding weights with uniform distribution
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        
        # Model components
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)

    def parse_batch(self, batch):
        """
        Parse and prepare batch data for training.
        
        Args:
            batch: Tuple containing (text_padded, input_lengths, mel_padded, 
                   gate_padded, output_lengths)
                   
        Returns:
            tuple: (inputs, targets) where:
                - inputs: (text_padded, input_lengths, mel_padded, max_len, output_lengths)
                - targets: (mel_padded, gate_padded)
        """
        text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths = batch
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths),
            (mel_padded, gate_padded))

    def parse_output(self, outputs, output_lengths=None):
        """
        Apply padding mask to model outputs for proper loss calculation.
        
        Args:
            outputs: List of model outputs [mel_outputs, mel_outputs_postnet, 
                     gate_outputs, alignments]
            output_lengths: Tensor containing actual lengths of each sequence
                           in the batch
                           
        Returns:
            list: Masked outputs with padding positions set to appropriate values
        """
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            # Mask mel outputs with 0.0
            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            # Mask gate energies with large value to avoid affecting softmax
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)

        return outputs

    def forward(self, inputs):
        """
        Forward pass through the complete Tacotron2 model.
        
        Args:
            inputs: Tuple containing (text_inputs, text_lengths, mels, max_len, output_lengths)
            
        Returns:
            list: Model outputs [mel_outputs, mel_outputs_postnet, gate_outputs, alignments]
        """
        text_inputs, text_lengths, mels, max_len, output_lengths = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data

        # Embed input characters
        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)

        # Encode input sequence
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        # Decode with attention to generate mel-spectrograms
        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=text_lengths)

        # Apply post-processing network to refine mel outputs
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths)

    def inference(self, inputs):
        """
        Inference pass for text-to-speech synthesis.
        
        Args:
            inputs: Text input tensor for synthesis
            
        Returns:
            list: Model outputs [mel_outputs, mel_outputs_postnet, gate_outputs, alignments]
        """
        # Embed input characters
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        
        # Encode without length information (for single sequence)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        
        # Decode with autoregressive generation
        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)

        # Apply post-processing
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = [mel_outputs, mel_outputs_postnet, gate_outputs, alignments]

        return outputs


def sqrt(x):
    """
    Square root function wrapper for math.sqrt.
    
    Args:
        x: Input value
        
    Returns:
        float: Square root of x
    """
    return math.sqrt(x)


def to_gpu(x):
    """
    Move tensor to GPU if available.
    
    Args:
        x: Input tensor
        
    Returns:
        Variable: Tensor moved to GPU with gradient tracking
    """
    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)


def get_mask_from_lengths(lengths):
    """
    Create a boolean mask from sequence lengths.
    
    Args:
        lengths: Tensor containing sequence lengths
        
    Returns:
        Tensor: Boolean mask where True indicates valid positions
                and False indicates padding positions
    """
    max_len = torch.max(lengths).item()
    if lengths.is_cuda:
        ids = torch.arange(0, max_len, device=lengths.device, dtype=torch.long)
    else:
        ids = torch.arange(0, max_len, dtype=torch.long)
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


class HParams:
    """
    Hyperparameters class for Tacotron2 model configuration.
    
    This class stores all the hyperparameters needed to configure
    the Tacotron2 model architecture and training process.
    """
    def __init__(self):
        # Symbols and embedding parameters
        self.n_symbols = 43  # Based on your character set (updated)
        self.symbols_embedding_dim = 512

        # Encoder parameters
        self.encoder_kernel_size = 5
        self.encoder_n_convolutions = 3
        self.encoder_embedding_dim = 512

        # Decoder parameters
        self.n_frames_per_step = 1
        self.decoder_rnn_dim = 1024
        self.prenet_dim = 256
        self.max_decoder_steps = 1000
        self.gate_threshold = 0.5
        self.p_attention_dropout = 0.1
        self.p_decoder_dropout = 0.1

        # Attention parameters
        self.attention_rnn_dim = 1024
        self.attention_dim = 128

        # Location Layer parameters
        self.attention_location_n_filters = 32
        self.attention_location_kernel_size = 31

        # Mel-post processing network parameters
        self.n_mel_channels = 80
        self.postnet_embedding_dim = 512
        self.postnet_kernel_size = 5
        self.postnet_n_convolutions = 5

        # Training parameters
        self.mask_padding = True
        self.fp16_run = False

        # Audio parameters
        self.max_wav_value = 32768.0
        self.sampling_rate = 22050
        self.filter_length = 1024
        self.hop_length = 256
        self.win_length = 1024
        self.mel_fmin = 0.0
        self.mel_fmax = 8000.0

        # Data parameters
        self.load_mel_from_disk = True
        self.text_cleaners = ['basic_cleaners']