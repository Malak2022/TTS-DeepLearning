"""
Tacotron2 Decoder Module

This module implements the decoder component of the Tacotron2 text-to-speech model.
The decoder generates mel-spectrogram frames autoregressively using attention mechanism
to focus on relevant parts of the encoder output.

Key Components:
- Prenet: Pre-processing network for mel-spectrogram inputs
- Attention RNN: LSTM that handles attention computation
- Attention Layer: Location-aware attention mechanism
- Decoder RNN: LSTM that generates mel-spectrogram frames
- Linear Projection: Converts RNN outputs to mel-spectrogram dimensions
- Gate Layer: Predicts when to stop generation

The decoder supports both teacher-forced training and autoregressive inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# Configure logger
logger = logging.getLogger(__name__)

try:
    from .layers import Attention, Prenet, LinearNorm
except ImportError:
    try:
        from tacotron2.models.layers import Attention, Prenet, LinearNorm
    except ImportError:
        logger.error("Could not import Attention, Prenet, and LinearNorm layers")
        raise ImportError("Required layers not found. Please check the import paths.")

class Decoder(nn.Module):
    """
    Tacotron2 Decoder module that generates mel-spectrograms from encoded text representations.
    
    The decoder uses an autoregressive approach with attention mechanism to generate
    mel-spectrogram frames one step at a time, attending to different parts of the
    encoder output at each generation step.
    
    Args:
        hparams (object): Hyperparameters object containing decoder configuration
        
    Attributes:
        n_mel_channels (int): Number of mel frequency channels
        n_frames_per_step (int): Number of frames generated per decoder step
        encoder_embedding_dim (int): Dimension of encoder embeddings
        attention_rnn_dim (int): Dimension of attention LSTM hidden state
        decoder_rnn_dim (int): Dimension of decoder LSTM hidden state
        prenet_dim (int): Dimension of prenet layers
        max_decoder_steps (int): Maximum number of steps during inference
        gate_threshold (float): Threshold for stopping generation
    """
    
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        # Store hyperparameters
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout

        # Initialize network components
        self.prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step,
            [hparams.prenet_dim, hparams.prenet_dim])

        self.attention_rnn = nn.LSTMCell(
            hparams.prenet_dim + hparams.encoder_embedding_dim,
            hparams.attention_rnn_dim)

        self.attention_layer = Attention(
            hparams.attention_rnn_dim, hparams.encoder_embedding_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(
            hparams.attention_rnn_dim + hparams.encoder_embedding_dim,
            hparams.decoder_rnn_dim, 1)

        self.linear_projection = LinearNorm(
            hparams.decoder_rnn_dim + hparams.encoder_embedding_dim,
            hparams.n_mel_channels * hparams.n_frames_per_step)

        self.gate_layer = LinearNorm(
            hparams.decoder_rnn_dim + hparams.encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')
        
        logger.debug("Decoder initialized with parameters: "
                    f"n_mel_channels={self.n_mel_channels}, "
                    f"n_frames_per_step={self.n_frames_per_step}, "
                    f"attention_rnn_dim={self.attention_rnn_dim}, "
                    f"decoder_rnn_dim={self.decoder_rnn_dim}")

    def get_go_frame(self, memory):
        """
        Create initial all-zero frames for the first decoder input.
        
        Args:
            memory (Tensor): Encoder outputs used to determine batch size and device
            
        Returns:
            Tensor: Zero-initialized frames of shape (B, n_mel_channels * n_frames_per_step)
        """
        B = memory.size(0)
        decoder_input = memory.new_zeros(
            B, self.n_mel_channels * self.n_frames_per_step)
        logger.debug(f"Created go frame with shape: {decoder_input.shape}")
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """
        Initialize all decoder states and buffers for a new sequence.
        
        Args:
            memory (Tensor): Encoder outputs of shape (B, T, D_encoder)
            mask (Tensor): Boolean mask for padded encoder outputs, None for inference
            
        Initializes:
            attention_hidden, attention_cell: Attention LSTM states
            decoder_hidden, decoder_cell: Decoder LSTM states
            attention_weights: Current attention weights
            attention_weights_cum: Cumulative attention weights
            attention_context: Current attention context vector
            processed_memory: Pre-processed encoder outputs for attention
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        # Initialize RNN states with zeros
        self.attention_hidden = memory.new_zeros(B, self.attention_rnn_dim)
        self.attention_cell = memory.new_zeros(B, self.attention_rnn_dim)

        self.decoder_hidden = memory.new_zeros(B, self.decoder_rnn_dim)
        self.decoder_cell = memory.new_zeros(B, self.decoder_rnn_dim)

        # Initialize attention-related buffers
        self.attention_weights = memory.new_zeros(B, MAX_TIME)
        self.attention_weights_cum = memory.new_zeros(B, MAX_TIME)
        self.attention_context = memory.new_zeros(
            B, self.encoder_embedding_dim)

        # Store references to encoder outputs
        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask
        
        logger.debug(f"Initialized decoder states for batch size {B}, max time {MAX_TIME}")

    def parse_decoder_inputs(self, decoder_inputs):
        """
        Prepare decoder inputs for teacher-forced training.
        
        Args:
            decoder_inputs (Tensor): Ground truth mel-spectrograms of shape (B, n_mels, T)
            
        Returns:
            Tensor: Processed decoder inputs of shape (T_out, B, prenet_dim)
        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        # Group frames according to n_frames_per_step
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        
        logger.debug(f"Parsed decoder inputs from shape {decoder_inputs.shape} "
                    f"to {decoder_inputs.shape}")
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """
        Convert decoder outputs from sequence format to final tensor format.
        
        Args:
            mel_outputs (list): List of mel output tensors from each time step
            gate_outputs (list): List of gate output tensors from each time step
            alignments (list): List of attention alignment tensors from each time step
            
        Returns:
            tuple: (mel_outputs, gate_outputs, alignments) in final tensor format
        """
        # Convert lists to tensors and transpose dimensions
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # Decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        logger.debug(f"Parsed decoder outputs: mel_shape={mel_outputs.shape}, "
                    f"gate_shape={gate_outputs.shape}, align_shape={alignments.shape}")
        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        """
        Perform a single decoder step using stored states and attention.
        
        Args:
            decoder_input (Tensor): Input for current decoder step
            
        Returns:
            tuple: (mel_output, gate_output, attention_weights) for current step
        """
        # Concatenate decoder input with previous attention context
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        
        # Forward pass through attention RNN
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)

        # Compute attention weights and context
        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)

        # Update cumulative attention weights
        self.attention_weights_cum += self.attention_weights
        
        # Prepare input for decoder RNN
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        
        # Forward pass through decoder RNN
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)

        # Generate mel output and gate prediction
        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths):
        """
        Forward pass for teacher-forced training.
        
        Args:
            memory (Tensor): Encoder outputs of shape (B, T, D_encoder)
            decoder_inputs (Tensor): Ground truth mel-spectrograms for teacher forcing
            memory_lengths (Tensor): Lengths of encoder sequences for masking
            
        Returns:
            tuple: (mel_outputs, gate_outputs, alignments) for the entire sequence
        """
        logger.debug(f"Starting forward pass: memory_shape={memory.shape}, "
                    f"decoder_inputs_shape={decoder_inputs.shape}")

        # Prepare initial frame and process decoder inputs
        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        # Initialize decoder states
        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths))

        # Generate outputs step by step
        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(
                decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [attention_weights]

        # Parse and return final outputs
        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)
        
        logger.debug("Forward pass completed successfully")
        return mel_outputs, gate_outputs, alignments

    def inference(self, memory):
        """
        Autoregressive inference pass for generating mel-spectrograms.
        
        Args:
            memory (Tensor): Encoder outputs of shape (B, T, D_encoder)
            
        Returns:
            tuple: (mel_outputs, gate_outputs, alignments) for the generated sequence
            
        Note:
            Generation stops when gate output exceeds threshold or max steps reached
        """
        logger.debug(f"Starting inference: memory_shape={memory.shape}")

        # Initialize with zero frame
        decoder_input = self.get_go_frame(memory)
        self.initialize_decoder_states(memory, mask=None)

        # Generate outputs autoregressively
        mel_outputs, gate_outputs, alignments = [], [], []
        step = 0
        
        while True:
            # Preprocess input and decode
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)

            # Store outputs
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            # Check stopping conditions
            gate_prob = torch.sigmoid(gate_output.data)
            if gate_prob > self.gate_threshold:
                logger.debug(f"Stopping generation at step {step}: gate_prob={gate_prob.item():.4f}")
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                logger.warning(f"Reached max decoder steps ({self.max_decoder_steps}) - stopping generation")
                break

            # Use generated output as next input
            decoder_input = mel_output
            step += 1

        # Parse and return final outputs
        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)
        
        logger.info(f"Inference completed: generated {len(mel_outputs)} steps, "
                   f"final gate probability: {gate_prob.item():.4f}")
        return mel_outputs, gate_outputs, alignments


def get_mask_from_lengths(lengths):
    """
    Create a boolean mask from sequence lengths.
    
    Args:
        lengths (Tensor): 1D tensor containing sequence lengths
        
    Returns:
        Tensor: Boolean mask of shape (B, max_len) where True indicates valid positions
    """
    max_len = torch.max(lengths).item()
    if lengths.is_cuda:
        ids = torch.arange(0, max_len, device=lengths.device, dtype=torch.long)
    else:
        ids = torch.arange(0, max_len, dtype=torch.long)
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask