"""
Tacotron2 model layers

This module contains the core neural network layers for the Tacotron2 text-to-speech model.
Includes encoder, decoder, attention mechanism, and post-processing components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
from scipy.signal import get_window
import librosa.util as librosa_util
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn
import logging

# Set up logger
logger = logging.getLogger(__name__)

class LinearNorm(torch.nn.Module):
    """
    Linear layer with Xavier uniform initialization and configurable gain.
    
    This layer applies a linear transformation to the input data with proper
    weight initialization for stable training.
    
    Args:
        in_dim (int): Input dimension size
        out_dim (int): Output dimension size
        bias (bool): Whether to include bias term
        w_init_gain (str): Weight initialization gain function ('linear', 'relu', 'tanh', etc.)
    """
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        """
        Forward pass through the linear layer.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, in_dim)
            
        Returns:
            Tensor: Output tensor of shape (batch_size, out_dim)
        """
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    """
    1D convolutional layer with proper padding and Xavier uniform initialization.
    
    This layer applies a 1D convolution to the input signal with automatic
    padding calculation and configurable weight initialization.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolutional kernel
        stride (int): Stride of the convolution
        padding (int, optional): Padding size. If None, calculates symmetric padding
        dilation (int): Dilation rate of the convolution
        bias (bool): Whether to include bias term
        w_init_gain (str): Weight initialization gain function
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        """
        Forward pass through the convolutional layer.
        
        Args:
            signal (Tensor): Input tensor of shape (batch_size, in_channels, sequence_length)
            
        Returns:
            Tensor: Output tensor of shape (batch_size, out_channels, sequence_length)
        """
        conv_signal = self.conv(signal)
        return conv_signal


class LocationLayer(nn.Module):
    """
    Location-aware attention layer that processes attention weights.
    
    This layer converts previous attention weights into location features
    that help the model attend to relevant positions in the sequence.
    
    Args:
        attention_n_filters (int): Number of filters for location convolution
        attention_kernel_size (int): Kernel size for location convolution
        attention_dim (int): Dimension of the attention space
    """
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        """
        Process concatenated attention weights through convolution and linear layers.
        
        Args:
            attention_weights_cat (Tensor): Concatenated previous and cumulative 
                attention weights of shape (batch_size, 2, max_time)
                
        Returns:
            Tensor: Processed location features of shape (batch_size, max_time, attention_dim)
        """
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    """
    Attention mechanism for aligning decoder outputs with encoder hidden states.
    
    Implements location-aware attention that considers both content and location
    information when computing alignment scores.
    
    Args:
        attention_rnn_dim (int): Dimension of attention RNN hidden state
        embedding_dim (int): Dimension of encoder embeddings
        attention_dim (int): Dimension of attention space
        attention_location_n_filters (int): Number of filters for location convolution
        attention_location_kernel_size (int): Kernel size for location convolution
    """
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        Compute alignment energies (scores) between query and memory.
        
        Args:
            query: Decoder output of shape (batch, n_mel_channels * n_frames_per_step)
            processed_memory: Processed encoder outputs of shape (B, T_in, attention_dim)
            attention_weights_cat: Cumulative and previous attention weights of shape (B, 2, max_time)

        Returns:
            Tensor: Alignment scores of shape (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        Compute attention context vector and attention weights.
        
        Args:
            attention_hidden_state: Attention RNN last output
            memory: Raw encoder outputs
            processed_memory: Processed encoder outputs
            attention_weights_cat: Previous and cumulative attention weights
            mask: Binary mask for padded data

        Returns:
            tuple: (attention_context, attention_weights)
                - attention_context: Context vector of shape (batch_size, embedding_dim)
                - attention_weights: Attention weights of shape (batch_size, max_time)
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Prenet(nn.Module):
    """
    Pre-net for processing input features before the decoder.
    
    Applies a series of linear layers with ReLU activations and dropout
    for non-linear transformation and regularization.
    
    Args:
        in_dim (int): Input dimension size
        sizes (list): List of layer sizes for the pre-net
    """
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        """
        Forward pass through the pre-net.
        
        Args:
            x (Tensor): Input tensor
            
        Returns:
            Tensor: Processed output tensor
        """
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class Postnet(nn.Module):
    """
    Post-processing network for refining mel-spectrogram predictions.
    
    Consists of multiple 1D convolutional layers with batch normalization
    and tanh activations (except the last layer) to add fine details to
    the generated spectrograms.
    
    Args:
        hparams: Hyperparameters object containing architecture parameters
    """

    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        # First convolution layer
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.n_mel_channels, hparams.postnet_embedding_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.postnet_embedding_dim))
        )

        # Middle convolution layers
        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.postnet_embedding_dim,
                             hparams.postnet_embedding_dim,
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int((hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_embedding_dim))
            )

        # Final convolution layer (linear activation)
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_embedding_dim, hparams.n_mel_channels,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hparams.n_mel_channels))
            )

    def forward(self, x):
        """
        Process mel-spectrogram predictions through the post-net.
        
        Args:
            x (Tensor): Input mel-spectrogram of shape (batch_size, n_mel_channels, time_steps)
            
        Returns:
            Tensor: Refined mel-spectrogram with added details
        """
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x


class Encoder(nn.Module):
    """
    Encoder module that processes input character embeddings.
    
    Consists of multiple 1D convolutional layers followed by a bidirectional LSTM
    to extract high-level features from input character sequences.
    
    Args:
        hparams: Hyperparameters object containing architecture parameters
    """
    def __init__(self, hparams):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hparams.encoder_embedding_dim,
                         hparams.encoder_embedding_dim,
                         kernel_size=hparams.encoder_kernel_size, stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hparams.encoder_embedding_dim,
                            int(hparams.encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        """
        Forward pass through the encoder with variable length sequences.
        
        Args:
            x (Tensor): Input character embeddings of shape (batch_size, embedding_dim, sequence_length)
            input_lengths (Tensor): Lengths of each sequence in the batch
            
        Returns:
            Tensor: Encoder outputs of shape (batch_size, sequence_length, encoder_embedding_dim)
        """
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs

    def inference(self, x):
        """
        Inference pass for single sequence without length masking.
        
        Args:
            x (Tensor): Input character embeddings for inference
            
        Returns:
            Tensor: Encoder outputs for inference
        """
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs