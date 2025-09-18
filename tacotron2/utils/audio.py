"""
Audio processing utilities for Tacotron2
"""

import torch
import torch.nn.functional as F
import numpy as np
import librosa
import logging
from scipy.signal import get_window
import librosa.util as librosa_util
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    Apply dynamic range compression (numpy version).

    Args:
        x (np.ndarray): Input signal.
        C (float): Compression constant.
        clip_val (float): Minimum value for clipping.

    Returns:
        np.ndarray: Compressed signal.
    """
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

def dynamic_range_decompression(x, C=1):
    """
    Apply dynamic range decompression (numpy version).

    Args:
        x (np.ndarray): Compressed signal.
        C (float): Compression constant.

    Returns:
        np.ndarray: Decompressed signal.
    """
    return np.exp(x) / C

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    Apply dynamic range compression (PyTorch version).

    Args:
        x (torch.Tensor): Input tensor.
        C (float): Compression constant.
        clip_val (float): Minimum value for clamping.

    Returns:
        torch.Tensor: Compressed tensor.
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    """
    Apply dynamic range decompression (PyTorch version).

    Args:
        x (torch.Tensor): Compressed tensor.
        C (float): Compression constant.

    Returns:
        torch.Tensor: Decompressed tensor.
    """
    return torch.exp(x) / C

def spectral_normalize_torch(magnitudes):
    """
    Normalize spectrogram magnitudes with dynamic range compression.

    Args:
        magnitudes (torch.Tensor): Spectrogram magnitudes.

    Returns:
        torch.Tensor: Normalized magnitudes.
    """
    return dynamic_range_compression_torch(magnitudes)

def spectral_de_normalize_torch(magnitudes):
    """
    De-normalize spectrogram magnitudes with dynamic range decompression.

    Args:
        magnitudes (torch.Tensor): Normalized magnitudes.

    Returns:
        torch.Tensor: De-normalized magnitudes.
    """
    return dynamic_range_decompression_torch(magnitudes)

mel_basis = {}
hann_window = {}

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    """
    Compute mel-spectrogram from waveform.

    Args:
        y (torch.Tensor): Input waveform tensor.
        n_fft (int): FFT size.
        num_mels (int): Number of mel bins.
        sampling_rate (int): Audio sampling rate.
        hop_size (int): Hop length.
        win_size (int): Window size.
        fmin (int): Minimum frequency.
        fmax (int): Maximum frequency.
        center (bool): Whether to pad waveform before STFT.

    Returns:
        torch.Tensor: Mel-spectrogram tensor.
    """
    if torch.min(y) < -1.:
        logger.warning("Min value is %f", torch.min(y).item())
    if torch.max(y) > 1.:
        logger.warning("Max value is %f", torch.max(y).item())

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec

class STFT(torch.nn.Module):
    """
    Short-Time Fourier Transform (STFT) and inverse transform module.

    Adapted from Prem Seetharaman's implementation:
    https://github.com/pseeth/pytorch-stft
    """
    def __init__(self, filter_length=1024, hop_length=512, win_length=1024,
                 window='hann'):
        """
        Initialize STFT module.

        Args:
            filter_length (int): FFT filter length.
            hop_length (int): Hop length.
            win_length (int): Window length.
            window (str): Window function type.
        """
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        if window is not None:
            assert(filter_length >= win_length)
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        """
        Compute the forward STFT.

        Args:
            input_data (torch.Tensor): Input audio tensor.

        Returns:
            tuple: Magnitude and phase tensors.
        """
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples

        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode='reflect')
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            torch.autograd.Variable(self.forward_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.autograd.Variable(
            torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase

    def inverse(self, magnitude, phase):
        """
        Compute the inverse STFT.

        Args:
            magnitude (torch.Tensor): Magnitude tensor.
            phase (torch.Tensor): Phase tensor.

        Returns:
            torch.Tensor: Reconstructed audio.
        """
        recombine_magnitude_phase = torch.cat(
            [magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            torch.autograd.Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        if self.window is not None:
            window_sum = window_sumsquare(
                self.window, magnitude.size(-1), hop_length=self.hop_length,
                win_length=self.win_length, n_fft=self.filter_length,
                dtype=np.float32)
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0])
            window_sum = torch.autograd.Variable(
                torch.from_numpy(window_sum), requires_grad=False)
            window_sum = window_sum.cuda() if magnitude.is_cuda else window_sum
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(self.filter_length/2):]
        inverse_transform = inverse_transform[:, :, :-int(self.filter_length/2):]

        return inverse_transform

    def forward(self, input_data):
        """
        Forward pass of STFT.

        Args:
            input_data (torch.Tensor): Input audio tensor.

        Returns:
            torch.Tensor: Reconstructed audio from STFT.
        """
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction

def pad_center(data, size, axis=-1, **kwargs):
    """
    Pad data on both sides to a target size.

    Args:
        data (np.ndarray): Input array.
        size (int): Target size.
        axis (int): Axis along which to pad.

    Returns:
        np.ndarray: Padded array.
    """
    kwargs.setdefault('mode', 'constant')
    n = data.shape[axis]
    lpad = int((size - n) // 2)
    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))
    if lpad < 0:
        raise ValueError(f"Target size ({size}) must be at least input size ({n})")
    return np.pad(data, lengths, **kwargs)

def window_sumsquare(window, n_frames, hop_length=512, win_length=None,
                     n_fft=2048, dtype=np.float32, norm=None):
    """
    Compute the squared sum of a window function across frames.

    Args:
        window (str): Window function type.
        n_frames (int): Number of frames.
        hop_length (int): Hop length.
        win_length (int): Window length.
        n_fft (int): FFT size.
        dtype: Data type.
        norm: Normalization.

    Returns:
        np.ndarray: Window sum.
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = normalize(win_sq, norm=norm)**2
    win_sq = pad_center(win_sq, n_fft)

    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x

def tiny(x):
    """
    Return the smallest positive usable number for a given dtype.

    Args:
        x (np.ndarray): Input array.

    Returns:
        float: Smallest positive number representable by dtype.
    """
    return np.finfo(x.dtype).tiny
