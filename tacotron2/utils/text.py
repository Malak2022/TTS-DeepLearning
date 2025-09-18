"""
Text processing utilities for Tacotron2
"""

import re
import unicodedata
import torch
import numpy as np

# Character mappings (matching preprocessing)
_pad = '_'
_eos = '~'
_characters = 'abcdefghijklmnopqrstuvwxyz1234567890!?., '

# Create mappings for symbols to IDs and vice versa
symbols = [_pad] + list(_characters) + [_eos]
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def clean_text(text):
    """
    Clean and normalize input text for Tacotron2 processing.

    Steps:
    - Convert text to lowercase.
    - Normalize unicode characters using NFKD normalization.
    - Remove characters not in the allowed set [a-zA-Z0-9\s.,!?].
    - Replace multiple spaces with a single space.

    Args:
        text (str): Input text string.

    Returns:
        str: Cleaned and normalized text.
    """
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def text_to_sequence(text, cleaner_names=['basic_cleaners']):
    """
    Convert a text string into a sequence of integer IDs.

    - Cleans the text.
    - Maps each character to its corresponding ID.
    - Appends an End-of-Sequence (EOS) token at the end.

    Args:
        text (str): Input text string.
        cleaner_names (list): List of text cleaners (not fully used here).

    Returns:
        list[int]: Sequence of integer IDs representing the text.
    """
    text = clean_text(text)
    sequence = []
    for char in text:
        if char in _symbol_to_id:
            sequence.append(_symbol_to_id[char])
        else:
            # Skip unknown characters
            continue
    
    sequence.append(_symbol_to_id[_eos])
    return sequence


def sequence_to_text(sequence):
    """
    Convert a sequence of IDs back to text.

    - Maps each integer ID to its corresponding character.
    - Stops at the EOS token.
    - Ignores padding tokens.

    Args:
        sequence (list[int]): Sequence of integer IDs.

    Returns:
        str: Reconstructed text string.
    """
    result = []
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            if s == _eos:
                break
            elif s != _pad:
                result.append(s)
    return ''.join(result)


def get_mask_from_lengths(lengths):
    """
    Create a boolean mask from input lengths.

    Each sequence is masked to indicate valid positions (True)
    and padded positions (False).

    Args:
        lengths (torch.Tensor): 1D tensor of sequence lengths.

    Returns:
        torch.BoolTensor: Boolean mask of shape [batch_size, max_len].
    """
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def to_gpu(x):
    """
    Move a tensor to GPU if available.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.autograd.Variable: Tensor placed on GPU if available.
    """
    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)


class TextMelLoader(torch.utils.data.Dataset):
    """
    A Dataset for loading text and audio pairs for Tacotron2.

    Responsibilities:
    1. Load audio-text pairs from disk.
    2. Normalize text and convert to sequences of IDs.
    3. Compute mel-spectrograms from audio files.

    Args:
        audiopaths_and_text (list): List of (audio_path, text) pairs.
        hparams (object): Hyperparameters object containing:
            - text_cleaners
            - max_wav_value
            - sampling_rate
            - load_mel_from_disk
            - filter_length
            - hop_length
            - win_length
            - n_mel_channels
            - mel_fmin
            - mel_fmax
            - seed
    """
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = audiopaths_and_text
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)

    def get_mel_text_pair(self, audiopath_and_text):
        """
        Process a single (audio, text) pair.

        Args:
            audiopath_and_text (tuple): (audio_path, text).

        Returns:
            tuple: (text_tensor, mel_spectrogram).
        """
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        return (text, mel)

    def get_mel(self, filename):
        """
        Compute or load mel-spectrogram from an audio file.

        Args:
            filename (str): Path to the audio file or saved mel-spectrogram.

        Returns:
            torch.FloatTensor: Mel-spectrogram tensor.
        """
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text):
        """
        Normalize and convert text to a tensor of symbol IDs.

        Args:
            text (str): Input text string.

        Returns:
            torch.IntTensor: Tensor of symbol IDs.
        """
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate:
    """
    Collate function to prepare batches of text and mel-spectrogram pairs.

    - Pads text sequences with zeros.
    - Pads mel-spectrograms with zeros.
    - Generates gate tensors for stop-token prediction.

    Args:
        n_frames_per_step (int): Number of frames per decoder step.
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """
        Collate a batch of (text, mel) pairs.

        Args:
            batch (list): List of (text_tensor, mel_spectrogram).

        Returns:
            tuple:
                - text_padded (LongTensor): Padded text sequences.
                - input_lengths (LongTensor): Lengths of input sequences.
                - mel_padded (FloatTensor): Padded mel-spectrograms.
                - gate_padded (FloatTensor): Gate values (stop tokens).
                - output_lengths (LongTensor): Lengths of mel-spectrograms.
        """
        # Right zero-pad all text sequences
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spectrograms
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))

        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, gate_padded, output_lengths
