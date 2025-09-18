"""
Loss functions for Tacotron2

This module contains various loss functions used in the Tacotron2 text-to-speech model.
The losses include mel spectrogram reconstruction loss, gate prediction loss, and
attention alignment losses to encourage proper alignment during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Tacotron2Loss(nn.Module):
    """
    Tacotron2 Loss Function
    Combines mel spectrogram loss and gate loss
    
    Args:
        mel_loss_weight (float): Weight for the mel spectrogram loss component
        gate_loss_weight (float): Weight for the gate loss component
    """
    def __init__(self, mel_loss_weight=1.0, gate_loss_weight=1.0):
        super(Tacotron2Loss, self).__init__()
        self.mel_loss_weight = mel_loss_weight
        self.gate_loss_weight = gate_loss_weight
        
    def forward(self, model_output, targets):
        """
        Calculate loss
        
        Args:
            model_output: Tuple of (mel_out, mel_out_postnet, gate_out, alignments)
            targets: Tuple of (mel_target, gate_target)
            
        Returns:
            total_loss, mel_loss, gate_loss
        """
        mel_target, gate_target = targets
        mel_out, mel_out_postnet, gate_out, _ = model_output
        
        # Mel spectrogram loss (L1 loss)
        mel_loss = F.l1_loss(mel_out, mel_target) + F.l1_loss(mel_out_postnet, mel_target)
        
        # Gate loss (BCE loss)
        gate_loss = F.binary_cross_entropy_with_logits(gate_out, gate_target)
        
        # Total loss
        total_loss = self.mel_loss_weight * mel_loss + self.gate_loss_weight * gate_loss
        
        return total_loss, mel_loss, gate_loss


class MelSpectrogramLoss(nn.Module):
    """
    Mel spectrogram L1 loss
    
    This loss measures the L1 distance between predicted and target mel spectrograms.
    It can be applied to both the pre- and post-net outputs.
    """
    def __init__(self):
        super(MelSpectrogramLoss, self).__init__()
        
    def forward(self, mel_out, mel_target, mel_out_postnet=None):
        """
        Calculate mel spectrogram loss
        
        Args:
            mel_out: Predicted mel spectrogram from the decoder
            mel_target: Target mel spectrogram
            mel_out_postnet: Post-processed mel spectrogram from the postnet (optional)
            
        Returns:
            mel_loss: L1 loss between predictions and targets
        """
        mel_loss = F.l1_loss(mel_out, mel_target)
        
        if mel_out_postnet is not None:
            mel_loss += F.l1_loss(mel_out_postnet, mel_target)
            
        return mel_loss


class GateLoss(nn.Module):
    """
    Gate prediction loss (binary cross entropy)
    
    This loss measures how well the model predicts the end of speech frames.
    
    Args:
        pos_weight (Tensor, optional): A weight of positive examples to handle class imbalance
    """
    def __init__(self, pos_weight=None):
        super(GateLoss, self).__init__()
        self.pos_weight = pos_weight
        
    def forward(self, gate_out, gate_target):
        """
        Calculate gate loss
        
        Args:
            gate_out: Predicted gate values (logits)
            gate_target: Target gate values (binary)
            
        Returns:
            gate_loss: Binary cross entropy loss
        """
        if self.pos_weight is not None:
            gate_loss = F.binary_cross_entropy_with_logits(
                gate_out, gate_target, pos_weight=self.pos_weight)
        else:
            gate_loss = F.binary_cross_entropy_with_logits(gate_out, gate_target)
            
        return gate_loss


class AttentionLoss(nn.Module):
    """
    Attention alignment loss to encourage monotonic attention
    
    This loss penalizes attention weights that violate the monotonic property
    (i.e., attention that moves backward in the input sequence).
    """
    def __init__(self):
        super(AttentionLoss, self).__init__()
        
    def forward(self, alignments):
        """
        Calculate attention loss
        
        Args:
            alignments: Attention weights [B, T_out, T_in]
            
        Returns:
            attention_loss: Penalty for non-monotonic attention
        """
        # Encourage monotonic attention by penalizing backward attention
        batch_size, max_time_out, max_time_in = alignments.size()
        
        # Create a mask for forward attention
        forward_mask = torch.triu(torch.ones(max_time_out, max_time_in), diagonal=0)
        if alignments.is_cuda:
            forward_mask = forward_mask.cuda()
            
        # Calculate loss as the sum of attention weights that violate monotonicity
        backward_attention = alignments * (1 - forward_mask)
        attention_loss = torch.sum(backward_attention) / batch_size
        
        return attention_loss


class GuidedAttentionLoss(nn.Module):
    """
    Guided attention loss to encourage diagonal attention alignment
    
    This loss encourages the attention matrix to follow a diagonal pattern,
    which is typical for speech synthesis where input and output are roughly aligned.
    
    Args:
        sigma (float): Width of the Gaussian mask around the diagonal
    """
    def __init__(self, sigma=0.4):
        super(GuidedAttentionLoss, self).__init__()
        self.sigma = sigma
        
    def forward(self, alignments, input_lengths, output_lengths):
        """
        Calculate guided attention loss
        
        Args:
            alignments: Attention weights [B, T_out, T_in]
            input_lengths: Input sequence lengths
            output_lengths: Output sequence lengths
            
        Returns:
            guided_attention_loss: Penalty for non-diagonal attention
        """
        batch_size, max_time_out, max_time_in = alignments.size()
        
        # Create guided attention matrix
        guided_attention = torch.zeros_like(alignments)
        
        for b in range(batch_size):
            T_in = input_lengths[b].item()
            T_out = output_lengths[b].item()
            
            for i in range(T_out):
                for j in range(T_in):
                    # Calculate the expected alignment position
                    expected_j = (j / T_in) * T_out
                    # Calculate Gaussian weight
                    w = 1.0 - torch.exp(-((i - expected_j) ** 2) / (2 * self.sigma ** 2))
                    guided_attention[b, i, j] = w
        
        # Calculate loss
        loss = torch.sum(alignments * guided_attention) / batch_size
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss function with multiple loss components
    
    This loss combines mel spectrogram loss, gate loss, and optional attention losses
    into a single weighted loss function.
    
    Args:
        mel_weight (float): Weight for mel spectrogram loss
        gate_weight (float): Weight for gate loss
        attention_weight (float): Weight for attention monotonicity loss
        guided_attention_weight (float): Weight for guided attention loss
    """
    def __init__(self, mel_weight=1.0, gate_weight=1.0, attention_weight=0.1, guided_attention_weight=0.1):
        super(CombinedLoss, self).__init__()
        self.mel_loss = MelSpectrogramLoss()
        self.gate_loss = GateLoss()
        self.attention_loss = AttentionLoss()
        self.guided_attention_loss = GuidedAttentionLoss()
        
        self.mel_weight = mel_weight
        self.gate_weight = gate_weight
        self.attention_weight = attention_weight
        self.guided_attention_weight = guided_attention_weight
        
    def forward(self, model_output, targets, input_lengths=None, output_lengths=None):
        """
        Calculate combined loss
        
        Args:
            model_output: Tuple of (mel_out, mel_out_postnet, gate_out, alignments)
            targets: Tuple of (mel_target, gate_target)
            input_lengths: Input sequence lengths (optional, needed for guided attention)
            output_lengths: Output sequence lengths (optional, needed for guided attention)
            
        Returns:
            Dictionary of losses with keys: total_loss, mel_loss, gate_loss, 
            and optionally attention_loss and guided_attention_loss
        """
        mel_target, gate_target = targets
        mel_out, mel_out_postnet, gate_out, alignments = model_output
        
        # Calculate individual losses
        mel_loss = self.mel_loss(mel_out, mel_target, mel_out_postnet)
        gate_loss = self.gate_loss(gate_out, gate_target)
        
        total_loss = self.mel_weight * mel_loss + self.gate_weight * gate_loss
        
        losses = {
            'total_loss': total_loss,
            'mel_loss': mel_loss,
            'gate_loss': gate_loss
        }
        
        # Add attention losses if enabled
        if self.attention_weight > 0:
            att_loss = self.attention_loss(alignments)
            losses['attention_loss'] = att_loss
            total_loss += self.attention_weight * att_loss
            
        if self.guided_attention_weight > 0 and input_lengths is not None and output_lengths is not None:
            guided_att_loss = self.guided_attention_loss(alignments, input_lengths, output_lengths)
            losses['guided_attention_loss'] = guided_att_loss
            total_loss += self.guided_attention_weight * guided_att_loss
            
        losses['total_loss'] = total_loss
        
        return losses