"""
FoldFormer: a simplified PyTorch implementation
------------------------------------------------

This module contains a high-level implementation of the core ideas behind the FoldFormer architecture proposed in the paper
"FoldFormer: sequence folding and seasonal attention for fine-grained long-term FaaS forecasting".
It includes time-to-latent folding, FFT-based convolutions, seasonal attention, and encoder/decoder blocks.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeToLatentFold(nn.Module):
    """
    Time-to-latent folding module. It folds a 1D sequence by a factor F and projects
    each folded block into a latent space of dimension d_model via a 1D convolution.
    The forward method returns a sequence of length seq_len // fold_factor, and the unfold
    method projects it back to the original sequence length using a learned linear projection.
    """
    def __init__(self, input_dim: int, d_model: int, fold_factor: int) -> None:
        super().__init__()
        if fold_factor <= 0:
            raise ValueError("fold_factor must be positive")
        self.fold_factor = fold_factor
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=d_model, kernel_size=fold_factor, stride=fold_factor)
        self.out_proj = nn.Linear(d_model, fold_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, input_dim = x.shape
        if seq_len % self.fold_factor != 0:
            raise ValueError("Sequence length must be divisible by fold_factor")
        x_c = x.transpose(1, 2)
        folded = self.conv(x_c)
        return folded.transpose(1, 2)

    def unfold(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n, d_model = x.shape
        x_proj = self.out_proj(x)
        return x_proj.reshape(bsz, n * self.fold_factor, 1)

class FFTConv(nn.Module):
    """
    Convolution layer operating in the frequency domain. It concatenates the real and imaginary
    parts of the FFT of the input and applies a 1D convolution across the frequency axis.
    """
    def __init__(self, d_model: int, hidden_dim: int, kernel_size: int) -> None:
        super().__init__()
        self.freq_conv = nn.Conv1d(in_channels=2 * d_model, out_channels=d_model, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_freq = torch.fft.rfft(x, dim=1)
        freq_concat = torch.cat([x_freq.real, x_freq.imag], dim=-1)
        freq_input = freq_concat.permute(0, 2, 1)
        freq_out = self.freq_conv(freq_input)
        freq_out = freq_out.permute(0, 2, 1)
        seq_len = x.size(1)
        out_interp = F.interpolate(freq_out.transpose(1, 2), size=seq_len, mode='nearest').transpose(1, 2)
        return out_interp

class SeasonalAttention(nn.Module):
    """
    Seasonal attention mechanism. It summarises each past sequence into a single token and computes
    attention scores between the current sequence summary and past summaries. The weighted sum of
    past summaries is expanded back to match the temporal dimension of the current sequence.
    """
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, current: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        bsz, num_periods, seq_len, d_model = past.shape
        past_summary = past.mean(dim=2)
        current_summary = current.mean(dim=1, keepdim=True)
        q = self.q_proj(current_summary)
        k = self.k_proj(past_summary)
        v = self.v_proj(past_summary)
        scores = torch.matmul(q, k.transpose(-2, -1)).squeeze(1) * self.scale
        weights = F.softmax(scores, dim=-1)
        aggregated = torch.matmul(weights.unsqueeze(1), v).squeeze(1)
        aggregated_expanded = aggregated.unsqueeze(1).expand(-1, seq_len, -1)
        return aggregated_expanded

class FoldFormerEncoderLayer(nn.Module):
    """
    Single encoder layer: self-attention, FFT convolution, and feed-forward network.
    """
    def __init__(self, d_model: int, nhead: int, ff_dim: int, kernel_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.fft_conv = FFTConv(d_model, hidden_dim=d_model, kernel_size=kernel_size)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        conv_out = self.fft_conv(x)
        x = x + self.dropout(conv_out)
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm3(x)
        return x

class FoldFormerDecoderLayer(nn.Module):
    """
    Single decoder layer: self-attention, seasonal cross-attention, FFT convolution, and feed-forward network.
    """
    def __init__(self, d_model: int, nhead: int, ff_dim: int, kernel_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.seasonal_attn = SeasonalAttention(d_model)
        self.fft_conv = FFTConv(d_model, hidden_dim=d_model, kernel_size=kernel_size)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, past_encodings: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        seasonal_out = self.seasonal_attn(x, past_encodings)
        x = x + self.dropout(seasonal_out)
        x = self.norm2(x)
        conv_out = self.fft_conv(x)
        x = x + self.dropout(conv_out)
        x = self.norm3(x)
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm4(x)
        return x

class FoldFormer(nn.Module):
    """
    FoldFormer model. It consists of:
    - Time-to-latent folding and embedding,
    - A stack of encoder layers for each past sequence,
    - A stack of decoder layers for the current sequence,
    - A linear projection to unfold the output back to fine-grained resolution.
    """
    def __init__(self, input_dim: int = 1, d_model: int = 128, fold_factor: int = 60,
                 num_encoder_layers: int = 2, num_decoder_layers: int = 2,
                 num_periods: int = 3, nhead: int = 4, ff_dim: int = 256, kernel_size: int = 7,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.num_periods = num_periods
        self.fold_factor = fold_factor
        self.fold = TimeToLatentFold(input_dim, d_model, fold_factor)
        self.encoders = nn.ModuleList([
            nn.Sequential(*[
                FoldFormerEncoderLayer(d_model, nhead, ff_dim, kernel_size, dropout)
                for _ in range(num_encoder_layers)
            ])
            for _ in range(num_periods)
        ])
        self.decoders = nn.ModuleList([
            FoldFormerDecoderLayer(d_model, nhead, ff_dim, kernel_size, dropout)
            for _ in range(num_decoder_layers)
        ])

    def forward(self, past_sequences: List[torch.Tensor], current_sequence: torch.Tensor) -> torch.Tensor:
        past_encodings = []
        for seq, encoder in zip(past_sequences, self.encoders):
            folded = self.fold(seq)
            enc_out = folded
            for layer in encoder:
                enc_out = layer(enc_out)
            past_encodings.append(enc_out)
        past_stack = torch.stack(past_encodings, dim=1)
        current_folded = self.fold(current_sequence)
        dec_out = current_folded
        for layer in self.decoders:
            dec_out = layer(dec_out, past_stack)
        out = self.fold.unfold(dec_out)
        return out

if __name__ == "__main__":
    # Example usage to verify shapes
    batch_size = 2
    seq_len = 120
    fold_factor = 60
    past_periods = 3
    model = FoldFormer(input_dim=1, d_model=32, fold_factor=fold_factor,
                       num_encoder_layers=2, num_decoder_layers=2,
                       num_periods=past_periods, nhead=4, ff_dim=64,
                       kernel_size=5, dropout=0.1)
    past_seqs = [torch.randn(batch_size, seq_len, 1) for _ in range(past_periods)]
    current_seq = torch.randn(batch_size, seq_len, 1)
    current_seq[:, seq_len//2:, :] = 0.0  # zero out forecast window
    output = model(past_seqs, current_seq)
    print("Output shape:", output.shape)
