import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LIFNeuron(nn.Module):
    def __init__(self, tau_m=10.0, tau_s=5.0, threshold=1.0, reset_value=0.0):
        super().__init__()
        self.tau_m = tau_m
        self.tau_s = tau_s
        self.threshold = threshold
        self.reset_value = reset_value
        self.v = None
        self.i = None

    def forward(self, x, dt=1.0):
        if self.v is None:
            self.v = torch.zeros_like(x)
            self.i = torch.zeros_like(x)

        self.i = self.i + (-self.i + x) * (dt / self.tau_s)
        self.v = self.v + ((self.reset_value - self.v) + self.i) * (dt / self.tau_m)
        spike = (self.v >= self.threshold).float()
        self.v = (1 - spike) * self.v + spike * self.reset_value

        return spike, self.v


class SNNLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.lif_neuron = LIFNeuron()

    def forward(self, x, dt=1.0):
        x = self.linear(x)
        spike, v = self.lif_neuron(x, dt)
        return spike, v


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (sequence_length, batch_size, hidden_size)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.temporal_attn = TemporalAttention(d_model, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=None):
        src2 = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            is_causal=is_causal,
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.temporal_attn(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm3(src)
        return src


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class CSnT(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, num_layers, nhead, dropout=0.1
    ):
        super().__init__()
        self.snn_layer = SNNLayer(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        encoder_layers = TransformerEncoderLayer(
            hidden_size, nhead, hidden_size * 4, dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output_layer = nn.Linear(hidden_size, output_size)

        # L1 regularization
        self.l1_lambda = 1e-5
        # L2 regularization is applied through weight_decay in the optimizer

    def forward(self, x, dt=1.0):
        batch_size, seq_len, window_size_ms, num_segments = x.shape
        x = x.view(batch_size * seq_len * window_size_ms, num_segments)
        spikes, voltages = self.snn_layer(x, dt)
        x = spikes + voltages
        x = x.view(batch_size * window_size_ms, seq_len, -1)
        x = x.permute(1, 0, 2)

        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        x = x.permute(1, 0, 2)
        output = self.output_layer(x)
        return output

    def l1_regularization(self):
        l1_loss = 0
        for param in self.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return self.l1_lambda * l1_loss
