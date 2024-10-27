import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class IonicCurrents(nn.Module):
    def __init__(self):
        super().__init__()
        # membrane currents
        self.g_Na = nn.Parameter(torch.tensor(120.0))
        self.g_K = nn.Parameter(torch.tensor(36.0))
        self.g_L = nn.Parameter(torch.tensor(0.3))

        # E_rev: reversal potential
        self.register_buffer("E_Na", torch.tensor(50.0, dtype=torch.float32))
        self.register_buffer("E_K", torch.tensor(-77.0, dtype=torch.float32))
        self.register_buffer("E_L", torch.tensor(-54.4, dtype=torch.float32))

        self.m = None
        self.h = None
        self.n = None

    def alpha_m(self, V):
        return 0.1 * (V + 40.0) / (1.0 - torch.exp(-(V + 40.0) / 10.0))

    def beta_m(self, V):
        return 4.0 * torch.exp(-(V + 65.0) / 18.0)

    def alpha_h(self, V):
        return 0.07 * torch.exp(-(V + 65.0) / 20.0)

    def beta_h(self, V):
        return 1.0 / (1.0 + torch.exp(-(V + 35.0) / 10.0))

    def alpha_n(self, V):
        return 0.01 * (V + 55.0) / (1.0 - torch.exp(-(V + 55.0) / 10.0))

    def beta_n(self, V):
        return 0.125 * torch.exp(-(V + 65.0) / 80.0)

    def forward(self, V, dt):
        if self.m is None:
            self.m = torch.ones_like(V) * 0.05
            self.h = torch.ones_like(V) * 0.6
            self.n = torch.ones_like(V) * 0.32

        dt = torch.tensor(dt, device=V.device, dtype=torch.float32)

        alpha_m, beta_m = self.alpha_m(V), self.beta_m(V)
        alpha_h, beta_h = self.alpha_h(V), self.beta_h(V)
        alpha_n, beta_n = self.alpha_n(V), self.beta_n(V)

        m = self.m + dt * (alpha_m * (1 - self.m) - beta_m * self.m)
        h = self.h + dt * (alpha_h * (1 - self.h) - beta_h * self.h)
        n = self.n + dt * (alpha_n * (1 - self.n) - beta_n * self.n)

        self.m = m.detach()
        self.h = h.detach()
        self.n = n.detach()

        # Ionic currents
        I_Na = self.g_Na * self.m**3 * self.h * (V - self.E_Na)
        I_K = self.g_K * self.n**4 * (V - self.E_K)
        I_L = self.g_L * (V - self.E_L)

        return I_Na + I_K + I_L


class SynapticPlasticity(nn.Module):
    def __init__(self, input_size, hidden_size, tau_pre=20.0, tau_post=20.0):
        super().__init__()
        self.register_buffer("tau_pre", torch.tensor(tau_pre, dtype=torch.float32))
        self.register_buffer("tau_post", torch.tensor(tau_post, dtype=torch.float32))

        self.weights = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.pre_trace = None
        self.post_trace = None

    def forward(self, pre_spikes, post_spikes, dt):
        # pre_spikes: [batch_size, seq_len, input_size]
        # post_spikes: [batch_size, seq_len, hidden_size]
        device = pre_spikes.device
        batch_size, seq_len, _ = pre_spikes.size()

        if self.pre_trace is None:
            self.pre_trace = torch.zeros_like(pre_spikes, device=device)
            self.post_trace = torch.zeros_like(post_spikes, device=device)

        dt = torch.tensor(dt, device=device, dtype=torch.float32)

        # update trace
        decay_pre = torch.exp(-dt / self.tau_pre)
        decay_post = torch.exp(-dt / self.tau_post)

        pre_trace = self.pre_trace * decay_pre + pre_spikes
        post_trace = self.post_trace * decay_post + post_spikes

        self.pre_trace = pre_trace.clone()
        self.post_trace = post_trace.clone()

        # synaptic connection
        output = F.linear(pre_spikes.view(-1, pre_spikes.size(-1)), self.weights)
        output = output.view(batch_size, seq_len, -1)

        # update STDP -> dw
        with torch.no_grad():
            dw = torch.zeros_like(self.weights)
            for b in range(batch_size):
                for t in range(seq_len):
                    pre_trace_t = self.pre_trace[b, t].unsqueeze(0)  # [1, input_size]
                    post_trace_t = self.post_trace[b, t].unsqueeze(
                        1
                    )  # [hidden_size, 1]
                    pre_spikes_t = pre_spikes[b, t].unsqueeze(0)  # [1, input_size]
                    post_spikes_t = post_spikes[b, t].unsqueeze(1)  # [hidden_size, 1]

                    dw = dw + torch.matmul(post_spikes_t, pre_trace_t)  # LTP
                    dw = dw - torch.matmul(post_trace_t, pre_spikes_t)  # LTD

            dw = dw / (batch_size * seq_len) * 0.01
            weights = torch.clamp(self.weights + dw, -1.0, 1.0)
            self.weights.data.copy_(weights)

        return output


class AdaptiveLIFNeuron(nn.Module):
    def __init__(
        self,
        tau_m=20.0,  # Membrance time
        tau_adapt=100.0,  # Adaptation time
        v_rest=-65.0,  # Resting membrane potential
        v_thresh=-50.0,  # Firing threshold
        v_reset=-65.0,  # Reset potential
        adapt_a=0.5,  # Adapt
        adapt_b=0.1,    # Spike-adapt
    ):
        super().__init__()

        self.register_buffer("tau_m", torch.tensor(tau_m, dtype=torch.float32))
        self.register_buffer("tau_adapt", torch.tensor(tau_adapt, dtype=torch.float32))
        self.register_buffer("v_rest", torch.tensor(v_rest, dtype=torch.float32))
        self.register_buffer("v_thresh", torch.tensor(v_thresh, dtype=torch.float32))
        self.register_buffer("v_reset", torch.tensor(v_reset, dtype=torch.float32))
        self.register_buffer("adapt_a", torch.tensor(adapt_a, dtype=torch.float32))
        self.register_buffer("adapt_b", torch.tensor(adapt_b, dtype=torch.float32))

        self.reset_states()

    def reset_states(self):
        self.v = None
        self.w = None

    def forward(self, I_in, dt):
        if self.v is None:
            self.v = torch.ones_like(I_in) * self.v_rest
            self.w = torch.zeros_like(I_in)

        dt = torch.tensor(dt, device=I_in.device, dtype=torch.float32)

        # Update mebrance potential
        dv = (-(self.v - self.v_rest) - self.w + I_in) * (dt / self.tau_m)
        _v = self.v + dv

        spike = (_v >= self.v_thresh).float()

        # Apply
        dw = (self.adapt_a * (_v - self.v_rest) - self.w + self.adapt_b * spike) * (
            dt / self.tau_adapt
        )
        _w = self.w + dw

        # _v = (1 - spike) * _v + spike * self.v_reset not work?
        _v_rs = torch.where(spike > 0.5, torch.full_like(_v, self.v_reset), _v)

        self.v = _v_rs.detach()
        self.w = _w.detach()

        return spike, _v_rs, _w


class BiologicalSNNLayer(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        tau_m=20.0, # Membrance time
        tau_adapt=100.0, # Adaptation time
        v_thresh=-50.0, # Firing threshold
        v_rest=-65.0, # Reset potential
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Ionic currents
        self.ionic_currents = IonicCurrents()

        # Synaptic plasticity
        self.synaptic_plasticity = SynapticPlasticity(input_size, hidden_size)

        self.neurons = nn.ModuleList(
            [
                AdaptiveLIFNeuron(
                    tau_m=tau_m, tau_adapt=tau_adapt, v_rest=v_rest, v_thresh=v_thresh
                )
                for _ in range(hidden_size)
            ]
        )

        # filter psp
        self.register_buffer("tau_syn", torch.tensor(5.0, dtype=torch.float32))
        self.psp = None

    def reset_states(self):
        self.psp = None
        for neuron in self.neurons:
            neuron.reset_states()

    def forward(self, x, dt=0.1):
        device = x.device
        batch_size, seq_len, _ = x.size()

        if self.psp is None:
            self.psp = torch.zeros(batch_size, seq_len, self.hidden_size, device=device)
        else:
            self.psp = self.psp.new_zeros(batch_size, seq_len, self.hidden_size)

        syn_input = self.synaptic_plasticity(x, self.psp, dt)

        # update psp
        dt_tensor = torch.tensor(dt, device=device, dtype=torch.float32)
        self.psp = self.psp * torch.exp(-dt_tensor / self.tau_syn) + syn_input

        I_ion = self.ionic_currents(self.psp, dt)

        spikes = []
        voltages = []
        adaptations = []

        for i, neuron in enumerate(self.neurons):
            spike, v, w = neuron(I_ion[..., i : i + 1] + self.psp[..., i : i + 1], dt)
            spikes.append(spike)
            voltages.append(v)
            adaptations.append(w)

        spikes = torch.cat(spikes, dim=-1)
        voltages = torch.cat(voltages, dim=-1)
        adaptations = torch.cat(adaptations, dim=-1)

        return spikes, voltages, adaptations


class NeuronalAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        # Synaptic decay
        self.tau_decay = nn.Parameter(torch.ones(num_heads) * 10.0)

        # Integrate dendritic
        self.dendritic_gates = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def compute_temporal_mask(self, seq_len, device):
        times = torch.arange(seq_len, device=device)
        time_diff = times.unsqueeze(1) - times.unsqueeze(0)
        masks = []
        for tau in self.tau_decay:
            mask = torch.exp(-torch.abs(time_diff) / tau)
            masks.append(mask)
        return torch.stack(masks)  # [num_heads, seq_len, seq_len]

    def forward(self, x, spikes=None, membrane_potential=None):
        batch_size, seq_len, _ = x.size()

        # Linear projections and reshape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply temporal masking
        temporal_mask = self.compute_temporal_mask(seq_len, x.device)
        attn_scores = attn_scores * temporal_mask.unsqueeze(0)

        # Spike modulation
        if spikes is not None:
            spike_mask = spikes.float()
            spike_mask = spike_mask.view(
                batch_size, seq_len, self.num_heads, self.head_dim
            )
            spike_mask = spike_mask.transpose(
                1, 2
            )  # [batch, num_heads, seq_len, head_dim]
            # Convert -> attention space
            spike_attn = torch.matmul(spike_mask, spike_mask.transpose(-2, -1))
            spike_attn = spike_attn / math.sqrt(self.head_dim)
            attn_scores = attn_scores * (1 + spike_attn)

        # Membrane potential modulation
        if membrane_potential is not None:
            v_mod = membrane_potential.view(
                batch_size, seq_len, self.num_heads, self.head_dim
            )
            v_mod = v_mod.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
            v = v * torch.sigmoid(v_mod)


        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, v)

        context = context.transpose(
            1, 2
        ).contiguous()  # [batch, seq_len, num_heads, head_dim]
        context = context.view(batch_size, seq_len, self.hidden_size)

        # Integrate dendritic
        if membrane_potential is not None:
            gate_input = torch.cat([context, membrane_potential], dim=-1)
            gate = torch.sigmoid(self.dendritic_gates(gate_input))
            context = context * gate

        return self.o_proj(context)


class BiologicalTransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.neuronal_attention = NeuronalAttention(hidden_size, num_heads, dropout)

        # dendritic-FFN model(WIP)
        self.dendritic_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.adaptive_threshold = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x, spikes=None, membrane_potential=None):
        if spikes is not None:
            spikes = spikes.float()

        # residual connection
        attended = x + self.neuronal_attention(
            self.norm1(x), spikes, membrane_potential
        )

        # residual
        output = attended + self.dendritic_ffn(self.norm2(attended))

        if membrane_potential is not None:
            threshold = self.adaptive_threshold * membrane_potential.mean(
                dim=-1, keepdim=True
            )
            output = output * (membrane_potential > threshold).float()

        return output


class BiologicalTransformer(nn.Module):
    def __init__(self, hidden_size, num_layers=6, num_heads=8, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                BiologicalTransformerLayer(hidden_size, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.spike_embedding = nn.Linear(1, hidden_size)
        # Membrane
        self.voltage_embedding = nn.Linear(1, hidden_size)

    def forward(self, x, spikes, membrane_potential):
        spike_features = self.spike_embedding(spikes.unsqueeze(-1))
        voltage_features = self.voltage_embedding(membrane_potential.unsqueeze(-1))

        # Combine (is it good way?)
        x = x + spike_features + voltage_features

        for layer in self.layers:
            x = layer(x, spikes, membrane_potential)

        return x


class FeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, x):
        return self.projection(x)


class MultiScaleTemporalConv(nn.Module):
    def __init__(self, hidden_size, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.hidden_size = hidden_size

        self.out_channels = hidden_size // len(kernel_sizes)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(hidden_size, self.out_channels, kernel_size=k, padding=k // 2)
                for k in kernel_sizes
            ]
        )

        # project to hidden_size
        self.projection = nn.Linear(self.out_channels * len(kernel_sizes), hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = x.transpose(1, 2)  # [batch_size, hidden_size, seq_len]

        conv_outputs = []
        for conv in self.convs:
            conv_outputs.append(conv(x))

        x = torch.cat(conv_outputs, dim=1)  # Combined to channel
        x = x.transpose(1, 2)  # [batch_size, seq_len, out_channels * len(kernel_sizes)]

        x = self.projection(x)

        return self.norm(x)


class OutputHead(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dense2 = nn.Linear(hidden_size // 2, output_size)
        self.norm = nn.LayerNorm(hidden_size // 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(F.gelu(self.dense1(x)))
        x = self.norm(x)
        return self.dense2(x)


class FeatureFusion(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fusion = nn.Linear(hidden_size * 3, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, spikes, voltages, adaptations):
        # Concatenate along feature dim
        fused = torch.cat([spikes, voltages, adaptations], dim=-1)
        fused = self.fusion(fused)
        return self.norm(fused)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pos = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(pos * div_term)
        pe[:, 0, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # input: [seq_len, batch_size, hidden_size]
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class CSnT(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_sizes, num_layers=6, nhead=8, dropout=0.1
    ):
        super().__init__()

        self.feature_extractor = FeatureExtractor(input_size, hidden_size)

        self.temporal_conv = MultiScaleTemporalConv(hidden_size)

        self.snn_layer = BiologicalSNNLayer(hidden_size, hidden_size)

        self.transformer_layers = nn.ModuleList(
            [
                BiologicalTransformerLayer(hidden_size, nhead, dropout)
                for _ in range(num_layers)
            ]
        )

        self.output_projections = nn.ModuleDict(
            {
                "spike": OutputHead(hidden_size, output_sizes[0]),
                "soma": OutputHead(hidden_size, output_sizes[1]),
                "DVT": OutputHead(hidden_size, output_sizes[2]),
            }
        )

        self.norm = nn.LayerNorm(hidden_size)

        self.pos_encoder = PositionalEncoding(hidden_size, dropout)

        self.feature_fusion = FeatureFusion(hidden_size)

    def reset_states(self):
        self.snn_layer.reset_states()

    def forward(self, x, dt=0.1):
        # Input shape: [batch_size, seq_len, input_size]
        batch_size, seq_len, _ = x.size()

        x = self.feature_extractor(x)

        temporal_features = self.temporal_conv(x)

        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)

        spikes, voltages, adaptations = self.snn_layer(x, dt)

        snn_features = self.feature_fusion(spikes, voltages, adaptations)
        x = x + snn_features

        for layer in self.transformer_layers:
            x = layer(x, spikes=spikes, membrane_potential=voltages)

        x = self.norm(x)

        return (
            self.output_projections["spike"](x),
            self.output_projections["soma"](x),
            self.output_projections["DVT"](x),
        )
