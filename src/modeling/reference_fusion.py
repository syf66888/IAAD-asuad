"""Linear-cost reference-conditioned feature enhancement for video tokens.

Input/output layout is always [batch, tokens, feature_dim].
"""

import torch
from torch import nn
from torch.nn import functional as F


def attention_mask_to_bias(mask, dtype, use_log=False):
    """Keep binary masking exact while allowing soft gates useful gradients.

    The legacy -10000 * (1-p) saturates softmax for almost every sigmoid p.
    log(p) instead expresses multiplicative attention priors. Calculate the
    logarithm in float32 before converting to the attention-score dtype.
    """
    if not use_log:
        return (1.0 - mask.to(dtype=dtype)) * -10000.0
    probabilities = mask.float()
    bias = probabilities.clamp_min(1e-6).log()
    return bias.masked_fill(probabilities <= 0, -10000.0).to(dtype=dtype)


def video_grid_to_tokens(features, feature_dim):
    """Convert channel-first 3D-backbone features without mixing channels."""
    if features.ndim == 5:
        if features.shape[1] != feature_dim:
            raise ValueError(f"Expected {feature_dim} backbone channels, got {features.shape}")
        temporal_steps = features.shape[2]
        tokens = features.permute(0, 2, 3, 4, 1).reshape(features.shape[0], -1, feature_dim)
        return tokens, temporal_steps
    if features.ndim == 3 and features.shape[-1] == feature_dim:
        return features, features.shape[1]
    raise ValueError(f"Expected [B,C,T,H,W] or [B,N,C], got {features.shape}")


class ReferenceFeatureFusion(nn.Module):
    """Approximate product -> hard selection -> soft enhancement cheaply.

    Same-clip references are spatially pooled at each time step and averaged
    over neighbouring time steps. Auxiliary references may instead be aligned
    per token, aligned per time step, or an unordered set pooled to one token.
    Auxiliary dimensions are explicitly projected; no attention matrix is
    formed. Hard gates use straight-through gradients during training.
    """

    def __init__(self, feature_dim, hidden_dim=128, reference_dim=None, dropout=0.1):
        super().__init__()
        reference_dim = reference_dim or feature_dim
        if feature_dim <= 0 or hidden_dim <= 0 or reference_dim <= 0:
            raise ValueError("All feature dimensions must be positive")
        self.feature_dim = feature_dim
        self.reference_dim = reference_dim
        self.input_norm = nn.LayerNorm(feature_dim)
        self.input_projection = nn.Linear(feature_dim, hidden_dim)
        if reference_dim == feature_dim:
            self.reference_norm = nn.LayerNorm(feature_dim)
            self.reference_projection = nn.Linear(feature_dim, hidden_dim)
            self.aux_norm = None
            self.aux_projection = None
        else:
            # Production uses the differently sized auxiliary references on
            # every step. Do not register an unused self-reference branch.
            self.reference_norm = None
            self.reference_projection = None
            self.aux_norm = nn.LayerNorm(reference_dim)
            self.aux_projection = nn.Linear(reference_dim, hidden_dim)
        self.hard_gate = nn.Linear(hidden_dim, 1)
        self.soft_gate = nn.Linear(hidden_dim, feature_dim)
        self.output_projection = nn.Linear(hidden_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        # Start with all references admitted while keeping gate gradients live.
        nn.init.zeros_(self.hard_gate.weight)
        nn.init.constant_(self.hard_gate.bias, 1.0)

    def forward(self, tokens, reference=None, temporal_steps=None):
        if tokens.ndim != 3 or tokens.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected [B,N,{self.feature_dim}], got {tokens.shape}")
        batch_size, token_count, _ = tokens.shape
        temporal_steps = int(temporal_steps or 1)
        if temporal_steps < 1 or token_count % temporal_steps:
            raise ValueError("Temporal steps must divide the number of video tokens")
        if reference is None:
            reference = tokens.reshape(batch_size, temporal_steps, -1, self.feature_dim).mean(2)
            if temporal_steps > 1:
                reference = F.avg_pool1d(
                    reference.transpose(1, 2), 3, stride=1, padding=1,
                    count_include_pad=False,
                ).transpose(1, 2)
            norm = self.reference_norm if self.reference_norm is not None else self.input_norm
            projection = self.reference_projection if self.reference_projection is not None else self.input_projection
            projected_reference = projection(norm(reference))
        else:
            if reference.ndim == 2:
                reference = reference.unsqueeze(1)
            if (reference.ndim != 3 or reference.shape[0] != batch_size
                    or reference.shape[-1] != self.reference_dim or reference.shape[1] == 0):
                raise ValueError(
                    f"Expected nonempty reference [B,R,{self.reference_dim}], got {reference.shape}"
                )
            reference = reference.to(device=tokens.device, dtype=tokens.dtype)
            if reference.shape[1] not in (1, temporal_steps, token_count):
                reference = reference.mean(1, keepdim=True)
            norm = self.aux_norm if self.aux_norm is not None else self.reference_norm
            projection = self.aux_projection if self.aux_projection is not None else self.reference_projection
            projected_reference = projection(norm(reference))

        reference_count = projected_reference.shape[1]
        if reference_count == temporal_steps and reference_count != token_count:
            projected_reference = projected_reference.repeat_interleave(token_count // temporal_steps, dim=1)
        hidden = F.gelu(self.input_projection(self.input_norm(tokens)) * projected_reference)
        probabilities = self.hard_gate(hidden).sigmoid()
        hard = (probabilities >= 0.5).to(probabilities.dtype)
        selection = hard + probabilities - probabilities.detach() if self.training else hard
        enhancement = selection * self.soft_gate(hidden).sigmoid() * self.output_projection(hidden)
        return tokens + self.residual_scale * self.dropout(enhancement)
