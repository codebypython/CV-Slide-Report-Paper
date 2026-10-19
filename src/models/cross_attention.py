"""Cross-Attention fusion mechanisms harmonizing micro-morphology and macro-structural context.

Mathematical Formulation:
Let f_s be the macro-structural representation (from Swin Transformer) and
f_c be the micro-morphological representation (from ConvNeXt).

In the directional cross-attention module:
    Q = W_q f_s
    K = W_k f_c
    V = W_v f_c
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    f_fused = LayerNorm(f_s + Dropout(Attention(Q, K, V)))

In the bidirectional module, symmetric attention streams exchange context both ways:
    (Macro queries Micro) and (Micro queries Macro)
followed by adaptive gating.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class CrossAttentionFusion(nn.Module):
    """Directional cross-attention where macro context queries micro morphology."""
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_feat: torch.Tensor,
        key_value_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:
            query_feat: [B, D] or [B, L_q, D]
            key_value_feat: [B, D] or [B, L_kv, D]
        Returns:
            fused_feat: [B, D] or [B, L_q, D]
            attn_weights: [B, L_q, L_kv]
        """
        # Ensure sequence dimension exists
        is_pooled = query_feat.dim() == 2
        q = query_feat.unsqueeze(1) if is_pooled else query_feat
        kv = key_value_feat.unsqueeze(1) if key_value_feat.dim() == 2 else key_value_feat

        attn_out, attn_weights = self.mha(query=q, key=kv, value=kv)
        out = self.norm(q + self.dropout(attn_out))

        if is_pooled:
            out = out.squeeze(1)

        return out, attn_weights


class BidirectionalCrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention exchanging micro and macro features with adaptive gating."""
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Swin queries ConvNeXt (macro queries micro)
        self.swin_to_conv = CrossAttentionFusion(embed_dim, num_heads, dropout)
        # ConvNeXt queries Swin (micro queries macro)
        self.conv_to_swin = CrossAttentionFusion(embed_dim, num_heads, dropout)

        # Gated fusion projection
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        feat_conv: torch.Tensor,
        feat_swin: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Args:
            feat_conv: [B, D]
            feat_swin: [B, D]
        Returns:
            fused: [B, D]
            (attn_s2c, attn_c2s)
        """
        s_queried, attn_s2c = self.swin_to_conv(feat_swin, feat_conv)
        c_queried, attn_c2s = self.conv_to_swin(feat_conv, feat_swin)

        concatenated = torch.cat([s_queried, c_queried], dim=-1)
        fused = self.final_norm(s_queried + self.gate_mlp(concatenated))

        return fused, (attn_s2c, attn_c2s)
