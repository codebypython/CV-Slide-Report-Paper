"""Proposed Dual-Branch Hybrid Architecture (ConvNeXt + Swin Transformer + Cross-Attention + FiLM).

Scientific Design:
- ConvNeXt Branch: Captures high-resolution micro-morphology (growth plate continuity, trabecular bone mineralization).
- Swin Transformer Branch: Models long-range macro-structural spatial correlations across distinct hand ossification centers.
- Cross-Attention Fusion: Enables macro query tokens to inspect fine-grained micro keys.
- FiLM Gender Modulation: Clinically conditions fused latent representations via biological sex embeddings.
"""

import torch
import torch.nn as nn
import timm
from typing import Optional, Tuple, Dict, Any

from .film import FiLMLayer
from .cross_attention import CrossAttentionFusion, BidirectionalCrossAttentionFusion


class DualBranchBoneAgeNet(nn.Module):
    """Proposed Dual-Branch ConvNeXt + Swin Transformer with Cross-Attention and FiLM."""
    def __init__(
        self,
        convnext_name: str = "convnext_small",
        swin_name: str = "swin_small_patch4_window7_224",
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout_rate: float = 0.3,
        drop_path_rate: float = 0.0,
        use_film: bool = True,
        use_cross_attention: bool = True,
        bidirectional_attention: bool = False,
        spatial_attention: bool = True,
        pretrained: bool = False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_film = use_film
        self.use_cross_attention = use_cross_attention
        self.bidirectional_attention = bidirectional_attention
        self.spatial_attention = spatial_attention

        # 1. Micro-morphology Branch: ConvNeXt
        self.convnext = timm.create_model(
            convnext_name, pretrained=pretrained, num_classes=0, drop_path_rate=drop_path_rate
        )
        convnext_dim = getattr(self.convnext, "num_features", 768)

        # 2. Macro-structural Branch: Swin Transformer
        self.swin = timm.create_model(
            swin_name, pretrained=pretrained, num_classes=0, drop_path_rate=drop_path_rate
        )
        swin_dim = getattr(self.swin, "num_features", 768)

        # 3. Projection to unified embedding space
        self.proj_conv = nn.Sequential(
            nn.Linear(convnext_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.proj_swin = nn.Sequential(
            nn.Linear(swin_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # 4. Cross-Attention interaction module
        if self.use_cross_attention:
            if self.bidirectional_attention:
                self.cross_attn = BidirectionalCrossAttentionFusion(
                    embed_dim=embed_dim, num_heads=num_heads, dropout=0.1
                )
            else:
                self.cross_attn = CrossAttentionFusion(
                    embed_dim=embed_dim, num_heads=num_heads, dropout=0.1
                )
        else:
            self.cross_attn = None

        # 5. Gender Conditioning Module
        if self.use_film:
            self.film = FiLMLayer(feature_dim=embed_dim, metadata_dim=1, hidden_dim=128)
            regressor_in_dim = embed_dim
        else:
            self.film = None
            # Standard concatenation with 1-dim gender vector (or 32-dim linear projection)
            self.gender_proj = nn.Linear(1, 32)
            regressor_in_dim = embed_dim + 32

        # 6. Deep Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(regressor_in_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1)  # Predicts continuous bone age in months
        )

    def extract_features(self, img_conv: torch.Tensor, img_swin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extracts pooled features from both backbones."""
        feat_conv = self.convnext(img_conv)  # [B, convnext_dim]
        feat_swin = self.swin(img_swin)      # [B, swin_dim]
        return feat_conv, feat_swin

    def extract_spatial_tokens(self, img_conv: torch.Tensor, img_swin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extracts 2D spatial feature tokens from both backbones."""
        # ConvNeXt: [B, C, H, W] -> normalize -> flatten to [B, H*W, C]
        feat_conv = self.convnext.forward_features(img_conv)
        if hasattr(self.convnext, "head") and hasattr(self.convnext.head, "norm"):
            feat_conv = self.convnext.head.norm(feat_conv)
        tokens_conv = feat_conv.flatten(2).permute(0, 2, 1)

        # Swin: [B, H, W, C] or [B, L, C] -> flatten to [B, L, C]
        feat_swin = self.swin.forward_features(img_swin)
        if feat_swin.dim() == 4:
            tokens_swin = feat_swin.view(feat_swin.size(0), -1, feat_swin.size(-1))
        else:
            tokens_swin = feat_swin

        return tokens_conv, tokens_swin

    def forward(
        self,
        img_conv: torch.Tensor,
        img_swin: torch.Tensor,
        gender: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        """Args:
            img_conv: [B, 3, H_c, W_c]
            img_swin: [B, 3, H_s, W_s]
            gender: [B, 1] float tensor (1.0 = male, 0.0 = female)
            return_attention: whether to return cross-attention weights
        Returns:
            bone_age_pred: [B] predicted age in months
            attn_info: attention weights if return_attention is True, else None
        """
        if self.spatial_attention:
            tokens_conv, tokens_swin = self.extract_spatial_tokens(img_conv, img_swin)
            f_c = self.proj_conv(tokens_conv)  # [B, L_c, embed_dim]
            f_s = self.proj_swin(tokens_swin)  # [B, L_s, embed_dim]

            attn_info = None
            if self.use_cross_attention and self.cross_attn is not None:
                # Directional: Swin (macro query) queries ConvNeXt (micro key/value)
                fused_tokens, attn_info = self.cross_attn(query_feat=f_s, key_value_feat=f_c)
                fused = fused_tokens.mean(dim=1)  # Spatial global pool over macro tokens: [B, embed_dim]
            else:
                fused = f_s.mean(dim=1) + f_c.mean(dim=1)
        else:
            # Pooled features
            feat_conv_raw, feat_swin_raw = self.extract_features(img_conv, img_swin)
            f_c = self.proj_conv(feat_conv_raw)  # [B, embed_dim]
            f_s = self.proj_swin(feat_swin_raw)  # [B, embed_dim]

            attn_info = None
            if self.use_cross_attention and self.cross_attn is not None:
                if self.bidirectional_attention:
                    fused, attn_info = self.cross_attn(f_c, f_s)
                else:
                    fused, attn_info = self.cross_attn(query_feat=f_s, key_value_feat=f_c)
            else:
                fused = f_s + f_c

        # Biological sex modulation via FiLM
        if self.use_film and self.film is not None:
            modulated = self.film(fused, gender)
        else:
            gender_emb = torch.relu(self.gender_proj(gender))
            modulated = torch.cat([fused, gender_emb], dim=-1)

        # Regression head
        pred = self.regressor(modulated).squeeze(-1)  # [B]

        if return_attention:
            return pred, attn_info
        return pred
