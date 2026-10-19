"""Grad-CAM (Gradient-weighted Class Activation Mapping) for ConvNeXt micro-morphology branch."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional


class ConvNeXtGradCAM:
    """Computes Grad-CAM saliency maps for the ConvNeXt branch in regression setting."""
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.gradients = None
        self.activations = None
        self.hook_handles = []

        # Default target layer: last stage of ConvNeXt
        if target_layer is None:
            if hasattr(model, "convnext"):
                conv_backbone = model.convnext
            elif hasattr(model, "backbone"):
                conv_backbone = model.backbone
            else:
                conv_backbone = model

            # Search for the last convolutional or norm layer in stages
            if hasattr(conv_backbone, "stages"):
                target_layer = conv_backbone.stages[-1]
            elif hasattr(conv_backbone, "norm_pre"):
                target_layer = conv_backbone.norm_pre
            else:
                # Fallback to last child module
                target_layer = list(conv_backbone.children())[-2]

        self.target_layer = target_layer
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_cam(
        self,
        img_conv: torch.Tensor,
        img_swin: torch.Tensor,
        gender: torch.Tensor,
        target_size: Tuple[int, int] = (512, 512)
    ) -> np.ndarray:
        """Generates Grad-CAM activation heatmap for a single sample."""
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        if hasattr(self.model, "extract_features") or "DualBranch" in self.model.__class__.__name__:
            pred = self.model(img_conv, img_swin, gender)
        else:
            pred = self.model(img_conv, gender)

        # Backward pass with respect to predicted age
        pred.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture gradients or activations in Grad-CAM.")

        # Compute importance weights alpha via global average pooling
        grads = self.gradients  # [B, C, H, W]
        acts = self.activations # [B, C, H, W]

        # In ConvNeXt, feature maps are [B, C, H, W]
        alpha = torch.mean(grads, dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
        cam = torch.sum(alpha * acts, dim=1, keepdim=True)   # [B, 1, H, W]
        cam = F.relu(cam)                                    # ReLU to keep positive influence

        # Normalize to [0, 1]
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # Resize to original image size
        cam_resized = cv2.resize(cam, target_size, interpolation=cv2.INTER_LINEAR)
        return cam_resized

    def remove_hooks(self):
        """Cleans up registered PyTorch hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
