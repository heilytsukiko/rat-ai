"""A small temporal detector: frozen pretrained prefix, trainable tail and head."""
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small


class TemporalDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # Pretrained weights are loaded explicitly from a verified local file.
        base = mobilenet_v3_small(weights=None)
        self.prefix = base.features[:-1]
        self.tail = base.features[-1]
        width = self.tail[0].out_channels
        self.head = nn.Sequential(nn.Linear(width * 3, 64), nn.Hardswish(),
                                  nn.Dropout(0.5), nn.Linear(64, 1))
        self.prefix.requires_grad_(False)

    def load_pretrained_backbone(self, state):
        base = mobilenet_v3_small(weights=None)
        base.load_state_dict(state, strict=True)
        self.prefix.load_state_dict(base.features[:-1].state_dict())
        self.tail.load_state_dict(base.features[-1].state_dict())

    def train(self, mode=True):
        super().train(mode)
        self.prefix.eval()
        # Tiny batches should not replace ImageNet normalization statistics.
        for layer in self.tail.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
        return self

    def forward_cached(self, features):
        batch, frames, channels, height, width = features.shape
        encoded = self.tail(features.reshape(batch * frames, channels, height, width))
        encoded = encoded.mean(dim=(-2, -1)).reshape(batch, frames, -1)
        temporal = torch.cat([encoded.mean(dim=1), encoded[:, -1] - encoded[:, 0],
                              (encoded[:, 1:] - encoded[:, :-1]).abs().mean(dim=1)], dim=1)
        return self.head(temporal).squeeze(-1)

    def forward(self, clips):
        batch, frames, channels, height, width = clips.shape
        with torch.no_grad():
            features = self.prefix(clips.reshape(batch * frames, channels, height, width))
        return self.forward_cached(features.reshape(batch, frames, *features.shape[1:]))
