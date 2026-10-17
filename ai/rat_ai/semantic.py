"""Frozen COCO object features and a trainable temporal projection/classifier."""
import torch
from torch import nn
from torchvision.models.detection import ssdlite320_mobilenet_v3_large


class ObjectFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.detector = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None, :, None, None])
        self.requires_grad_(False)

    def forward(self, images):
        # Input shares the existing RGB letterbox/ImageNet video preprocessing.
        images = (images * self.std + self.mean).clamp(0, 1)
        predictions = self.detector(list(images))
        features = []
        for image, prediction in zip(images, predictions):
            keep = prediction['scores'] >= .2
            scores, labels, boxes = (prediction[name][keep] for name in ('scores', 'labels', 'boxes'))
            height, width = image.shape[-2:]
            cx = (boxes[:, 0] + boxes[:, 2]) / (2 * width)
            cy = (boxes[:, 1] + boxes[:, 3]) / (2 * height)
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / (width * height)
            stats = images.new_zeros(6, 91)
            stats[0].scatter_reduce_(0, labels, scores, reduce='amax', include_self=True)
            for row, values in enumerate((scores, scores * cx, scores * cy, scores * area, torch.ones_like(scores)), 1):
                stats[row].scatter_add_(0, labels, values)
            features.append(stats.flatten())
        return torch.stack(features)


class SemanticTemporalDetector(nn.Module):
    input_size = 320

    def __init__(self):
        super().__init__()
        self.prefix = ObjectFeatures()
        self.tail = nn.Sequential(nn.Linear(546, 64), nn.Hardswish())
        self.head = nn.Sequential(nn.Linear(192, 32), nn.Hardswish(), nn.Dropout(.3), nn.Linear(32, 1))

    def load_pretrained_backbone(self, state):
        self.prefix.detector.load_state_dict(state, strict=True)

    def train(self, mode=True):
        super().train(mode)
        self.prefix.eval()
        return self

    def forward_cached(self, features):
        encoded = self.tail(features)
        temporal = torch.cat([encoded.mean(dim=1), encoded[:, -1] - encoded[:, 0],
                              (encoded[:, 1:] - encoded[:, :-1]).abs().mean(dim=1)], dim=1)
        return self.head(temporal).squeeze(-1)

    def forward(self, clips):
        batch, frames, channels, height, width = clips.shape
        with torch.no_grad():
            features = self.prefix(clips.reshape(batch * frames, channels, height, width))
        return self.forward_cached(features.reshape(batch, frames, -1))
