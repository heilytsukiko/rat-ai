# Experimental detector v2

`detector-v2.safetensors` contains only numeric model tensors. The accompanying
JSON contains the architecture, preprocessing, checksum and public pretrained
weight provenance. Raw videos, frames, sample annotations, feature caches and
developer file paths are omitted. The original training checkpoint stays local.

Architecture: MobileNetV3 Small frozen prefix, fine-tuned final visual block,
and a temporal binary classifier over three frames spanning two seconds.
167,169 parameters were optimized on CPU. The export uses the best checkpoint
from the second 15-epoch run. Base torchvision weights:
https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth

Validation used eight videos, with exact duplicate files kept in one split.
At threshold 0.5: TP=4, TN=0, FP=4, FN=0, accuracy=0.50, precision=0.50,
recall=1.00. Every validation video triggered a detection. The weights are
an experimental collaboration baseline, not a reliable violation detector.
They do not assign event types or regulatory sources. Qwen is a separate,
unfinished experiment and was not fine-tuned.

Full-video ROC AUC for v2 on this repeatedly used validation set is 0.0.
A third training experiment reduced false alarms but missed every positive
video; it was not promoted. See `../../EXPERIMENTS.md` for the comparison.

Source attribution and license: `TORCHVISION-LICENSE.txt`. This directory does
not grant any additional rights over the hackathon source dataset. Keep source
data and private training records in the team's authorized environment.
