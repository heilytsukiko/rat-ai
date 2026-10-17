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
With raw maxima at threshold 0.5: TP=4, TN=0, FP=4, FN=0.
The published manifest now requires two consecutive positive scan windows and
selects the window nearest the group's mean time, using that window's score.
At scan stride 2 seconds: TP=4, TN=1, FP=3, FN=0, accuracy=0.625;
negative-video candidates decreased from 17 to 7, and point matches stayed
at 3/4 within two seconds. This is postprocessing; numeric weights are unchanged.
Short events can be filtered out. The policy was selected on this same repeatedly
used validation set; no independent test set exists. The weights are
an experimental collaboration baseline, not a reliable violation detector.
They do not assign event types or regulatory sources. Qwen is a separate,
unfinished experiment and was not fine-tuned.

Full-video ROC AUC for v2 on this repeatedly used validation set is 0.0.
Training experiments v3–v5 were not promoted because they increased misses.
See `../../EXPERIMENTS.md` and `../../evaluation/v2-confirmed.json`.

Source attribution and license: `TORCHVISION-LICENSE.txt`. This directory does
not grant any additional rights over the hackathon source dataset. Keep source
data and private training records in the team's authorized environment.
