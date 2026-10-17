"""Offline window scoring for the trained compact detector."""
from pathlib import Path
import hashlib
import json
import time


def load_detector(checkpoint=None):
    import torch
    from .temporal import TemporalDetector
    checkpoint = Path(checkpoint) if checkpoint is not None else Path(__file__).parent / "artifacts/detector-v2.safetensors"
    if checkpoint.suffix == ".safetensors":
        from safetensors.torch import load_file
        config = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
        with checkpoint.open("rb") as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
            if digest.hexdigest() != config["weights_sha256"]:
                raise ValueError("Detector checkpoint checksum mismatch")
        payload = {"config": config, "state_dict": load_file(str(checkpoint), device="cpu")}
    else:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    architecture = payload["config"]["architecture"]
    if architecture == "ssdlite_coco_temporal_v1":
        from .semantic import SemanticTemporalDetector
        model = SemanticTemporalDetector()
    elif architecture == "mobilenet_v3_small_temporal_v1":
        model = TemporalDetector()
    else:
        raise ValueError("Unsupported detector checkpoint")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload["config"]


class VideoDetector:
    """Load once in a local worker; return raw candidates without invented rules."""
    def __init__(self, checkpoint=None, threads=4):
        import torch
        if not isinstance(threads, int) or not 1 <= threads <= 8:
            raise ValueError("threads must be between 1 and 8")
        torch.set_num_threads(threads)
        self.model, self.config = load_detector(checkpoint)

    def analyze_video(self, path, stride_sec=2, threshold=None):
        chosen_threshold = self.config["threshold"] if threshold is None else threshold
        policy = validate_candidate_policy(self.config.get("candidate_policy"))
        # Validate before decoding a potentially long video.
        candidate_events({"scan_times_sec": [], "scan_scores": []}, chosen_threshold, stride_sec, **policy)
        scored = score_video(self.model, path, stride_sec=stride_sec)
        return {**scored, "candidates": candidate_events(scored, chosen_threshold, stride_sec, **policy),
                "candidate_policy": policy,
                "threshold": chosen_threshold, "experimental": True,
                "model": self.config["architecture"]}


def score_video(model, path, duration_sec=None, stride_sec=2):
    import cv2
    import numpy as np
    import torch
    from .video import clip_tensor

    if not Path(path).is_file() or not np.isfinite(stride_sec) or stride_sec < 0.25:
        raise ValueError("Existing video and scan stride >= 0.25 seconds required")
    if duration_sec is None:
        capture = cv2.VideoCapture(str(path))
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            if not capture.isOpened() or not np.isfinite(fps) or fps <= 0 or count <= 0:
                raise ValueError("Video cannot be decoded")
            duration_sec = count / fps
        finally:
            capture.release()
    if not np.isfinite(duration_sec) or duration_sec <= 0:
        raise ValueError("Invalid video duration")
    started = time.perf_counter()
    centers = np.arange(min(0.5, duration_sec / 2), duration_sec, stride_sec).tolist()
    scores, times = [], []
    model.eval()
    with torch.inference_mode():
        for center in centers:
            clip, actual_time = clip_tensor(str(path), center, duration_sec, size=getattr(model, "input_size", 224))
            scores.append(float(model(clip[None]).sigmoid()[0]))
            times.append(actual_time)
    peak = int(np.argmax(scores))
    return {"video_score": scores[peak], "peak_time_sec": times[peak],
            "scan_times_sec": times, "scan_scores": scores,
            "processing_sec": time.perf_counter() - started}


def validate_candidate_policy(policy=None):
    policy = {} if policy is None else policy
    if not isinstance(policy, dict) or set(policy) - {"min_windows", "representative"}:
        raise ValueError("Invalid candidate policy")
    minimum, representative = policy.get("min_windows", 1), policy.get("representative", "peak")
    if (type(minimum) is not int or not 1 <= minimum <= 10
        or not isinstance(representative, str) or representative not in {"peak", "center"}):
        raise ValueError("Invalid candidate policy")
    return {"min_windows": minimum, "representative": representative}


def candidate_events(scored, threshold=0.5, stride_sec=2, min_windows=1, representative="peak"):
    """Merge consecutive positive windows; these are statistical review candidates."""
    import math
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0, 1]")
    validate_candidate_policy({"min_windows": min_windows, "representative": representative})
    groups, current = [], []
    for timestamp, score in zip(scored["scan_times_sec"], scored["scan_scores"]):
        if score >= threshold:
            if current and timestamp - current[-1][0] > stride_sec * 1.5:
                groups.append(current)
                current = []
            current.append((timestamp, score))
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    result = []
    for group in groups:
        if len(group) < min_windows:
            continue
        if representative == "peak":
            timestamp, score = max(group, key=lambda pair: pair[1])
        else:
            center = sum(t for t, _ in group) / len(group)
            timestamp, score = min(group, key=lambda pair: abs(pair[0] - center))
        result.append({"timecode_sec": timestamp, "score": score,
                       "first_scored_time_sec": group[0][0], "last_scored_time_sec": group[-1][0]})
    return result
