from pathlib import Path
import math


def sample_frames(path: Path, start: float, duration: float, count: int, max_side: int):
    import cv2
    import numpy as np

    if (not math.isfinite(start) or not math.isfinite(duration) or start < 0
            or duration <= 0 or not 1 <= count <= 16 or max_side < 56):
        raise ValueError("Invalid video sampling parameters")
    if not path.is_file():
        raise ValueError("Video does not exist")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("Video decoder could not open the file")
        fps = capture.get(cv2.CAP_PROP_FPS)
        total = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or fps <= 0 or total <= 0:
            raise ValueError("Video metadata is unavailable")
        end = min(start + duration, (total - 1) / fps)
        if start > end:
            raise ValueError("Requested clip starts outside the video")
        frames, timestamps = [], []
        fallbacks = 0
        for timestamp in np.linspace(start, end, count):
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000)
            ok, frame = capture.read()
            actual_time = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000 if ok else None
            if not ok:
                # MOV can have variable frame intervals: mean FPS is not an exact
                # mapping to the final presentation timestamp. Decode a short tail
                # and retain its actual last timestamp instead of assuming CFR.
                fallbacks += 1
                capture.set(cv2.CAP_PROP_POS_MSEC, max(0, float(timestamp) - 0.5) * 1000)
                for _ in range(max(2, math.ceil(fps) + 2)):
                    tail_ok, tail_frame = capture.read()
                    if not tail_ok:
                        break
                    frame = tail_frame
                    actual_time = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    ok = True
                    if actual_time >= timestamp:
                        break
            if not ok:
                raise ValueError("Failed to decode a requested frame")
            height, width = frame.shape[:2]
            scale = min(1.0, max_side / max(height, width))
            if scale < 1:
                frame = cv2.resize(frame, (round(width * scale), round(height * scale)),
                                   interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamps.append(round(actual_time, 4))
        return frames, timestamps, {"fps": fps, "frame_count": int(total),
                                    "seek_fallbacks": fallbacks}
    finally:
        capture.release()


def clip_tensor(path, center_sec, duration_sec, size=224):
    """Three RGB frames with aspect ratio preserved and ImageNet normalization."""
    import cv2
    import numpy as np
    import torch

    start = max(0, center_sec - 1)
    end = min(duration_sec - 0.001, center_sec + 1)
    frames, times, _ = sample_frames(Path(path), start, max(0.001, end - start), 3, 448)
    images = []
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    for frame in frames:
        height, width = frame.shape[:2]
        scale = size / max(height, width)
        resized = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))),
                             interpolation=cv2.INTER_AREA)
        square = np.full((size, size, 3), mean * 255, dtype=np.float32)
        top, left = (size - resized.shape[0]) // 2, (size - resized.shape[1]) // 2
        square[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
        images.append(torch.from_numpy(((square / 255 - mean) / std).transpose(2, 0, 1).copy()))
    return torch.stack(images), times[1]
