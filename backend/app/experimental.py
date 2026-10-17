"""Explicit raw-detector integration, without violation types or legal verdicts."""
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Candidate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    timecode_sec: float = Field(ge=0, allow_inf_nan=False)
    score: float = Field(ge=0, le=1, allow_inf_nan=False)
    first_scored_time_sec: float = Field(ge=0, allow_inf_nan=False)
    last_scored_time_sec: float = Field(ge=0, allow_inf_nan=False)


class CandidatePolicy(BaseModel):
    model_config = ConfigDict(extra='forbid')
    min_windows: int = Field(default=1, ge=1, le=10, strict=True)
    representative: Literal['peak', 'center'] = 'peak'


class RawPrediction(BaseModel):
    model_config = ConfigDict(extra='forbid')
    experimental: Literal[True]
    model: Literal['mobilenet_v3_small_temporal_v1', 'ssdlite_coco_temporal_v1']
    threshold: float = Field(ge=0, le=1, allow_inf_nan=False)
    video_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    peak_time_sec: float = Field(ge=0, allow_inf_nan=False)
    scan_times_sec: list[float] = Field(min_length=1, max_length=1000)
    scan_scores: list[float] = Field(min_length=1, max_length=1000)
    candidates: list[Candidate] = Field(max_length=100)
    processing_sec: float = Field(ge=0, allow_inf_nan=False)
    total_elapsed_sec: float = Field(ge=0, allow_inf_nan=False)
    note: str = Field(max_length=1000)
    candidate_policy: CandidatePolicy = Field(default_factory=CandidatePolicy)


def adapt_prediction(text, duration, extract_frame):
    prediction = RawPrediction.model_validate_json(text)
    times, scores = prediction.scan_times_sec, prediction.scan_scores
    if len(times) != len(scores) or not all(math.isfinite(t) and 0 <= t < duration for t in times):
        raise ValueError('Invalid experimental scan timestamps')
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Experimental scan timestamps must increase')
    if not all(math.isfinite(s) and 0 <= s <= 1 for s in scores):
        raise ValueError('Invalid experimental scan scores')
    peak = max(range(len(scores)), key=scores.__getitem__)
    if prediction.video_score != scores[peak] or prediction.peak_time_sec != times[peak]:
        raise ValueError('Experimental peak disagrees with scan')
    # Reproduce the declared policy using only numbers; no AI package is needed
    # in the API environment when the detector runs in its own Python environment.
    groups, current = [], []
    for timestamp, score in zip(times, scores):
        if score >= prediction.threshold:
            if current and timestamp - current[-1][0] > 3:
                groups.append(current)
                current = []
            current.append((timestamp, score))
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    expected = []
    policy = prediction.candidate_policy
    for group in groups:
        if len(group) < policy.min_windows:
            continue
        center = sum(t for t, _ in group) / len(group)
        point = (max(group, key=lambda pair: pair[1]) if policy.representative == 'peak'
                 else min(group, key=lambda pair: abs(pair[0] - center)))
        expected.append((group[0][0], group[-1][0], *point))
    received = [(e.first_scored_time_sec, e.last_scored_time_sec, e.timecode_sec, e.score)
                for e in prediction.candidates]
    if received != expected:
        raise ValueError('Experimental candidates disagree with declared policy')
    candidates = []
    previous_end = -1
    for event in prediction.candidates:
        if not previous_end < event.first_scored_time_sec <= event.timecode_sec <= event.last_scored_time_sec < duration:
            raise ValueError('Experimental candidate outside video or out of order')
        if event.score < prediction.threshold or (event.timecode_sec, event.score) not in zip(times, scores):
            raise ValueError('Experimental candidate disagrees with scan')
        previous_end = event.last_scored_time_sec
        candidates.append({**event.model_dump(), 'evidence_frames': [extract_frame(event.timecode_sec)]})
    return {'analysis_mode': 'experimental', 'experimental': True,
            'candidate_policy': policy.model_dump(),
            'candidates': candidates, 'duration_sec': duration,
            'threshold': prediction.threshold, 'model': prediction.model,
            'video_score': prediction.video_score, 'processing_sec': prediction.processing_sec}
