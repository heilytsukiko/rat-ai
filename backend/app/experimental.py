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


class RawPrediction(BaseModel):
    model_config = ConfigDict(extra='forbid')
    experimental: Literal[True]
    model: Literal['mobilenet_v3_small_temporal_v1']
    threshold: float = Field(ge=0, le=1, allow_inf_nan=False)
    video_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    peak_time_sec: float = Field(ge=0, allow_inf_nan=False)
    scan_times_sec: list[float] = Field(min_length=1, max_length=1000)
    scan_scores: list[float] = Field(min_length=1, max_length=1000)
    candidates: list[Candidate] = Field(max_length=100)
    processing_sec: float = Field(ge=0, allow_inf_nan=False)
    total_elapsed_sec: float = Field(ge=0, allow_inf_nan=False)
    note: str = Field(max_length=1000)


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
    if not prediction.candidates and prediction.video_score >= prediction.threshold:
        raise ValueError('Experimental candidates missing despite positive scan')
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
            'candidates': candidates, 'duration_sec': duration,
            'threshold': prediction.threshold, 'model': prediction.model,
            'video_score': prediction.video_score, 'processing_sec': prediction.processing_sec}
