import copy
import json
from pathlib import Path

import pytest

from app.experimental import adapt_prediction
from app import pipeline
from test_pipeline import configure


def raw():
    return {'experimental': True, 'model': 'mobilenet_v3_small_temporal_v1',
            'threshold': .5, 'video_score': .8, 'peak_time_sec': 2.5,
            'scan_times_sec': [.5, 2.5, 4.5], 'scan_scores': [.1, .8, .1],
            'processing_sec': .1, 'total_elapsed_sec': 1, 'note': 'Raw candidates only',
            'candidates': [{'timecode_sec': 2.5, 'score': .8,
                            'first_scored_time_sec': 2.5, 'last_scored_time_sec': 2.5}]}


def test_real_pipeline_experimental_mode_extracts_evidence_without_legal_verdict(monkeypatch):
    configure(monkeypatch, raw())
    monkeypatch.setenv('ANALYSIS_MODE', 'experimental')
    result = pipeline.analyze(Path('test.mov'))
    assert result['analysis_mode'] == 'experimental'
    assert result['experimental'] is True
    assert 'verdict' not in result and 'violations' not in result
    assert 'source' not in result['candidates'][0]
    assert result['candidates'][0]['evidence_frames'][0]['timestamp_sec'] == 2.5


def test_strict_mode_cannot_accept_raw_detector_output(monkeypatch):
    configure(monkeypatch, raw())
    monkeypatch.setenv('ANALYSIS_MODE', 'strict')
    with pytest.raises(ValueError):
        pipeline.analyze(Path('test.mov'))


def test_empty_experimental_result_does_not_become_no_violation():
    prediction = raw()
    prediction.update(video_score=.1, peak_time_sec=.5, scan_scores=[.1, .1, .1], candidates=[])
    result = adapt_prediction(json.dumps(prediction), 20, lambda _: pytest.fail('No frame expected'))
    assert result['candidates'] == []
    assert 'verdict' not in result


@pytest.mark.parametrize('change', [
    {'experimental': False}, {'scan_times_sec': [.5, .5, 4.5]}, {'scan_times_sec': [.5, 2.5]},
    {'scan_scores': [.1, 2, .1]}, {'video_score': .9}, {'peak_time_sec': 4.5},
    {'scan_times_sec': [.5, 2.5, 22]}, {'candidates': []}, {'verdict': 'no_violation'},
])
def test_inconsistent_raw_outputs_are_rejected_before_frame_extraction(change):
    prediction = {**raw(), **change}
    with pytest.raises(ValueError):
        adapt_prediction(json.dumps(prediction), 20, lambda _: pytest.fail('Invalid output must not extract frames'))


@pytest.mark.parametrize('change', [
    {'timecode_sec': 21}, {'score': .3}, {'score': .9}, {'first_scored_time_sec': 3},
    {'last_scored_time_sec': 22}, {'source': {'title': 'Invented'}},
])
def test_invalid_candidates_fail(change):
    prediction = copy.deepcopy(raw())
    prediction['candidates'][0].update(change)
    with pytest.raises(ValueError):
        adapt_prediction(json.dumps(prediction), 20, lambda _: pytest.fail('Invalid candidate'))
