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


@pytest.mark.parametrize('architecture', ['mobilenet_v3_small_temporal_v1', 'ssdlite_coco_temporal_v1',
                                         'qwen2_5_vl_3b_qlora_v1', 'qwen3_vl_2b_qlora_v1',
                                         'qwen2_5_vl_7b_qlora_v1', 'qwen3_vl_4b_qlora_v1'])
def test_real_pipeline_experimental_mode_extracts_evidence_without_legal_verdict(monkeypatch, architecture):
    configure(monkeypatch, {**raw(), 'model': architecture})
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


def test_optional_gpu_description_keeps_result_inconclusive_and_adds_context_frames():
    prediction=raw()
    prediction['observation_model']='Qwen/Qwen2.5-VL-7B-Instruct'
    prediction['candidates'][0]['observation_draft']='A synthetic rectangle moves.'
    result=adapt_prediction(json.dumps(prediction),20,lambda t:{'timestamp_sec':t})
    assert result['candidates'][0]['observation_draft']=='A synthetic rectangle moves.'
    assert [f['timestamp_sec'] for f in result['candidates'][0]['evidence_frames']]==[2.5,0,6]
    assert 'verdict' not in result and 'source' not in result['candidates'][0]


@pytest.mark.parametrize('value',['', ' '*4, 'x'*1601, {'unsafe':'object'}])
def test_malformed_observation_is_rejected(value):
    prediction=raw()
    prediction['observation_model']='Qwen/Qwen2.5-VL-7B-Instruct'
    prediction['candidates'][0]['observation_draft']=value
    with pytest.raises(ValueError):
        adapt_prediction(json.dumps(prediction),20,lambda _:pytest.fail('Invalid description'))


def test_observation_requires_a_declared_known_model():
    prediction=raw()
    prediction['candidates'][0]['observation_draft']='Synthetic action'
    with pytest.raises(ValueError):
        adapt_prediction(json.dumps(prediction),20,lambda _:pytest.fail('Missing model'))


def test_empty_experimental_result_does_not_become_no_violation():
    prediction = raw()
    prediction.update(video_score=.1, peak_time_sec=.5, scan_scores=[.1, .1, .1], candidates=[])
    result = adapt_prediction(json.dumps(prediction), 20, lambda _: pytest.fail('No frame expected'))
    assert result['candidates'] == []
    assert 'verdict' not in result


def test_confirmation_filters_a_high_isolated_peak_without_a_legal_verdict():
    prediction = raw()
    prediction.update(candidate_policy={'min_windows': 2, 'representative': 'center'}, candidates=[])
    result = adapt_prediction(json.dumps(prediction), 20, lambda _: pytest.fail('Filtered spike'))
    assert result['video_score'] == .8 and result['candidates'] == []
    assert 'verdict' not in result


def test_center_policy_extracts_the_center_frame_with_its_actual_score():
    prediction = raw()
    prediction.update(video_score=.9, peak_time_sec=.5, scan_scores=[.9, .6, .7],
                      candidate_policy={'min_windows': 2, 'representative': 'center'},
                      candidates=[{'timecode_sec': 2.5, 'score': .6,
                                   'first_scored_time_sec': .5, 'last_scored_time_sec': 4.5}])
    result = adapt_prediction(json.dumps(prediction), 20, lambda t: {'timestamp_sec': t})
    assert result['candidates'][0]['score'] == .6
    assert result['candidates'][0]['evidence_frames'][0]['timestamp_sec'] == 2.5
    prediction['candidates'][0].update(timecode_sec=.5, score=.9)
    with pytest.raises(ValueError):
        adapt_prediction(json.dumps(prediction), 20, lambda _: pytest.fail('Wrong representative'))


@pytest.mark.parametrize('policy', [{'min_windows': True}, {'min_windows': 0},
                                   {'representative': 'invented'}, {'unknown': 1}])
def test_invalid_confirmation_policy_is_rejected(policy):
    with pytest.raises(ValueError):
        adapt_prediction(json.dumps({**raw(), 'candidate_policy': policy}), 20,
                         lambda _: pytest.fail('Invalid policy'))


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


def reported():
    prediction = raw()
    prediction['observation_model'] = 'Qwen/Qwen2.5-VL-7B-Instruct'
    prediction['candidates'][0]['observation_report'] = {
        'summary': 'Тестовый прямоугольник смещается.',
        'actions': [{'timestamp_sec': 0, 'description': 'Прямоугольник слева.'},
                    {'timestamp_sec': 6, 'description': 'Прямоугольник справа.'}],
        'object_description': 'Прямоугольник.', 'assessment': 'no_transfer_visible',
        'reason': 'Людей и передачи не видно.', 'limitations': ['Только выбранные кадры.'],
        'sampled_timestamps_sec': [0, 1, 2, 2.5, 3, 4, 5, 6]}
    return prediction


def test_report_preserves_actions_and_extracts_real_context_frames():
    result = adapt_prediction(json.dumps(reported()), 20, lambda t: {'timestamp_sec': t})
    event = result['candidates'][0]
    assert event['observation_report']['assessment'] == 'no_transfer_visible'
    assert [f['timestamp_sec'] for f in event['evidence_frames']] == [2.5, 0, 6]
    assert 'verdict' not in result


@pytest.mark.parametrize('field,value', [
    ('assessment', 'confirmed_bribe'), ('summary', ' '), ('limitations', [' ']),
    ('actions', [{'timestamp_sec': 1.1, 'description': 'Not a sampled frame'}]),
    ('actions', [{'timestamp_sec': 6, 'description': 'Later'}, {'timestamp_sec': 0, 'description': 'Earlier'}]),
    ('sampled_timestamps_sec', [0, 1, 2, 2.5, 3, 4, 5, 19]),
])
def test_report_cannot_invent_frame_references_or_final_verdict(field, value):
    prediction = reported()
    prediction['candidates'][0]['observation_report'][field] = value
    with pytest.raises(ValueError):
        adapt_prediction(json.dumps(prediction), 20, lambda _: pytest.fail('Invalid report must not extract evidence'))


def test_report_requires_model_attribution():
    prediction = reported()
    del prediction['observation_model']
    with pytest.raises(ValueError):
        adapt_prediction(json.dumps(prediction), 20, lambda _: pytest.fail('Missing model'))
