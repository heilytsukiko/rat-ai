from pathlib import Path
import copy
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from rat_ai.inference import candidate_events, validate_candidate_policy
from evaluate_candidate_policy import evaluate
from compare_detectors import compare_reports
from test_comparison import report


class CandidatePolicyTests(unittest.TestCase):
    def test_confirmation_removes_isolated_spikes_and_keeps_actual_center_score(self):
        scored = {'scan_times_sec': [.5, 2.5, 4.5, 6.5, 8.5],
                  'scan_scores': [.8, .1, .9, .6, .7]}
        events = candidate_events(scored, min_windows=2, representative='center')
        self.assertEqual(events, [{'timecode_sec': 6.5, 'score': .6,
                                  'first_scored_time_sec': 4.5, 'last_scored_time_sec': 8.5}])
        # Windows separated by a missing scan cannot confirm each other.
        self.assertEqual(candidate_events({'scan_times_sec': [.5, 6.5], 'scan_scores': [.8, .8]},
                                          min_windows=2), [])

    def test_invalid_policy_cannot_silently_change_results(self):
        for policy in ([], {'min_windows': True}, {'min_windows': 0}, {'min_windows': 11},
                       {'representative': 'invented'}, {'unknown': 1}):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                validate_candidate_policy(policy)

    def test_evaluation_preserves_raw_scores_and_rejects_tampered_decisions(self):
        original = report([.9, .9, .9, .9])
        for row in original['validation_video_results']:
            row['scan_times_sec'] = [.5, 2.5, 4.5]
            row['scan_scores'] = [.1, .9, .9 if row['label'] else .1]
            row['annotated_event_times_sec'] = [2.5] if row['label'] else []
        saved = copy.deepcopy(original)
        calibrated = evaluate(original, {'min_windows': 2, 'representative': 'center'})
        self.assertEqual(original, saved)
        result = compare_reports(original, calibrated)
        self.assertTrue(result['candidate_eligible_for_packaging'])
        self.assertEqual(result['candidate']['fp'], 0)
        self.assertEqual(calibrated['validation_video_results'][0]['video_score'], .9)
        calibrated['validation_video_results'][0]['decision_score'] = .9
        with self.assertRaisesRegex(ValueError, 'Decision score'):
            compare_reports(original, calibrated)


if __name__ == '__main__':
    unittest.main()
