import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from compare_gpu_models import compare
from test_comparison import report


class GpuComparisonTests(unittest.TestCase):
    def fixtures(self):
        first, second = report([.9, .9, .9, .9]), report([.1, .1, .9, .9])
        for item in (first, second):
            for row in item['validation_video_results']:
                row['scan_times_sec'] = [.5, 2.5]
                row['scan_scores'] *= 2
                row['annotated_event_times_sec'] = [1.5] if row['label'] else []
        return first, second

    def test_ensemble_does_not_automatically_replace_better_single_model(self):
        first, second = self.fixtures()
        saved = copy.deepcopy(first)
        result = compare(first, second)
        self.assertEqual(result['recommended_by_validation'], 'qwen3')
        self.assertEqual(result['results']['equal_mean']['metrics']['fp'], 2)
        self.assertEqual(first, saved)
        self.assertNotIn('PRIVATE', str(result))

    def test_different_inputs_are_not_a_fair_competition(self):
        first, second = self.fixtures()
        second['validation_video_results'][0]['scan_times_sec'] = [.5, 3.5]
        with self.assertRaisesRegex(ValueError, 'identical'):
            compare(first, second)


if __name__ == '__main__':
    unittest.main()
