import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from train_gpu_probe import fit_probe, score_probe


class ProbeTests(unittest.TestCase):
    def test_validation_never_changes_fit(self):
        rows = [{'split':'train','group':0,'label':0}, {'split':'train','group':1,'label':1},
                {'split':'validation','group':2,'label':0}]
        x = np.array([[-1.,0.], [1.,0.], [50.,100.]])
        before = fit_probe(x, rows)
        rows[-1]['label'] = 1
        x[-1] = [-900.,800.]
        after = fit_probe(x, rows)
        for key in before:
            np.testing.assert_allclose(before[key], after[key])

    def test_class_balance_with_different_window_counts(self):
        rows = [{'split':'train','group':0,'label':0}, {'split':'train','group':1,'label':1}]
        x = np.array([[-1.], [1.]])
        # Weighting preserves class balance even with several windows in one video.
        fit = fit_probe(np.repeat(x, [5,1], axis=0), [rows[0]]*5+[rows[1]])
        self.assertAlmostEqual(float(fit['mean'][0]), 0.)
        self.assertAlmostEqual(fit['bias'], 0.)
        self.assertLess(score_probe(x, fit)[0], .5)
        self.assertGreater(score_probe(x, fit)[1], .5)

    def test_rejects_nonfinite_features(self):
        with self.assertRaises(ValueError):
            fit_probe([[float('nan')]], [{'split':'train','group':0,'label':0}])
