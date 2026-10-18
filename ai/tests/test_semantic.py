import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Requires detector dependencies')
class SemanticFeatureTests(unittest.TestCase):
    def test_projection_trains_while_object_detector_remains_frozen_and_offline(self):
        import torch
        from rat_ai.semantic import SemanticTemporalDetector
        torch.set_num_threads(2)
        with patch('urllib.request.urlopen', side_effect=AssertionError('Network in model initialization')):
            model = SemanticTemporalDetector()
            model.train()
            before = model.tail[0].weight.detach().clone()
            optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01)
            model.forward_cached(torch.randn(2, 3, 546)).square().mean().backward()
            optimizer.step()
            self.assertGreater(float((model.tail[0].weight - before).abs().sum().detach()), 0)
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.prefix.parameters()))
            self.assertFalse(model.prefix.training)
            model.eval()
            with torch.inference_mode():
                scores = model(torch.zeros(1, 3, 3, 320, 320))
            self.assertEqual(tuple(scores.shape), (1,))
            self.assertTrue(bool(torch.isfinite(scores).all()))
            from export_detector import export_checkpoint
            from rat_ai.inference import load_detector
            scratch = Path(__file__).resolve().parents[1] / 'local/tests'
            scratch.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=scratch) as temporary:
                source, public = Path(temporary) / 'private.pt', Path(temporary) / 'public.safetensors'
                torch.save({'state_dict': model.state_dict(), 'epoch': 1, 'config': {
                    'architecture': 'ssdlite_coco_temporal_v1', 'preprocessing': 'synthetic fixture',
                    'threshold': .5, 'pretrained_url': 'https://example.invalid/fixture',
                    'pretrained_sha256': 'synthetic', 'sample_spec': [{'video_path': 'PRIVATE_SENTINEL'}]}}, source)
                metadata = export_checkpoint(source, public)
                self.assertEqual(metadata['input_size'], 320)
                self.assertNotIn('PRIVATE_SENTINEL', public.with_suffix('.json').read_text(encoding='utf-8'))
                restored, _ = load_detector(public)
                self.assertTrue(torch.equal(model.tail[0].weight, restored.tail[0].weight))


if __name__ == '__main__':
    unittest.main()
