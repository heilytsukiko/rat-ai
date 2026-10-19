import copy
import json
import unittest
from rat_ai.observation import parse_report, report_prompt


def response():
    return {'summary': 'На тестовых кадрах перемещается прямоугольник.',
            'actions': [{'frame_index': 0, 'description': 'Прямоугольник слева.'},
                        {'frame_index': 7, 'description': 'Прямоугольник справа.'}],
            'object_description': 'Прямоугольник; людей и передачи не видно.',
            'assessment': 'no_transfer_visible', 'reason': 'В кадрах нет людей.',
            'limitations': ['Не анализировались движения между выбранными кадрами.']}


class ObservationTests(unittest.TestCase):
    def test_actions_reference_actual_decoded_timestamps(self):
        result = parse_report(json.dumps(response()), [0, .3, .6, .9, 1.2, 1.5, 1.8, 2.1])
        self.assertEqual([a['timestamp_sec'] for a in result['actions']], [0, 2.1])
        self.assertNotIn('frame_index', result['actions'][0])
        self.assertEqual(result['assessment'], 'no_transfer_visible')

    def test_json_fence_is_supported(self):
        self.assertEqual(parse_report('```json\n'+json.dumps(response())+'\n```', list(range(8)))['summary'], response()['summary'])

    def test_invalid_or_invented_reports_are_rejected(self):
        for field, value in [('assessment', 'confirmed_bribe'), ('assessment', []),
                             ('actions', [{'frame_index': 8, 'description': 'invented'}]),
                             ('actions', [{'frame_index': True, 'description': 'boolean'}]),
                             ('actions', [{'frame_index': 7, 'description': 'later'}, {'frame_index': 0, 'description': 'earlier'}]),
                             ('limitations', []), ('summary', '   ')]:
            with self.subTest(field=field, value=value):
                data=copy.deepcopy(response());data[field]=value
                with self.assertRaises(ValueError):
                    parse_report(json.dumps(data), list(range(8)))

    def test_prompt_requests_visible_facts_and_missing_context(self):
        prompt=report_prompt(8)
        self.assertIn('Индексы кадров от 0 до 7',prompt)
        self.assertIn('Не называй эпизод доказанной взяткой',prompt)

    def test_empty_scan_does_not_load_gpu_dependencies(self):
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location('describe_gpu_candidates', Path(__file__).resolve().parents[1] / 'tools/describe_gpu_candidates.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.describe('unused.mov', [], 'unused.json'), [])
        with self.assertRaisesRegex(ValueError, 'Too many candidates'):
            module.describe('unused.mov', [{}]*9, 'unused.json')
