import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from auto_annotate_gpu import parse_answer, consensus
from merge_auto_annotations import merge


class AutoAnnotationTests(unittest.TestCase):
    def valid(self, **changes):
        value = {'action':'cash_counting','visibility':'clear','review':'ordinary_visible_action',
                 'observation':'Виден пересчёт банкнот.','uncertainty':''}
        value.update(changes)
        return parse_answer(json.dumps(value))

    def test_disagreement_abstains(self):
        output = consensus(self.valid(), self.valid(action='phone_handling'))
        self.assertEqual(output['status'], 'uncertain')
        self.assertEqual(output['review'], 'unclear')

    def test_agreement_never_becomes_human_or_legal_confirmation(self):
        output = consensus(self.valid(review='possible_issue'), self.valid(review='possible_issue'))
        self.assertEqual(output['status'], 'auto_agreement')
        self.assertFalse(output['human_verified'])
        self.assertIsNone(output['legal_verdict'])

    def test_partial_visibility_abstains(self):
        self.assertEqual(consensus(self.valid(), self.valid(visibility='partial'))['status'], 'uncertain')

    def test_invalid_response_abstains(self):
        for text in ('not json', '[]', '{"action":"theft"}', '{"observation": 3}'):
            output = parse_answer(text)
            self.assertFalse(output['parsed'])
            self.assertEqual(consensus(self.valid(), output)['status'], 'uncertain')

    def test_mismatched_review_is_unclear_even_with_same_action(self):
        output = consensus(self.valid(), self.valid(review='possible_issue'))
        self.assertEqual(output['action'], 'cash_counting')
        self.assertEqual(output['review'], 'unclear')

    def merge_fixture(self):
        dataset={'samples':[{'group':i,'split':split,'center_sec':.5,'file':f'{i}.npz'}
                            for i,split in enumerate(('train','validation'))],
                 'groups':[{'group':i,'samples':[i],'annotated_event_times_sec':[], 'duration_sec':1}
                           for i in range(2)]}
        passes=[{'model':model,'dataset_sha256':'digest','prompt_sha256':'prompt','data_kind':'synthetic',
                 'predictions':[dict(self.valid(),sample_index=i) for i in range(2)]}
                for model in ('Qwen/Qwen2.5-VL-7B-Instruct','Qwen/Qwen3-VL-4B-Instruct')]
        return dataset,passes

    def test_validation_never_enters_silver_training_export(self):
        dataset,passes=self.merge_fixture()
        annotations,silver,drafts,summary=merge(dataset,*passes,'digest')
        self.assertEqual([row['sample_index'] for row in silver],[0])
        self.assertEqual(summary['silver_validation_windows'],0)
        self.assertTrue(all(row['review_status']=='pending' for row in drafts))
        self.assertTrue(all(row['legal_verdict'] is None for row in annotations))

    def test_incomplete_or_same_model_pass_rejected(self):
        dataset,passes=self.merge_fixture()
        with self.assertRaises(ValueError):
            merge(dataset,passes[0],passes[0],'digest')
        passes[1]['predictions'].pop()
        with self.assertRaises(ValueError):
            merge(dataset,*passes,'digest')
