import itertools
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from audit_video_rule_chain import build_tasks, select_windows, summarize
from auto_annotate_gpu import MODELS
from rat_ai.evidence import CONDITIONS
from rat_ai.sampled_evidence import PROTOCOL, LIMITS, assess_sampled


def fixture():
    dataset={'frames_per_clip':4,'clip_span_sec':3,
             'groups':[{'group':0,'split':'validation','samples':[0,1,2,3,4],
                        'label':1,'annotated_event_times_sec':[99]}],
             'samples':[{'group':0,'split':'validation','center_sec':i} for i in range(5)]}
    passes=[{'model':model,'dataset_sha256':'hash','prompt_sha256':'prompt','data_kind':'bank_review',
             'sampling':'existing four-frame three-second windows',
             'predictions':[{'sample_index':i,'parsed':True,'observation':'Синтетический пример.',
                             'uncertainty':'Действие может быть вне кадра.'} for i in range(5)]}
            for model in MODELS.values()]
    return dataset,passes


class VideoRuleChainTests(unittest.TestCase):
    def test_labels_and_event_times_never_change_inputs(self):
        dataset,passes=fixture()
        before=build_tasks(dataset,passes,'hash')
        dataset['groups'][0].update(label=0,annotated_event_times_sec=[0,1,2,3])
        self.assertEqual(before,build_tasks(dataset,passes,'hash'))
        self.assertEqual(select_windows(dataset),[0,2,4])
        self.assertTrue(all('Действие может быть вне кадра.' in t['observation'] for t in before[0]))

    def test_provenance_and_coverage_must_match(self):
        dataset,passes=fixture()
        with self.assertRaises(ValueError):
            build_tasks(dataset,passes,'different')
        passes[0]['predictions'].pop()
        with self.assertRaises(ValueError):
            build_tasks(dataset,passes,'hash')

    def test_other_sampling_protocol_is_rejected(self):
        dataset,passes=fixture()
        dataset['frames_per_clip']=16
        with self.assertRaises(ValueError):
            build_tasks(dataset,passes,'hash')
        dataset['frames_per_clip']=4
        passes[0]['sampling']='continuous video'
        with self.assertRaises(ValueError):
            build_tasks(dataset,passes,'hash')

    def test_invalid_caption_is_not_converted_into_clean_result(self):
        dataset,passes=fixture()
        passes[0]['predictions'][0]['parsed']=False
        tasks,selected,skipped=build_tasks(dataset,passes,'hash')
        self.assertEqual(skipped,1)
        self.assertEqual(len(tasks),30)
        report=summarize([{'parsed':False,'rule_id':'NBK231-P12'}])
        self.assertEqual(report['invalid_outputs'],1)
        self.assertEqual(report['bounded_status_counts'],{})

    def test_protocol_cannot_promote_and_preserves_unverified_status(self):
        for rule in CONDITIONS:
            for facts in itertools.product(('yes','no','unknown'),repeat=3):
                result=assess_sampled(rule,list(facts),protocol=PROTOCOL)
                self.assertNotEqual(result['status'],'possible_issue')
                self.assertFalse(result['human_verified'])
                self.assertFalse(result['legal_verdict'])
                for condition in LIMITS[rule]:
                    self.assertEqual(result['facts'][condition],'unknown')
        with self.assertRaises(ValueError):
            assess_sampled('NBK231-P20',['yes']*3,protocol='full_verified_video')


if __name__=='__main__':
    unittest.main()
