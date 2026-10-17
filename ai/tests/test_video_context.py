import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from compare_video_context import nominal_refs, build_messages, parse_context_answer, tiles, summary_for
from build_context_review import render_prediction


def fixture():
    return {'frames_per_clip':4,'clip_span_sec':3,
            'groups':[{'group':0,'split':'validation','duration_sec':12,
                       'samples':list(range(6)),'label':1,'annotated_event_times_sec':[7]}],
            'samples':[{'group':0,'split':'validation','center_sec':1+2*i} for i in range(6)]}


def answer():
    return {'action':'computer_use','visibility':'clear','review':'ordinary_visible_action',
            'observation':'Работа за компьютером.','uncertainty':'',
            'objects':['computer'],'evidence_frames':[1]}


class VideoContextTests(unittest.TestCase):
    def test_context_preserves_target_and_uses_unique_sorted_real_references(self):
        dataset=fixture()
        four=nominal_refs(dataset,3,4)
        extended=nominal_refs(dataset,3,16)
        self.assertTrue(all(r in extended for r in four))
        self.assertGreater(len(extended),4)
        self.assertLessEqual(len(extended),16)
        times=[r['nominal_sec'] for r in extended]
        self.assertEqual(times,sorted(set(times)))
        self.assertTrue(all(1<=t<=12 for t in times))
        self.assertGreater(times[-1]-times[0],four[-1]['nominal_sec']-four[0]['nominal_sec'])

    def test_label_changes_cannot_change_context(self):
        dataset=fixture()
        before=nominal_refs(dataset,2,16)
        dataset['groups'][0].update(label=0,annotated_event_times_sec=[1,11])
        for row in dataset['samples']:
            row['label']=0
        self.assertEqual(before,nominal_refs(dataset,2,16))

    def test_cross_group_references_rejected_and_short_clip_not_padded(self):
        dataset=fixture()
        dataset['samples'][4]['group']=99
        with self.assertRaises(ValueError):
            nominal_refs(dataset,2,16)
        dataset=fixture()
        dataset['groups'][0]['samples']=[0]
        self.assertEqual(len(nominal_refs(dataset,0,16)),4)

    def test_tiles_are_explicitly_same_instant_and_do_not_add_frame_numbers(self):
        from PIL import Image
        refs=nominal_refs(fixture(),2,16)
        chat,anchor=build_messages(refs,5,True)
        self.assertEqual(sum(x['type']=='image' for x in chat[0]['content']),len(refs)+4)
        texts=[x['text'] for x in chat[0]['content'] if x['type']=='text']
        self.assertEqual(sum('same instant, not a new frame' in x for x in texts),4)
        result=tiles(Image.new('RGB',(336,336),'red'))
        self.assertEqual(len(result),4)
        self.assertTrue(all(x.size==(336,336) for x in result))

    def test_unlocatable_or_tile_references_fail_without_creating_labels(self):
        for indices in ([],[0],[5],[True],[1,1]):
            value=answer()
            value['evidence_frames']=indices
            self.assertFalse(parse_context_answer(json.dumps(value),4)['parsed'])
        self.assertTrue(parse_context_answer(json.dumps(answer()),4)['parsed'])

    def test_prompt_placement_preserves_images_references_and_instruction(self):
        refs=nominal_refs(fixture(),2,16)
        before,anchor=build_messages(refs,5,True)
        after,new_anchor=build_messages(refs,5,True,prompt_at_end=True)
        self.assertEqual(anchor,new_anchor)
        self.assertEqual(after[0]['content'][-1],before[0]['content'][0])
        self.assertEqual(after[0]['content'][:-1],before[0]['content'][1:])

    def test_extra_valid_references_are_retained_but_unknown_objects_abstain(self):
        value=answer()
        value['evidence_frames']=[1,2,3,4]
        parsed=parse_context_answer(json.dumps(value),4)
        self.assertTrue(parsed['parsed'])
        self.assertTrue(parsed['extra_evidence_references'])
        self.assertFalse(parse_context_answer(json.dumps(value),4,strict_schema=True)['parsed'])
        value['objects']=['computer','unrecognized_object']
        parsed=parse_context_answer(json.dumps(value),4)
        self.assertTrue(parsed['parsed'])
        self.assertEqual(parsed['objects'],['computer','unclear'])
        self.assertEqual(parsed['visibility'],'partial')
        self.assertEqual(parsed['review'],'unclear')

    def test_incomplete_comparison_cannot_be_published_as_improvement(self):
        with self.assertRaises(ValueError):
            summary_for([], [0])

    def test_generated_observations_cannot_execute_in_offline_review(self):
        row=answer()
        row.update(parsed=True,observation='<script>alert(1)</script>',
                   uncertainty='<img src=x onerror=alert(1)>',evidence_nominal_sec=[1.0])
        html=render_prediction(row)
        self.assertNotIn('<script>',html)
        self.assertNotIn('<img src=x',html)
        self.assertIn('&lt;script&gt;',html)


if __name__=='__main__':
    unittest.main()
