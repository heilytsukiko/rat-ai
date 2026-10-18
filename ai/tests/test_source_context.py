from pathlib import Path
import copy
import hashlib
import json
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from prepare_source_context import letterbox
from compare_source_context import read_source_window
from compare_video_context import MODELS
from build_context_review import load_source_review


class SourceContextTests(unittest.TestCase):
    def test_source_review_requires_matching_frames_models_and_complete_results(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            root=Path(folder)
            context={'source_dataset_sha256':'fixture','source_mapping_verified':True,
                     'records':[{'sample_index':0}]}
            raw=json.dumps(context).encode()
            (root/'context.json').write_bytes(raw)
            baseline={'dataset_sha256':'fixture','selected_windows':[0],
                      'models':{k:{'model':v,'revision':'test'} for k,v in MODELS.items()},
                      'prompt_sha256':'test','parser_policy':'test'}
            protocol=copy.deepcopy(baseline)
            protocol.update(condition='source16_tiles',source_context_manifest_sha256=hashlib.sha256(raw).hexdigest())
            (root/'protocol-private.json').write_text(json.dumps(protocol))
            rows=[{'model_key':key,'mode':'source16_tiles','sample_index':0,'parsed':False,
                   'action':'unclear','visibility':'unclear','review':'unclear','objects':[],
                   'elapsed_sec':0,'frame_count':16} for key in MODELS]
            (root/'predictions-private.json').write_text(json.dumps({'predictions':rows}))
            records,loaded=load_source_review(root,root,baseline)
            self.assertEqual((len(records),len(loaded)),(1,2))
            wrong=copy.deepcopy(baseline)
            wrong['models']['qwen25-7b']['revision']='different'
            with self.assertRaises(ValueError):
                load_source_review(root,root,wrong)
            (root/'predictions-private.json').write_text(json.dumps({'predictions':rows[:1]}))
            with self.assertRaises(ValueError):
                load_source_review(root,root,baseline)
            (root/'context.json').write_bytes(raw+b' ')
            with self.assertRaises(ValueError):
                load_source_review(root,root,baseline)

    def test_portrait_keeps_aspect_ratio_and_neutral_padding(self):
        image=np.full((848,464,3),[200,20,10],dtype=np.uint8)
        result=letterbox(image)
        self.assertEqual(result.shape,(448,448,3))
        np.testing.assert_array_equal(result[224,0],[127,127,127])
        np.testing.assert_array_equal(result[224,224],[200,20,10])

    def test_source_times_and_crop_anchor_are_checked(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            root=Path(folder)
            np.savez_compressed(root/'clip.npz',frames=np.zeros((16,448,448,3),dtype=np.uint8),
                tiles=np.zeros((4,448,448,3),dtype=np.uint8),decoded_time_sec=np.arange(16,dtype=float))
            row={'file':'clip.npz','center_sec':7,'anchor_index':7}
            frames,crops,times=read_source_window(root,row)
            self.assertEqual((len(frames),len(crops),times[7]),(16,4,7))
            row['anchor_index']=0
            with self.assertRaises(ValueError):
                read_source_window(root,row)

    def test_external_bundle_paths_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            with self.assertRaises(ValueError):
                read_source_window(Path(folder),{'file':'../other.npz'})


if __name__=='__main__':
    unittest.main()
