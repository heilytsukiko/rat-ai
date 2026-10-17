from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from prepare_source_context import letterbox
from compare_source_context import read_source_window


class SourceContextTests(unittest.TestCase):
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
