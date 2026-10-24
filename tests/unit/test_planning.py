import unittest
from gdn_prefill import CheckpointRequest as R, Geometry, make_plan
from gdn_prefill.integrations.sglang import requests_from_tracking


class PlanningTests(unittest.TestCase):
    def test_measured_dispatch(self):
        cases = [([8192], 'v2', 'v2'), ([8192,1024,1024,1024], 'v2','v2'),
                 ([4096,4096], 'v2','full'), ([8192,8192], 'v2','full'), ([1024]*8, 'full','full')]
        for lengths, a, b in cases:
            requests = [R(i,n//128*64,i) for i,n in enumerate(lengths)]
            for geometry, expected in [(Geometry(4,16),a),(Geometry(16,32),b)]:
                self.assertEqual(make_plan(lengths,geometry,requests).backend,expected)

    def test_no_checkpoint_routes(self):
        self.assertEqual(make_plan([8192,1024,1024,1024],Geometry(4,16)).backend,'mixed_cp')
        self.assertEqual(make_plan([8192],Geometry(16,32)).backend,'original')
        self.assertEqual(make_plan([4096,4096],Geometry(4,16)).backend,'original')

    def test_multi_target_dense_mapping_and_sparse_slots(self):
        p=make_plan([0,129,8193],Geometry(4,16),[R(1,64,2),R(1,128,0),R(2,8192,5)])
        self.assertEqual(p.backend,'original')
        self.assertEqual(p.checkpoints.dense_ids,(0,1,129))
        self.assertEqual(p.checkpoints.slots,(2,0,5))
        self.assertEqual(p.checkpoints.capacity,6)

    def test_capability_fallback_and_experimental_gate(self):
        self.assertEqual(make_plan([512],Geometry(8,16),[R(0,64,0)]).backend,'original')
        with self.assertRaises(ValueError):make_plan([512],Geometry(4,16),[R(0,64,0)],backend='cp_checkpoint')
        self.assertEqual(make_plan([512],Geometry(4,16),[R(0,64,0)],backend='cp_checkpoint',policy='experimental').backend,'cp_checkpoint')
        with self.assertRaises(ValueError):make_plan([512],Geometry(4,16),[R(0,64,0)],backend='cp')
        with self.assertRaises(ValueError):make_plan([512],Geometry(4,16),[R(0,64,0),R(0,128,1)],backend='v2')

    def test_invalid_requests(self):
        for reqs in [[R(0,65,0)],[R(0,576,0)],[R(1,64,0)],[R(0,64,-1)],[R(0,64,0),R(0,128,0)]]:
            with self.assertRaises(ValueError):make_plan([512],Geometry(4,16),reqs)
        for lengths in [[],[0,0],[-1,512],[1.5]]:
            with self.assertRaises(ValueError):make_plan(lengths,Geometry(4,16))

    def test_tracking_preserves_consumer_order(self):
        reqs=requests_from_tracking(lengths=[1024,2048,512],prefix_lengths=[4096,0,512],
                                   track_lengths=[4992,2048,960],track_mask=[True,True,True],interval=256)
        self.assertEqual(reqs,(R(0,768,0),R(2,256,1)))
        with self.assertRaises(ValueError):requests_from_tracking(lengths=[64],prefix_lengths=[1000],
                          track_lengths=[1010],track_mask=[True],interval=256)

if __name__=='__main__':unittest.main()
