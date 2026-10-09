"""Contract tests with fake adapters, not model-accuracy tests."""
import unittest
import numpy as np
from perception import PerceptionPipeline


class Adapter:
    def __init__(self,result=None,error=False,mutate=False):self.result=result or [];self.error=error;self.mutate=mutate;self.seen=[]
    def predict(self,frame):
        self.seen.append(frame.copy())
        if self.mutate:frame[:]=255
        if self.error:raise RuntimeError('fixture failure')
        return self.result


class ContractTests(unittest.TestCase):
    def setUp(self):self.frame=np.zeros((20,30,3),dtype=np.uint8)
    def test_same_frame_identity_and_nonmutating_adapters(self):
        a=Adapter(mutate=True);b=Adapter();p=PerceptionPipeline(a,b,clock=lambda:10.)
        r=p.process(self.frame,7,10.)
        self.assertEqual(r.frame_id,7);self.assertEqual(r.received_at_monotonic,10.)
        self.assertFalse(self.frame.any());self.assertFalse(b.seen[0].any());self.assertEqual(r.status,'ok')
    def test_failed_branch_does_not_masquerade_as_successful_negative(self):
        person={'confidence':.9,'bbox_xyxy_pixels':[1,2,10,15]}
        p=PerceptionPipeline(Adapter(error=True),Adapter([person]),clock=lambda:10.)
        r=p.process(self.frame,0,10.)
        self.assertEqual(r.status,'degraded');self.assertEqual(len(r.person_candidates),1)
        self.assertEqual(r.branches['openings'],'error:RuntimeError');self.assertFalse(r.flight_ready)
    def test_stale_frame_skips_both_models(self):
        a=Adapter();b=Adapter();r=PerceptionPipeline(a,b,clock=lambda:10.).process(self.frame,0,8.)
        self.assertEqual(r.status,'stale');self.assertEqual(a.seen,[]);self.assertEqual(b.seen,[])
    def test_expired_result_discards_outputs(self):
        clock=iter([10.,10.,10.1,10.1,12.,12.])
        p=PerceptionPipeline(Adapter(),Adapter([{'confidence':.8,'bbox_xyxy_pixels':[0,0,10,10]}]),clock=lambda:next(clock))
        r=p.process(self.frame,0,10.)
        self.assertEqual(r.status,'stale');self.assertEqual(r.person_candidates,[])
    def test_invalid_box_is_explicit_branch_error(self):
        r=PerceptionPipeline(Adapter(),Adapter([{'confidence':.9,'bbox_xyxy_pixels':[0,0,50,10]}]),clock=lambda:10.).process(self.frame,0,10.)
        self.assertEqual(r.status,'degraded');self.assertEqual(r.person_candidates,[])
    def test_invalid_frame_and_future_timestamp(self):
        p=PerceptionPipeline(Adapter(),Adapter(),clock=lambda:10.)
        for frame,stamp in [(np.zeros((10,10)),10.),(self.frame,11.)]:
            with self.assertRaises(ValueError):p.process(frame,0,stamp)
    def test_nested_openings_remain_separate(self):
        polygon=[[.1,.1],[.9,.1],[.9,.9],[.1,.9]]
        r=PerceptionPipeline(Adapter([{'confidence':.9,'polygon_normalized':polygon},{'confidence':.6,'polygon_normalized':[[.3,.3],[.6,.3],[.6,.6]]}]),Adapter(),clock=lambda:10.).process(self.frame,0,10.)
        self.assertEqual(len(r.opening_candidates),2)


if __name__=='__main__':unittest.main()
