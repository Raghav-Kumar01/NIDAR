"""Frame-freshness and failure regression tests; fake masks are not accuracy data."""
from dataclasses import dataclass
import io
import json
from types import SimpleNamespace
import unittest
import numpy as np
from full_openings import OpeningHook


@dataclass
class HumanResult:
    people: tuple = ()
    inference_ms: float = 1.


class HookTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.
        self.output = io.StringIO()
        self.polygon = {'confidence': .9, 'polygon_normalized': [[.1,.1],[.9,.1],[.9,.9],[.1,.9]]}
        self.adapter = SimpleNamespace(predict=lambda frame: [self.polygon])
        self.hook = OpeningHook(self.adapter, self.output, clock=lambda: self.now)
        self.frame = np.zeros((100,100,3),np.uint8)
        self.snapshot = SimpleNamespace(packet=SimpleNamespace(frame=self.frame,sequence=3,received_at=10.),connections=1)

    def record(self):
        self.hook.record(self.snapshot,HumanResult(),None,None,None,None,1.,'mission-a')
        return json.loads(self.output.getvalue().splitlines()[-1])

    def test_same_frame_and_overlay_do_not_mutate_capture(self):
        self.hook.predict(self.snapshot)
        rendered=self.hook.overlay(self.frame,self.snapshot,1.)
        self.assertFalse(self.frame.any())
        self.assertTrue(rendered.any())
        result=self.record()
        self.assertEqual(result['status'],'ok')
        self.assertEqual(result['frame_id'],3)
        self.assertEqual(result['opening_candidates'],[self.polygon])

    def test_stale_result_drops_all_observations(self):
        self.hook.predict(self.snapshot)
        self.now=12.
        result=self.record()
        self.assertEqual(result['status'],'stale')
        self.assertEqual(result['opening_candidates'],[])
        self.assertIsNone(result['humans'])

    def test_new_connection_cannot_reuse_old_masks(self):
        self.hook.predict(self.snapshot)
        self.snapshot.connections=2
        self.assertFalse(self.hook.current(self.snapshot,1.))

    def test_wrong_frame_cannot_reuse_old_masks(self):
        self.hook.predict(self.snapshot)
        self.snapshot.packet.sequence=4
        self.assertFalse(self.hook.current(self.snapshot,1.))

    def test_failure_preserves_human_result_and_marks_degraded(self):
        def fail(frame): raise RuntimeError('fixture')
        self.adapter.predict=fail
        self.hook.predict(self.snapshot)
        result=self.record()
        self.assertEqual(result['status'],'degraded')
        self.assertEqual(result['opening_status'],'ERROR:RuntimeError')
        self.assertIsNotNone(result['humans'])

    def test_reset_clears_opening_history(self):
        self.hook.predict(self.snapshot)
        self.hook.clear()
        self.assertEqual(self.hook.status,'WAITING')
        self.assertFalse(self.hook.current(self.snapshot,1.))

    def test_connection_lost_during_inference_discards_results(self):
        self.hook.predict(self.snapshot)
        self.hook.record(self.snapshot, HumanResult(), None, None, None, None,
                         1., 'mission-a', SimpleNamespace(packet=None, connections=2))
        result = json.loads(self.output.getvalue())
        self.assertEqual(result['status'], 'stale')
        self.assertEqual(result['opening_candidates'], [])
        self.assertIsNone(result['humans'])

    def test_invalid_polygon_is_failed_branch(self):
        self.polygon['polygon_normalized']=[[0,0],[0,0],[0,0]]
        self.hook.predict(self.snapshot)
        self.assertEqual(self.hook.status,'ERROR:ValueError')


if __name__=='__main__': unittest.main()
