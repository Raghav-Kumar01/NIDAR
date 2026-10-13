"""Run pinned upstream regressions and a combined-loop fixture; no accuracy claim."""
import argparse
import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/perception.local.json')
    args = parser.parse_args()
    from run_full_perception import prepare
    prepare(args.config)
    import numpy as np
    import full_runtime as runtime
    from full_openings import OpeningHook
    from full_missions import rotate_mission
    from nidar_survivor_demo.camera.phone_stream import PhoneStream
    from nidar_survivor_demo.config import CameraConfig
    from nidar_survivor_demo.dashboard import DashboardState
    from nidar_survivor_demo.reid import ReIDConfig, AppearanceMatcher
    from nidar_survivor_demo.survivors import SurvivorManager, ManagerConfig
    from nidar_survivor_demo.tracking import PersonTracker, TrackerConfig
    from nidar_survivor_demo.verification import TemporalVerifier, VerificationConfig
    from nidar_survivor_demo.mission_report import read_report
    from nidar_survivor_demo.tests.test_phase1 import FakeCapture
    from nidar_survivor_demo.tests.test_phase2 import FakeDetector
    from nidar_survivor_demo.tests.test_phase6 import Encoder

    class CombinedTests(unittest.TestCase):
        def test_shared_frame_pipeline_and_report(self):
            output = io.StringIO()
            hook = OpeningHook(SimpleNamespace(predict=lambda frame: [
                {'confidence': .9, 'polygon_normalized': [[.1,.1],[.9,.1],[.9,.9],[.1,.9]]}
            ]), output)
            state = DashboardState()
            with tempfile.TemporaryDirectory() as temp, \
                 patch.object(runtime, 'AppearanceEncoder', return_value=Encoder()), \
                 patch.object(runtime.ReIDConfig, 'load', return_value=replace(ReIDConfig(), sample_interval=.001, retry_interval=.001)), \
                 patch.object(runtime, 'PersonDetector', return_value=FakeDetector()), \
                 patch.object(runtime, 'PhoneStream', return_value=PhoneStream(CameraConfig(), lambda _: FakeCapture())), \
                 patch('nidar_survivor_demo.reid.quality_crop', return_value=(np.ones((100,50,3),np.uint8), 'good')), \
                 patch('nidar_survivor_demo.dashboard.DashboardState', return_value=state), \
                 patch('nidar_survivor_demo.dashboard.DashboardServer') as server, \
                 contextlib.redirect_stdout(io.StringIO()):
                server.return_value.url = 'http://127.0.0.1:8765'
                report = Path(temp) / 'report.json'
                result = runtime.run(['--dashboard','--headless','--seconds','1.2',
                                      '--mission-db',str(Path(temp)/'mission.sqlite3')],
                                     opening_hook=hook, report_path=report)
                self.assertEqual(result, 0)
                self.assertEqual(json.loads(report.read_text())['unique_survivors'], 1)
                server.return_value.close.assert_called_once()
            rows = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertTrue(rows)
            self.assertTrue(any(row['opening_candidates'] and row['tracking'] and row['verification']
                                and row['reid'] and row['survivors'] for row in rows))
            self.assertTrue(all(not row['flight_ready'] for row in rows))
            self.assertEqual(state.snapshot()['status'], 'STOPPED')

        def test_reset_preserves_old_database_in_output_area(self):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                old = SurvivorManager(root/'mission.sqlite3', AppearanceMatcher(ReIDConfig(), Encoder(), retain_gallery=True), ManagerConfig())
                previous_id = old.mission_id
                replacement, _, _, _, previous = rotate_mission(
                    old, PersonTracker(TrackerConfig()), TemporalVerifier(VerificationConfig()), root)
                try:
                    self.assertEqual(previous['mission_id'], previous_id)
                    self.assertNotEqual(replacement.mission_id, previous_id)
                    self.assertTrue(replacement.path.resolve().is_relative_to(root.resolve()))
                    self.assertEqual(read_report(old.path)['mission_id'], previous_id)
                    self.assertEqual(replacement.snapshot()['unique_survivors'], 0)
                finally:
                    replacement.close()

    modules = ['test_phase3','test_phase4','test_phase5','test_phase6','test_phase7',
               'test_mission_reset','test_multi_identity_scaling','test_identity_core',
               'test_identity_regression','test_counting_input','test_sample_memory']
    suite = unittest.defaultTestLoader.loadTestsFromNames(['nidar_survivor_demo.tests.' + name for name in modules])
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(CombinedTests))
    return 0 if unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
