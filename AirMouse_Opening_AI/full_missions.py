"""Rotate missions inside this application's output area; preserve old databases."""
from pathlib import Path
import uuid


def rotate_mission(manager, tracker, verifier, output_dir):
    from nidar_survivor_demo.reid import AppearanceMatcher
    from nidar_survivor_demo.survivors import SurvivorManager
    from nidar_survivor_demo.tracking import PersonTracker
    from nidar_survivor_demo.verification import TemporalVerifier
    matcher = AppearanceMatcher(manager.matcher.config, manager.matcher.encoder, retain_gallery=True)
    next_tracker = PersonTracker(tracker.config)
    next_verifier = TemporalVerifier(verifier.config)
    path = Path(output_dir) / 'missions' / str(uuid.uuid4()) / 'mission.sqlite3'
    replacement = SurvivorManager(path, matcher, manager.config)
    try:
        manager.close()
    except Exception:
        replacement.close()
        raise
    return replacement, matcher, next_tracker, next_verifier, manager.snapshot()
