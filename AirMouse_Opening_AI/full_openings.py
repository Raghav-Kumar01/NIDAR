"""Opening branch attached to Raj's full human runtime, with frame-scoped output."""
from dataclasses import asdict
import json
import time

import cv2
import numpy as np

from perception import validate_observations


class OpeningHook:
    def __init__(self, adapter, output, clock=time.monotonic):
        self.adapter = adapter
        self.clock = clock
        self.output = output
        self.clear()

    def clear(self):
        self.sequence = None
        self.connection = None
        self.values = []
        self.status = 'WAITING'
        self.elapsed_ms = None

    def predict(self, snapshot):
        self.clear()
        packet = snapshot.packet
        self.sequence, self.connection = packet.sequence, snapshot.connections
        started = self.clock()
        try:
            values = self.adapter.predict(packet.frame.copy())
            h, w = packet.frame.shape[:2]
            validate_observations(values, 'openings', w, h)
            self.values = values
            self.status = 'ACTIVE'
        except Exception as exc:
            self.status = 'ERROR:' + type(exc).__name__
        self.elapsed_ms = (self.clock() - started) * 1000

    def current(self, snapshot, max_age):
        return (snapshot.packet is not None and self.sequence == snapshot.packet.sequence
                and self.connection == snapshot.connections
                and 0 <= self.clock() - snapshot.packet.received_at <= max_age)

    def overlay(self, frame, snapshot, max_age):
        output = frame.copy()
        valid = self.current(snapshot, max_age)
        values = self.values if valid else []
        h, w = output.shape[:2]
        for index, value in enumerate(values, 1):
            polygon = np.rint(np.asarray(value['polygon_normalized']) * [w - 1, h - 1]).astype(np.int32)
            cv2.polylines(output, [polygon], True, (255, 220, 0), 2, cv2.LINE_AA)
            x, y = polygon[0]
            cv2.putText(output, f'Opening {index}: {value["confidence"]:.2f}',
                        (int(x), max(20, int(y))), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 220, 0), 1)
        status = self.status if valid else 'STALE'
        label = f'Openings: {len(values) if status == "ACTIVE" else "?"} | {status} | EXPERIMENTAL - NO FLIGHT CLEARANCE'
        cv2.rectangle(output, (0, h - 30), (w, h), (20, 20, 20), -1)
        cv2.putText(output, label, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 220, 0), 1)
        return output

    def record(self, snapshot, detection, tracking, verification, reidentification, managed, max_age, mission_id, health=None):
        valid = (self.current(snapshot, max_age) and
                 (health is None or (health.packet is not None and health.connections == snapshot.connections)))
        data = {
            'schema_version': 'full-perception-1.0', 'frame_id': snapshot.packet.sequence,
            'received_at_monotonic': snapshot.packet.received_at, 'connection': snapshot.connections,
            'mission_id': mission_id, 'status': ('ok' if self.status == 'ACTIVE' else 'degraded') if valid else 'stale',
            'opening_status': self.status if valid else 'STALE',
            'opening_candidates': self.values if valid else [], 'opening_ms': self.elapsed_ms,
            'humans': asdict(detection) if valid and detection else None,
            'tracking': asdict(tracking) if valid and tracking else None,
            'verification': asdict(verification) if valid and verification else None,
            'reid': asdict(reidentification) if valid and reidentification else None,
            'survivors': asdict(managed) if valid and managed else None,
            'opening_model_status': 'experimental_release_failed',
            'identity_scope': 'mission_appearance_estimate', 'flight_ready': False,
        }
        self.output.write(json.dumps(data, allow_nan=False) + '\n')
        self.output.flush()
