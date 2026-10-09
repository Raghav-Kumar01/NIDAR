"""Shared-frame opening and person perception; no tracking or flight commands."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import math
import subprocess
import sys
import time
import numpy as np

UPSTREAM_COMMIT = '0d112b4b1bb28e8e1511a9e994a3de78e3182356'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class FrameResult:
    schema_version: str
    frame_id: int
    received_at_monotonic: float
    width: int
    height: int
    status: str
    branches: dict
    opening_candidates: list
    person_candidates: list
    timing_ms: dict
    opening_model_status: str = 'experimental_release_failed'
    identity_scope: str = 'frame_only_no_unique_survivor_count'
    flight_ready: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class PerceptionPipeline:
    """Injectable adapters provide predict(frame) -> lists of validated observations."""
    def __init__(self, opening, human, max_age_ms=1000.0, clock=time.monotonic):
        if not math.isfinite(max_age_ms) or max_age_ms <= 0:
            raise ValueError('max_age_ms must be finite and positive')
        self.opening, self.human = opening, human
        self.max_age_ms, self.clock = max_age_ms, clock

    def process(self, frame, frame_id: int, received_at: float) -> FrameResult:
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
            raise ValueError('Expected nonempty uint8 BGR frame')
        if type(frame_id) is not int or frame_id < 0 or not math.isfinite(received_at):
            raise ValueError('Invalid frame identity/timestamp')
        start = self.clock()
        if received_at > start:
            raise ValueError('Timestamp must come from the same monotonic clock and cannot be in the future')
        h,w = frame.shape[:2]
        states={}; timings={}; outputs={}
        if (start-received_at)*1000 > self.max_age_ms:
            return FrameResult('1.0',frame_id,received_at,w,h,'stale',
                {'openings':'skipped_stale','humans':'skipped_stale'},[],[],{'age_ms':(start-received_at)*1000,'total_ms':0.0})
        for name,adapter in [('openings',self.opening),('humans',self.human)]:
            before=self.clock()
            try:
                # Each adapter owns a copy; neither can mutate the camera or the other branch's input.
                values=adapter.predict(frame.copy())
                validate_observations(values,name,w,h)
                outputs[name]=values;states[name]='ok'
            except Exception as exc:
                outputs[name]=[];states[name]='error:'+type(exc).__name__
            timings[name]=(self.clock()-before)*1000
        end=self.clock();timings.update(total_ms=(end-start)*1000,age_ms=(end-received_at)*1000)
        status='ok' if all(v=='ok' for v in states.values()) else 'degraded'
        if timings['age_ms'] > self.max_age_ms:
            status='stale';outputs={'openings':[],'humans':[]}
        return FrameResult('1.0',frame_id,received_at,w,h,status,states,outputs['openings'],outputs['humans'],timings)


def validate_observations(values, branch, width, height):
    if not isinstance(values,list):raise ValueError('Adapter must return a list')
    for value in values:
        if not 0 <= value['confidence'] <= 1:raise ValueError('Invalid confidence')
        if branch=='humans':
            x1,y1,x2,y2=value['bbox_xyxy_pixels']
            if not all(math.isfinite(v) for v in (x1,y1,x2,y2)) or not (0<=x1<x2<=width and 0<=y1<y2<=height):
                raise ValueError('Invalid person box')
        else:
            p=np.asarray(value['polygon_normalized'],dtype=float)
            if p.ndim!=2 or p.shape[1]!=2 or len(p)<3 or not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():
                raise ValueError('Invalid opening polygon')
            area=abs(float(np.dot(p[:,0],np.roll(p[:,1],-1))-np.dot(p[:,1],np.roll(p[:,0],-1))))/2
            if area<=0:raise ValueError('Degenerate opening polygon')


class OpeningAdapter:
    def __init__(self, model, device, confidence):
        from ultralytics import YOLO
        self.model=YOLO(str(model));self.device=device;self.confidence=confidence
        if self.model.task!='segment' or self.model.names!={0:'candidate_opening'}:
            raise ValueError('Opening checkpoint must be class-0 candidate_opening segmentation')

    def predict(self, frame):
        result=self.model.predict(frame,imgsz=640,conf=self.confidence,device=self.device,verbose=False)[0]
        if not len(result.boxes):return []
        if result.masks is None or len(result.masks)!=len(result.boxes):raise ValueError('Missing masks')
        return [{'confidence':float(c),'polygon_normalized':p.tolist()} for c,p in zip(result.boxes.conf.cpu().tolist(),result.masks.xyn)]


class HumanAdapter:
    def __init__(self, detector):self.detector=detector
    def predict(self, frame):
        result=self.detector.predict(frame)
        return [{'confidence':float(d.confidence),'bbox_xyxy_pixels':list(d.xyxy)} for d in result.people]


def load_pipeline(config_path: Path):
    """Use pinned friend code directly; never copy or rewrite its detector implementation."""
    path=config_path.resolve();cfg=json.loads(path.read_text(encoding='utf-8'))
    def resolve(value):return (path.parent/Path(value)).resolve()
    repo=resolve(cfg['human_repository']);opening=resolve(cfg['opening_model'])
    if not opening.is_file():raise ValueError('Opening checkpoint missing; supply a trusted local checkpoint')
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if commit!=UPSTREAM_COMMIT:raise ValueError('Human repository revision differs from tested integration pin')
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('Human repository tracked files differ from pinned source')
    ml=repo/'Raj Gupta/ML'
    manifest=json.loads((ml/'models/manifest.json').read_text())
    models={r['path']:r for r in manifest['models']}
    required=['models/yolo11n.pt','models/nidar-person-posture-specialist-yolo11n.pt']
    for name in required:
        if digest(ml/name)!=models[name]['sha256']:raise ValueError('Human checkpoint hash mismatch')
    device=cfg.get('device','cpu')
    if device not in ('cpu','0'):raise ValueError('device must be cpu or 0')
    for key in ('opening_confidence','human_confidence'):
        if not math.isfinite(cfg[key]) or not 0<cfg[key]<=1:raise ValueError('Confidence must be in (0,1]')
    sys.path.insert(0,str(ml))
    from nidar_survivor_demo.detector import DetectorConfig
    from nidar_survivor_demo.hybrid_detector import HybridPersonDetector
    import nidar_survivor_demo.detector as module
    if Path(module.__file__).resolve().parent!=ml/'nidar_survivor_demo':raise ValueError('Unexpected loaded human package')
    human=HybridPersonDetector(DetectorConfig(model=ml/required[0],confidence=cfg['human_confidence'],device=device),
        specialist_model=ml/required[1],fusion_config=ml/'nidar_survivor_demo/settings/hybrid_detector.json')
    opening_adapter=OpeningAdapter(opening,device,cfg['opening_confidence'])
    pipeline=PerceptionPipeline(opening_adapter,HumanAdapter(human),cfg.get('max_age_ms',1000))
    provenance={'human_repo_url':'https://github.com/RajGoyal01/NIDAR.git','human_commit':commit,
        'human_model_hashes':{name:models[name]['sha256'] for name in required},'opening_sha256':digest(opening),
        'config':cfg,'human_mode':'upstream_hybrid_general_plus_posture','identity_scope':'frame_only',
        'opening_status':'experimental_release_failed','flight_ready':False}
    return pipeline,provenance
