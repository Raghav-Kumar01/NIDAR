"""One combined preview/JSONL stream from opening segmentation and upstream human detection."""
from pathlib import Path
import argparse,json,time
import cv2
import numpy as np
from perception import load_pipeline


def draw(frame,result):
    image=frame.copy();h,w=image.shape[:2]
    for item in result.opening_candidates:
        points=np.round(np.asarray(item['polygon_normalized'])*[w,h]).astype('int32')
        cv2.polylines(image,[points],True,(255,220,0),2)
    for item in result.person_candidates:
        x1,y1,x2,y2=map(round,item['bbox_xyxy_pixels'])
        cv2.rectangle(image,(x1,y1),(x2,y2),(30,230,60),2)
        cv2.putText(image,f'person {item["confidence"]:.2f}',(x1,max(20,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,(30,230,60),1)
    header=np.zeros((65,w,3),dtype='uint8')
    cv2.putText(header,f'Frame {result.frame_id} | {result.status} | openings {len(result.opening_candidates)} | people {len(result.person_candidates)}',
        (8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
    cv2.putText(header,'EXPERIMENTAL opening model | person candidates, no unique count',(8,48),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,190,255),1)
    return np.vstack([header,image])


def run(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path(__file__).parent/'config/perception.example.json')
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--source',type=Path,help='Local image, image folder or recorded video')
    source.add_argument('--camera-index',type=int,help='USB webcam; uses upstream latest-frame capture')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-frames',type=int,default=100)
    p.add_argument('--seconds',type=float,default=30,help='Camera session limit including waiting')
    p.add_argument('--show',action='store_true')
    args=p.parse_args(argv)
    if args.output.exists():p.error('Output already exists; choose a new directory')
    if args.max_frames<1 or args.seconds<=0:p.error('Limits must be positive')
    if args.source is not None and not args.source.exists():p.error('Local source does not exist')
    if args.camera_index is not None and args.camera_index<0:p.error('Invalid camera index')
    pipeline,provenance=load_pipeline(args.config)
    args.output.mkdir(parents=True)
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2))
    frames=[];source_video=None;stream=None;count=0;timings=[];statuses={};people_frames=opening_frames=0
    try:
        if args.camera_index is not None:
            from nidar_survivor_demo.config import CameraConfig
            from nidar_survivor_demo.camera.phone_stream import PhoneStream
            stream=PhoneStream(CameraConfig(source=args.camera_index));stream.start()
        elif args.source.is_dir():frames=sorted(f for f in args.source.iterdir() if f.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.webp'})
        elif args.source.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.webp'}:frames=[args.source]
        else:
            source_video=cv2.VideoCapture(str(args.source))
            if not source_video.isOpened():raise ValueError('Could not open recorded video')
        start=time.monotonic();last_sequence=-1
        with (args.output/'frames.jsonl').open('w',encoding='utf-8') as log:
            while count<args.max_frames:
                if stream:
                    if time.monotonic()-start>args.seconds:break
                    snapshot=stream.snapshot();packet=snapshot.packet
                    if packet is None or packet.sequence==last_sequence:
                        time.sleep(.005);continue
                    last_sequence=packet.sequence;frame=packet.frame;received=packet.received_at;frame_id=packet.sequence
                elif source_video:
                    ok,frame=source_video.read()
                    if not ok:break
                    received=time.monotonic();frame_id=count
                else:
                    if count>=len(frames):break
                    frame=cv2.imread(str(frames[count]));received=time.monotonic();frame_id=count
                    if frame is None:raise ValueError('Unreadable image')
                result=pipeline.process(frame,frame_id,received)
                record=result.to_dict()
                record['source_kind']='live_camera' if stream else 'recorded_video' if source_video else 'image'
                record['source_frame']=str(frames[count].name) if frames else frame_id
                log.write(json.dumps(record)+'\n');log.flush()
                preview=draw(frame,result)
                if not cv2.imwrite(str(args.output/f'frame_{count:05d}.jpg'),preview):raise OSError('Preview write failed')
                statuses[result.status]=statuses.get(result.status,0)+1
                timings.append(result.timing_ms['total_ms']);people_frames+=bool(result.person_candidates);opening_frames+=bool(result.opening_candidates)
                count+=1
                if args.show:
                    cv2.imshow('NIDAR combined experimental perception',preview)
                    if cv2.waitKey(1)&255==27:break
    finally:
        if source_video:source_video.release()
        if stream:stream.stop()
        if args.show:cv2.destroyAllWindows()
    summary={'frames':count,'statuses':statuses,'frames_with_people':people_frames,'frames_with_openings':opening_frames,
        'inference_p50_ms':float(np.median(timings)) if timings else None,'inference_p95_ms':float(np.percentile(timings,95)) if timings else None,
        'timing_scope':'combined adapters and CPU mask transfer; excludes decode, preview rendering and writes; includes initial cold opening inference',
        'semantic_accuracy':'not measured','flight_ready':False}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
    return 0 if count and all(k=='ok' for k in statuses) else 2


if __name__=='__main__':raise SystemExit(run())
