"""Read-only checkpoint/data inspection with isolated release-test outputs."""
from pathlib import Path
import csv, hashlib, json, time, platform, html
from collections import Counter
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/airmouse/v3_release_review'
MODEL=ROOT/'runs/airmouse/opening_seg_v3_baseline/weights/best.pt'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def phash(p):
    gray=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
    d=cv2.dct(cv2.resize(gray,(32,32)).astype(np.float32))[:8,:8]
    bits=d.flatten();return bits>np.median(bits[1:])


def main():
    OUT.mkdir(exist_ok=False,parents=True)
    (OUT/'preview').mkdir()
    records=[];base=ROOT/'datasets/annotation_full_v1'
    for p in sorted((base/'reviews').glob('*.json')):
        a=json.loads(p.read_text())
        if a['method']=='explicit_browser_review' and a['status'] in ('positive','negative'):
            r=json.loads((base/'records'/p.name).read_text())
            records.append(dict(id=a['id'],path=r['image'],source=r['source'],suite='human_decisions',
                expected=a['status'],historical_review=str(p),expected_sha=a['image_sha256']))
    for r in csv.DictReader((ROOT/'datasets/v3_pilot/frames_manifest.csv').open()):
        p=ROOT/'datasets/v3_pilot/frames'/r['filename']
        records.append(dict(id=p.stem,path=str(p),source=r['source_video'],suite='video_frames',expected='unlabeled',expected_sha=sha(p)))
    assert Counter(r['suite'] for r in records)=={'human_decisions':50,'video_frames':87}
    training=list(csv.DictReader((ROOT/'datasets/v3_training_baseline/split_manifest.csv').open()))
    eligible=[r for r in training if r['split']!='quarantine']
    known=[]
    for r in eligible:
        p=ROOT/'datasets/v3_training_baseline/images'/r['split']/r['filename']
        assert sha(p)==r['image_sha256']
        label=ROOT/'datasets/v3_training_baseline/labels'/r['split']/(p.stem+'.txt')
        assert sha(label)==r['label_sha256']
        known.append((r,phash(p)))
    cross=[]
    for i,(a,ha) in enumerate(known):
        for b,hb in known[i+1:]:
            if a['split']!=b['split'] and np.count_nonzero(ha!=hb)<=6:
                cross.append(dict(a=a['filename'],b=b['filename'],distance=int(np.count_nonzero(ha!=hb))))
    for r in records:
        p=Path(r['path']);r['sha256']=sha(p);assert r['sha256']==r['expected_sha']
        assert r['sha256'] not in {x['image_sha256'] for x in eligible},'Exact overlap'
        h=phash(p); nearest=min(known,key=lambda x:np.count_nonzero(h!=x[1]))
        r['nearest_v3_phash_distance']=int(np.count_nonzero(h!=nearest[1]));r['nearest_v3_image']=nearest[0]['filename']
    completion=json.loads((MODEL.parent.parent/'completion_report.json').read_text())
    assert sha(MODEL)==completion['best_sha256']
    model=YOLO(str(MODEL));assert model.task=='segment' and model.names=={0:'candidate_opening'}
    results=[];thumbs=[]
    for i,r in enumerate(records):
        image=cv2.imread(r['path']);assert image is not None
        pred=model.predict(image,imgsz=640,conf=0.25,iou=0.7,device=0,verbose=False)[0]
        boxes=pred.boxes;num=len(boxes);polys=[]
        if num:
            assert pred.masks is not None and len(pred.masks)==num
            for poly in pred.masks.xyn:
                assert np.isfinite(poly).all() and (poly>=0).all() and (poly<=1).all()
                polys.append(poly.tolist())
        row=dict(r,index=i,count=num,confidences=boxes.conf.cpu().tolist(),polygons=polys)
        row['outcome']='unscored' if r['expected']=='unlabeled' else ('TP' if num else 'FN') if r['expected']=='positive' else ('FP' if num else 'TN')
        results.append(row)
        orig=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));orig.thumbnail((400,300))
        plot=Image.fromarray(cv2.cvtColor(pred.plot(),cv2.COLOR_BGR2RGB));plot.thumbnail((400,300))
        canvas=Image.new('RGB',(800,340),'#18202b');canvas.paste(orig,(0,35));canvas.paste(plot,(400,35))
        d=ImageDraw.Draw(canvas);d.text((8,5),f'{i:03} | {r["suite"]} | expected {r["expected"]} | {num} predictions | {row["outcome"]}',fill='white')
        canvas.save(OUT/'preview'/f'{i:03}.jpg',quality=88)
        small=canvas.resize((600,255));thumbs.append(small)
        if i%20==0: print('predicted',i,'of',len(records),flush=True)
    for offset in range(0,len(thumbs),8):
        sheet=Image.new('RGB',(1200,1020),'#18202b')
        for j,im in enumerate(thumbs[offset:offset+8]):sheet.paste(im,((j%2)*600,(j//2)*255))
        sheet.save(OUT/'preview'/f'sheet_{offset//8+1:02}.jpg',quality=88)
    # Fixed diagnostic perturbations of the already exposed synthetic test set; no retraining.
    stress=[]
    for p in sorted((ROOT/'datasets/v3_training_baseline/images/test').glob('*.png')):
        im=cv2.imread(str(p))
        variants={'original':im,'dark_45pct':(im*0.45).astype('uint8'),'gaussian_blur_7':cv2.GaussianBlur(im,(7,7),2),
            'jpeg_quality_35':cv2.imdecode(cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,35])[1],1)}
        for kind,frame in variants.items():
            pred=model.predict(frame,imgsz=640,conf=0.25,iou=0.7,device=0,verbose=False)[0]
            stress.append(dict(filename=p.name,variant=kind,count=len(pred.boxes),confidences=pred.boxes.conf.cpu().tolist()))
    # Warmed batch-one latency includes predict preprocessing, model and postprocessing; excludes disk/decode/plot.
    frames=[cv2.imread(r['path']) for r in records[:10]]
    for n in range(10):model.predict(frames[n%10],imgsz=640,conf=0.25,iou=0.7,device=0,verbose=False)
    times=[];torch.cuda.reset_peak_memory_stats()
    for n in range(100):
        torch.cuda.synchronize();start=time.perf_counter()
        model.predict(frames[n%10],imgsz=640,conf=0.25,iou=0.7,device=0,verbose=False)
        torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
    c=Counter(r['outcome'] for r in results if r['suite']=='human_decisions')
    stats=dict(checkpoint_sha256=sha(MODEL),confidence=0.25,nms_iou=0.7,imgsz=640,
        human_image_level_counts=dict(c),positive_hit_rate=c['TP']/(c['TP']+c['FN']),
        negative_false_alarm_rate=c['FP']/(c['FP']+c['TN']),
        frames_tested=87,frames_with_predictions=sum(r['count']>0 for r in results if r['suite']=='video_frames'),
        inference_contract='finite in-bounds mask vertices, masks present for detections, class mapping correct',
        cross_split_phash_candidates=cross,external_near_duplicates=[{k:r[k] for k in ('id','nearest_v3_image','nearest_v3_phash_distance')} for r in records if r['nearest_v3_phash_distance']<=6],
        timing=dict(warmup=10,iterations=100,p50_ms=float(np.percentile(times,50)),p95_ms=float(np.percentile(times,95)),mean_ms=float(np.mean(times)),min_ms=min(times),max_ms=max(times),peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,scope='warm batch-one predict incl pre/post; excludes camera IO, disk decode and rendering'),
        gpu=torch.cuda.get_device_name(0),python=platform.python_version(),limitations='Historical human image-level decisions, not re-adjudicated segmentation GT; correlated video frames unlabeled; pHash is a heuristic, not proof of scene independence.')
    for r in records: assert sha(Path(r['path']))==r['sha256']
    (OUT/'predictions.json').write_text(json.dumps(results,indent=2))
    (OUT/'stress.json').write_text(json.dumps(stress,indent=2))
    (OUT/'summary.json').write_text(json.dumps(stats,indent=2))
    with (OUT/'results.csv').open('w',newline='') as f:
        fields=['index','id','path','suite','source','expected','count','outcome','nearest_v3_phash_distance']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(results)
    cards=''.join(f'<article><h3>{r["index"]:03} {html.escape(r["outcome"])} | {html.escape(r["source"])}</h3><p>{html.escape(r["path"])}</p><img loading="lazy" src="preview/{r["index"]:03}.jpg"></article>' for r in results)
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>V3 release review</title><style>body{background:#18202b;color:white;font:16px sans-serif}img{max-width:100%}article{margin:30px 0}p{overflow-wrap:anywhere}</style><h1>V3 release review</h1><p>Left: source. Right: prediction at confidence 0.25. Video frames are unlabeled; historical image decisions are not mask ground truth.</p>'+cards,encoding='utf-8')
    print(json.dumps(stats,indent=2),flush=True)


if __name__=='__main__':main()
