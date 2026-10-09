"""Local polygon editor for a fixed, curated manual reference set."""
from __future__ import annotations
import argparse
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import secrets
import time
from urllib.parse import urlparse

from full_dataset import review_payload, yolo_text

HTML = r'''<!doctype html><html><meta charset="utf-8"><title>Manual opening reference labels</title>
<style>body{background:#131920;color:#edf4fa;font:16px system-ui;margin:24px}button{padding:10px;margin:4px;background:#223244;color:white;border:1px solid #718398;border-radius:5px;cursor:pointer}canvas{display:block;max-width:95vw;max-height:78vh;margin-top:16px}#details{overflow-wrap:anywhere;color:#afc5d7}</style>
<h2>Manual reference labels — visible opening region</h2>
<p>Draw around the visible free-space opening. Exclude door panels, frames, glass and visible foreground obstructions. Click vertices, then Finish polygon. You may draw multiple openings. Use Ambiguous if unsure, or Negative if no opening is visible. These labels are stored separately from the existing review set.</p>
<div><button id="undo">Undo vertex</button><button id="finish">Finish polygon</button><button id="clear">Clear polygons</button>
<button data-s="positive">Save positive</button><button data-s="negative">Save negative</button><button data-s="ambiguous">Save ambiguous</button><button data-s="reject">Reject image</button><button id="skip">Skip</button><span id="counter"></span></div>
<p id="details"></p><div id="msg"></div><canvas id="c"></canvas><script>
const token='__TOKEN__',c=document.querySelector('#c'),ctx=c.getContext('2d');let items=[],i=0,r,img=new Image(),polys=[],line=[],busy=false;
function msg(t){document.querySelector('#msg').textContent=t}function draw(){ctx.clearRect(0,0,c.width,c.height);ctx.drawImage(img,0,0,c.width,c.height);for(const p of [...polys,line]){if(!p.length)continue;ctx.beginPath();p.forEach(([x,y],j)=>j?ctx.lineTo(x*c.width,y*c.height):ctx.moveTo(x*c.width,y*c.height));if(p!==line)ctx.closePath();ctx.strokeStyle=p===line?'#ffe083':'#00e5b4';ctx.lineWidth=3;ctx.stroke();if(p!==line){ctx.fillStyle='#00dcaa30';ctx.fill()}}}
function load(){if(!items.length)return; r=items[i]; fetch('/image/'+r.id).then(x=>x.blob()).then(b=>{img=new Image();img.onload=()=>{let s=Math.min(1200/img.width,780/img.height,1);c.width=Math.round(img.width*s);c.height=Math.round(img.height*s);polys=[];line=[];draw();document.querySelector('#counter').textContent=` ${i+1}/${items.length}`;document.querySelector('#details').textContent=`${r.source} | ${r.image}`;msg('Annotate from the image, without looking at previous model polygons.')};img.src=URL.createObjectURL(b)})}
c.onclick=e=>{if(busy)return;let b=c.getBoundingClientRect();line.push([(e.clientX-b.left)/b.width,(e.clientY-b.top)/b.height]);draw()};
document.querySelector('#finish').onclick=()=>{if(line.length<3){msg('A polygon needs at least 3 points.');return}polys.push(line);line=[];draw()};document.querySelector('#undo').onclick=()=>{line.pop();draw()};document.querySelector('#clear').onclick=()=>{polys=[];line=[];draw()};document.querySelector('#skip').onclick=()=>{if(i<items.length-1)i++;load()};
document.querySelectorAll('[data-s]').forEach(b=>b.onclick=async()=>{if(busy)return;let s=b.dataset.s;if(line.length){msg('Finish or clear the active polygon first.');return}if(s==='positive'&&!polys.length){msg('Positive needs at least one polygon.');return}if(s!=='positive'&&polys.length){msg('Clear polygons for this decision.');return}busy=true;try{let x=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':token},body:JSON.stringify({id:r.id,status:s,polygons:polys})}),v=await x.json();if(!x.ok)throw Error(v.error);items=items.filter(a=>a.id!==r.id);i=0;msg('Saved.');if(items.length)load();else msg('All reference images have a saved decision.')}catch(e){msg(e.message)}finally{busy=false}});
(async()=>{items=await(await fetch('/items')).json();if(!items.length){msg('All reference images are already reviewed.');return}load()})();</script></html>'''


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def serve(package: Path, selection: Path, port: int) -> None:
    records = json.loads(selection.read_text(encoding='utf-8'))
    by_id = {r['id']: r for r in records}
    output = package/'manual_reference_set_v1'
    (output/'reviews').mkdir(parents=True, exist_ok=True)
    (output/'labels').mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, data, mime='application/json', code=200):
            self.send_response(code)
            self.send_header('Content-Type', mime)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == '/':
                self.send(HTML.replace('__TOKEN__', token).encode(), 'text/html; charset=utf-8')
            elif path == '/items':
                pending = [r for r in records if not (output/'reviews'/f"{r['id']}.json").exists()]
                self.send(json.dumps(pending).encode())
            elif path.startswith('/image/') and path.split('/')[-1] in by_id:
                r = by_id[path.split('/')[-1]]
                self.send(Path(r['image']).read_bytes(), 'image/png' if Path(r['image']).suffix.lower()=='.png' else 'image/jpeg')
            else:
                self.send(b'{"error":"not found"}', code=404)

        def do_POST(self):
            if self.path != '/save' or self.headers.get('X-Review-Token') != token:
                self.send(b'{"error":"unauthorized"}', code=403)
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size < 1_000_000:
                    raise ValueError('Invalid payload size')
                data = json.loads(self.rfile.read(size))
                r = by_id[data['id']]
                review_payload(data['status'], data['polygons'])
                image = Path(r['image'])
                if sha(image) != r['sha256']:
                    raise ValueError('Image hash mismatch; no label saved')
                target = output/'reviews'/f"{r['id']}.json"
                if target.exists():
                    raise ValueError('Reference decision already saved; overwrite disabled')
                review = {'id': r['id'], 'status': data['status'], 'polygons': data['polygons'],
                          'method': 'explicit_human_reference_review', 'reviewer': 'user',
                          'image_sha256': r['sha256'], 'saved_utc_epoch': time.time(),
                          'source': r['source'], 'image': r['image'], 'split': 'unassigned'}
                with target.open('x', encoding='utf-8') as f:
                    json.dump(review, f, indent=2)
                if data['status'] in {'positive','negative'}:
                    with (output/'labels'/f"{r['id']}.txt").open('x', encoding='utf-8') as f:
                        f.write(yolo_text(data['polygons']))
                self.send(b'{"ok":true}')
            except (ValueError, KeyError, TypeError) as exc:
                self.send(json.dumps({'error': str(exc)}).encode(), code=400)

    print(f'Manual reference annotation page: http://127.0.0.1:{port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=Path('datasets/annotation_full_v1'))
    parser.add_argument('--selection', type=Path, default=Path('datasets/annotation_full_v1/manual_reference_set_v1/selection.json'))
    parser.add_argument('--port', type=int, default=8767)
    args = parser.parse_args()
    serve(args.package, args.selection, args.port)
