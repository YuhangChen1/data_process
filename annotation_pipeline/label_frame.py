"""Label one image with object instances and semantics: GPT names what is in the frame, SAM3 segments it.

Same method as the household global pipeline, for a single frame:
  1. GPT semantic audit (global_pipeline.discovery.AUDIT + clean_concepts): the object types / surfaces visible in the
     image, each with 1-3 short English SAM3 prompts. Skipped when --concepts is given.
  2. SAM3 text grounding through the resident service (annotation_pipeline.sam_service /segment_concepts): one image
     embedding, ordered fallback prompts per type, same-type NMS (IoU >= 0.80), exclusive pixel ownership by score.
  3. Export: labels.json (objects with id, label, thing/stuff, score, stability, bbox, area, prompt), instances.png
     (uint16 instance id, 0 = unlabeled), semantic.png (uint8 class id, 0 = unlabeled), overlay.jpg, sam3.npz/json.

    cd /mnt/cfs/cyh/cvpr_ws
    python -m annotation_pipeline.label_frame path/to/frame.png                        # GPT names + SAM3 masks
    python -m annotation_pipeline.label_frame frame.png --concepts "sofa,table,floor"  # no GPT: given types only
    python -m annotation_pipeline.label_frame frame.png --add "door,exit sign"         # GPT types + extra ones
    python -m annotation_pipeline.label_frame frame.png --out outputs/label_frame/my_frame --port 18766

Needs a running SAM3 service (ports listed in services/global_settings.json; start one with
`SAM3_GPU=<id> bash services/start_sam3.sh`). GPT uses the gateway key in /mnt/cfs/cyh/auth.json (never written out).
"""
import argparse
import colorsys
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PORTS = [18763, 18766, 18767, 18768, 18769, 18770, 18771, 18764]


def pick_service(port=None):
    s = requests.Session()
    s.trust_env = False
    for p in ([port] if port else PORTS):
        try:
            h = s.get(f'http://127.0.0.1:{p}/health', timeout=2).json()
            if h.get('ready'):
                return f'http://127.0.0.1:{p}', h
        except requests.RequestException:
            continue
    sys.exit('No ready SAM3 service; start one: SAM3_GPU=<gpu> bash services/start_sam3.sh')


def gpt_concepts(image, out, model):
    from annotation_pipeline.vlm import VLM
    from global_pipeline.discovery import AUDIT, clean_concepts
    answer = VLM(model=model).call(AUDIT, [image], out / 'gpt_inventory.json', max_tokens=4500)
    return clean_concepts(answer)


def given_concepts(text):
    from global_pipeline.discovery import SURFACES
    names = [c.strip().lower() for c in text.split(',') if c.strip()]
    return [dict(semantic_label=n, category_type='stuff' if n in SURFACES else 'thing', sam3_prompts=[n],
                 visible=True, source='user') for n in dict.fromkeys(names)]


def color(i):
    r, g, b = colorsys.hsv_to_rgb((i * 0.61803398875) % 1.0, 0.75, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def font(size):
    for f in ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', '/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf'):
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def export(image, out, seg):
    npz = np.load(out / 'sam3.npz')
    inst = npz['local_instances'].astype(np.uint16)
    classes = {}
    for d in seg['detections']:
        classes.setdefault(d['semantic_label'], len(classes) + 1)
    sem = np.zeros(inst.shape, np.uint8)
    objects = []
    for d in sorted(seg['detections'], key=lambda d: d['exclusive_local_id']):
        i = d['exclusive_local_id']
        m = inst == i
        ys, xs = np.nonzero(m)
        sem[m] = classes[d['semantic_label']]
        objects.append(dict(id=int(i), label=d['semantic_label'], category_type=d['category_type'],
                            semantic_id=classes[d['semantic_label']], score=round(d['score'], 4),
                            stability_score=round(d['stability_score'], 4), prompt=d['prompt'],
                            bbox_xyxy=[int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                            area_px=int(m.sum()), centroid_xy=[round(float(xs.mean()), 1), round(float(ys.mean()), 1)]))
    Image.fromarray(inst).save(out / 'instances.png')
    Image.fromarray(sem).save(out / 'semantic.png')
    rgb = np.asarray(Image.open(image).convert('RGB'), np.float32)
    over = rgb.copy()
    for o in objects:
        m = inst == o['id']
        over[m] = 0.45 * rgb[m] + 0.55 * np.array(color(o['id']), np.float32)
    vis = Image.fromarray(over.clip(0, 255).astype(np.uint8))
    dr = ImageDraw.Draw(vis)
    f = font(max(12, vis.width // 70))
    for o in sorted(objects, key=lambda o: -o['area_px']):
        x, y = o['centroid_xy']
        text = f"{o['id']} {o['label']}"
        box = dr.textbbox((x, y), text, font=f, anchor='mm')
        dr.rectangle(box, fill=(0, 0, 0))
        dr.text((x, y), text, fill=color(o['id']), font=f, anchor='mm')
    vis.save(out / 'overlay.jpg', quality=92)
    labels = dict(image=str(image), size_wh=[int(inst.shape[1]), int(inst.shape[0])],
                  semantic_classes={k: v for k, v in classes.items()}, objects=objects,
                  concepts_requested=seg['concepts'], unlabeled_fraction=round(float((inst == 0).mean()), 4),
                  sam3=dict(checkpoint_sha256=seg['checkpoint_sha256'], confidence=seg['confidence'],
                            nms_iou=seg['nms_iou'], min_area=seg['min_area'], seconds=round(seg['seconds'], 2),
                            errors=seg['errors'], prompt_attempts=seg['prompt_attempts']),
                  files=dict(instances='instances.png (uint16, 0 = unlabeled)', semantic='semantic.png (uint8, 0 = unlabeled)',
                             overlay='overlay.jpg', raw_masks='sam3.npz:raw_masks', sam3_record='sam3.json',
                             gpt_record='gpt_inventory.json (if GPT was used)'))
    (out / 'labels.json').write_text(json.dumps(labels, indent=1, ensure_ascii=False))
    return labels


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('image')
    ap.add_argument('--out', help='output dir (default outputs/label_frame/<image stem>)')
    ap.add_argument('--concepts', help='comma-separated object types; skips GPT')
    ap.add_argument('--add', help='comma-separated object types added to the GPT inventory')
    ap.add_argument('--port', type=int, help='SAM3 service port (default: first ready one)')
    ap.add_argument('--model', default='gpt-6-sol', help='GPT model for the semantic audit')
    ap.add_argument('--confidence', type=float, default=0.40)
    ap.add_argument('--nms-iou', type=float, default=0.80)
    ap.add_argument('--min-area', type=int, default=80)
    a = ap.parse_args()
    src = Path(a.image).resolve()
    out = Path(a.out).resolve() if a.out else ROOT / 'outputs/label_frame' / src.stem
    out.mkdir(parents=True, exist_ok=True)
    image = out / ('input' + src.suffix.lower())   # the SAM3 service only reads files inside the workspace
    if src != image:
        shutil.copyfile(src, image)
    concepts = given_concepts(a.concepts) if a.concepts else gpt_concepts(image, out, a.model)
    if a.add:
        have = {c['semantic_label'] for c in concepts}
        concepts += [c for c in given_concepts(a.add) if c['semantic_label'] not in have]
    if not concepts:
        sys.exit('No object types found (GPT returned none usable); pass --concepts to name them yourself')
    url, health = pick_service(a.port)
    s = requests.Session()
    s.trust_env = False
    r = s.post(url + '/segment_concepts', json=dict(image_path=str(image), output_path=str(out / 'sam3.npz'),
                                                     concepts=concepts, confidence=a.confidence, nms_iou=a.nms_iou,
                                                     min_area=a.min_area), timeout=600)
    r.raise_for_status()
    labels = export(image, out, r.json())
    print(f"{len(labels['objects'])} objects, {len(labels['semantic_classes'])} classes, "
          f"{labels['unlabeled_fraction']:.0%} pixels unlabeled ({url}, {labels['sam3']['seconds']} s) -> {out}")
    for o in labels['objects']:
        print(f"  {o['id']:3d} {o['label']:28s} {o['category_type']:5s} score {o['score']:.2f} area {o['area_px']}")


if __name__ == '__main__':
    main()
