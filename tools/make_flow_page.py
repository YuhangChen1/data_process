"""Render docs/pipeline_flow.html: the pipeline flowchart + every operator's interface, from the live registries.

  python tools/make_flow_page.py          # reads ops3d.OPS, operators.list_operators(), the pilot scores
"""
from pathlib import Path
import collections, html, json, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCENE = 'KS-IN-OE-S-F1_01_202609121414'

# ------------------------------------------------------------------ overview diagram (hand-placed on a grid)
NODES = {  # id: (x, y, w, h, title, subtitle, kind)   kind: ext | op | opt | cond
    'upstream.scan': (16, 60, 250, 58, '海塞扫描', 'gaussian.ply · mesh · data.json', 'ext'),
    'upstream.manyou_probe': (16, 150, 250, 58, '董婷 manyou · 探针 + 低清评分', 'probes.json · metrics.json', 'ext'),
    'upstream.manyou_walkthrough': (16, 240, 250, 58, '董婷 manyou · 主漫游', 'poses.json · frames_hd · mp4', 'ext'),
    'upstream.capture_aware': (16, 330, 250, 58, '采集感知漫游 videos/', 'data.json → ~2300 位姿（未使用）', 'ext'),
    'scan.source': (336, 60, 260, 58, '登记扫描', 'scan.source', 'op'),
    'geom.analyze': (336, 150, 260, 58, '调平 · 楼层 · 可通行图', 'geom.analyze', 'op'),
    'geom.floor_check': (336, 240, 260, 58, '楼层实测', 'geom.floor_check', 'op'),
    'traj.align_walkthrough': (336, 420, 124, 58, '对齐自带漫游', 'traj.align', 'op'),
    'traj.astar_walkthrough': (472, 420, 124, 58, 'A* 漫游', 'traj.astar', 'op'),
    'geom.render_views': (336, 590, 260, 58, '俯视图 · 全景', 'geom.render_views', 'op'),
    'anno.frames': (666, 60, 260, 52, '标注帧索引', 'anno.frames', 'op'),
    'anno.inventory': (666, 140, 260, 52, 'GPT 逐帧类别清单', 'anno.inventory', 'op'),
    'anno.sam3_segment': (666, 220, 260, 52, 'SAM3 分割', 'anno.sam3_segment', 'op'),
    'anno.superpoints': (666, 300, 260, 52, '超点 + z-buffer 投影', 'anno.superpoints', 'op'),
    'anno.instances_3d': (666, 380, 260, 52, '3D 实例融合', 'anno.instances_3d', 'op'),
    'anno.export': (666, 460, 260, 52, '投票 → GPT 补 unknown → 剔碎片', 'anno.export', 'op'),
    'anno.text': (666, 540, 260, 52, '文字层：描述 · 五档问答', 'anno.text（可选）', 'opt'),
    'anno.verify': (666, 620, 260, 52, '校验 + 场景清单', 'anno.verify', 'op'),
    'model': (1000, 60, 216, 52, '被测模型 · 4DGS', '外部输入', 'ext'),
    'eval.case': (1000, 150, 216, 58, '评测 case', '10 输入 · 6 关键点 · 3+3 动作', 'op'),
    'eval.render_model': (1000, 260, 216, 52, '按位姿 + t 渲染', 'eval.render_model', 'op'),
    'eval.score': (1000, 370, 216, 52, '72 个评测算子', 'eval.score · A–H', 'op'),
    'eval.results': (1000, 480, 216, 52, '交付视图', 'results/ · OPERATORS.md', 'op'),
}
COLS = [(16, '上游（外部）'), (336, '3D 几何 + 轨迹'), (666, '3D 标注'), (1000, '评测 evaluate1')]


def node_svg(nid):
    x, y, w, h, t, sub, kind = NODES[nid]
    href = f'#op-{nid}' if nid != 'model' else '#eval'
    cls = {'ext': 'n ext', 'op': 'n op', 'opt': 'n opt', 'cond': 'n'}[kind]
    return (f'<a href="{href}" class="nl"><rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="6"/>'
            f'<text class="nt" x="{x + w / 2}" y="{y + h / 2 - 4}" text-anchor="middle">{html.escape(t)}</text>'
            f'<text class="ns" x="{x + w / 2}" y="{y + h / 2 + 13}" text-anchor="middle">{html.escape(sub)}</text></a>')


def arrow(points, label=None, lx=None, ly=None, anchor='middle', dashed=False):
    d = 'M' + ' L'.join(f'{a},{b}' for a, b in points)
    s = f'<path class="e{" dash" if dashed else ""}" d="{d}" marker-end="url(#ah)"/>'
    if label: s += f'<text class="el" x="{lx}" y="{ly}" text-anchor="{anchor}">{html.escape(label)}</text>'
    return s


def overview():
    p = []
    for x, t in COLS: p.append(f'<text class="ch" x="{x}" y="34">{t}</text>')
    # upstream internal
    p.append(arrow([(141, 118), (141, 148)], 'gaussian.ply', 148, 137, 'start'))
    p.append(arrow([(141, 208), (141, 238)], '锚点 · 评分', 148, 227, 'start'))
    p.append(arrow([(30, 118), (30, 328)], dashed=True))
    p.append('<text class="el" x="36" y="322">data.json</text>')
    # upstream -> 3D geometry
    p.append(arrow([(266, 89), (334, 89)], 'ply · obj', 300, 82))
    p.append(arrow([(266, 262), (334, 262)], 'poses.json', 300, 255))
    p.append(arrow([(266, 282), (300, 282), (300, 449), (334, 449)], 'frames_hd', 304, 330, 'start'))
    # geometry chain + decision
    p.append(arrow([(466, 118), (466, 148)]))
    p.append(arrow([(466, 208), (466, 238)]))
    p.append(arrow([(466, 298), (466, 330)]))
    p.append('<polygon class="n cond" points="466,332 532,362 466,392 400,362"/>'
             '<text class="nt" x="466" y="367" text-anchor="middle">自带漫游？</text>')
    p.append(arrow([(420, 383), (398, 418)], '是', 396, 404, 'end'))
    p.append(arrow([(512, 383), (534, 418)], '否', 538, 404, 'start'))
    p.append(arrow([(596, 179), (616, 179), (616, 560), (612, 600), (598, 619)], 'maps', 620, 548, 'start'))
    # trajectory -> annotation / evaluation
    p.append('<path class="e" d="M398,478 L398,500 L640,500"/><path class="e" d="M534,478 L534,500"/>')
    p.append(arrow([(640, 500), (640, 86), (664, 86)], '轨迹 clip', 560, 514, 'middle'))
    p.append(arrow([(596, 196), (630, 196), (630, 326), (664, 326)], 'mesh', 636, 262, 'start'))
    # annotation chain
    ys = [60, 140, 220, 300, 380, 460, 540, 620]
    for a, b in zip(ys, ys[1:]):
        p.append(arrow([(796, a + 52), (796, b - 2)], dashed=(a in (460, 540))))
    p.append('<text class="el" x="804" y="531">WITH_TEXT=1</text>')
    # annotation -> evaluation
    p.append(arrow([(926, 86), (962, 86), (962, 168), (998, 168)], '轨迹', 966, 128, 'start'))
    p.append(arrow([(926, 486), (950, 486), (950, 192), (998, 192)], '实例 · mask', 944, 449, 'end'))
    # evaluation chain
    p.append(arrow([(1108, 208), (1108, 258)], 'cameras.npz', 1114, 238, 'start'))
    p.append(arrow([(1216, 86), (1230, 86), (1230, 286), (1218, 286)]))
    p.append(arrow([(1108, 312), (1108, 368)], 'frames/*.png', 1114, 344, 'start'))
    p.append(arrow([(1000, 200), (988, 200), (988, 396), (998, 396)], dashed=True))
    p.append('<text class="el" x="984" y="420" text-anchor="end">case 真值</text>')
    p.append(arrow([(1108, 422), (1108, 478)], 'scores.json', 1114, 454, 'start'))
    p += [node_svg(n) for n in NODES]
    defs = ('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            '<path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker></defs>')
    return (f'<svg class="flow" viewBox="0 0 1244 690" role="img" aria-label="anno_eval 数据管线：海塞扫描与董婷漫游输出进入 3D 几何与轨迹，'
            f'再经 3D 标注，最后与被测 4DGS 一起进入评测">{defs}{"".join(p)}</svg>')


# ------------------------------------------------------------------ page
def page():
    import ops3d, operators as O
    ops = ops3d.describe(); rows = O.list_operators()
    scores = {}
    sp = ROOT / 'evaluate1' / 'runs' / SCENE / 'oracle' / 'scores.json'
    if sp.exists(): scores = json.loads(sp.read_text())['results']
    groups = collections.OrderedDict()
    for r in rows: groups.setdefault((r['group'], r['group_title']), []).append(r)
    data = {'stages': ops3d.STAGES, 'stage_zh': ops3d.STAGE_ZH, 'ops': ops,
            'groups': [{'code': c, 'title': t, 'ops': [{'name': r['name'], 'zh': r['zh'], 'inputs': r['inputs'], 'hib': r['higher_is_better'],
                                                         'unit': r['unit'], 'status': scores.get(r['name'], {}).get('status'),
                                                         'value': scores.get(r['name'], {}).get('value')} for r in rs]}
                       for (c, t), rs in groups.items()]}
    counts = collections.Counter(r['status'] for r in scores.values())
    tpl = (ROOT / 'tools' / 'flow_page_template.html').read_text()
    out = (tpl.replace('/*DATA*/', json.dumps(data, ensure_ascii=False)).replace('<!--SVG-->', overview())
              .replace('{{N_OPS}}', str(sum(1 for o in ops if not o['external']))).replace('{{N_EXT}}', str(sum(1 for o in ops if o['external'])))
              .replace('{{N_METRICS}}', str(len(rows))).replace('{{SCENE}}', SCENE)
              .replace('{{PILOT}}', f"oracle：{counts.get('ok', 0)} 出分 · {counts.get('n/a', 0)} n/a"))
    (ROOT / 'docs' / 'pipeline_flow.html').write_text(out)
    print('wrote', ROOT / 'docs' / 'pipeline_flow.html', len(out))


if __name__ == '__main__':
    page()
