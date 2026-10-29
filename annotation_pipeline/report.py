from .common import ROOT,read,dump,utc,rel
from pathlib import Path
from collections import Counter
import csv,json,html
import numpy as np,cv2

PAGE='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Household 自动标注 v0.1</title><style>body{margin:0;background:#f1f4f8;color:#192636;font:15px/1.7 system-ui,sans-serif}header{background:#172b40;color:white;padding:22px 32px}header h1{margin:0;font-size:25px}main{max-width:1500px;margin:auto;padding:24px}section{background:white;padding:20px;border:1px solid #dce3ed;border-radius:10px;margin-bottom:18px}.toolbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap}select,input{font:inherit;padding:7px 10px;border:1px solid #c8d1df;border-radius:6px}a{color:#2266ae}.note{background:#fff2cf;border-radius:6px;padding:12px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:14px}.card{border:1px solid #dde3eb;border-radius:8px;padding:12px}.card img{width:100%;height:190px;object-fit:contain;background:#f5f5f5}.card p{margin:6px 0}.tag{display:inline-block;background:#edf3fa;border-radius:5px;padding:2px 7px;margin:3px}.warn{color:#934500;background:#fff0d9}.overview{display:grid;grid-template-columns:1.5fr 1fr;gap:20px}.overview img{max-width:100%;max-height:640px;object-fit:contain}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:7px;border-bottom:1px solid #dfe5ed}details{padding:7px;border-bottom:1px solid #e5e8ef}small{color:#637187}.issues{max-height:420px;overflow:auto}@media(max-width:900px){.overview{grid-template-columns:1fr}main{padding:12px}}</style>
<header><h1>Household 自动基础标注 · v0.1</h1><div>来源：SAM3 文本分割 + mesh 深度 + 多视图融合 + VLM 视觉描述</div></header><main><section><div class="toolbar"><select id="scene"></select><select id="floor"></select><input id="filter" placeholder="搜索物体 ID / 类别"><label><input type="checkbox" id="issuesOnly">仅显示有审核问题的物体</label><a href="README.md">方法与限制</a></div><p id="stats"></p><p class="note">全部结果尚未人工审核。三维框覆盖观测表面，不是完整物体尺寸；分数不是校准概率。米制仍为推断；地图与图片中 +X/+Y 不代表东/北。这里只读自动文件，不写 human/ 或 final/。</p><p id="description"></p><div id="downloads"></div></section><section class="overview"><div><img id="map"></div><div><h3>区域与连接</h3><div id="regions"></div></div></section><section><h3>优先审核问题</h3><div id="issues" class="issues"></div></section><section><h3>物体候选与证据</h3><div id="objects" class="grid"></div></section><section><h3>所有场景概览</h3>SUMMARY</section></main><script>const DATA=DATA_JSON,$=x=>document.getElementById(x);let item;const esc=s=>String(s??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');function path(s){return item.scene_id+'/'+s}function refs(x){return Array.isArray(x.ref)?x.ref:[x.ref]}
function show(){item=DATA[Number($('scene').value)];const g=item.graph;$('floor').innerHTML='<option value="all">全部楼层</option>'+g.floors.map(f=>'<option value="'+f.id+'">'+f.id+' · z='+f.z_floor.toFixed(2)+'</option>').join('');$('stats').textContent=item.scene_id+' · '+g.floors.length+' 层 · '+g.regions.length+' 区域 · '+g.connections.length+' 连接候选 · '+g.objects.length+' 物体 · '+g.issues.length+' 条审核问题';$('description').textContent=g.scene.description_pred?.long||'';$('downloads').innerHTML='<a href="'+path('scene_graph.json')+'">scene_graph.json</a> · <a href="'+path('run_info.json')+'">运行与失败记录</a> · <a href="'+path('evidence/keyframes.json')+'">关键帧相机</a> · <a href="'+path('evidence/object_points.npz')+'">物体点云 NPZ</a>';$('issues').innerHTML=g.issues.map(x=>'<details><summary>'+esc(x.type)+' · '+esc(refs(x).join(', '))+' · 优先级 '+Number(x.priority||0).toFixed(2)+'</summary><div>'+esc(x.detail)+'</div><small>source: '+esc(x.source)+' · score: '+Number(x.score).toFixed(2)+'</small></details>').join('');render()}
function render(){const g=item.graph,fl=$('floor').value,filter=$('filter').value.toLowerCase(),issueRefs=new Set(g.issues.flatMap(refs));const first=fl==='all'?g.floors[0].id:fl;$('map').src=path('evidence/overview_'+first+'.jpg');$('regions').innerHTML=g.regions.filter(r=>fl==='all'||r.floor===fl).map(r=>'<p><span class="tag">'+r.id+' · '+esc(r.name_auto)+' · '+r.type_pred.score.toFixed(2)+'</span><a href="'+path(r.panorama)+'">全景</a><br><small>'+esc(r.type_pred.label)+' · '+r.floor+' · '+r.object_ids.length+' 个物体 · '+r.area.toFixed(1)+' m² 可通行候选</small></p>').join('')+'<details><summary>连接候选</summary>'+g.connections.filter(c=>fl==='all'||c.floor===fl||c.floors?.includes(fl)).map(c=>'<p>'+c.id+' · '+c.regions.join(' ↔ ')+' · '+c.type_pred.label+' ('+c.score.toFixed(2)+') · 地图连通='+c.connected_in_map+'</p>').join('')+'</details>';
$('objects').innerHTML=g.objects.filter(o=>(fl==='all'||o.floor===fl)&&(!filter||(o.id+' '+o.category_pred.label).toLowerCase().includes(filter))&&(!$('issuesOnly').checked||issueRefs.has(o.id))).map(o=>{const b=o.bbox,a=o.attributes_pred||{},warn=issueRefs.has(o.id),p=o.category_vlm_check||{};return '<article class="card"><a href="'+path(o.best_view_metadata.full_view_path)+'"><img loading="lazy" src="'+path(o.best_view)+'"></a><p><b>'+o.id+' · '+esc(o.category_pred.label)+'</b> <span class="tag '+(warn?'warn':'')+'">'+(warn?'待重点审核':'unreviewed')+'</span></p><p>'+o.floor+' / '+o.region+' · SAM3 '+o.category_pred.score.toFixed(2)+' · '+o.n_views+' 视图</p><small>中心 ['+b.center.map(x=>x.toFixed(2)).join(', ')+']<br>观测框跨度 ['+b.size.map(x=>x.toFixed(2)).join(', ')+']<br>颜色 '+esc(a.color)+' · 材质 '+esc(a.material)+'<br>VLM '+(p.confirmed===false?'类别有冲突：'+esc(p.alternative):p.confirmed===true?'类别认可':'待确认')+'</small><p><a href="'+path(o.best_view_metadata.context_path)+'">原始裁图</a> · <a href="'+path(o.best_view_metadata.full_view_path)+'">整帧掩码</a> · <a href="'+path(o.best_view_metadata.mask_path)+'">mask</a></p></article>'}).join('')}
DATA.forEach((x,i)=>{$('scene').innerHTML+='<option value="'+i+'">'+x.scene_id+'</option>'});$('scene').onchange=show;$('floor').onchange=render;$('filter').oninput=render;$('issuesOnly').onchange=render;show();</script></html>'''

def overview(scene,out,graph):
    for fl in graph['floors']:
        folder=ROOT/'outputs'/scene/f'floor_{int(fl["id"][1:])}';image=cv2.imread(str(folder/'topdown_3dgs.png'));camera=read(folder/'topdown_camera.json');origin=np.array(camera['origin_xy']);extent=np.array(camera['extent_xy']);h,w=image.shape[:2]
        def pixel(xy):return tuple(np.rint((np.array(xy)-origin)/extent*[w,-h]+[0,h]).astype(int))
        for r in graph['regions']:
            if r['floor']!=fl['id']:continue
            for poly in r['geometry']['coordinates']:
                for ring in poly:
                    p=np.array([pixel(x) for x in ring],np.int32);cv2.polylines(image,[p],True,(170,75,20),2,cv2.LINE_AA)
            point=pixel(r['center']);cv2.circle(image,point,8,(255,255,255),-1);cv2.putText(image,r['id'],(point[0]+8,point[1]),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),2,cv2.LINE_AA)
        for o in graph['objects']:
            if o['floor']!=fl['id']:continue
            point=pixel(o['bbox']['center'][:2]);cv2.circle(image,point,4,(0,70,235),-1);cv2.putText(image,o['id'],point,cv2.FONT_HERSHEY_SIMPLEX,.28,(0,35,190),1,cv2.LINE_AA)
        for c in graph['connections']:
            if fl['id'] not in c.get('floors',[c.get('floor')]):continue
            point=pixel(c['position'][:2]);cv2.drawMarker(image,point,(0,210,0),cv2.MARKER_DIAMOND,14,2)
        cv2.imwrite(str(out/'evidence'/('overview_'+fl['id']+'.jpg')),image)


def main():
    entries=[];summary=[]
    for p in sorted((ROOT/'auto').glob('*/scene_graph.json')):
        graph=read(p);run=read(p.parent/'run_info.json');overview(p.parent.name,p.parent,graph)
        entries.append({'scene_id':p.parent.name,'graph':graph});summary.append({'scene_id':p.parent.name,**run['counts'],'keyframes':read(p.parent/'evidence/segmentation_summary.json')['keyframes'],'scale_check':graph['scene']['scale_check']['status'],'status':run['status']})
    if not entries:return
    dump(ROOT/'auto/summary.json',summary)
    with (ROOT/'auto/summary.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    headers=['scene_id','floors','regions','connections','objects','texts','issues','keyframes','scale_check','status'];table='<table><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr>'
    for row in summary:table+='<tr>'+''.join('<td>'+html.escape(str(row[k]))+'</td>' for k in headers)+'</tr>'
    table+='</table>'
    (ROOT/'auto/index.html').write_text(PAGE.replace('SUMMARY',table).replace('DATA_JSON',json.dumps(entries,ensure_ascii=False).replace('</','<\\/')))
    print('Reported',len(entries),'scenes',sum(x['objects'] for x in summary),'objects',flush=True)

if __name__=='__main__':main()
