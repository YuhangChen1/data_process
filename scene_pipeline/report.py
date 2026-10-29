"""Offline review page and reproducible summary of measured results."""
from pathlib import Path
import json,csv,html
import numpy as np
from .geometry import save_json

HTML='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>家庭场景 · 标注工作台</title>
<style>body{margin:0;background:#f3f5f8;color:#182331;font:15px/1.65 system-ui,sans-serif}header{padding:24px 32px;background:#142538;color:white}header h1{margin:0;font-size:25px}header p{color:#bdcedf;margin:6px 0}main{max-width:1480px;margin:auto;padding:24px}section{background:white;border:1px solid #dce3ec;border-radius:12px;padding:20px;margin-bottom:20px}.toolbar{display:flex;gap:10px;flex-wrap:wrap;align-items:center}select,input,button{font:inherit;padding:7px 12px;border:1px solid #ccd5e0;border-radius:6px;background:white}button{cursor:pointer}button.primary{background:#225faf;color:white}a{color:#225faf}canvas{width:100%;height:auto;display:block;background:#eee;touch-action:none}.work{display:grid;grid-template-columns:minmax(400px,2fr) minmax(300px,1fr);gap:22px}.preview{width:100%;max-height:420px;object-fit:contain}.pill{background:#eef3fb;border-radius:6px;padding:4px 10px}.muted{color:#65738b}.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px}.gallery img{width:100%}video{width:100%;max-height:300px}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:7px;border-bottom:1px solid #e0e5ec}#list{max-height:240px;overflow:auto}#list button{padding:2px 6px;margin:3px}.note{background:#fff4da;padding:12px;border-radius:6px}code{font-size:13px}@media(max-width:900px){.work{grid-template-columns:1fr}main{padding:12px}}</style>
<header><h1>家庭场景 · 标注工作台</h1><p>原始 3DGS 渲染 · mesh 深度 · 保守可通行地图 · 标注专用轨迹</p></header><main>
<section><div class="toolbar"><select id="scene"></select><select id="layer"><option value="annotation_overlay.png">3DGS + mesh 高度叠图</option><option value="topdown_3dgs.png">3DGS 俯视图</option><option value="mesh_height.png">mesh 高度图</option><option value="regions_and_routes.png">可通行区域与路径</option><option value="mesh_open_boundaries.png">mesh 开放边界（红色）</option></select><label><input type="checkbox" id="show" checked>显示区域</label><a href="REPORT.md">完整测量报告</a></div><p id="stats"></p><p class="note">坐标已统一为 Z 朝上；米制由住宅层高推断，未得到真实尺寸独立验证。彩色区域是自动几何候选，不是已确认的房间；未知地面保持不可通行。所有漫游轨迹仅用于标注，不属于评测 Q。</p></section>
<div class="work"><section><canvas id="canvas"></canvas><p class="muted">点击添加多边形顶点 → 输入名称 → 保存新区域。选择已有区域可命名或删除。导出后保存的 GeoJSON 使用本地平面坐标，不是经纬度。</p><div class="toolbar"><input id="label" placeholder="如 living_room / 卧室"><button id="save" class="primary">保存新区域</button><button id="undo">撤销顶点</button><button id="cancel">清空草稿</button><button id="export">导出本层标注</button></div></section>
<section><h3>区域列表</h3><div id="list"></div><div class="toolbar"><button id="rename">命名选中区域</button><button id="remove">删除选中区域</button></div><p class="muted">修改只保存在当前页面内存中，离开前请导出。</p><h3>空间检查</h3><img id="hist" class="preview"><div id="links"></div><h3>同相机深度对比</h3><a id="depthlink"><img id="depth" class="preview"></a></section></div>
<section><h3>区域 360° 全景</h3><p class="muted">2048 × 1024 等距柱状图；中心经度朝本地 +X，向右转向 +Y。可直接交给 VLM 描述区域。编号对应自动候选区域。</p><div id="panos" class="gallery"></div></section>
<section><h3>标注专用漫游</h3><p class="muted">每个可通行连通分量单独一段；相机高 1.5 m。深度原始数据位于各视频目录的 mesh_depth_mm/，相机与时间戳位于 cameras.npz。</p><div id="videos" class="gallery"></div></section>
<section><h3>全部场景检查结果</h3>SUMMARY_TABLE</section></main>
<script>const DATA=SCENE_DATA;const $=id=>document.getElementById(id),canvas=$('canvas'),ctx=canvas.getContext('2d');let active=null,image=new Image(),draft=[],selected=-1;const edits={};
function world(p){const c=active.camera;return[c.origin_xy[0]+p[0]/canvas.width*c.extent_xy[0],c.origin_xy[1]+(1-p[1]/canvas.height)*c.extent_xy[1]]}
function pixel(p){const c=active.camera;return[(p[0]-c.origin_xy[0])/c.extent_xy[0]*canvas.width,(1-(p[1]-c.origin_xy[1])/c.extent_xy[1])*canvas.height]}
function features(){return edits[active.key].features}
function draw(){ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(image,0,0,canvas.width,canvas.height);if($('show').checked){features().forEach((f,index)=>{ctx.beginPath();const polygons=f.geometry.type==='Polygon'?[f.geometry.coordinates]:f.geometry.coordinates;for(const polygon of polygons){for(const ring of polygon){ring.forEach((p,i)=>{const q=pixel(p);i?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.closePath()}}ctx.fillStyle=index===selected?'#ffad4338':'#2469d51a';ctx.fill('evenodd');ctx.strokeStyle=index===selected?'#ff8b00':'#386bbe';ctx.lineWidth=index===selected?3:1.5;ctx.stroke()})}if(draft.length){ctx.beginPath();draft.forEach((p,i)=>{const q=pixel(p);i?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.strokeStyle='#ff4433';ctx.lineWidth=3;ctx.stroke();draft.forEach(p=>{const q=pixel(p);ctx.beginPath();ctx.arc(...q,4,0,2*Math.PI);ctx.fillStyle='#ff4433';ctx.fill()})}}
function list(){const div=$('list');div.replaceChildren();features().forEach((f,i)=>{const b=document.createElement('button');b.textContent=(f.properties.region_id||'region')+' · '+(f.properties.semantic_label||'未命名');if(i===selected)b.style.background='#fff1cc';b.onclick=()=>{selected=i;$('label').value=f.properties.semantic_label||'';list();draw()};div.appendChild(b)})}
function loadLayer(){image=new Image();image.onload=draw;image.src=active.path+'/'+$('layer').value}
function load(){active=DATA[Number($('scene').value)];draft=[];selected=-1;if(!edits[active.key])edits[active.key]=structuredClone(active.geojson);canvas.width=active.camera.width;canvas.height=active.camera.height;$('stats').textContent=active.summary;$('hist').src=active.scene+'/height_histogram.png';$('depth').src=active.path+'/alignment_views/contact_sheet.jpg';$('depthlink').href=$('depth').src;$('links').innerHTML='<a href="'+active.path+'/maps.npz">地图 NPZ</a> · <a href="'+active.path+'/occupancy.yaml">occupancy YAML</a> · <a href="'+active.path+'/annotation_routes.json">轨迹 JSON</a> · <a href="'+active.scene+'/analysis.json">测量 JSON</a>';$('panos').replaceChildren();for(const r of active.regions){const box=document.createElement('div'),a=document.createElement('a'),img=document.createElement('img'),caption=document.createElement('div');a.href=active.path+'/panoramas/'+r.region_id+'.jpg';img.src=a.href;img.loading='lazy';a.append(img);caption.textContent=r.region_id+' · '+r.area_m2_assuming_meters.toFixed(1)+' m²';box.append(a,caption);$('panos').append(box)}$('videos').replaceChildren();for(const clip of active.clips){const box=document.createElement('div'),v=document.createElement('video'),a=document.createElement('a'),p=document.createElement('p');v.controls=true;v.preload='none';v.src=active.path+'/annotation_walkthrough/'+clip+'/rgb.mp4';a.href=active.path+'/annotation_walkthrough/'+clip+'/depth_preview.mp4';a.textContent='mesh 深度预览';p.textContent=clip;box.append(v,p,a);$('videos').append(box)}list();loadLayer()}
DATA.forEach((d,i)=>{const o=document.createElement('option');o.value=i;o.textContent=d.key;$('scene').append(o)});$('scene').onchange=load;$('layer').onchange=loadLayer;$('show').onchange=draw;
canvas.onclick=e=>{const r=canvas.getBoundingClientRect();draft.push(world([(e.clientX-r.left)*canvas.width/r.width,(e.clientY-r.top)*canvas.height/r.height]));draw()};$('undo').onclick=()=>{draft.pop();draw()};$('cancel').onclick=()=>{draft=[];draw()};$('save').onclick=()=>{if(draft.length<3)return alert('至少需要三个顶点');const ring=[...draft,draft[0]];features().push({type:'Feature',properties:{region_id:'manual_'+Date.now(),semantic_label:$('label').value.trim()||null,status:'human_annotated',floor_id:active.floor_id,purpose:'annotation'},geometry:{type:'Polygon',coordinates:[ring]}});draft=[];selected=features().length-1;list();draw()};$('rename').onclick=()=>{if(selected<0)return;features()[selected].properties.semantic_label=$('label').value.trim()||null;features()[selected].properties.status='human_reviewed';list();draw()};$('remove').onclick=()=>{if(selected<0)return;features().splice(selected,1);selected=-1;list();draw()};$('export').onclick=()=>{const data={...edits[active.key],scene:active.scene,floor_id:active.floor_id,world_to_canonical:active.transform,unit_status:'meters_inferred_not_independently_verified'};const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/geo+json'}));const a=document.createElement('a');a.href=url;a.download=active.key.replaceAll('/','_')+'_annotated.geojson';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};load();
</script></html>'''

def main():
    root=Path(__file__).resolve().parents[1];out=root/'outputs';rows=[];scenes=[];data=[]
    for p in sorted(out.glob('*/analysis.json')):
        d=json.loads(p.read_text());name=p.parent.name;a=d['alignment'];q=d['mesh_quality']
        row={'scene':name,'floors':d['floor_count'],'gaussians':a['gaussian_count'],'surface_median_mm':round(a['before_point_to_surface']['median']*1000,2),'surface_p90_mm':round(a['before_point_to_surface']['p90']*1000,2),'icp_translation_mm':round(a['icp_translation']*1000,2),'icp_rotation_deg':round(a['icp_rotation_degrees'],3),'local_scale_ratio':round(a['local_similarity_scale'],6),'ceiling_gaps_m':'; '.join(f'{f["ceiling_gap"]:.3f}' for f in d['floors']),'navigable_area_m2':round(sum(f['navigable_area'] for f in d['floors']),2),'regions':sum(f['regions'] for f in d['floors']),'clips':sum(f['connected_components'] for f in d['floors']),'open_boundary_components_over_1m':q['boundary_components_perimeter_over_1m'],'has_colors':q['vertex_colors'],'same_frame_supported':a['same_frame_supported']}
        depths=[]
        for fl in d['floors']:
            folder=p.parent/f'floor_{fl["floor_id"]}'
            comparison=folder/'depth_alignment.json'
            if comparison.exists():depths += json.loads(comparison.read_text())['views']
            if not (folder/'topdown_camera.json').exists():continue
            geo=json.loads((folder/'regions.geojson').read_text());regions=json.loads((folder/'regions.json').read_text());cam=json.loads((folder/'topdown_camera.json').read_text())
            clips=sorted(x.name for x in (folder/'annotation_walkthrough').glob('component_*') if (x/'rgb.mp4').exists())
            data.append({'key':f'{name}/floor_{fl["floor_id"]}','scene':name,'path':f'{name}/floor_{fl["floor_id"]}','floor_id':fl['floor_id'],'camera':cam,'geojson':geo,'regions':regions,'clips':clips,'transform':d['world_to_canonical'],
                         'summary':f'地面高度 {fl["height"]:.2f} · 层内高度约 {fl["ceiling_gap"]:.2f} m · 保守可通行 {fl["navigable_area"]:.1f} m² · {fl["regions"]} 个候选区域 · {fl["connected_components"]} 个连通分量'})
        vals=[r['median_abs_depth_difference'] for r in depths if r['median_abs_depth_difference'] is not None]
        row['median_of_view_median_depth_error_mm']=round(float(np.median(vals))*1000,2) if vals else None
        rows.append(row);scenes.append(d)
    if not rows:return
    with (out/'summary.csv').open('w',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save_json(out/'summary.json',rows)
    table=['| 场景 | 候选楼层 | 层内高度 m | 中心到表面中位数 / P90 mm | ICP 位移 mm | 尺度比 | 可通行 m² | 区域 / 视频 |','|---|---:|---|---|---:|---:|---:|---:|']
    for r in rows:table.append(f'| {r["scene"]} | {r["floors"]} | {r["ceiling_gaps_m"]} | {r["surface_median_mm"]} / {r["surface_p90_mm"]} | {r["icp_translation_mm"]} | {r["local_scale_ratio"]} | {r["navigable_area_m2"]} | {r["regions"]} / {r["clips"]} |')
    text='''# 家庭场景处理结果：第 0–2 步

入口：[标注工作台](index.html)。本报告对应 8 个家庭及 household_3 的 0929_update 独立版本；更新版没有与旧版混用。

## 第 0 步：坐标、尺度、质量

所有 mesh 均带顶点颜色。各场景主竖直方向为原始 **−Y**。先将 `[x,y,z]` 映射到 `[x,z,-y]`，再用主地面面片的面积加权法向做小角度调平；精确齐次变换保存在各场景 `analysis.json` 的 `world_to_canonical`。mesh 与高斯一起使用同一变换，没有对任何一方额外应用 ICP 修正。

使用固定随机种子，筛选 mesh 主体包围盒内、opacity > 0.5 且最大高斯标准差 < 0.25 的中心点，取至多 4 万个，测到三角面表面的距离；随后对 mesh 表面采样做修剪 ICP，并估计局部相似尺度比。此采样诊断不代表远处离群高斯，也不能独立证明物理单位。完整条件、变换与残差分布见 `alignment.json` 和 `alignment_samples.npz`。

以下长度和面积均基于 **1 个坐标单位 = 1 米的推断**。住宅层内高度提供一致性证据，但没有已知门高、标尺或原始标定，不能宣称米制已经独立确认。短于常见 2.4–3 m 的层必须复核。家庭7上层俯视图能看到床和卧室布局，支持存在夹层；其约 2 m 的层内高度仍需真实尺寸复核。

'''+ '\n'.join(table)+'''

**同一相机深度验证：** 每层面积最大的至多 3 个候选区域中心、每个中心 4 个方向，同时渲染 mesh first-hit Z-depth 与 3DGS alpha-normalized expected Z-depth。`alignment_views/` 提供 RGB、两种深度及误差并排图，以及相机、原始 float 深度 NPZ；`depth_alignment.json` 统计 alpha > 0.8 的有效像素。透明窗、孔洞和遮挡边界处，两种深度估计本来就可能不同，不能要求逐像素完全相等。家庭3两版还包含室外庭院和植被；相应视角误差明显更大，必须与室内视角分别检查。

**孔洞和颜色：** `mesh_quality.json` 列出开放边数、非流形边、退化面、总开放边长、最大开放边界的包围盒和跨度。`boundary_edges.npz` 保留边界段位置，每层 `mesh_open_boundaries.png` 将这些开放边界以红线叠到地面范围上。边界连通分量并不一定是闭合孔洞，也可能是门窗、扫描范围边缘或缺失外壳，报告不会把它们全部当作可通行门洞；所有候选地图均由观测支撑限制。

## 第 1 步：楼层与可通行地图

对近水平面片按面积累积 2.5 cm 高度直方图，分别统计朝上地面候选和朝下天花板候选。找峰后，要求地面有相应层内净高证据，且相邻候选楼层高度至少间隔 1.9 m；家具台面不会仅凭一个峰被当成楼层。`height_histogram.png` 和 `analysis.json` 保留完整峰值信息。

地图分辨率 0.05 m。地面支持来自候选高度 ±0.22 m 内朝上的三角面投影；障碍来自距地 **0.10–1.80 m** 高度带内精确裁剪后的三角形投影，投影加入半格保守边缘。随后按 **0.25 m** 人体半径用距离变换收缩。只修补一格裂缝和小于 0.04 m² 的孔隙，推断支撑单独保存在 `inferred_floor`；未知区域不视为自由空间。

每层 `maps.npz` 包含 observed_floor、inferred_floor、support、obstacles、clearance、navigable、floor_height、raw_occupancy、occupancy。`occupancy.npy` / ROS 风格 PGM+YAML 使用 -1 未知、0 自由、100 障碍；导出的 occupancy 已按人体半径膨胀，后续使用不要无意重复膨胀。NumPy 行索引朝 +Y，图片做上下翻转；精确约定在 `floor.json`。

自动区域是距离变换分水岭得到的**几何候选**，种子平台按 8 邻域合并，并将相邻小于 0.8 m² 的碎片合并到共享边界最多的候选；它们不是语义房间真值。小通道、家具和扫描缺口可能使一个房间分成多个候选，或使多个房间合并。`regions.geojson` 可在工作台审核、命名、删除或添加多边形后导出；新标注不会自动覆盖已有全景和轨迹，需要据新中心重新渲染。

## 第 2 步：标注素材

- **俯视图：** `topdown_3dgs.png` 是真实各向异性 3D Gaussian splatting；按每层地面至 min(地面+1.8 m, 天花板−0.2 m) 裁高斯中心，移除上层与天花板。`mesh_height.png` 是从同一裁剪高度往下射线得到的高度图；`annotation_overlay.png` 叠加二者并标候选区域编号。原始高度、alpha 与相机保存在 `topdown.npz` / `topdown_camera.json`。高斯按中心裁剪，跨裁剪平面的大椭球可能留下少量边缘伪影。
- **漫游：** `annotation_walkthrough/component_NNN/rgb.mp4` 为 960×540、8 fps 的真实 3DGS RGB。区域中心取可通行范围内净空最大的点，以禁止斜穿墙角的 A* 连接，同一连通分量内平滑并重新检查线段。相机高 1.5 m，朝轨迹前方；只有单个中心的分量原地环视。每个分量单独视频，不跨墙、不传送、不自动跨楼层走楼梯。
- **深度：** 同路径 `mesh_depth_mm/NNNNN.png` 为 16-bit 毫米值，0 无效，是相机 Z 深度而非到相机的欧氏距离；`depth_preview.mp4` 只是着色预览。`cameras.npz` 保存逐帧 canonical / 原始坐标相机位姿、内参与时间戳。全部 `purpose` 明确为 `annotation_only_not_evaluation_Q`，没有创建或混用评测 Q。
- **全景：** `panoramas/candidate_NNN.jpg` 为 2048×1024 360° 等距柱状图，由 6 个 90° 3DGS 视图投影而成，可供 VLM 或人工写描述。元数据记录中心和立方体相机位姿；尚未编造语义描述。原始高斯文件仅有 DC 颜色，无高阶球谐项，因此这些渲染使用 SH degree 0。

GPU 渲染会过滤非有限数据、opacity ≤0.005、最大 sigma ≥2 和远离 mesh 主体的高斯，数量与规则见 `render_info.json`。原始模型文件保持不变。局部未重建区域在素材中可能留白，不能把留白解释成真实开口。

## 复现与输出核验

在工作目录运行：

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=6 python -m scene_pipeline.run --stage geometry
CUDA_VISIBLE_DEVICES=7 TORCH_CUDA_ARCH_LIST=8.0 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=6 python -m scene_pipeline.run --stage render
python -m scene_pipeline.report
python -m scene_pipeline.verify
```

`--scene household_3__0929_update` 可单独处理更新版；`--force` 重算，默认跳过已完成场景。所用 GPU 号可按空闲情况调整。数值参数集中在 `scene_pipeline/geometry.py`；若实际单位不是米或人体半径需要修改，应重新计算地图与素材。

本地直接打开 `index.html` 即可浏览、画区域与导出，页面不依赖外部资源。也可以在工作目录执行 `python -m http.server 8000`，浏览 `http://localhost:8000/outputs/`。本次没有发布网站。
'''
    verification=out/'verification.json'
    if verification.exists():
        checked=json.loads(verification.read_text())
        text += f'\n最终核验通过：{len(rows)} 套场景、{sum(r["floors"] for r in rows)} 个候选楼层、{checked["total_panoramas"]} 张全景、{checked["total_clips"]} 段漫游和 {checked["total_frames"]} 张逐帧深度图。所有视频已完整解码核对，全部轨迹与相机位置通过网格碰撞检查。核验详情见 `verification.json`。\n'
    (out/'REPORT.md').write_text(text)
    headers=['场景','层数','表面中位数 mm','P90 mm','尺度比','层内高度 m','候选区域','>1 m 开放边界']
    tablehtml='<table><thead><tr>'+''.join('<th>'+x+'</th>' for x in headers)+'</tr></thead><tbody>'
    for r in rows:
        values=[r['scene'],r['floors'],r['surface_median_mm'],r['surface_p90_mm'],r['local_scale_ratio'],r['ceiling_gaps_m'],r['regions'],r['open_boundary_components_over_1m']]
        tablehtml+='<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in values)+'</tr>'
    tablehtml+='</tbody></table>'
    if data:(out/'index.html').write_text(HTML.replace('SUMMARY_TABLE',tablehtml).replace('SCENE_DATA',json.dumps(data,ensure_ascii=False).replace('</','<\\/')))
    print(f'Reported {len(rows)} scenes / {len(data)} rendered levels')

if __name__=='__main__':main()
