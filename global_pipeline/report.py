"""Compact reproducible delivery report for all global-instance scenes."""
from .common import ROOT,read,dump,utc
import csv
from collections import Counter

def main():
    rows=[];failures=[];model_requests={}
    expected={p.parent.name for p in (ROOT/'outputs').glob('household*/analysis.json')}
    for folder in sorted((ROOT/'auto').glob('household*/global_instances')):
        required=['frames.json','global_instances.json','coverage.json','qa/reprojection_summary.json','segmentation_summary.json']
        if not all((folder/name).is_file() for name in required):continue
        v=read(folder/'validation.json') if (folder/'validation.json').exists() else {}
        local=read(folder/'local_validation.json') if (folder/'local_validation.json').exists() else {}
        descriptions=read(folder/'descriptions.json') if (folder/'descriptions.json').exists() else {}
        frames=read(folder/'frames.json');instances=read(folder/'global_instances.json')['instances']
        r=read(folder/'run_info.json');c=read(folder/'coverage.json');q=read(folder/'qa/reprojection_summary.json')
        inv=read(folder/'concepts.json')
        segmentation=read(folder/'segmentation_summary.json')
        for evidence in (folder/'evidence').glob('gpt_*/*.json'):
            record=read(evidence)
            if record.get('status')=='ok' and record.get('response_id'):model_requests[record['response_id']]=record
        current_frames=len(frames['frames']);clips=len(frames['clips']);expected_changes=sum(max(0,len(c['frames'])-1) for c in frames['clips'])
        def current(check):
            return bool(check.get('passed') and check.get('frames')==current_frames and check.get('clips')==clips
                        and check.get('global_instances')==len(instances))
        frame_ids={f['id'] for f in frames['frames']}
        pairs={(a,b) for clip in frames['clips'] for a,b in zip(clip['frames'][:-1],clip['frames'][1:])}
        level1=len({x['frame_id'] for x in descriptions.get('level1',[])}&frame_ids)
        level2=len({(x['from_frame'],x['to_frame']) for x in descriptions.get('level2',[])}&pairs)
        # Early full validation records predate the optional scope field;
        # they still contain both exact description coverage counts.
        verified=bool(current(v) and v.get('scope','full_outputs')=='full_outputs' and v.get('level1')==current_frames
                      and v.get('level2')==expected_changes and level1==current_frames and level2==expected_changes
                      and not descriptions.get('failures') and inv.get('all_frames_direct_gpt_audited',False)
                      and r.get('status')=='complete')
        rows.append({'scene_id':folder.parent.name,'status':r.get('status'), 'verified':verified,
            'local_verified':current(local) or verified,
            'frames':current_frames,'clips':clips,'global_instances':len(instances),
            'semantic_unknown':q['semantic_unknown_instances'],'level1':level1,'level2':level2,
            'expected_level1':current_frames,'expected_level2':expected_changes,
            'point_coverage':c['point_coverage'],'visible_point_coverage':c.get('visible_point_coverage'),
            'superpoint_coverage':c['superpoint_coverage'],'direct_gpt_frames':len(inv['audit_frames']),
            'raw_sam_masks':segmentation['raw_masks'],'exclusive_sam_masks':segmentation['exclusive_masks'],
            'frames_with_no_usable_masks':segmentation['empty_frames'],
            'mask_assignment_rate':q['accepted_mask_assignments']/max(q['total_masks'],1),
            'point_reprojection_iou':q['metrics'].get('visible_point_iou'),
            'hard_cluster_conflicts':read(folder/'coarse_summary.json')['hard_conflicts_inside_clusters']})
        if not verified or r.get('status')!='complete':failures.append(folder.parent.name)
    failures=sorted(set(failures)|(expected-{r['scene_id'] for r in rows}))
    models=dict(Counter(r.get('actual_model','unknown') for r in model_requests.values()))
    usage={key:sum(r.get('usage',{}).get(key,0) for r in model_requests.values()) for key in ['input_tokens','output_tokens','total_tokens']}
    dump(ROOT/'auto/global_summary.json',{'created_at':utc(),'expected_scenes':sorted(expected),'scenes':rows,'incomplete_scenes':failures,
        'gpt_evidence':{'unique_successful_response_ids':len(model_requests),'actual_model_counts':models,'token_usage':usage,
                        'scope':'Saved successful responses for exported scenes, deduplicated by response ID; failed requests and discarded attempts are not counted.'}})
    if rows:
        with (ROOT/'auto/global_summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    totals={key:sum(r[key] for r in rows) for key in ['frames','clips','global_instances','semantic_unknown','level1','level2','raw_sam_masks','exclusive_sam_masks','frames_with_no_usable_masks']}
    report=['# 全局实例和两级描述结果','',f'生成时间：{utc()}。动态浏览：<http://10.0.0.184:18765/global>；本机：<http://127.0.0.1:18765/global>。','',
        '方法、坐标与复现命令见 [流水线文档](../global_pipeline/README.md)。','',
        '表中描述数量仅统计正式 descriptions.json，不把运行中成功缓存预览冒充全量产物。浏览器可查看运行中部分文字。','',
        '| 场景 | 验收状态 | 帧 | 轨迹 | Global 实例 | Unknown 语义 | 全点覆盖 | 可见点覆盖 | 单帧/变化描述 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        visible='—' if r['visible_point_coverage'] is None else '%.1f%%'%(100*r['visible_point_coverage'])
        state='全部通过' if r['verified'] else ('几何/视频通过；文字待验收' if r['local_verified'] else '待验收')
        report.append('| [%s](%s/global_instances/global_instances.json) | %s | %d | %d | %d | %d | %.1f%% | %s | %d / %d |'%(r['scene_id'],r['scene_id'],state,r['frames'],r['clips'],r['global_instances'],r['semantic_unknown'],100*r['point_coverage'],visible,r['level1'],r['level2']))
    report+=['',f'已导出 {len(rows)} 个场景，其中 {sum(r["local_verified"] for r in rows)} 个通过几何/视频验收，{sum(r["verified"] for r in rows)} 个通过包含文字的全量验收。','',
        f'全部导出场景合计：{totals}。完整文字目标：{sum(r["expected_level1"] for r in rows)} 条 Level 1、{sum(r["expected_level2"] for r in rows)} 条 Level 2。','',
        f'已保存的成功 GPT 响应按 response ID 去重：{len(model_requests)} 次；实际模型计数：{models}。用量：{usage}，不含失败或未保存的尝试。','',
        '所有输出均未人工审核，global ID 的唯一来源是 fine_instances.npz:point_fine_ids。0 为未知。三维框与点云仍按 inferred_meter 解释。','',
        '每帧直接经 GPT 做类别发现，并经 SAM3 做掩码分割。Level 1 是当前帧物体描述；Level 2 是同轨迹相邻帧的可见变化。原始房屋为静态重建，不把视角、遮挡或检测支持变化写成真实物体运动。','',
        'SAM3 文本 grounding 接口没有独立预测 mask IoU。硬冲突筛选与投票使用真实检测分数和阈值稳定度代理，sam_iou 保存 null。无人工真值时，重投影一致性不是标注准确率。','',
        '只读视觉抽查已观察到空调部分标成 curtain、墙面标成 tree、柜门标成 door 等语义错误；global ID 数量是自动候选实例数，不能当作真实物体计数。截图和帧/ID证据见 [最终只读抽查](../services/viewer_validation/FINAL_READONLY_AUDIT.md)。该抽查不修改自动标签，也不构成人工审核。','',
        '每个场景目录有 projection_video.mp4 预览；videos/ 下按轨迹保存 RGB/overlay/instance/semantic 视频；rendered/ 保存每帧数字标签和颜色图片；labeled_points.ply 可在三维工具中打开。视频预览在不连通区域和楼层之间硬切，不声称存在可通行连接。','',
        f'未完成或验收失败场景：{failures or "无"}。']
    (ROOT/'auto/GLOBAL_REPORT.md').write_text('\n'.join(report)+'\n')
    print('Global report',len(rows),'scenes',totals,'incomplete',failures,flush=True)

if __name__=='__main__':main()
