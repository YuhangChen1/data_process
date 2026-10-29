# 最终算子（72 个）

入口 `operators.py`：`list_operators()`、`load_context(case, frames)`、`run_operator(name, ctx)`、`run_all(ctx)`；
命令行 `python operators.py list | doc | run`。每个算子是 `evaluate1/metrics/*.py` 中按 `metrics/base.py` 约定注册的函数，
返回 `ok` / `n/a`（case 不具备所需输入，附原因）/ `error`（附 traceback）。
方向 ↑ 表示越高越好。WBench 类按 WBench 的 0–100 归一；R2M 类的值是关键点重访对的 NMR（raw 中另有 MG、R2M 挖掘配对的结果）。
最后一列是试点场景 KS-IN-OE-S-F1_01_202609121414 上 `oracle`（扫描原始 3DGS，作为上界参考）的值。

公式、模型权重、与原 benchmark 的差异：`evaluate1/docs/wbench_ops.md`、`evaluate1/docs/r2m_revisit_ops.md`。

### A. 画面质量（WBench V）

| 算子 | 含义 | 方向 | 单位 | 输入 | 试点 oracle |
|---|---|---|---|---|---|
| `wbench.V1_aesthetic_quality` | 美学质量：LAION aesthetic（CLIP ViT-L/14），2 fps 取帧平均，/10 | ↑ | — | frames | 60.33 |
| `wbench.V2_imaging_quality` | 成像质量：MUSIQ，长边缩到 512，2 fps 平均，/100 | ↑ | — | frames | 74.86 |
| `wbench.V3_temporal_flickering` | 时间闪烁：10 fps 相邻帧平均绝对差，(255−d)/255，越高越不闪 | ↑ | — | frames | 87.78 |
| `wbench.V4_dynamic_degree` | 动态程度：RAFT 光流 top-5% 幅值超过阈值的连续帧对是否足够（0/1） | ↑ | — | frames | 100 |
| `wbench.V5_motion_smoothness` | 运动平滑：AMT 用 I(t−1)、I(t+1) 插出中间帧，与真实 I(t) 的误差 | ↑ | — | frames | 95.37 |
| `wbench.V6_hpsv3_norm` | 人类偏好：HPSv3 奖励（20 帧均匀采样、空提示词），按 p1–p99 百分位归一 | ↑ | — | frames | 89.12 |

### B. 输入帧保真（10 帧真值）

| 算子 | 含义 | 方向 | 单位 | 输入 | 试点 oracle |
|---|---|---|---|---|---|
| `fidelity.brisque` | 10 个输入位姿上渲染帧的无参考画质 BRISQUE（越低越好） | ↓ | — | frames, input_index, input_gt | 41.75 |
| `fidelity.dinov2` | 10 个输入位姿上，渲染 vs 真值的 DINOv2 全局特征余弦相似度 | ↑ | — | frames, input_index, input_gt | 0.9944 |
| `fidelity.dists` | 10 个输入位姿上，渲染 vs 真值的 DISTS 结构/纹理感知距离 | ↓ | — | frames, input_index, input_gt | 0.005705 |
| `fidelity.dreamsim` | 10 个输入位姿上，渲染 vs 真值的 DreamSim 感知距离 | ↓ | — | frames, input_index, input_gt | 0.003067 |
| `fidelity.lpips_alex` | 10 个输入位姿上，渲染 vs 真值的 LPIPS（AlexNet）感知距离 | ↓ | — | frames, input_index, input_gt | 0.01446 |
| `fidelity.lpips_vgg` | 10 个输入位姿上，渲染 vs 真值的 LPIPS（VGG）感知距离 | ↓ | — | frames, input_index, input_gt | 0.02352 |
| `fidelity.ms_ssim` | 10 个输入位姿上，渲染 vs 真值的多尺度 SSIM | ↑ | — | frames, input_index, input_gt | 0.9884 |
| `fidelity.niqe` | 10 个输入位姿上渲染帧的无参考画质 NIQE（越低越好） | ↓ | — | frames, input_index, input_gt | 4.037 |
| `fidelity.psnr` | 10 个输入位姿上，模型渲染 vs 输入真值帧的 PSNR | ↑ | dB | frames, input_index, input_gt | 32.29 |
| `fidelity.ssim` | 10 个输入位姿上，渲染 vs 输入真值帧的 SSIM | ↑ | — | frames, input_index, input_gt | 0.984 |

### C. 设定遵循（WBench S）

| 算子 | 含义 | 方向 | 单位 | 输入 | 试点 oracle |
|---|---|---|---|---|---|
| `wbench.S1_scene_adherence` | 场景遵循：GPT 看第 1 段帧，评可见场景元素是否保持(1–5)、画外元素是否出现(0/1)，用 case 的 visible/offscreen 描述 | ↑ | — | frames, turns | n/a |
| `wbench.S2_subject_adherence` | 主体遵循：主体外观保持 + 动作符合描述（需要主体，静态场景 n/a） | ↑ | — | frames, turns | n/a |

### D. 交互遵循（WBench I）

| 算子 | 含义 | 方向 | 单位 | 输入 | 试点 oracle |
|---|---|---|---|---|---|
| `wbench.I1_navigation_score` | 导航分：MegaSaM 从画面估计相机位姿，与每个动作的指令轨迹比 nATE（对齐+弧长重采样），再加重复动作的跨轮一致性 | ↑ | — | frames, keypoints | 98.06 |
| `wbench.I2_event_editing_adherence` | 事件编辑遵循：每个事件 turn 5 项递进判定（需要事件 turn，静态场景 n/a） | ↑ | — | frames, turns | n/a |
| `wbench.I3_subject_action_adherence` | 主体动作遵循：每个主体动作 turn 5 项递进判定（需要主体，n/a） | ↑ | — | frames, turns | n/a |
| `wbench.I4_perspective_switching_adherence` | 视角切换遵循：转场可见、目标类型一致、新视角结构合规三项全过的比例（需要视角切换 turn，n/a） | ↑ | — | frames, turns | n/a |

### E. 时空一致性（WBench C）

| 算子 | 含义 | 方向 | 单位 | 输入 | 试点 oracle |
|---|---|---|---|---|---|
| `wbench.C1_spatial_consistency` | 空间一致：每个关键点出发帧 vs 返回帧（MegaSaM 选旋转最接近的帧）的 DreamSim 相似度 | ↑ | — | frames, keypoints | 99.5 |
| `wbench.C2_gated_spatial_consistency` | 门控空间一致：C1 × 门控（往返途中画面确实变化过，防止原地不动拿高分） | ↑ | — | frames, keypoints | 96.91 |
| `wbench.C3_segment_continuity` | 片段连续：TransNetV2 检测不到硬切为 1 | ↑ | — | frames | 100 |
| `wbench.C4_perspective_consistency` | 视角一致：SAM2 跟踪主体，质心在画面中的稳定度（需要主体，n/a） | ↑ | — | frames | n/a |
| `wbench.C5_geometric_consistency` | 几何一致：DA3 每帧深度点重投影到其它帧的相对深度误差 | ↑ | — | frames | 97.11 |
| `wbench.C6_photometric_consistency` | 光度一致：DA3 点云颜色重投影到其它帧后的 PSNR，1−10^(−PSNR/20) | ↑ | — | frames | 79.3 |
| `wbench.C7_subject_consistency` | 主体一致：主体裁剪的 DINOv2 相邻帧 + CLIP 首帧锚定相似度（需要主体，n/a） | ↑ | — | frames | n/a |
| `wbench.C8_background_consistency` | 背景一致：CLIP ViT-B/32 相邻帧与首帧相似度（VBench 定义） | ↑ | — | frames | 92.57 |

### F. 物理合理性（WBench P）

| 算子 | 含义 | 方向 | 单位 | 输入 | 试点 oracle |
|---|---|---|---|---|---|
| `wbench.P1_causal_fidelity` | 因果合理性：GPT 两阶段物理/因果评分（静态场景无物理维度时只用阶段 1） | ↑ | — | frames | n/a |
| `wbench.P2_visual_plausibility` | 视觉合理性：PAVRM（Qwen3-VL-30B 微调）1–5 分；权重未就绪时用 gpt-6.1-sol 代评并标记 PROXY | ↑ | — | frames | n/a |

### G. 关键点重访（本项目：位姿 / 画面 / 物体 / 往返镜像）

| 算子 | 含义 | 方向 | 单位 | 输入 | 试点 oracle |
|---|---|---|---|---|---|
| `revisit.excursion.dinov2` | 往返镜像：动作 k 之后与其反向动作之前的同位姿帧 DINOv2 相似度 | ↑ | — | frames, keypoints, c2w_cmd | 1 |
| `revisit.excursion.lpips` | 往返镜像：动作 k 之后与其反向动作之前的同位姿帧 LPIPS | ↓ | — | frames, keypoints, c2w_cmd | 0 |
| `revisit.excursion.pixel_disp_px` | 往返镜像：动作 k 之后与其反向动作之前的同位姿帧匹配点中位像素位移 | ↓ | px | frames, keypoints, c2w_cmd | 0 |
| `revisit.excursion.psnr` | 往返镜像：动作 k 之后与其反向动作之前的同位姿帧 PSNR | ↑ | dB | frames, keypoints, c2w_cmd | 100 |
| `revisit.image.dinov2` | 画面：出发帧 vs 返回帧 DINOv2 余弦相似度 | ↑ | — | frames, keypoints | 1 |
| `revisit.image.dreamsim` | 画面：出发帧 vs 返回帧 DreamSim 距离 | ↓ | — | frames, keypoints | -1.987e-08 |
| `revisit.image.lpips` | 画面：出发帧 vs 返回帧 LPIPS | ↓ | — | frames, keypoints | 0 |
| `revisit.image.psnr` | 画面：出发帧 vs 返回帧 PSNR（完全一致封顶 100） | ↑ | dB | frames, keypoints | 100 |
| `revisit.image.ssim` | 画面：出发帧 vs 返回帧 SSIM | ↑ | — | frames, keypoints | 1 |
| `revisit.object.agnostic_iou` | 物体：不依赖标签，SAM2 自动 mask 匈牙利匹配后的面积加权 IoU | ↑ | IoU | frames, keypoints | 1 |
| `revisit.object.area_consistency` | 物体：匹配物体 mask 面积比 min/max | ↑ | ratio | frames, keypoints, objects | 1 |
| `revisit.object.centroid_shift_px` | 物体：匹配物体 mask 质心位移（像素） | ↓ | px | frames, keypoints, objects | 0 |
| `revisit.object.mask_iou` | 物体：标注物体在返回帧按类别重分割（GroundingDINO+SAM2）与标注 mask 的 IoU | ↑ | IoU | frames, keypoints, objects | 0.4161 |
| `revisit.object.pair_iou` | 物体：同一检测器在出发帧/返回帧对同一物体的 mask IoU（理想 1） | ↑ | IoU | frames, keypoints, objects | 1 |
| `revisit.object.precision` | 物体：返回帧检出的同类物体中，与出发帧位置吻合的比例 | ↑ | ratio | frames, keypoints, objects | 1 |
| `revisit.object.recall` | 物体：出发帧能检出的标注物体，返回帧仍在原位（IoU≥0.5）的比例 | ↑ | ratio | frames, keypoints, objects | 1 |
| `revisit.pose.cmd_rot_err_deg` | 自检：指令轨迹中出发/返回的旋转差（动作程序闭合时为 0） | ↓ | deg | keypoints, c2w_cmd | 0 |
| `revisit.pose.cmd_trans_err_m` | 自检：指令轨迹中出发/返回的相机中心距离（应为 0） | ↓ | m | keypoints, c2w_cmd | 0 |
| `revisit.pose.est_drift_rot_deg` | 估计轨迹漂移：MegaSaM 轨迹 Sim(3) 对齐到指令后，出发→返回的旋转漂移 | ↓ | deg | keypoints, c2w_est, c2w_cmd | 0.003417 |
| `revisit.pose.est_drift_trans_m` | 估计轨迹漂移：MegaSaM 轨迹 Sim(3) 对齐后，出发→返回的相机中心漂移（米） | ↓ | m | keypoints, c2w_est, c2w_cmd | 0.0001606 |
| `revisit.pose.match_ratio` | 位姿：SuperPoint+LightGlue 匹配率 \|M\|/min(Ka,Kb) | ↑ | ratio | frames, keypoints | 1 |
| `revisit.pose.pixel_disp_px` | 位姿：出发帧与返回帧匹配点的中位像素位移（精确重访为 0） | ↓ | px | frames, keypoints | 0 |
| `revisit.pose.rot_err_deg` | 位姿：从出发帧/返回帧画面估计的相对旋转，与指令（单位阵）的角度误差 | ↓ | deg | frames, keypoints, K | 2.012e-07 |
| `revisit.pose.trans_dir_err_deg` | 位姿：本质矩阵平移方向误差；只在指令基线非零时有定义（精确重访为 n/a） | ↓ | deg | frames, keypoints, K, c2w_cmd | n/a |
| `revisit.pose.trans_err_m` | 位姿：用出发帧深度做 PnP 求返回相机，与指令位置的平移误差（米，需要 depth） | ↓ | m | frames, keypoints, K, depth | 4.512e-08 |

### H. 重访记忆 R2M（revisit vs baseline/short → MG / NMR）

| 算子 | 含义 | 方向 | 单位 | 输入 | 试点 oracle |
|---|---|---|---|---|---|
| `r2m.appearance.lpips` | 外观保真：LPIPS（Alex）→ NMR | ↑ | NMR | frames, pairs | 4.998 |
| `r2m.appearance.psnr` | 外观保真：PSNR 在 revisit / gap-matched baseline / short 三组配对上的均值 → NMR | ↑ | NMR | frames, pairs | 17.37 |
| `r2m.appearance.ssim` | 外观保真：SSIM（7×7）→ NMR | ↑ | NMR | frames, pairs | 7.845 |
| `r2m.geometry.inlier_ratio` | 局部几何：RANSAC 基础矩阵内点率 → NMR | ↑ | NMR | frames, pairs | 1.711 |
| `r2m.geometry.match_ratio` | 局部几何：SuperPoint+LightGlue 匹配率 → NMR | ↑ | NMR | frames, pairs | 2.237 |
| `r2m.object_identity.appearance` | 物体身份：GroundingDINO+SAM2 物体裁剪的 DINOv2 外观持续性（消失记 0）→ NMR | ↑ | NMR | frames, pairs | 2.717 |
| `r2m.object_identity.appearance_similarity` | 物体身份：仅对仍存在物体的外观相似度 → NMR | ↑ | NMR | frames, pairs | 2.193 |
| `r2m.object_identity.existence` | 物体身份：首帧物体在配对帧中的存在率 → NMR | ↑ | NMR | frames, pairs | n/a |
| `r2m.object_identity.semantic` | 物体身份：GDINO 置信度 × CLIP 图文相似度的语义持续性 → NMR | ↑ | NMR | frames, pairs | 5.572 |
| `r2m.object_identity.semantic_similarity` | 物体身份：仅对仍存在物体的语义相似度 → NMR | ↑ | NMR | frames, pairs | 1.782 |
| `r2m.persistent_state.vlm` | 状态持续：GPT 按 R2M 量表判场景关系/状态是否持续（1–5）→ NMR | ↑ | NMR | frames, pairs | n/a |
| `r2m.scene_identity.boq` | 场景身份：BoQ 视觉位置识别描述子相似度 → NMR | ↑ | NMR | frames, pairs | 2.329 |
| `r2m.scene_identity.dinov2` | 场景身份：DINOv2-B/14 全局特征相似度 → NMR | ↑ | NMR | frames, pairs | 2.226 |
| `r2m.scene_identity.mutualvpr` | 场景身份：MutualVPR 描述子相似度 → NMR | ↑ | NMR | frames, pairs | 1.421 |
| `r2m.summary.nmr` | R2M 总分：五族各自取 NMR 均值后等权平均（不可用的族跳过） | ↑ | NMR | frames, pairs | 4.546 |
