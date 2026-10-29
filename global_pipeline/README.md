# Household 全局实例、逐帧投影和两级 GPT 描述

新增：当前帧 Level 1 的五档问答（识别、属性、相对位置、世界坐标、相机远近），见 [FRAME_QA.md](FRAME_QA.md)。每帧的已标注物体、global ID 和空间真值会直接交给 GPT 生成答案，网页 `/global` 展示逐题依据和真实请求证据。进度与结果见 `auto/FRAME_QA_SUMMARY.json`；原跨帧 Level 2 描述保留。

浏览入口：<http://10.0.0.184:18765/global>；本机入口：<http://127.0.0.1:18765/global>。`/` 为实时进度工作台，原基础场景图浏览位于 `/scenes`。新结果位于 `auto/<scene_id>/global_instances/`，原 v0.1 `scene_graph.json` 不被覆盖。启动和远程访问方法见 [网站说明](../services/ANNOTATION_SITE.md)。

## 数据流与坐标

`RGB/mesh depth/相机 → GPT 每帧类别审计 → SAM3 每帧独立 mask → superpoint → coarse → fine → global ID 重投影 → 跨视角语义投票 → PNG/视频/两级描述/QA`。

global ID 的唯一来源是 `fine_instances.npz:point_fine_ids`，不是 SAM3 的单帧局部编号。该数组第 i 项对应同文件 `points[i]`。`point_superpoint_ids` 和 `superpoint_fine_ids` 保留映射。ID 为场景内整数，0 表示 unknown；不同家庭的数字 ID 不表示同一物体。

坐标为 canonical，Z 向上。沿用 geometry analysis 的世界变换，不把楼层逐层归零。单位始终是 `inferred_meter`。相机使用 OpenCV +X 右/+Y 下/+Z 前；深度为相机 Z 距离、uint16 毫米、0 无效。投影采用像素中心约定。

## 轨迹和图片来源

复用 `outputs/` 中原有碰撞检查通过的 A* 平滑漫游轨迹，导出其中每一帧。为补足转向和视角覆盖，额外在每个可通行候选区域的中心用真实 3DGS 渲染 32 帧 360° 扫视，同时以同相机渲染 mesh 深度。镜头高为离地 1.5 个推断米制单位；扫视轻微向下 0.08 斜率。

所有原漫游与扫视各自是独立 clip，不连通区域和楼层之间没有虚构的过渡路径。场景总览 `projection_video.mp4` 将 clips 明确硬切串接；连续轨迹应使用 `videos/<clip>/`。本次是标注素材，不是评测 Q。

`frames.json` 是统一索引，包含图像、源深度、时间戳、K、camera_to_canonical、源视频和帧号。`frame_id_registry.json` 保留旧帧 ID；追加轨迹不会改已有帧 ID。

## GPT 与 SAM3

默认每帧都用 GPT 做类别审计，使用用户提供的完整库存提示词，返回类别、thing/stuff 和 1–3 个提示词。代码只做字段校验、排除零件/装饰类以及明确同义词/颜色材质变体的归一。相邻帧的 prompt 变体可以作为召回备用，当前帧是否存在该对象仍由 SAM3 检测确认。`concepts.json` 记录直接审计覆盖情况。

模型使用提供的网关 GPT 网关（`VLM_ENDPOINT` / `services/global_settings.local.json`） 和 `/mnt/cfs/cyh/auth.json` 中 `OPENAI_API_KEY`，默认 `gpt-6-sol`。不使用 Qwen。请求图像按网关体积限制压缩；小物体细节仍可能不足。原始请求提示词、图像哈希、实际模型、请求 ID、响应和解析结果保存在 evidence，不保存密钥。

网关限流或故障时，`services/global_settings.json` 中的 `gpt_request_model` 可指定另一个已验证的 GPT 路由。当前恢复任务使用已验证的 `gpt-6-sol` 路由；早期缓存中可能同时存在 `gpt-6-astra` 路由返回的 `gpt-6.1-sol` 响应。每次响应分别记录 requested/routed/actual model，答案的 source 使用实际模型。跨场景共享请求槽限制总并发，重试写入状态而不删除成功缓存。

SAM3 每帧只提取一次图像特征，按类别依序尝试参考校准 prompt、基础 prompt、备用 prompt。首个返回有效实例的 prompt 后停止该类别。没有人工 top-k 实例截断，允许多个独立实例。默认分数阈值 0.40、面积至少 80 像素，同类 mask IoU≥0.80 做 NMS。全图按分数从高到低分配唯一像素归属，重叠像素不能属于两个局部实例。

`masks/<clip>/<frame>.npz` 同时保留 `raw_masks` 和 `local_instances`；JSON 保存 NMS 前后检测、提示词尝试与分数。

**SAM3 文本 grounding 接口不暴露独立预测 mask IoU**。因此 `sam_iou` 记录 null，不能把检测分数冒充 IoU。本版保留真实分数并计算 `stability_score = area(p>0.55)/area(p>0.45)`。硬冲突质量要求稳定度≥0.90、检测分数≥0.80、点支持≥20、superpoint≥2；这是公开记录的替代条件。

参考中另外列出的 SAM 自动网格 `points_per_side/crop_n_layers/pred_iou_thresh` 属于另一类自动分割接口，本版执行的是用户指定的 GPT+SAM3 文本发现路径，没有宣称调用那套网格参数。

## 三维实例

- mesh 顶点在 canonical 中按 4 cm 体素聚合位置和颜色；构建 k=8、半径 0.13 的邻接图。边代价是空间距离 + 0.35×RGB 颜色距离。FH 阈值 k=0.14，超点最多 300 点、跨度最多 0.65，降低超大跨物体区域的风险。
- 投影先保留每像素最近的三维点，再检查 `abs(z-mesh_z) <= 0.06 + 0.02*mesh_z`。mask 支持至少 20 点；单个超点须有至少 2 点且占当前可见超点支持≥0.20。
- Proposal 共享超点 IoU≥0.25 或 IoMin≥0.65 建立正边。同帧高质量独立 mask、同帧空间分离 proposal 形成 cannot-link。合并检查整个 cluster 的冲突集，防止 A–B、B–C 传递合并越过 A–C 冲突。超点按质量和点支持加权投票生成 coarse ID。
- Fine 固定 coarse 实例拓扑，只允许相邻边界超点迁移。0.70 mask evidence + 0.30 邻居一致性，最低分 0.50、增益≥0.08、至少两个独立视角，且不能搬空源实例。独立视角按 0.25 位置栅格与 30° 朝向分箱，避免把近乎重复帧当作独立证据。
- Fine ID 在相同确定性点云上按加权点支持 IoU≥0.25 匹配旧版，未匹配者从 registry 最大值后发号，不回收旧号。拓扑变化仍可能导致新 ID，记录见 `global_id_matching.json`。

## 回投影和语义

每帧重新统计 fine global ID 在排他 2D mask 中的可见支持。支持≥20 且纯度≥0.65 才接受，整张 2D mask 的排他边界保持不变。若同帧两个高质量 thing mask 声称同一 global ID，仅保留支持更强的一个，其余为 unknown 并记录冲突；不修改 fine 拓扑。

跨帧语义权重为 `sqrt(mask_score * quality_measure) * instance_cover * log(1+support)`；当前 quality_measure 是上述稳定度代理。类别归一后累积票数，至少两个独立视角、winner/runner-up margin≥0.15 才赋语义，否则 unknown。没有强行填充黑色未知区域。

数字 `instance/*.png` 为 uint16 global ID；数字 `semantic/*.png` 为 uint16 semantic ID，词表见 `semantic_registry.json`。颜色图不可当数字标签使用。`labeled_points.ply` 保存坐标、实例颜色、global_id、semantic_id、superpoint_id，可直接导入点云查看器。

## 两级文字

`descriptions.json`：

- `level1`：每个实际输出帧一条，描述此帧可见物体、连续表面和空间布局，关联可见 global IDs。
- `level2`：同一 clip 的每一对连续帧一条，描述进入/离开视野、尺度、遮挡和视角变化。不跨轨迹切点、不跨层推断运动。

GPT 看到按时间顺序标帧号的 contact sheet，最初使用四帧批次；网关恢复阶段对尚未完成的帧使用最多 12 帧，继续复用有效的四帧响应。批次重叠或覆盖缺失的相邻对，确保所有相邻帧对都被同次查看。所有输出均有模型、score、证据、unreviewed 和 private 默认值。全场景是静态重建，提示词禁止把相机变化说成物体真实移动/门开关。

家庭 8 在网关恢复阶段对剩余审计使用每批最多四张图，逐图单独返回类别清单；每个帧 ID 都有对应原 RGB 哈希与真实批次响应路径。批次审计不把多张图的类别合并后冒充逐帧可见性。联系图压缩可能损失细节，原分辨率 RGB 独立保留。

## 文件结构

```text
auto/<scene>/global_instances/
  frames.json, frame_id_registry.json, trajectory.json
  frames/<clip>/gNNNNNN.jpg
  supplemental_trajectories/floor_N/sweep_rNN/{rgb.mp4,cameras.npz,mesh_depth_mm/,metadata.json}
  concepts.json, masks/<clip>/*.{json,npz}
  superpoints.{json,npz}, proposals.{json,npz}
  coarse_instances.npz, fine_instances.npz, labeled_points.ply
  coarse_summary.json, fine_summary.json, coverage.json
  global_instances.json, semantic_registry.json, global_id_registry.json
  frame_assignments.json, rendered_frames.json, descriptions.json
  rendered/<clip>/{instance,semantic,instance_color,semantic_color,sam2d,overlay}/
  videos/<clip>/{rgb,overlay,instance,semantic}.mp4
  projection_video.mp4, preview.jpg
  evidence/{gpt_inventory,gpt_descriptions}/
  qa/{hard_mask_conflicts,merge_conflicts,fine_boundary_migrations,reprojection_metrics,frame_stats,same_frame_global_id_collisions}.csv
  qa/reprojection_summary.json, qa/issues.json, validation.json, run_info.json
```

## 运行与验收

```bash
cd /mnt/cfs/cyh/cvpr_ws
bash services/start_global_workers.sh
# 实际新视角渲染：仅在未生成区域扫视时运行。
CUDA_VISIBLE_DEVICES=2 TORCH_CUDA_ARCH_LIST=8.0 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m global_pipeline.trajectory
# 两个场景并行执行，worker 组由 global_settings.json 指定；失败阶段缓存保留。
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m global_pipeline.batch
# 单场景/单阶段可重试，禁止同场景并发写入。
GLOBAL_VLM_MODEL=gpt-6-sol GLOBAL_SAM_URLS=http://127.0.0.1:18763,http://127.0.0.1:18764 python -m global_pipeline.run --scene household_1
python -m global_pipeline.run --scene household_1 --stage describe
python -m global_pipeline.run --scene household_1 --stage verify
# 强制更新派生结果，仍复用有效 API 和 SAM3 缓存。
python -m global_pipeline.run --scene household_1 --force
# 只恢复尚未完成的阶段；已导出视频的场景只补文字和验收。
python -m global_pipeline.resume --workers 4
python -m global_pipeline.report
python -m unittest global_pipeline.test_global -v
bash services/start_viewer.sh
```

SAM3 worker 仅监听 loopback，各自串行处理请求；GPT 是独立网络请求。`services/global_settings.json` 记录实际端口、GPU 和每场景 worker 组，也可调整 GPT 并发。当前启用 8 个 SAM3 worker，每场景使用 4 个，端口为 18763/18764/18766/18767/18768/18769/18770/18771。当前环境沿用 `.deps/sam3`、torch、gsplat、Open3D、SciPy、Numba，不修改共享环境版本。

每场景有文件锁；另一进程重复调度同场景时等待现有写入者完成，再复用已通过验收的结果，防止两个任务覆盖同一组产物。可在当前场景进入 CPU 几何阶段后，让空闲 SAM3 worker 处理另一场景。

验收覆盖全部帧的 uint16 标签、mask 边界、global ID 注册表、点数组对应关系、fine 拓扑保持、硬冲突、视频帧数、每帧直接 GPT 审计、两级描述完整性和真实 API 证据。全点覆盖和可见点覆盖分别报告；模型间重投影一致性不等于人工准确率。所有模型低置信度/混合纯度/语义票差不足都保留 unknown 和 QA 证据。

`qa/issues.json` 按优先级汇总未知语义、低投票置信度、少视角支持和物体观测几何不连通等复核线索。缺失 mesh 或遮挡也可能导致不连通，因此只标记，不自动拆分。3D 框仅包住已观测表面，可能存在 mask 泄漏或遮挡导致的范围误差，不视作人工精确框。

文字 QA 也会标记低分描述、Level 1 引用上一帧，以及视角左右措辞。保留 GPT 原始证据，不把这些措辞自动转换为世界坐标关系。`auto/GLOBAL_REPORT.md` 和 `global_summary.{json,csv}` 区分几何/视频验收与包含两级文字的全量验收；正式文字数量只来自 `descriptions.json`，运行中的缓存预览不计入正式完成数。

运行 `python services/make_global_gallery.py` 可生成 `services/global_all_scene_examples.jpg` 和对应帧索引 JSON。每场景选取已知 thing ID 较多的一帧作为浏览索引，不作为随机准确率样本。

后端 `/global` 可同步播放视频、逐帧浏览、显示两级描述，并切换 superpoint/coarse/fine/语义点云。原人工 review 和 final 目录不写入。
