# anno_eval — 室内扫描标注 + 4DGS 漫游重访评测（交付说明）

本目录把**数据处理（标注）**和**评测（evaluate1）**合在一起，是一个可以独立运行的工作区。原始数据只读引用，不复制。

- **最终算子**：72 个，入口 [`operators.py`](operators.py)，清单见 [`OPERATORS.md`](OPERATORS.md)。
- **试点场景**：`KS-IN-OE-S-F1_01_202609121414`（图书阅览室，办公或教育空间 / 教育科研）。结果在 [`results/`](results/)。

```
anno_eval/
├── README.md  OPERATORS.md  operators.py      交付说明 / 最终评测算子清单 / 评测算子统一入口
├── ops3d.py                  管线算子（上游、3D 几何、轨迹、3D 标注、评测）统一接口：list / run / describe
├── docs/pipeline_flow.html   流程图网页（由 tools/make_flow_page.py 从 ops3d 与 operators 注册表生成）
├── hesai/                    hesai 扫描适配：stage.py（几何+漫游对齐）、annotate.sh（标注）、package.py（场景清单）
├── scene_pipeline/           几何：调平、楼层、可通行图、区域、全景、俯视图；A* 漫游（仅当扫描没有自带轨迹）
├── global_pipeline/          标注：GPT 类别清单 → SAM3 → 3D 实例 → 回投影 → unknown 重标/碎片剔除 → 描述 → 五档问答
├── annotation_pipeline/      SAM3 服务、GPT 网关客户端（共用）
├── evaluate1/                评测：case.py（协议）、render.py（参考渲染）、run.py（评测）、metrics/（72 个算子）、docs/、tests/
├── tools/                    内网 GPU 机器运行脚本（remote_exec / remote_launch / gpu_wait）
├── services/                 global_settings.json（gpt-6.1-sol、并发）
├── results/<scene>/          试点交付：场景清单、标注摘要、评测 case 摘要、oracle 分数
└── 数据目录                  hesai_dataset（原始扫描，链接）、sources、outputs、auto、scenes、evaluate1/cases、evaluate1/runs
```

## 一、数据处理（标注）

目的：把 hesai 扫描（gaussian.ply + mesh_post.obj + 漫游视频）变成和 household 同一套标准的标注，给 VLN、世界模型评测、问答等下游任务直接用。

```
hesai_dataset/<...>/<scene>/  ──hesai/stage.py──▶ outputs/<key>/  ──hesai/annotate.sh──▶ auto/<key>/global_instances/
   原始扫描                      几何 + 对齐的漫游 clip              逐帧 mask、3D 实例（可选：描述、问答）
```

标注流程（`hesai/annotate.sh`）就三步：

1. **SAM3 标注**：GPT 逐帧给出类别清单，作为 SAM3 的提示词 → SAM3 出 mask → 融合成 3D 实例 → 多视角投票定类别。
2. **unknown 过一遍 GPT**：投票后仍是 unknown 的实例，交给 gpt-6.1-sol 标一次。
3. **剔除碎片**：任何一帧都看不到的实例直接放弃，不标注。

逐帧描述和五档问答是可选的文字层，设 `WITH_TEXT=1` 才会生成。试点场景已经生成过。

**轨迹规则**：扫描自带前进漫游（`manyou_videos/main_walkthrough` 里有 poses.json 和 frames_hd）就直接用，不再生成 A*；没有才在可通行图上用 A* 规划区域漫游并渲染。标注帧和评测轨迹都来自同一条轨迹。90 个 hesai 场景里有 86 个自带漫游，4 个需要 A*（`KS-IN-RS-S-F2_00_202609141608`、`KS-IN-CE-S-F1_01_202610041743`、`KS-IN-CE-S-F1_01_202610061320`、`KS-IN-OE-S-F1_03_202610021436`，最后一个只有改名的 dataaware 版本）。

| 层 | 产物 | 说明 |
|---|---|---|
| 几何 | `outputs/<key>/analysis.json`、`floor_*/maps.npz`、`regions.geojson`、`panoramas/`、`topdown_3dgs.png` | 原始数据 −Y 朝上，调平成 Z 朝上；0.05 m 可通行栅格（半径 0.25 m）；区域候选；全景；俯视图 |
| 轨迹 | `outputs/<key>/hesai_walkthrough/main/`，没有自带轨迹时为 `floor_*/annotation_walkthrough/component_*/` | 自带漫游（试点 205 帧），附 canonical 位姿、内参、mesh 深度，3DGS 重渲染校验 PSNR 中位数 26.4 dB；A* 漫游附 3DGS 渲染、深度、相机 |
| 逐帧标注 | `auto/<key>/global_instances/rendered/*/{instance,semantic}/` | uint16 实例/语义 PNG（0 = 未知），附叠加图、视频 |
| 3D 实例 | `global_instances.json`、`fine_instances.npz`、`labeled_points.ply` | global ID、类别（含原始细粒度名 `fine_label`）、观测表面 AABB、最佳视角 |
| 文字 | `descriptions.json`、`frame_qa/` | 每帧描述 + 相邻帧变化描述；单帧五档问答（识别、属性、相对位置、世界坐标、相机远近） |
| 证据 | `evidence/`、`qa/` | 每次 GPT 请求/响应（含图像哈希、实际模型，不含密钥）；自动复核线索 |

相对 household 流水线的改动：

1. **楼层**：在漫游相机正下方实测地面高度。和自动选出的最大平面相差超过 8 cm 时，按实测高度重建可通行图（试点场景就是这种情况：阅览室比相邻区域高 16 cm，原来整间屋子被当成障碍）。
2. **标注帧 / 轨迹**：有自带漫游就用它，和评测轨迹逐帧对齐；没有才用 A* 规划的漫游。
3. **类别归一**：多视角投票前先合并同义类别名，原名保留在 `fine_label`。
4. **unknown 后处理**：投票后仍为 unknown 的实例，在原帧和放大图上画出轮廓，交给 gpt-6.1-sol 判断（可见、是单个物体、置信度 ≥ 0.6 才采纳）。
5. **碎片剔除**：任何一帧都看不见的实例（没有被接受的 mask，且每帧可见点都少于 30）**不标注**，直接剔除。它们的点在 fine/coarse 数组里都清零，剔除记录在 `global_instances.json:dropped_instances`。
6. 全部 GPT 调用使用 **gpt-6.1-sol**。SAM3 只在分割阶段占卡。

试点场景的结果：

- 205 帧，SAM3 共 12,598 个 mask。
- 3D 实例 113 个，其中 112 个有类别：多视角投票 87 个，GPT 补标 25 个。剩 1 个 unknown（轮廓里是好几本书）。
- 剔除不可见碎片 24 个。
- 描述 205 + 204 条，五档问答 1025 条。
- 自动校验全部通过。按惯例，所有标注的审核状态都是 `unreviewed`。

## 二、评测 evaluate1

借鉴 [WBench](https://github.com/meituan-longcat/WBench)（22 个自动子指标）和 [R2M-Bench](https://github.com/AMAP-ML/R2MBench)（重访记忆），加上本项目的关键点重访指标。模型输出是 4DGS，所以评测方式是**按给定相机轨迹和时间渲染 4DGS 再打分**。

### 协议（`python -m evaluate1.case --scene <key>`）

- **轨迹**：场景自带的前进漫游，每帧一步（12 fps）；没有自带轨迹的场景用最长的一段 A* 漫游（8 fps）。可用 `--trajectory auto|provided|astar` 指定。
- **输入 10 帧**：漫游上均匀取 10 帧（位姿 + 画面），模型在这 10 个位姿上的渲染和真值比画面质量与保真度。
- **重访 6 个关键点**：沿漫游均匀选取，优先选可见标注物体多、平移不撞墙的位置。每个关键点的流程是：
  1. 暂停 6 步。
  2. 执行 3 个动作（WBench 词表：W/S/A/D 平移 0.4 m，LEFT/RIGHT 偏航 30°，UP/DOWN 俯仰 15°，每个动作 12 步）。
  3. 按相反顺序执行 3 个反向动作，回到关键点的精确位姿（返回帧）。
  4. 继续沿漫游前进。

  6 个关键点的动作模板合起来覆盖全部 8 个动作。平移会先经过 mesh 射线和可通行图检查。
- **配对**：每个关键点的出发帧/返回帧是一对重访。R2M 规则另外挖 revisit / gap-matched baseline / short 三组配对，用来算 MemoryGain / NMR。
- **case 内容**：`case.json`（参数、关键点、WBench 式 turns、GPT 场景描述、关键点处的标注物体 mask）、`trajectory.json`、`cameras.npz`（`c2w_source` / `c2w_canonical`、`K`、`t`）、`input/`。试点 case 共 679 步，带 313 个物体 mask。

### 模型怎么接

按 `cameras.npz` 的每一步（位姿 + 时间 `t`）渲染 4DGS，写到 `evaluate1/runs/<key>/<model>/frames/<step:05d>.png`。深度可选，放在 `depth/<step:05d>.npy`。自带参考：`python -m evaluate1.render --case ... --model oracle`（扫描原始 3DGS，作为上界）或 `--model ply:<path>`。

### 打分

```bash
python operators.py list                                         # 72 个算子
python -m evaluate1.run --case evaluate1/cases/<key> --frames evaluate1/runs/<key>/<model>/frames
python -m evaluate1.run ... --vlm-only --merge                   # 只补跑 GPT 裁判类并合并
python -m evaluate1.run --case evaluate1/cases/<key> --report    # 多模型对比表
```

- 每个算子返回 `ok` / `n/a`（case 不具备所需输入，例如静态场景没有主体）/ `error`（附 traceback），单个算子出问题不会中断整次评测。
- 先跑 WBench：它的 MegaSaM 位姿会作为 `c2w_est` 交给重访漂移算子。
- 算子分组：A 画面质量 · B 输入帧保真 · C 设定遵循 · D 交互遵循 · E 时空一致性 · F 物理合理性 · G 关键点重访 · H R2M 重访记忆。详见 `OPERATORS.md`。

## 三、运行环境

- GPU 任务跑在内网机器上，用 `tools/remote_launch.sh <host:port> <gpu> <name> <cmd>`（CFS 上的 Python 环境，放在私有 mount namespace 里，不改机器本身）。
  - 内网 GPU 机器的卡是 Exclusive_Process 模式，一张卡同时只能跑一个进程，`tools/gpu_wait.sh` 会等卡空出来再启动。SSH 密钥等本机设置放在 `tools/local.env`（不入库，示例见 `tools/local.env.example`）。
- 评测依赖的位置：
  - 额外 Python 包：`/mnt/cfs/cyh/tmp/data_pipeline/pylib_{wbench,r2m,megasam}`
  - 权重：`/mnt/cfs/cyh/tmp/data_pipeline/weights/`（运行时离线读取；部分是指向本机已有 WBench/MegaSaM 权重的链接）
  - 第三方代码：`/mnt/cfs/cyh/tmp/data_pipeline/third_party/`（与 WBench、R2M 固定版本一致）
- GPT：GPT 网关（地址由环境变量 `VLM_ENDPOINT` 或本地文件 `services/global_settings.local.json` 提供，不入库），密钥从 `/mnt/cfs/cyh/auth.json` 读取，不落盘。

## 四、已知限制

- 标注全部是自动生成，状态都是 `unreviewed`。米制单位是推断值。3D 框只覆盖观测到的表面。
- C5/C6 暂时用 DA3-SMALL：DA3-GIANT 权重没下载完。
- P2 暂时用 gpt-6.1-sol 代评（结果标记 PROXY）：PAVRM 的 62 GB 权重没下载完。
- R2M 的 NMR 动态范围小时数值噪声较大，见 `evaluate1/docs/r2m_revisit_ops.md`。
- GPT 裁判类算子（S1、S2、I2、I3、I4、P1、P2、r2m.persistent_state.vlm）已实现，但本次试点没有用 GPT 打分：S1、P1、P2 记为“未运行”，其余几个本来就因为静态场景没有主体/事件而是 n/a。需要时用 `python -m evaluate1.run ... --vlm-only --merge` 补跑。
- 静态场景下，WBench 的主体、事件、视角切换类算子（C4、C7、S2、I2、I3、I4）为 n/a。它们已经完整实现，并在构造的输入上测试通过。
