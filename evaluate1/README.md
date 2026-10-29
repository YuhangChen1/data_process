# evaluate1 — 4DGS 漫游 + 重访评测

借鉴 [WBench](https://github.com/meituan-longcat/WBench)（22 个自动子指标）和 R2M-Bench（[AMAP-ML/R2MBench](https://github.com/AMAP-ML/R2MBench)，重访记忆）。我们的模型输出是 4DGS，所以评测不让模型“生成视频”，而是**按给定相机轨迹和时间渲染 4DGS**，再对渲染结果算指标。

## 协议（case）

```bash
python -m evaluate1.case --scene KS-IN-OE-S-F1_01_202609121414     # 生成 cases/<scene>/
```

- **轨迹**：场景自带的前进漫游视频（hesai `manyou_videos/main_walkthrough`，205 帧，12 fps）。标注流水线标的也是这条视频，所以每一帧都有实例/语义标签。
- **输入 10 帧**：漫游上均匀取 10 帧，给出相机位姿和画面（`input/`）。模型在这 10 个位姿上的渲染要和真值比画面质量与保真度。
- **重访**：沿漫游均匀选 6 个关键点。每个关键点的流程是：
  1. 暂停 6 帧。
  2. 执行 3 个动作（WBench 导航词表：`W/S/A/D` 平移 0.4 m，`LEFT/RIGHT` 偏航 30°，`UP/DOWN` 俯仰 15°；每个动作 12 帧）。
  3. 按相反顺序执行这 3 个动作的反向动作，相机精确回到关键点位姿。
  4. 继续沿漫游前进。

  6 个关键点的动作模板合起来覆盖全部 8 个动作。平移会先用 mesh 射线和可通行地图检查，不会穿墙。
- **配对**：
  - 每个关键点的出发帧（`visit`）和返回帧（`return`）构成一对重访（`pairs.keypoint_revisit`）。
  - 运行时另外按 R2M 规则挖 revisit / gap-matched baseline / short 三组配对，用来算 MemoryGain / NMR。
- **文件**：

| 文件 | 内容 |
|---|---|
| `case.json` | 参数、输入帧、关键点与动作程序、WBench 式 turns、场景文字（gpt-6.1-sol 生成）、关键点处的标注物体、配对 |
| `trajectory.json` | 每一步的类型（walk / pause / action / inverse / return）、动作、时间 t、`camera_to_source`、`camera_to_canonical`、相对上一步的位姿 |
| `cameras.npz` | `c2w_source`、`c2w_canonical` [T,4,4]（OpenCV 相机：+X 右、+Y 下、+Z 前）、`K`、`t`、宽高 |
| `objects/` | 关键点出发帧上标注物体的二值 mask（来自 global instance 标注） |

## 模型怎么接

模型拿到输入视频（或那 10 帧）后，按 `cameras.npz` 的每一步（位姿 + 时间 `t`）渲染自己的 4DGS，输出：

```
runs/<scene>/<model>/frames/<step:05d>.png       # 每一步一张，共 T 张
runs/<scene>/<model>/depth/<step:05d>.npy        # 可选：相机 Z 深度（米），任意子集
```

`render.py` 自带参考模型 `oracle`：扫描原始 3DGS，静态，重访完全一致，作为上界；`ply:<path>` 渲染任意静态高斯 ply 作为基线。

```bash
python -m evaluate1.render --case evaluate1/cases/<scene> --model oracle
python -m evaluate1.render --case evaluate1/cases/<scene> --model ply:/path/model.ply --frame source
```

## 评测

```bash
python -m evaluate1.run --case evaluate1/cases/<scene> --frames evaluate1/runs/<scene>/<model>/frames
python -m evaluate1.run --case evaluate1/cases/<scene> --report      # 多模型对比表 runs/<scene>/report.md
```

- `metrics/` 下每个模块注册若干算子（`metrics/base.py` 的 `@operator`），runner 全部执行，不筛选。
- 某个算子当前 case 用不上时（比如静态场景没有主体），记为 `n/a` 并写明原因；出错记为 `error` 并保留 traceback。都不会中断整次运行。
- VLM 裁判统一用 gpt-6.1-sol（GPT 网关（地址由环境变量 `VLM_ENDPOINT` 或本地文件 `services/global_settings.local.json` 提供，不入库），密钥读 `/mnt/cfs/cyh/auth.json`，不落盘）。

全部 72 个算子的清单见上一级的 `OPERATORS.md`，统一入口是 `operators.py`。公式和与原 benchmark 的差异见 `docs/wbench_ops.md`（WBench 22 项）和 `docs/r2m_revisit_ops.md`（R2M 五大类 + 重访位姿/物体重合 + 输入帧保真度）。补跑 GPT 裁判类：`--vlm-only --merge`。

## 运行环境

- GPU 任务放在内网机器上跑（`tools/remote_exec.sh` / `tools/remote_launch.sh`，CFS 上的 Python 环境），机器本身不改动。
- 内网 GPU 机器的卡是 Exclusive_Process 模式，一张卡同时只能跑一个进程。
- 额外的 Python 包装在 `/mnt/cfs/cyh/tmp/data_pipeline/pylib_*`，权重放在 `/mnt/cfs/cyh/tmp/data_pipeline/weights/`。
