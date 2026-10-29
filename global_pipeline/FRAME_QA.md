# Household Level 1 帧内五档问答

浏览入口：`http://127.0.0.1:18765/global`。继续使用原来的 SSH 转发即可。在“Level 1 · 当前帧五档问答”选择难度；每题下面可以查看 global ID、属性与空间标注，点击物体标签可定位点云。另有问答、事实、固定问题和真实 GPT 请求/响应下载。

五档都属于 `annotation_level: 1`，`temporal_scope: single_frame`。原来的 `descriptions.json:level2` 仍指相邻帧变化，与难度 2 不是一回事。

| difficulty_level | 内容 | 依据 |
| --- | --- | --- |
| 1 | 当前可见物体类别及连续表面 | 该帧已接受的 mask→global ID 与跨帧语义 |
| 2 | 识别三件主要对象，再描述颜色和可见几何特征 | 当前 RGB 内该物体排他 mask 的颜色、轮廓范围 |
| 3 | 物体特征、参照对象、相对位置 | 图像中心左右/上下，另附 canonical 中心差量及三维距离 |
| 4 | 物体特征、世界坐标和绝对位置 | canonical Z-up 世界坐标中的观测表面框中心及范围 |
| 5 | 先识别两个独立物体，再比较相机远近 | 当前相机到深度一致可见表面点的欧氏距离中位数 |

问题是 GPT 生成的固定问句，不把具体对象名称、物体 ID、坐标或距离答案写进问题。答案生成时直接提供该帧的已标注对象、ID、颜色、三维位置、相机位姿、相对关系、距离及可核验事实底稿，要求 GPT 依据这些事实组织完整中文回答，无需重新猜物体。不使用 Qwen。

固定问题全文见 [FRAME_QA_PROMPTS.md](FRAME_QA_PROMPTS.md)。

## 数据与来源

```text
auto/<scene>/global_instances/frame_qa/
  facts/gNNNNNN.json             # 当前 RGB/mask/投影测得的逐帧事实
  frames/gNNNNNN.json            # 五档问题、答案和 answer_annotations
  prompts.json                  # 五档细则及早期 GPT 回答模板
  stable_questions.json         # 正式五档固定问题
  glossary.json                 # GPT 类别翻译
  evidence/stable_questions.json # 实际 GPT 问题请求和响应
  evidence/gpt_answers/*.json    # 逐帧真值输入、实际模型、请求 ID、响应
  manifest.json                 # 帧索引和 GPT 回答覆盖进度
  gpt_run_info.json, run_info.json, validation.json
```

每条问答保存：`id`、`difficulty_level`、`question`、`answer`、`answer_status`、`score`、`source`、`review=unreviewed`。`answer_annotations` 另存场景内 `global_ids`、带场景前缀的 `object_ids`、类别、颜色来源、图像框及相应三维/相机信息，避免靠自然语言重新解析真值。

实际 GPT 答案标记 `generation_mode=gpt_answer_from_supplied_frame_ground_truth`，保留 `answer_reference` 事实底稿、`evidence` 和 `evidence_sha256`。还未发送 GPT 的事实参考答案标记为 `gpt_template_instantiation_not_independent_image_inference`，网页也明确显示“等待 GPT 回答”。不能把两类统计混为 GPT 已完成数量。

原始输入 `frames.json`、`fine_instances.npz`、`global_instances.json`、`frame_assignments.json`、场景图、原两级描述和人工审核文件均不写回。global ID 仍来自 `fine_instances.npz:point_fine_ids`，不重新编号。

## 坐标、属性与不确定性

canonical 的 Z 朝上；X/Y 不代表已测量的东西南北，单位为 `inferred_meter`。第 4 档应向被评模型提供世界参考/三维数据，单张 RGB 不足以恢复绝对坐标。第 5 档的数值回答需要深度、相机标定或对应三维信息；距离是相机到当前可见表面的统计量，不等于完整物体中心距离、导航路径距离或世界 Z。

颜色是渲染 RGB 中当前排他 mask 的 HSV 统计，有光照与分割误差；不据此补造材质。轮廓覆盖画面比例与观测表面框不代表完整实物尺寸。初始全局实例可能存在分片，因此盘点类别而不把分片数当精确实物数量。没有足够对象、有效三维点或远近差异的题保留明确拒答；这些条目不应直接作为可回答 benchmark 题纳入。

发现的旧帧清单 RGB 哈希不符单独记录 `source_integrity` 和 `validation.warnings`，保存实际 RGB 字节哈希并显示来源警告。原清单不被静默修改。校验通过表示来源、ID、数字、覆盖和响应对应正确，不等于人工确认语义及物理尺度正确。

## 复现与恢复

```bash
# 整理 9 个场景的测量事实和可核验参考答案，成功 GPT 词表缓存可复用
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -m global_pipeline.frame_qa --workers 4

# 将每一帧的真值直接交给 GPT，逐帧生成五档自然语言答案，断点续跑
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -m global_pipeline.frame_qa_grounded --workers 4 --batch-size 4

# 也可以用 --scene household_1 限定场景，--limit 4 做 pilot
python -m global_pipeline.frame_qa --stage verify
python -m unittest global_pipeline.test_frame_qa -v
```

网关及密钥复用 `annotation_pipeline.vlm.VLM`：GPT 网关（`VLM_ENDPOINT` / `services/global_settings.local.json`），当前认证读取 `/root/workspace/clash/codex-local/home/auth.json`。路径通过 `services/global_settings.json:vlm_auth_file` 配置，或由 `VLM_AUTH_FILE` 环境变量覆盖；实际路由遵循同一设置文件。请求与证据不写入密钥。成功批次按真值内容指纹缓存，续跑会重新核对事实和 GPT 响应哈希；批次中通过校验的帧分别写入，只重新请求不合格帧。`--recover-cache-only` 可恢复已有响应中独立通过校验的帧，不调用 API。永久 HTTP 拒绝会记录网关原因并停止后续批次；超时、限流等暂时错误按策略重试。共享并发保持 4。

`python services/resume_frame_qa.py` 负责续跑、有限次失败重试和全量验收。若认证失败，它等待认证文件发生变化再探测；若网关给出下次额度补充时间，它等待到该时间再探测，不持续提交失败请求。状态记录在 `services/frame_qa_completion.json`。

HTTP：

- `GET /api/scenes/<scene>/global/frame_qa`：索引与实际 GPT 回答进度。
- `GET /api/scenes/<scene>/global/frames/<frame>/qa`：该帧五档问答、来源和下载。
- 原逐帧 `.../global/frames/<frame>` 也附带 `frame_qa`，原 API 字段继续保留。

全局汇总见 `auto/FRAME_QA_SUMMARY.json` 和 `auto/FRAME_QA_REPORT.md`。这次是额外的帧内标注层，原 VLN 任务与评测分母保持独立；需审核后再选取题目升级 benchmark。
