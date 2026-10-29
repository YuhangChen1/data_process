# WBench operators in evaluate1

The code is in `evaluate1/metrics/wbench_{quality,consistency,interaction,physics}.py`. It ports all 22 automatic sub-metrics of WBench (meituan-longcat/WBench, arXiv 2605.25874). None are filtered out.

- **Reference code:** `/mnt/cfs/cyh/tmp/refs/WBench-main`. Each operator's `source` field cites the file and lines it follows.
- **Third-party code:** pinned submodule commits copied from `/mnt/cfs/cyh/WBench/third_party` to `/mnt/cfs/cyh/tmp/data_pipeline/third_party/wbench`. MegaSaM's `droid_backends` and `lietorch_backends` are built for sm_80 inside that copy.
- **Weights:** the WBench-weights layout lives under `/mnt/cfs/cyh/tmp/data_pipeline/weights/wbench`. They are fetched from hf-mirror by `wbench_work/dl_curl.sh`, with resume and size checks.
- **Extra pip packages:** pyiqa 0.1.14.1, dreamsim 0.2.1, fire and trl 0.29.1, installed `--no-deps` into `/mnt/cfs/cyh/tmp/data_pipeline/pylib_wbench`.

## Common conventions

- **Score scale:** `value` is WBench's own score in [0,1] multiplied by 100, which is the paper's 0-100 scale. `raw.score01` keeps the [0,1] number.
- **Frame sampling:** WBench's per-metric fps (evaluator.py `FRAME_SAMPLING_CONFIG`), with the case fps (12) as the video fps.
- **Turns:** WBench "interactions" map to `ctx.turns`. Frame ranges are exact; WBench splits the video into equal-length turns.
- **Loop cases:** each revisit keypoint (visit → 3 actions → 3 inverse actions) is one WBench loop/navigation case. Results are averaged over keypoints.
- **GPU use:** all models run in-process. On GPUs in `Exclusive_Process` mode a subprocess cannot reach the GPU this process already holds, so MegaSaM's three scripts run through `runpy`.
- **VLM judge:** gpt-6.1-sol through the data_pipeline gateway client, cached under `<case>/vlm_cache/`.
  - WBench prompts are copied verbatim.
  - Frames are sent as numbered contact sheets, at most 3 sheets of 16 frames, because the gateway only accepts ~100 KiB requests.
  - Answers are requested as JSON, because the client demands one JSON object. WBench used a base64 video plus free text.

## Operators

| operator | measures | model / checkpoint | inputs | formula → value | deviation from WBench |
|---|---|---|---|---|---|
| `wbench.V1_aesthetic_quality` | aesthetics | CLIP ViT-L/14 + LAION `sa_0_4_vit_l_14_linear` | frames @2 fps | mean(head(norm f)/10) | none |
| `wbench.V2_imaging_quality` | technical quality | pyiqa MUSIQ-KonIQ | frames @2 fps, long edge ≤512 | mean(MUSIQ/100) | pyiqa 0.1.14.1 |
| `wbench.V3_temporal_flickering` | flicker | none | frames @10 fps | (255 − mean \|Iₜ−Iₜ₊₁\|)/255 | none |
| `wbench.V4_dynamic_degree` | is there motion | RAFT-things | frames @8 fps, min side 512 | 1 if ≥ round(4N/16) pairs have top-5% flow > 6·s/256, else 0 | Farneback fallback, as in WBench |
| `wbench.V5_motion_smoothness` | interpolation error | AMT-S | frames @10 fps, min side 512 | (255 − mean \|AMT(i,i+2) − i+1\|)/255 | none |
| `wbench.V6_hpsv3_norm` | human preference | HPSv3 (Qwen2-VL-7B) | 20 uniform frames, empty prompt | clip((mean − (−5.21)) / (8.61 + 5.21), 0, 1) | none (needs 33 GB of weights) |
| `wbench.C1_spatial_consistency` | return-frame similarity | DreamSim + MegaSaM poses | per keypoint: visit frame vs best-rotation frame in the last inverse turn | mean 1/(1+DreamSim) | loop = keypoint excursion |
| `wbench.C2_gated_spatial_consistency` | C1 with a gate against static video | same | 10 samples from visit to return | ret_sim · min(1, (1 − min_sim)/0.15) | as C1 |
| `wbench.C3_segment_continuity` | hard cuts | TransNetV2 | all frames at 48×27 | 0 if any cut (p > 0.5, merge < 10 frames), else 1 | none |
| `wbench.C4_perspective_consistency` | subject centroid stability | SAM2.1-b+ video tracker (+ DA3 depth, reported only) | subject mask (`ctx.extra['subject_mask']`), 5 fps | max(0, 1 − c_std/0.3) · presence | needs a subject; n/a on static scenes |
| `wbench.C5_geometric_consistency` | cross-view depth agreement | DA3-GIANT-1.1 | frames @3 fps, all views jointly | 1/(1 + mean median \|z_reproj − d\|/d) | point subsampling is seeded |
| `wbench.C6_photometric_consistency` | cross-view colour agreement | DA3-GIANT-1.1 | same | 1 − 10^(−PSNR/20) (main.py:901) | as C5 |
| `wbench.C7_subject_consistency` | subject identity | SAM2 + DINOv2 ViT-B/14 + CLIP B/16 | masked crops, 5 fps | mean (DINO adjacent + CLIP vs first)/2 | needs a subject |
| `wbench.C8_background_consistency` | global stability | CLIP ViT-B/32 | frames @2 fps | mean (cos prev + cos first)/2 | none |
| `wbench.S1_scene_adherence` | scene kept, off-screen content appears | VLM | Turn-1 frames @2 fps + `case.setting` | (maintenance/5 + offscreen)/2 | judge, sheets, JSON |
| `wbench.S2_subject_adherence` | subject kept and acting | VLM | Turn-1 frames + subject text | (maintenance/5 + action)/2 | needs a subject |
| `wbench.I1_navigation_score` | camera follows the actions | MegaSaM (fallback DA3 per keypoint clip) | the 6 action/inverse turns of each keypoint | (Acc + Cons)/2; Acc = 1 − (nATE_t + nATE_r)/2, Cons from mirrored turn pairs | one case per keypoint; walk turns excluded like WBench's nav_mask |
| `wbench.I2_event_editing_adherence` | instructed event happens | VLM, 5 binary questions | event_edit turns @3 fps | mean of correct/5 | needs event_edit turns |
| `wbench.I3_subject_action_adherence` | subject performs the action | VLM, 5 binary questions | subject_action turns | mean of correct/5 | needs subject_action turns |
| `wbench.I4_perspective_switching_adherence` | viewpoint switch | VLM, Q1/Q3/Q4/Q_cam | early, late and full frame sets | fraction of turns with Q1 ∧ Q3 ∧ Q4 | needs perspective_switch turns |
| `wbench.P1_causal_fidelity` | physics and causality | VLM, two tracks | frames @3 fps | (T1 + mean T2)/2/3; T1 alone without annotated dims | judge, sheets (48 of 170 frames), JSON |
| `wbench.P2_visual_plausibility` | artifacts, plausibility | PAVRM (Qwen3-VL-30B-A3B, bf16 ~62 GB) | video @2 fps, max_pixels 602112 | E[rating over 5 anchors]/5 | gpt-6.1-sol proxy, flagged in the note, until the weights are complete |

## Pose source (I1, C1, C2)

The pose source is recorded in `raw.pose_source`. Sources are tried in this order:

1. `ctx.c2w_est`, if the caller set it.
2. MegaSaM on the whole video. The result is cached in `/mnt/cfs/cyh/tmp/data_pipeline/wbench_cache/megasam_<fingerprint>.npz`.
3. DA3-GIANT per keypoint clip.
4. Commanded poses. C1 and C2 use this only to pick the return frame.

## Environment adaptations

These were found while testing on box 695 with torch 2.5.1 and transformers 4.57.6.

- **Weights reused from local copies:** many files are symlinked from local copies instead of downloaded. The sources are only read, never modified.
  - From `/mnt/cfs/cyh/any2videobench/.runtime/weights`: CLIP L/14 and B/32, DreamSim, DINOv2-B/14, `HPSv3.safetensors`, and DA3-SMALL.
  - From `/mnt/cfs/cyh/v2allbench/motion_tools/weights/mega-sam`: `megasam_final.pth`, Depth-Anything ViT-L, and UniDepth-V2.
  - All downloaded files are checked against the HF LFS sha256 values in `wbench_work/lfs_sha256.json`. One corrupted DreamSim file was replaced by the local copy.
- **CLIP ViT-B/16 (C7):** the local copy is `pytorch_model.bin`. transformers refuses `torch.load` with torch < 2.6, so it was converted once to `model.safetensors`, with identical tensors.
- **MegaSaM:**
  - `droid_backends` and `lietorch` are built for sm_80.
  - The venv's `torch_scatter` is ABI-broken for torch 2.5.1. The pt25cu121 wheel in `/mnt/cfs/cyh/tmp/data_pipeline/pylib_megasam` is put first on `sys.path` during tracking.
  - The torch-hub `dinov2_vitl14` and `metric_depth` checkpoints turned out not to be needed.
- **HPSv3 (V6):** WBench pins transformers < 4.52. Under 4.57, three adaptations apply:
  1. Checkpoint keys are renamed with transformers' own `Qwen2VLForConditionalGeneration._checkpoint_conversion_mapping`.
  2. The reward model gets `.model = language_model` and `.visual = model.visual`, so HPSv3's forward sees the old layout.
  3. If the Qwen2-VL-7B base shards are absent, the architecture is built from its `config.json`. This gives identical results, because `HPSv3.safetensors` is loaded with `strict=True` and overwrites every parameter.
- **DA3 (C4/C5/C6, I1 fallback):** DA3-GIANT-1.1 is used when its checkpoint is complete. Until then, the local DA3-SMALL is used and every affected result's `note` says so.
- **PAVRM (P2):** the 62 GB checkpoint is only partly downloaded. Until it is complete, P2 is the gpt-6.1-sol proxy, and its `note` says PROXY.

## Running

```bash
# real case, WBench only (box 695, GPU 5)
python3 -m evaluate1.run --case evaluate1/cases/<scene> --frames <run>/frames --only wbench
# smoke test: all 22 operators plus subject/turn-type paths on augmented inputs
python3 evaluate1/tests/smoke_wbench.py [--only V1 C4 ...] [--no-vlm]
```
