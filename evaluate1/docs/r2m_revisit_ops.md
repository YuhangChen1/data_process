# evaluate1 — R2M, revisit and fidelity operators

Code: `evaluate1/metrics/{pairs,r2m,revisit,fidelity}.py`. Every operator follows `metrics/base.py` (`fn(ctx) -> Result`,
registered in `REGISTRY`). Reference implementation for the R2M families: R2M-Bench (arXiv 2608.27328,
github.com/AMAP-ML/R2MBench), local copy `/mnt/cfs/cyh/tmp/refs/R2MBench-main`.

**Offline weight store.** Every operator reads weights only from `/mnt/cfs/cyh/tmp/data_pipeline/weights/`; nothing is
downloaded at evaluation time (a missing file raises `FileNotFoundError` naming it):

| dir | content | source |
|---|---|---|
| `dinov2/` | `dinov2_vitb14_pretrain.pth` + `facebookresearch_dinov2_main/` (torch-hub local repo) | dl.fbaipublicfiles.com, GitHub |
| `boq/dinov2_12288.pth` | BoQ DINOv2 12288-d | GitHub release v1.0 |
| `mutualvpr/MutualVPR_model_512.pth` | MutualVPR 512-d | Google Drive (author link) |
| `clip/ViT-B-32.pt` | openai CLIP | openaipublic |
| `groundingdino/groundingdino_swint_ogc.pth` | GroundingDINO SwinT-OGC (includes all 200 BERT tensors) | GitHub release |
| `bert-base-uncased/` | config + tokenizer only (git-blob hashes match HF) | ModelScope |
| `sam2/sam2.1_hiera_large.pt` | SAM2.1-L (sha256 2647878d… = HF blob) | ModelScope mirror |
| `torch_hub/checkpoints/` | SuperPoint v1, LightGlue superpoint v0.1_arxiv, AlexNet, VGG16 (LPIPS, DISTS) | GitHub, download.pytorch.org |
| `pyiqa/` | NIQE, BRISQUE, DISTS parameters | pyiqa releases |
| `wbench/dreamsim/` | DreamSim ensemble (shared with the WBench operators) | DreamSim release |

Fill / refresh it once with `python3 evaluate1/tests/smoke_r2m.py --prefetch` (network allowed only there,
`EVAL_ALLOW_DOWNLOAD=1`); big checkpoints come from `/mnt/cfs/cyh/tmp/data_pipeline/r2m_download.sh`. Backend code
(GroundingDINO, sam2, LightGlue, Bag-of-Queries, MutualVPR, openai/clip) is R2M's vendored copy at
`/mnt/cfs/cyh/tmp/data_pipeline/third_party/r2m/`; GroundingDINO's deformable-attention CUDA op is JIT-built once into
`torch_ext/`. Extra pip packages (dreamsim, piq, pyiqa) are in `pylib_r2m/`.

## Pairs and MemoryGain / NMR (`pairs.py`)

| function | R2M source | what it does |
|---|---|---|
| `mine_pairs(c2w, ...)` | `src/sample_pairs.py:90-210`, `src/eval_revisit.py:67-432`, `src/pair_sampling.py` | Revisit pairs from commanded poses: static-tail trim, adaptive tolerances `tau_rot = clip(2·P95(Δyaw), 1°, 5°)`, `tau_pos = clip(2·P95(Δt), 0.1, 1.0) m`, motion-segment matching (3D fallback when < 3 pairs), gap `≥ 0.2·T`, ≤ 100 sampled (seed 42). Gap-matched baselines: same temporal bin as the anchor (10 bins), `|gap − rev_gap| ≤ max(10, 0.3·rev_gap)`, NotNearReturn radius `max(10, T//20)`, relaxed 0.5 fallback to 20 pairs. Short pairs: gap 2..max(6, 0.02·T), `min(100, max(20, #revisit))` anchors |
| `memory_block(sets, metric)` | `compute_nmr_summary.py:210-231`, `eval_gpu_metrics.py:68-102` | per-set mean/std/median/count; `MG = R − B`, `DR = S − B` (reversed for LPIPS); invalid when `DR ≤ 1e-8`; `NMR = MG / (DR + 1e-8)`; R2M's per-video-file value (`0` when `DR ≤ 1e-5`) kept as `nmr_per_video_file` |
| `overall_nmr(metric_nmr)` | `compute_nmr_summary.py:270` | mean over families (Appearance, Scene ID, Local Geo., State, Object) of the mean metric NMR |

Adaptation: R2M's yaw is `atan2(R[0,2], R[0,0])` in a y-down world. Our canonical world is Z-up, so poses are first
rotated by `A` (`A·up = (0,−1,0)`, det +1). With one case per trajectory, "video-balanced" aggregation reduces to the
case's own MG and DR.

## R2M families (`r2m.py`)

Every operator: inputs `frames`, `pairs`; it evaluates four pair sets (first `max_pairs_per_type`, default 100, as
`presampled_pairs.select_for_video`): `keypoint_revisit` (our 6 keypoint visit/return pairs), `revisit` (R2M-mined from
the commanded trajectory), and the shared `baseline` and `short` sets. MG / DR / NMR are computed for **both** revisit
sets against the same baseline and short sets. **value = keypoint-revisit NMR** (our headline; falls back to the
R2M-mined NMR when undefined), higher is better. `raw` holds one block per revisit set (`keypoint_revisit`, `revisit`:
revisit / baseline / short stats, `mg`, `dr`, `nmr`, `nmr_valid`), flat copies `nmr_keypoint_revisit`, `nmr_revisit`,
`mg_*`, and `headline_set`; `per_item` has every pair score. `r2m.summary.nmr` reports the Overall NMR for both sets
(`overall_keypoint_revisit`, `overall_revisit`, per-family values).

| operator | R2M key | measures | model / checkpoint | formula |
|---|---|---|---|---|
| `r2m.appearance.psnr` | psnr | pixel fidelity | skimage | PSNR, data_range 255 (identical frames capped at 100 dB) |
| `r2m.appearance.ssim` | ssim | structure | skimage | SSIM, win 7, channel_axis 2 |
| `r2m.appearance.lpips` | lpips | perceptual distance | lpips 0.1.4, AlexNet | LPIPS on [-1,1] full-resolution frames (NMR reversed) |
| `r2m.scene_identity.dinov2` | dino_similarity | same place | DINOv2 ViT-B/14 `dinov2_vitb14_pretrain.pth` | resize 518 bicubic + centre crop 518, cosine of L2-normalised CLS output |
| `r2m.scene_identity.boq` | boq_similarity | place recognition | BoQ `dinov2_12288.pth` (DINOv2-B + 2 BoQ layers, 64 queries, 12288-d) | 322×322, cosine |
| `r2m.scene_identity.mutualvpr` | mvpr_similarity | place recognition | MutualVPR `MutualVPR_model_512.pth` (DINOv2-B + GeM + Linear 512) | 518×518, cosine |
| `r2m.geometry.match_ratio` | match_ratio | local correspondence | SuperPoint (1024 kp) + LightGlue (`superpoint_lightglue` v0.1_arxiv) | `|M| / max(min(K_a, K_b), 1)` |
| `r2m.geometry.inlier_ratio` | ransac_inlier_ratio | epipolar consistency | OpenCV | RANSAC F (3 px, 0.99, ≥ 8 matches), `|inliers| / |M|` |
| `r2m.object_identity.appearance` | object_appearance_persistence | object survives with same look | GroundingDINO SwinT-OGC + SAM2.1-L + DINOv2-B | anchors: GDINO on the default prompt (box 0.25, text 0.20, NMS 0.60, ≤ 10, area ≥ 0.002) + SAM2 box masks; per anchor re-detect `"<concept> ."` (≤ 5); app = clip01((cos_DINO(white-masked crops)+1)/2); match if 0.5·app + 0.5·sem ≥ 0.5; mean over anchors, unmatched = 0 |
| `r2m.object_identity.semantic` | object_semantic_persistence | object keeps its category | + CLIP ViT-B/32 | sem = 0.5·GDINO conf + 0.5·clip01((cos_CLIP(crop, concept)+1)/2); mean over anchors, unmatched = 0 |
| `r2m.object_identity.existence` | object_existence_rate | fraction of anchors re-found | same | matched / anchors |
| `r2m.object_identity.appearance_similarity` | object_appearance_similarity | look of matched objects | same | mean app over matches |
| `r2m.object_identity.semantic_similarity` | object_semantic_similarity | category of matched objects | same | mean sem over matches |
| `r2m.persistent_state.vlm` | state_score | scene state persistence | **gpt-6.1-sol** (R2M: gemini-3.1-pro-preview) | R2M rubric verbatim, score 1–5 clamped, ≤ 10 pairs per type (`run_parallel_eval.sh:228`), 3 workers, ≤ 4 attempts |
| `r2m.summary.nmr` | Overall NMR | — | — | mean over the 5 families of mean metric NMR (`compute_nmr_summary.py:270`) |

Deviations from R2M:
- PSNR of bit-identical frames is `inf` in skimage; it is capped at 100 dB (a static 3DGS rendered at the same pose gives identical frames).
- VLM images go through the gateway client (`annotation_pipeline/vlm.py`): JPEG ≤ 42 KB per image (the gateway's ~100 KiB
  request limit) instead of lossless PNG; the R2M system prompt is sent inside the user prompt; temperature is the
  gateway default.
- GroundingDINO's deformable-attention CUDA op is JIT-compiled (`torch.utils.cpp_extension.load`) from R2M's vendored
  sources; if compilation fails the op falls back to GroundingDINO's own PyTorch implementation (`Result.raw` of the
  object operators does not change; the path used is `get_grounded_sam().ops`).
- BERT for GroundingDINO is built from its config and then filled from the GroundingDINO checkpoint, which carries all
  BERT parameters (R2M loads pretrained BERT first and then overwrites the same tensors with `strict=False`; the code
  checks that no parameter is left uncovered), so the separate 440 MB BERT file is not needed.
- SAM2's image embedding is reused across prompts on the same frame (R2M re-encodes per call; same masks).
- With one case per trajectory, R2M's "video-balanced" mean over videos reduces to this case's own MG and DR.

## Keypoint revisit (`revisit.py`, ours)

Inputs: `keypoints` (each `{'visit', 'return'}` step; the camera pauses, runs 3 actions + 3 inverse actions and must
return to the visit pose), `frames`, plus `K`, `c2w_cmd`, `depth`, `c2w_est`, `objects` where noted. Value is the mean
over keypoints (objects: over objects × keypoints); `per_item` lists each keypoint / object.

| operator | measures | method | ideal |
|---|---|---|---|
| `revisit.pose.rot_err_deg` | camera rotation drift at return | SuperPoint+LightGlue matches; model selection as in ORB-SLAM / Torr: rotation-only homography `K R K⁻¹` (RANSAC 2 px) when its inliers ≥ 0.8 × the essential matrix's (1 px) — the zero-baseline case where E is degenerate — else E + recoverPose; angle(R_cmdᵀ R_est) | 0 |
| `revisit.pose.pixel_disp_px` | misregistration | median ‖p_return − p_visit‖ of matched keypoints (raw: normalised by image diagonal) | 0 |
| `revisit.pose.match_ratio` | correspondence | R2M match ratio on the visit/return pair | 1 |
| `revisit.pose.trans_dir_err_deg` | translation direction | essential-matrix t vs commanded, only when the commanded baseline ≥ 5 cm (n/a for exact revisits) | 0 |
| `revisit.pose.trans_err_m` | metric translation drift | back-project visit keypoints with `depth[visit]`, PnP-RANSAC (2 px) in the return frame; ‖t − t_cmd‖ (raw: PnP rotation error) | 0 |
| `revisit.pose.cmd_rot_err_deg`, `cmd_trans_err_m` | sanity of the action program | commanded visit vs return pose | 0 |
| `revisit.pose.est_drift_rot_deg`, `est_drift_trans_m` | drift of an estimated trajectory | Umeyama Sim(3) alignment of `c2w_est` camera centres to `c2w_cmd`, then relative visit→return error (raw: ATE RMSE, scale) | 0 |
| `revisit.image.psnr / ssim / lpips / dreamsim / dinov2` | image similarity visit vs return | as in R2M appearance / scene identity; DreamSim ensemble distance | ∞(100) / 1 / 0 / 0 / 1 |
| `revisit.object.mask_iou` | 画面物体重合 vs annotation | IoU(GT mask at visit, best label-prompted GDINO+SAM2 mask in the return frame); 0 when not re-detected. Absolute — bounded by detector/annotation agreement (`raw.oracle_ceiling` = same IoU on the visit frame) | ceiling |
| `revisit.object.pair_iou` | same object, same place | for objects the detector re-finds at visit (IoU ≥ 0.5 with GT): IoU(predicted visit mask, predicted return mask); missing at return = 0 — detector bias cancels | 1 |
| `revisit.object.centroid_shift_px` | object displacement | centroid distance of the predicted visit and return masks | 0 |
| `revisit.object.area_consistency` | object scale change | min(area)/max(area) | 1 |
| `revisit.object.recall` | object presence | of the objects found at visit, share still there at return (pair IoU ≥ 0.5); `raw.recall_vs_gt` against annotation masks | 1 |
| `revisit.object.precision` | spurious / duplicated objects | share of return-frame detections (labels of the visible annotated objects) matching a same-label visit detection (IoU ≥ 0.5); `raw.precision_vs_gt` against annotation masks | 1 |
| `revisit.object.agnostic_iou` | label-free layout overlap | SAM2.1-L automatic masks (32×32 points, area ≥ 0.2 %) on both frames, Hungarian matching on IoU, area-weighted mean matched IoU over all masks (raw: recall/precision at 0.5) | 1 |
| `revisit.excursion.psnr / lpips / dinov2 / pixel_disp_px` | consistency inside the excursion | identical commanded poses reached by action k and again by its inverse (pose classes inside [visit, return], keypoint pose excluded) | ∞(100) / 0 / 1 / 0 |

## Input-frame fidelity (`fidelity.py`, ours)

Inputs `frames`, `input_index` (the 10 input steps), `input_gt` (their ground-truth images). Renders are resized to the
GT size if needed. Value = mean over the 10 frames.

| operator | metric | lower is better |
|---|---|---|
| `fidelity.psnr`, `fidelity.ssim` | skimage (win 7), PSNR capped at 100 dB | no |
| `fidelity.ms_ssim` | piq multi-scale SSIM | no |
| `fidelity.lpips_alex`, `fidelity.lpips_vgg` | lpips 0.1.4 | yes |
| `fidelity.dists` | pyiqa DISTS | yes |
| `fidelity.dreamsim` | DreamSim ensemble distance | yes |
| `fidelity.dinov2` | DINOv2-B global cosine | no |
| `fidelity.niqe`, `fidelity.brisque` | pyiqa no-reference on renders (raw `gt_mean`: same metric on the GT) | yes |
