# KS-IN-OE-S-F1_01_202609121414 — 图书阅览室

类别：办公或教育空间 / 教育科研。楼层 1，可通行 155.3 m²（推断米制）。
标注：205 帧，113 个 3D 实例（112 个有类别），校验 通过，全部 unreviewed。

## 0_source

| 路径 | 内容 | 格式 |
|---|---|---|
| `sources/KS-IN-OE-S-F1_01_202609121414` | symlinks to the raw scan: gaussian.ply (3DGS, SH0, ~7.9M Gaussians) + mesh_post1.obj (= mesh_post.obj, vertex colours) | PLY / OBJ |

## 1_scene_geometry

| 路径 | 内容 | 格式 |
|---|---|---|
| `outputs/KS-IN-OE-S-F1_01_202609121414/analysis.json` | world_to_canonical (scan -Y-up -> leveled Z-up), floors, ceiling gaps, unit status (inferred metres) | JSON |
| `outputs/KS-IN-OE-S-F1_01_202609121414/floor_0/maps.npz` | navigation grid 0.05 m: observed/inferred floor, obstacles (0.10-1.80 m slab), clearance, navigable (r=0.25 m) | NPZ [row,col] -> origin+([col,row]+.5)*res |
| `outputs/KS-IN-OE-S-F1_01_202609121414/floor_0/occupancy.pgm` | ROS-style occupancy (+ .yaml) | PGM/YAML |
| `outputs/KS-IN-OE-S-F1_01_202609121414/floor_0/regions.geojson` | automatic region candidates (watershed on free space; not semantic rooms) | GeoJSON canonical XY |
| `outputs/KS-IN-OE-S-F1_01_202609121414/floor_0/topdown_3dgs.png` | top-down 3DGS render (ceiling clipped) + mesh height map | PNG |
| `outputs/KS-IN-OE-S-F1_01_202609121414/floor_0/panoramas` | 360° equirectangular 3DGS panoramas at region centres + metadata | JPG 2048x1024 + JSON |
| `outputs/KS-IN-OE-S-F1_01_202609121414/hesai_walkthrough/main` | trajectory = the scan's own forward walkthrough, aligned: cameras.npz (camera_to_canonical, camera_to_source, K), mesh_depth_mm/, 3DGS re-render check | NPZ/PNG/JSON |

## 2_annotations

| 路径 | 内容 | 格式 |
|---|---|---|
| `auto/KS-IN-OE-S-F1_01_202609121414/global_instances/frames.json` | frame index of the annotated clip: image, mesh depth, K, camera_to_canonical, timestamp | JSON |
| `auto/KS-IN-OE-S-F1_01_202609121414/global_instances/frames` | annotated RGB frames (JPEG q92 of the walkthrough PNGs) | JPG |
| `auto/KS-IN-OE-S-F1_01_202609121414/global_instances/global_instances.json` | 3D object instances: global_id, semantic_label (+fine_label, vote details), category thing/stuff, observed-surface AABB, best view | JSON canonical metres |
| `auto/KS-IN-OE-S-F1_01_202609121414/global_instances/fine_instances.npz` | point cloud (4 cm voxels of the mesh) with per-point global ID / superpoint ID — the source of truth for IDs | NPZ |
| `auto/KS-IN-OE-S-F1_01_202609121414/global_instances/labeled_points.ply` | same point cloud with instance colours, global_id, semantic_id, superpoint_id | PLY |
| `auto/KS-IN-OE-S-F1_01_202609121414/global_instances/frame_assignments.json` | per frame: which global IDs are visible, mask support, purity, accepted flag | JSON |
| `auto/KS-IN-OE-S-F1_01_202609121414/global_instances/rendered` | per frame numeric instance / semantic masks (uint16 PNG; 0 = unknown) + colour/overlay visualisations | PNG |
| `auto/KS-IN-OE-S-F1_01_202609121414/global_instances/semantic_registry.json` | semantic_id <-> label vocabulary | JSON |
| `auto/KS-IN-OE-S-F1_01_202609121414/global_instances/descriptions.json` | GPT (gpt-6.1-sol) text: level1 per-frame description with visible IDs, level2 adjacent-frame change description | JSON |
| `auto/KS-IN-OE-S-F1_01_202609121414/global_instances/frame_qa` | single-frame five-level QA (recognition, attributes, relative position, world coordinates, camera distance) grounded on measured facts | JSON |
| `auto/KS-IN-OE-S-F1_01_202609121414/global_instances/videos` | per-clip rgb / overlay / instance / semantic videos | MP4 |
| `auto/KS-IN-OE-S-F1_01_202609121414/global_instances/qa` | automatic QA evidence: conflicts, reprojection metrics, issue list for human review | CSV/JSON |
| `auto/KS-IN-OE-S-F1_01_202609121414/global_instances/evidence` | every GPT request/response with image hashes and actual model (no keys) | JSON |

## 3_evaluation

| 路径 | 内容 | 格式 |
|---|---|---|
| `evaluate1/cases/KS-IN-OE-S-F1_01_202609121414` | evaluate1 case: 10 input frames, 6 keypoint revisits (3 actions + 3 inverses), trajectory/cameras for rendering a 4DGS, object masks, scene text | JSON/NPZ/PNG |

## 约定

- **canonical**：leveled Z-up, units = inferred metres (not independently verified); world_to_canonical in analysis.json maps scan coordinates
- **camera**：OpenCV camera (+X right, +Y down, +Z forward), camera-to-world matrices; pixel centres at (u+.5, v+.5)
- **depth**：mesh first-hit camera-Z depth, uint16 millimetres, 0 = invalid
- **ids**：global IDs are per scene integers from fine_instances.npz:point_fine_ids; 0 = unknown
- **review**：everything is automatic and marked unreviewed; qa/issues.json lists review leads
