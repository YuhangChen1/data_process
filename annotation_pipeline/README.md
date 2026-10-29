# Automatic annotation pipeline

See [the delivery documentation](../auto/README.md) for methods, commands, coordinate conventions and known limitations. The entry point is:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m annotation_pipeline.pipeline
```

Modules: `prepare` builds geometry and panorama semantics; `run` performs SAM3/mesh instance extraction and VLM semantics; `recover_vlm` retries missing semantics; `postprocess` applies deterministic text/confidence rules; `report` builds the offline browser; `verify` checks actual deliverables against the JSON schema and source evidence. All automatic writes stay outside `human/` and `final/`.

The independent read-only live viewer is `services/viewer_app.py`, started with `bash services/start_viewer.sh`.
