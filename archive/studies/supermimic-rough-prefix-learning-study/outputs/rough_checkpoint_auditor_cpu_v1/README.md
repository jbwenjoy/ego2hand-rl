# Saved format104 health audit

`scripts/audit_rough_checkpoint_cpu.py` reads saved tensors and files on CPU. It constructs no environment, simulator, optimizer, or contact query. `CUDA_VISIBLE_DEVICES=-1` is set before importing Torch. Existing review directories cannot be overwritten.

```powershell
& D:/workspace/dex-supermimic-research/.venv/Scripts/python.exe `
  D:/workspace/supermimic-rough-prefix-learning-study/scripts/audit_rough_checkpoint_cpu.py `
  --config ACTUAL_AUTHORIZED_CONFIG.json --run-dir ACTUAL_RUN_DIR `
  --checkpoint ACTUAL_RUN_DIR/train/model_final.pt --output NEW_REVIEW_DIR
```

The smoke budget is 128 worlds, 16 warmup steps, ten iterations at UTD `.4`: 11 collect calls and 41 control intervals per world, or 5248 actual transitions. Expected optimizer steps are WM20, `fail_head`20, actor8, critic32. Hook counts, sidecar/index, serialized optimizer steps and actual collection evidence are checked separately.

The audit binds config/protocol/input/source archive, format104/controller signatures, new16 reference and analytic requested initialization, actual8 raw readbacks, module shapes and finite tensors. Raw constructor-return CPU parameters independently reconstruct initial hashes and checkpoint parameter changes; no new initialization RNG is drawn. Controller evidence covers the complete JSONL prefix named by the checkpoint, with independent FP32 projection/cache/actual/replay checks on the declared sparse traces. Later complete JSONL rows or a partially appended suffix do not alter an earlier checkpoint's prefix.

Saved policy start phases, H8 active masks, critic normalization eligibility and minibatch mask counts are independently reconstructed. Native replay-window continuity itself is reported at its actual evidence strength: source-validated saved summary, not a new inspection of an unsaved raw window. Requested initialization bindings are not relabeled as independent raw simulator reset readbacks.

Capacity records are recomputed from raw integer flags, OR history and unknown counts. Checkpoint prefix is nonfinal and cannot claim complete-run capacity. Only a final-iteration audit requires `training_completion.json`, the separate finalized run capacity record, and exactly one end read. Earlier checkpoints use their own prefix; later capacity results remain a different scope. No flag shape is interpreted as a per-world attribution.

These27 checks are small CPU helper fixtures, **not an actual training checkpoint audit**. They cover positive metadata/tensor/control cases and corrupted marker, replay, mask, fingerprint, delta, overflow, unknown and finalization cases. Real smoke artifacts were not available when this receipt was created. Full real audit remains pending and must use the same source hash or a separately recorded revision.

PASS means saved engineering health only. It does not establish physical task success, contact-force prediction, useful imagination gradients, learning benefit, full motion success, or real ego-data generalization. Frozen prior metadata retains the earlier local-proxy and compiled-precision failures.
