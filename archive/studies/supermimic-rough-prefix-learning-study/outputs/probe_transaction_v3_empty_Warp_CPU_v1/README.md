# v3 empty Warp-array metadata correction

This is a new implementation correction stage. The retained physical attempt and v1/v2 sources remain unchanged; no physical rerun is authorized here.

`probe_transaction_v3.py` differs from frozen v2 only by `import math` and the Warp branch of `_meta`: it preserves `ptr=None` when the array shape has zero elements, rejects a null pointer for a nonempty shape, and otherwise retains the previous integer pointer identity. All other top-level code and all non-Warp metadata branches are AST-identical. Empty buffers remain distinguishable by dtype, shape and device.

The new checker ran once with Warp 1.14.0 and `CUDA_VISIBLE_DEVICES=-1`: **21/21 PASS**, [report.json](D:/workspace/supermimic-rough-prefix-learning-study/outputs/probe_transaction_v3_empty_Warp_CPU_v1/report.json), SHA `987ec351e8a4ad42292ea3ac562db8981f460e9f148a7f90d2b44e647e6f4719`.

It used nine real Warp **CPU** empty arrays: float32 shapes `(0,)`, `(0,3)`, `(2,0)`, `(0,0)`; int32 and bool `(0,)`; vec3 `(0,)` and `(0,2)`; mat33 `(0,)`. For each, actual `ptr` was None, NumPy/raw bytes were empty, cloning/copying restored an empty buffer successfully and metadata remained unchanged. A real nonempty CPU array preserved its integer pointer and signed-zero bytes. Three nonempty null-pointer stand-ins were rejected before reading memory. The production native snapshot pair-list metadata comprehension was exercised using these real CPU buffers; no native snapshot/Env was constructed.

Warp initialization printed CUDA error 100 / no CUDA-capable device under the mask, then enumerated only CPU. The report records zero CUDA devices; every allocation/clone/copy used `device='cpu'`. No GPU arrays or kernels, environment, MuJoCo/Warp physics, WM, or original 40/33 fake contract suite was run. This validates the narrow empty-buffer ABI, not native full restore/baseline/dynamics qualification.

The original failure had zero scheduled donor/control/branch steps before the metadata TypeError, but its constructor/graph/reset API costs are separately recorded by Root. This CPU fix does not erase that attempt or relabel the whole prior process as zero physics work. Any future worker/entry/supervisor/output must be separately versioned, reviewed and authorized.
