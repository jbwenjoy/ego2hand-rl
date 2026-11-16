# Fixed real-ego mug RGB projection audit V1

Use original fixed-revision HOT3D Aria clip001884,objectBOP9 UID223371871635142,
RGB stream214-1,ALL frames47..107 and their exact source timestamps. Display
frames47/77/107 fixed before computing metrics. No fitting/pose adjustment/
image selection. Preserve original encoded JPEG bytes for display samples.
Existing source mesh/BODY scale and actual compiled128-piece asset unchanged.

Pin official Meta camera source9064a2fe7fda51bb1f1451a0d3d0da7f3ef5e64e,
Apache-2.0/LICENSE,plus exact HOT3D clip-format/source snapshots. No dependency
installation or installed source change. Use pure NumPy FP64 Aria15-parameter
Fisheye624 against pinned official camera oracle. Same-eye projection tolerance
1e-8px;full pipeline including official FP32 camera-pose serialization1e-3px.
Camera +Z front/OpenCV xy convention,integer pixel centers;unmodified source
world-from-camera and world-from-object transforms,not robot-frame mappings.

Recover actual collider BODY vertices from compiled mesh and geom transforms,
compare each of128 pieces to the fixed decomposition within2e-7m. No native
forward/integration. Subdivide planar triangles only for image rasterization,
max edge5mm for source and union;3D geometry/physics/physical parameters are
unchanged. Use CPU PIL triangle coverage at original1408x1408 resolution. This
is linearized fisheye triangle rasterization,not a new photometric renderer;
report pixels/boundary differences descriptively,do not tune a silhouette gate.

Decode source masks using one-based start/length,C-order runs,strict bounds,
and exact re-encode roundtrip. Dataset amodal masks were rendered from the SAME
model/annotated poses with vignette clipping;modal masks came from SAM2-based
processing. Neither is independent manual truth. Compare source raster against
both,compiled union against source/amodal,retain ALL61 frames and metrics.
Reported IoUs are image/asset/annotation consistency diagnostics,not recovered
pose accuracy,calibrated metric geometry,collision/contact-force truth or task
success. RGB contours may be visually inspected without claiming independent
reconstruction or zero annotation error.

Contact sheet uses one fixed global crop from all annotation boxes plus margin,
original RGB,provided mask contours and source-vs-compiled contours. Preserve
frame IDs/timestamps/crop coordinates and raw sample image hashes. Keep source
uncertainty,occlusion/vignette/rasterization limitations explicit.

No gravity/mass/inertia/friction calibration,policy/contact/WM fitting,native
integration or GPU kernels. Current pen training source/config/backend unchanged.
Full M0–M5,strict control evaluation,multi-reference/seed costs,independent RGB
reconstruction/source dynamics and hardware observation contracts remain open.
