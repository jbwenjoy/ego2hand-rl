# Isolated scratch and inconsistent-tetrahedron recovery variants V1

The new entry trace distinguishes face-buffer exhaustion (256 slots) from
projection degeneracy, correcting the previous ambiguous attach-face label.
The silent local case enters EPA with GJK dim4/dist0 but _polytope4 status12.
No installed module/driver/CUDA bytes are edited and no old state is rebound.

Fixed44 scenes: ALL36 previous world/local-placement/ULP/control geometries,
plus8 object translations around the base scene's independent native witness
normal. Six offsets use +/-(abs(base native depth)+epsilon), epsilon0.25/2/10um;
two more use+/-50um. Translation computation FP64 then FP32 geometry cast is
explicit, and signed reference distance determines actual overlap/separation.

Five variants, same shapes/materials and input bytes: installed24/factor5;
horizon96 with face factors20/40; and those same two scratch variants with the
existing closest-face fallback reused when GJK dim4 disagrees with _test_tetra.
Copy source with explicit unique edits, preserve licenses and exact diffs.
Each of face/index, face projection and norm arrays changes capacity together.
The algorithmic retry does not invent a contact; it retries EPA initialization.

Maximum220 single-pair GPU queries and44 native FP64 signed-distance queries
at50 iterations, no FK/integration/Adam. Use installed convex dispatch for the
installed control, isolated convex copies for others; only pair1,55/world31.
Record raw flags, all finite contact witnesses/frames/distances, penetration
and separation misses, distance error, old positive-control byte differences,
all failures/compilation/wall. Near-boundary cases within1um are reported
separately, never silently changed into positives/negatives to pass a test.

No variant may be admitted merely for clearing flags or passing the44 scenes.
Any promising variant still needs independent geometry/positive controls,
full-scene integration/state/gradient qualification and a new matched control
study before production or training adoption. Do not relax task/no-aid gates.
