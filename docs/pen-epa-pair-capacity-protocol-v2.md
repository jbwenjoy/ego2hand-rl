# Pair capacity V2: explicit input fields after persistence failure

All V1 physical/numerical/geometry/pair/ULP/capacity/query limits remain.
V1 is terminal exit1 after1 full narrowphase replay,12 isolated candidates,
1 capacity24 query,0 FK/integrations/Adam. A duplicate contact_worldid NPZ
keyword stopped artifact writing. Full replay and sole failing pair[1,55]
were preserved; capacity query work is charged even without its output file.
Do not replace its output or mark it successful.

V2 reads only qpos/geom_xpos/geom_xmat as collision input, so old response
fields cannot collide with newly serialized contact/overflow fields. The
helper checks shape, finite FP32 and copies bytes. Preserve V1 exact frozen
source and pin its failure as an input. Re-run under a new registration and
output with the same budget; no installed package or native contract change.

Original plan: pen-epa-pair-capacity-protocol-v1.md.
