# New-backend dataset normalization and numeric packing V1

Require both completed raw cohorts, independent packet preparation and the
registered full-plant/coverage review before dataset registration. Verify exact
raw/config/registration/backend/role/source-receipt references; retain all source,
branch and physical-row exclusions. Freeze those actual reviewed inputs before
any normalization. The production CLI rejects old schemas rather than relabeling
old records or using their fixed 3866/1935/949 counts.

Use all valid new TRAIN points for the maintained point/head scale formulas.
Use only complete, physically qualified new TRAIN signed pairs with both original
repeats for finite-contrast supervision. Preserve actual source/warm/dt equality,
PD direction tolerance4e-7rad, metadata/repeat identities, and every empty,
incomplete or masked declared group. Pair membership must exactly equal the
independent review's TRAIN rank4/rank20 physical eligibility, without selecting by
response magnitude/stability or validation labels. This rank names input direction
coverage, not output Jacobian rank or infinitesimal-gradient validity.

Validation remains inspected development NEVER_FIT, not held-out original
reference/ego/final test. Its own basis is checked separately. Deliberately alter
all validation x/y/head values and invalid TRAIN values in independent copies;
require bitwise-identical TRAIN scales, head scales, pair targets/indices and
subsets. Count both normalization computations, one original and one isolation
check. Original arrays/masks and metadata remain unchanged and pinned. Reject an
empty or pairless TRAIN bank; do not invent examples to enable fitting.

The output backend_pen_response_common_data:1 owns the actual backend contract,
dynamic counts and provenance. Existing old dataset/model loaders do not acquire
permission to use it. Model training requires a separate new-backend adapter and
fixed-budget registration after these products are reviewed; no fitting or Actor
permission follows from this protocol. No physical steps, neural optimization,
speedup, full-task success or ego/hardware claim.
