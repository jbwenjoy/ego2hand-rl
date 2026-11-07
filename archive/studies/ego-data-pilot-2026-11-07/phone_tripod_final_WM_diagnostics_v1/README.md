# Fixed offline-final WM diagnostics

`compare_final_WM_v2.py --protocol protocol_CPUFalse_v2.json` performs JSON/hash checks only. The four final-file inputs are pending; this protocol cannot execute inference. Root must bind the actual completed paired1000/WM1100 protocol, result, health and terminal in a new protocol and explicitly authorize CPU execution. The first source/protocol/metadata v1 are preserved; `source_revision_v2.json` and `v1_to_v2.diff` enumerate the catalog/pad guards and explicit geometry-asset read accounting.

The only predicted models are the shared WM from that fixed offline final and its fixed physical-WM5000 parent. Two trusted CPU checkpoint loads and two WM constructors are allowed; no Actor/Critic constructor, optimizer, AD, native backend, fitting or GPU call is allowed.

Four inference batches are declared: both models on the same current-geometry MON52 input; final on CHECK1472; final on fresh228. There are 1804 predicted rows and four structured decodes. Warm CHECK/fresh predictions remain the existing saved references. Reusing their actual FP32 normalized366 inputs requires raw equality of final and warm normalizers and physical context before inference. MON geometry is computed once for the two models.

Five data NPZ files are read once: tripod MON, saved CHECK predictions/labels, original CHECK rows, saved CHECK side responses, and saved fresh predictions/labels. The existing current-geometry provider additionally reads two frozen asset NPZ files once; these are separately counted. All1476 CHECK and240 fresh identities/masks remain in outputs, including four unknown CHECK and twelve partial fresh rows with NaN predictions. No unavailable row is promoted.

`endpoint_statistics.json` contains each dataset's ALL, both native-dt cohorts, available family cohorts and both model roles. CHECK also includes baseline and action-side endpoints. It reuses frozen `state_stats`/`pair_stats`; fresh has state and original two-pad coverage truth only. No meanFn22/J22 truth is fabricated for fresh.

`response_statistics.json` covers all1436 defined finite-action CHECK sides and all20 action axes, native-dt and original step strata. Six physical head blocks retain separate units. Responses and actual-du secants are separate; exact-zero native COM responses retain predicted phantom response magnitudes. These are known-state finite-action comparisons, not infinitesimal Jacobians or independent episodes. Raw side/baseline identities, actual denominators and all1440 masks are saved.

The runner saves predictions, normalized inputs, native targets/masks, MON raw operands, per-cohort statistics, normalizer/context equality and immutability results. Fixed final checkpoint choice does not imply policy benefit, overall improvement, faithful ego contact or held-out trajectory qualification. No backend inference has been executed during source preparation.
