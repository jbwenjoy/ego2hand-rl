# Explicit Fn6 response supervision — 2026-11-16

Eight registered study5 arms completed2000 updates. Adding Fn6 response
supervision improves contact-response errors,but transfers errors to existing
object/wrench responses or forward predictions. No model qualifies for imagined
Actor optimization. A separate longer model-free control baseline is now
running;it does not use these WMs.

Same3866 valid TRAIN points,949 pairs,initial NN,seeds,units,forward coefficient1,
auxiliary.1 and physical masks. At each outer weight.1 or1,FN adds only the Fn6
mean to the previous object+wrench sum;old coefficients are not divided by3.
Reported forward MSE/AD relative L2 continue using POINT y_scale,not RMS loss
units. Physical-screen pair counts are object248,wrench266,Fn284;zero response=1.

| Arm | Forward object | Forward Fn6 | AD object | AD wrench | AD Fn6 |
|---|---:|---:|---:|---:|---:|
| RMS01_DIRECT |2.45937|.24738|.56760|.71454|1.02979|
| FN01_DIRECT |.88358|.41033|.95176|.92820|.90074|
| RMS01_IMPULSE |6.40998|.19847|.70732|.86252|1.04114|
| FN01_IMPULSE |5.50364|.33322|.79409|.96532|.57169|
| RMS1_DIRECT |4.11759|.22563|.49509|.71193|1.00235|
| FN1_DIRECT |1.83529|.78921|.81471|.77353|.61675|
| RMS1_IMPULSE |14.02707|.23960|.83329|.76579|1.00338|
| FN1_IMPULSE |4.54904|1.71144|.64471|.90909|.53219|

FN1_IMPULSE improves contact AD1.003->.532 and object AD.833->.645,but wrench
AD worsens.766->.909 and forward Fn MSE.240->1.711. Its exploratory candidate
counts are102/248 object,35/266 wrench,141/284 Fn. FN01_DIRECT improves object
forward2.46->.884 while losing most object action response (AD.952 and0/248
candidate object directions). No single column establishes whole-model success.

TRAIN all949-pair Fn AD becomes FN01_DIRECT.903,FN01_IMPULSE.639,FN1_DIRECT.628,
FN1_IMPULSE.500. These TRAIN pools differ from screened validation pools;do not
interpret their difference as an unbiased generalization estimate. The data
remain one original reference/fixed actor seed,inspected NEVER_FIT development
histories. Old324 CHECK and31 masked raw rows are unchanged.

## Component audit and coverage boundary

All component probes use the complete949 TRAIN pair table at100/500/1000/2000.
At2000,normalized Fn component MSE is84.29/7171.65 for the unused RMS01 direct/
impulse heads,and.560/.519 after explicit FN01 supervision. At weight1,the
corresponding174.80/7155.22 becomes.664/.438. Existing object/wrench components
also worsen in several arms. This verifies actual supervision and interference,
not a claim that every gradient component improved.

TRAIN derivative RMS for Fn columns84:90 is approximately
[3.6563,2.3131,3.7950,2.5551,0,0]N/rad. The last two group channels use the
existing positive floor. That exposes missing active-contact coverage in this
early source bank;it is not proof of a sole failure cause and must not be
silently solved by dropping channels or hardcoding zero outputs for all tasks.
Longer actual policy execution and broader contact/source coverage remain needed.

All four RMS controls reproduce study4 at100/2000 for NN,Adam,point/pair sampler,
gradient snapshot and logs,all48 field comparisons exact. First/final187/187,
407 core tests and410 tests including new long-control budget checks pass.
Evaluation980/980,diagnosis11/11,independent review1243/1243 include component
arithmetic,final scalar recomputation,unchanged masks and block/cache identities.
Arithmetic/health passes do not replace gradient/control admission.

## Cost and artifacts

- `runs/pen_fn_response_fit_v1`:16000 live+32 clone QA=16032 optimizer updates,
  0 native/Actor. Fit+QA97.2227s,runner100.1875s. First checkpoint estimated
  89.518s remaining compute. Frozen old units reused;loader point/head and
  response-unit reconstructions reported separately (one each).
- `runs/pen_fn_response_evaluation_v1`:all480 groups/475 valid,1920 model/source/
  direction JVPs,all predictions;0 optimizer/native.
- `runs/pen_fn_response_diagnosis_v1`:all949 TRAIN responses/JVPs and component
  curve;0 optimizer/native.
- `runs/pen_fn_response_review_v1`:independent saved arithmetic/control replay;
  0 new updates. Status:NO_MODEL_QUALIFIED_FOR_ACTOR_OPTIMIZATION.
- Exact closed FN-study ledger:`provenance/pen_fn_response_results`.

The four component probes add diagnostic endpoint forwards,not optimizer draws.
Equal updates do not establish equal wall/FLOPs or amortized speed. No new
native data or unit-statistic preparation was performed. Model-free long PPO
qualification/training is a separate cohort,not included in these16032 updates.

## Actual control task next

The prior64-round policy fails the full7.36s task. Register and run two fresh,
shared-initialization model-free geometry/support arms2000 rounds each,qualified
CACHED32 sampling,original physics/no aids/geometry gates. Current qualification
75/75 passes;first optimized checkpoint byte restoration is checked before
continuing. Intermediate training rewards/phases are not task success.

Do not change captured source/config/protocol bytes while the long job runs.
After its fixed budget,require fresh LIVE W1 mean-policy execution,repeats and
strict full-duration/position/orientation/geometry/no-aid evaluation before new
video claims. This is not model-based A1/A2 or ego data. Preserve independent
trajectories/seeds,total cost,asset admission and hardware goals. No driver/
CUDA/package changes or external publish.
