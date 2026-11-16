# Saved interval target recovery V1

Recover the four path maxima from ALL4860 existing new-backend saved controls,
not endpoint geometry:object penetration and hand self-penetration at each of
the8 pre-integration forwards,joint excursion at each post-integration pose,
and maximum over ticks of summed nonhand-object positive normal force. Preserve
units m/m/rad/N and original limits .002/.0002/.001/.1. Reconstruct the first3
from saved diagnostic tick arrays and fourth from the raw contact table plus
compiled geom/body/name map;compare against authoritative native window bytes.
This is saved-tick/window correspondence,not new mesh collision calibration.

Reconstruct original float32 geometry loss,geometry failure and reward penalty;
compare with native window and previously checked full native reward chain.
Retain metric/global error bits and failure labels as separate arrays. Their
absence or lack of positive coverage is NOT an error predictor qualification.
No weaker physical gate,geometric mesh approximation,endpoint replacement,
source epoch,simulator step/forward,optimizer or normalizer fit.

Exact common dataset/preparation/raw/reward identities and source snapshots
must match before extraction. All3240 TRAIN and1620 development validation
rows including38/29 original impulse failures remain aligned. Preserve original
valid masks,all960/480 groups including incomplete/empty groups. Original valid
TRAIN rows and complete qualified TRAIN pairs alone may supply future fitting;
development validation is already inspected and never fitted,not final test.

Save raw path4 and dimensionless log1p((path/limit)^2)4,raw error/failure labels,
physical mask and case/row identities. Sum transformed components reproduces
the original soft geometry term;hard failure is retained separately and not
treated as a smooth derivative. Keep actual0 labels,zero response directions,
all pair denominators and unavailable groups in coverage summaries. Per-PD
finite contrast uses2epsilon;RAW conversion is the previously verified0.25
controller gain,not a change to perturbation amplitudes.

Registration freezes only this read-only recovery/coverage design;it does not
register new NN architecture/loss or admit any WM to Actor. Fit/gradient and
multistep/native reward improvement require separate controlled experiments.
Full M0–M5/A1/A2,task,multireference/seed/total costs,real ego assets/gravity/
dynamics/RGB and hardware contracts stay active. No driver/CUDA/install edits.
