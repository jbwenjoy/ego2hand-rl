# Saved phone contact wrench labels

The once-only export completed from the immutable `acquired_v1` cache: **960 rows,
912 full H20ms intervals and48 actual partial prefixes**,89679 PRE ticks and47077
phone-boundary contacts. Three analytic rotation/sign/origin fixtures passed.
No model loading, native forward/FK/step, neural model, AD or training occurred.
These are simulator-decoded contact labels, not real human/phone force measurements.

`completed_v1/all960_interval_wrench_labels_and_residuals.npz` is sorted by its
stable `row_id`; identities, family/episode, original split, complete and training
eligibility masks remain intact. All vector arrays are F64. Six-vectors always
order **force3 then torque3**, with torque measured about the explicitly named COM.
Means use N/Nm; integrals use Ns/Nms.

| Field prefix | Axes and torque origin |
|---|---|
| `world_*` | World axes; each PRE tick's moving physical COM |
| `instantaneous_BODY_*` | Each PRE body orientation; moving COM. Its integral is not a single-frame impulse |
| `source_BODY_axes_moving_COM_*` | Fixed interval-source body axes; moving COM |
| `source_BODY_fixed_origin_*` | Fixed interval-source body axes and **source COM** origin; adds the COM-shift force lever |

BODY axes are physical body/material axes, not principal inertial axes and not
the body-root position. The source-frame origin is source COM, not qpos rootp.
`all_PRE_tick_wrenches.npz` also retains all four corresponding tick series,
PRE COM/bodyR, integer ticks, native h and the original Reader world-wrench series.

For every generated phone-boundary contact, the stored local decoded force and
direct moment are rotated by `frame.T`, with positive sign for phone=geom2 and
negative for phone=geom1. Torque is `(contact_pos−COM_PRE)×F + direct_moment`.
All phone contacts are included, including nonpad/table and weak/zero normal
forces. There is no Fn threshold, raw-constraint/address lookup or pyramid
re-decoding. PRE qpos/qvel come from episode initial state and previous POST
states; no extra native forward is used.

Mass .2kg, principal inertia and body_iquat are pinned to both selected native
source records, COM offset to the acquisition model audit, and simulator gravity
to both source support records. `Ibody=R(body_iquat) diag(Iprincipal) R.T` and
`Lworld=Rbody Ibody Rbody.T omegaWorld`; body and inertial orientations are distinct.
Parameters remain assumptions of the frozen native model, not calibrated real GT.

World residuals are saved, never forced to zero:

`rP = m(vCOM_end−vCOM_start) − mg*(n*h) − sum(h*Fcontact_world)`

`rL = Lworld_end−Lworld_start − sum(h*tauCOM_world)`.

Partial intervals use their actual returned `n*h`, not a fabricated H target or
native clock rounding. Only world moving-COM torque enters this angular balance.
Rotating-body endpoint subtraction needs transport terms; a fixed-origin balance
would additionally need orbital momentum and gravity-lever terms.

Across all912 complete intervals, residual vector RMS is **6.648e−8Ns** linear
and **1.036e−7Nms** angular, equivalent to **3.324e−6N /5.182e−6Nm** after division
by H. Maximum norms are3.659e−7Ns/6.029e−7Nms. Recomputed world wrench versus the
saved Reader capture differs by at most3.553e−15; rebuilt POST state58 matches
exactly. Per-dt, source-branch, TRAIN/VALIDATION/TEST and partial statistics are
preserved in `momentum_residual_statistics.json` and CSV. These are consistency
descriptors, not a newly invented acceptance tolerance.

The Reader's no-applied-force/callback/gravcomp checks are source-backed; separate
per-step passive/applied-force arrays were not exported, so they are not newly
certified here. Discrete integration and any omitted terms remain part of the
observed residual. An interval mean/impulse constrains momentum increments but
does not uniquely determine full position/orientation transitions without force
timing, kinematics and history. Exporting labels neither proves holding/support
nor authorizes a new model fit or Actor/CHORD use.
