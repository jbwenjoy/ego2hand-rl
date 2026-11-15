# Registered Windows source-epoch restore benchmark v1

Before timing freeze current sources/config/protocol. Default native4 restore
keeps full source/engine SHA verification on every call. The optional epoch
acquires GENERIC_READ/FILE_SHARE_READ handles for canonical observed,snapshot
and engine paths,hashes all at entry and exit,and checks handle/path identity
and exact source/engine manifest before every use. All old task/RNG/array/layout/
alias/pointer/observation checks remain. A changed/foreign/expired token rejects.
Outputs are provisional until successful context exit. No epoch is serialized.
Unsupported backends or lock failures reject;no fallback to unchecked loading.

Windows sharing semantics follow Microsoft's CreateFileW documentation:
https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilew
File identity follows GetFileInformationByHandle:
https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-getfileinformationbyhandle
This is a bounded scientific I/O transaction,not a security claim against
malicious native code/preexisting writable memory mappings. No permanent
permissions changes. All handles close on failures as well as normal exit.

Changing loader source bytes must not silently resume historical native states.
Generate6 NEW current-code W1 sources from frozen final64 actors,at phases
10,30,60 for residual/support. Transfer actor/norm only; verify historical source
snapshots and full current W1 Model against the recorded W1 context. Never load
old optimizer/critic/native/RNG. W1/W32 derived solver fields(body_invweight0,
dof_invweight0,actuator_acc0) differ;do not call this full W32-plant equivalence.
These new prefixes are uninterrupted with no post-start reset/restore.

After source generation,one separate W1 counterfactual runtime uses the SAME
six complete sources for both modes. One strict warmup/source,then3 timed
restores/mode/source,alternating mode order with source+repeat parity. Keep an
epoch open while both modes execute so source conditions match. Report all36
timings plus lock/entry/exit hash cost;do not add nested profiling times or call
the restore-only ratio PPO speedup. After each restore verify bytes and source
observations. Negative altered-source-manifest and immutable-target tests must
reject without native state changes; expired tokens reject after exit.

Then replay the existing27 declared profiles/source/mode(two fixed directions,
two PD amplitudes,three repeats and3 baseline repeats). Alternate mode order
per source/profile. Save all324 native8-step packets and windows,actual raw/PD,
warm,mocap/EFC/physics. Default task/geometry/no-aid/force/impulse gates remain;
these counterfactual resets are not full-task evaluation or expert data.
Physical budget6 sources:2*59*8=944 prefix +324*8=2592 branches,total3536.
No neural/Actor/optimizer updates. All records stay CHECK/NEVER_FIT.

Independently decode all packets and compare modes at identical sources. Use
the frozen existing WM TRAIN y scales only for dimensionless response reporting.
Compare each profile's endpoint output mean across modes against the sum of
their observed3-repeat radii,plus normalized L2 floor1e-5. Keep every failed
comparison; observed finite spread is not a probability bound. Also require
unchanged active raw/PD and task/physics checks. A restore-only timing gain is
reported separately even if numerical future-response equivalence is inconclusive;
do not silently relax the floor to enable fast data collection.
