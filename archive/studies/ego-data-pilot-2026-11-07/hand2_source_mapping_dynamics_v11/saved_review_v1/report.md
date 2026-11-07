# Source-mapping dynamic saved review

All four cases ABORT_LOSS_OF_HOLD. These are loss-of-hold prefixes, not stable holding durations. Neither dt pair completed2s; all original tail metrics remain null.

|Mapping/dt ms|Steps / abort s|Thumb <=.1N PRE s|Index <=.1N PRE s|Nonpad >.1N PRE s|COM >5mm POST s|Angle >5deg POST s|
|---|---|---|---|---|---|---|
|exact_source_original_hand/0.3125|361 / 0.1128125|0.004375 (tick14)|0.0025 (tick8)|0.04125 (tick132)|0.039375 (tick126)|0.0328125 (tick105)|
|exact_source_original_hand/0.15625|728 / 0.11375|0.004375 (tick28)|0.0025 (tick16)|0.0575 (tick368)|0.039375 (tick252)|0.03265625 (tick209)|
|source1mm_opposed_side_original_hand/0.3125|622 / 0.194375|0.0084375 (tick27)|0.1190625 (tick381)|0.1034375 (tick331)|0.0734375 (tick235)|0.075625 (tick242)|
|source1mm_opposed_side_original_hand/0.15625|1248 / 0.195|0.0084375 (tick54)|0.1196875 (tick766)|0.1040625 (tick666)|0.07390625 (tick473)|0.0759375 (tick486)|

Each row i stores solverPRE force/energy at tick i and POST state at tick i+1; integer nominal times are distinct from accumulated native clocks. JSON identifies first/peak nonpad contacts via saved22-row geometry ID/name mappings. No model was loaded to infer names. Terminal/pins,2959steps,2581contact rows, PRE aggregation and all original case gates were checked. No missing tail/root/COM RMS was filled. Noaid relies on raw initial evidence and the preserved runtime/source guard; unsaved per-tick applied-force vectors are not independently reconstructed.
