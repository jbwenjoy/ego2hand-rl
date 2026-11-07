# Repository scope

This is the development home for the Wuji Hand2 SuperMimic + CHORD research.
Make shared implementation changes in `src/`; express experiment differences
as configurations. Do not create another top-level study directory in
`D:/workspace` for ordinary iterations.

`archive/` contains historical source snapshots and provenance. Preserve their
bytes. They are reference material, not additional implementations to patch.
Large datasets, checkpoints, simulator assets and environments stay outside Git
and are located through the artifact registry. Never silently change a frozen
experiment's data, source pins, physics, split, or reported validation scope.

Current legacy training and evaluation processes may still use original paths.
Do not move, rewrite, replace with junctions, or uninstall those dependencies.
New runtime code must use package imports and explicit artifact paths rather
than `D:/workspace/<study>` imports or sibling-directory discovery.

No GPU performance changes are part of the initial consolidation. Verify
numerical behavior and provenance first. Later performance changes should be
measured on the same data and update budget, with geometry/gradient checks and
scientific limitations retained.

Continue routine authorized, reversible work without adding approval gates.
Never send external messages or publish/push a repository without user scope.
