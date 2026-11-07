# Environment boundaries

The repository combines source, not incompatible installed environments.
`environments/profiles.toml` records the relevant profiles. The full original
project configurations and lockfiles are historical records in the archive;
they must not be mechanically merged into one dependency solver input.

The lightweight root project provides the repository CLI and shared package.
Learning and native extras declare broad package requirements; the current
Hand2 numerical baseline is the exact installed profile reported by `doctor`.
They are not a replacement for that platform-specific environment lock.

During migration, use the existing Hand2 interpreter to start this checkout's
bootstrap in a fresh process:

```powershell
& ..\dex-supermimic-research\.venv\Scripts\python.exe scripts\research.py doctor
& ..\dex-supermimic-research\.venv\Scripts\python.exe scripts\research.py artifacts --verify
```

The bootstrap inserts this checkout's `src` only in its child process. It does
not edit the old environment's `.pth` files, reinstall the old editable
projects, initialize CUDA for inspection, or change any active legacy worker.
Do not install the new `dex_supermimic` namespace into that live environment
while a frozen process still verifies the old imported module path.

Use `paths.local.toml` for explicit machine data-root overrides. The default
manifest is relative to `configs/artifacts.toml`, so this workspace works
without hardcoded study paths inside production Python. Nested historical data
pins can be explicitly relocated by `ArtifactRegistry.resolve_pin` without
editing the original JSON; size and SHA256 remain required.
