"""Run this checkout without installing over a live legacy editable package."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supermimic_research.cli import main

raise SystemExit(main())
