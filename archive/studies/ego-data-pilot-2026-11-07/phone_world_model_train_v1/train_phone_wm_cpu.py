"""Metadata-only default; Root alone authorizes --run after source/data review."""
import argparse
import json
from pathlib import Path
import train_phone_wm as m

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.protocol is None:
        if args.run:
            parser.error("--run requires --protocol")
        print(json.dumps(m.declaration()))
        return
    if args.run:
        result = m.run_training(args.protocol.resolve(strict=True))
        print(json.dumps({"status": "COMPLETED", "report": str(result)}))
    else:
        p, d, out = m.metadata_preflight(args.protocol.resolve(strict=True))
        print(json.dumps({"status": "METADATA_VALID_NO_TORCH_ARRAY_OR_MODEL_EXECUTION", "output": str(out),
            "family_counts": {k: len(v) for k, v in d["family_split"].items()}, "updates": 5000,
            "first_checkpoint": 250, "actual_execution": 0}))

if __name__ == "__main__":
    main()
