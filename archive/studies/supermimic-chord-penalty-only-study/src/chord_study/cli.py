import argparse
import json
from pathlib import Path
import numpy as np
from .reference import Reference
from .evaluation import evaluate
from .experiments import perturb_reference, save_reference


def main():
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command", required=True)
    inspect = subs.add_parser("inspect")
    inspect.add_argument("reference", type=Path)
    inspect.add_argument("--contact-required", action="store_true")
    ev = subs.add_parser("evaluate")
    ev.add_argument("reference", type=Path)
    ev.add_argument("rollout", type=Path)
    ev.add_argument("--output", type=Path, required=True)
    perturb = subs.add_parser("perturb")
    perturb.add_argument("reference",type=Path)
    perturb.add_argument("--output",type=Path,required=True)
    perturb.add_argument("--joint-bias-rad",type=float,default=0.)
    perturb.add_argument("--time-offset-s",type=float,default=0.)
    perturb.add_argument("--seed",type=int,default=0)
    args = parser.parse_args()
    ref = Reference.load(args.reference)
    if args.command == "inspect":
        result = ref.validate(contact_required=args.contact_required)
    elif args.command == "perturb":
        ref = perturb_reference(ref,joint_bias_rad=args.joint_bias_rad,time_offset_s=args.time_offset_s,seed=args.seed)
        save_reference(ref,args.output)
        result={"source_id":ref.meta["source_id"],"output":str(args.output),"split_group":ref.meta["split_group"],**ref.validate()}
    else:
        with np.load(args.rollout,allow_pickle=False) as data:
            arrays = {k:data[k] for k in data.files}
        meta = json.loads(args.rollout.with_suffix(".json").read_text(encoding="utf-8-sig"))
        result = evaluate(ref,arrays,meta)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__ == "__main__":
    main()
