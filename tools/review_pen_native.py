"""Review saved native objective comparisons, with explicit historical mappings."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from supermimic_research.mechanics.pen_native_search_review import main

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-relocations',type=Path)
    main(p.parse_args())
