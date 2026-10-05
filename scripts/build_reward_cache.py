"""Build a training-caption reward cache for BDDX or MMAU SCST."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', '--dataset_name', type=str.upper, choices=['BDDX', 'MMAU'], default='BDDX')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'datasets')
    parser.add_argument('--train_yaml', type=Path, help='Optional explicit training annotation YAML.')
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    from src.tasks.scst_reward import build_training_cache
    training_yaml = args.train_yaml or args.data_root / args.dataset / 'training_32frames.yaml'
    cache = build_training_cache(str(training_yaml), str(args.cache), dataset_name=args.dataset)
    print(json.dumps({'cache': str(args.cache), 'dataset': args.dataset,
                      'training_captions': cache['tasks']['des']['ref_len']}, indent=2))

if __name__ == '__main__':
    main()
