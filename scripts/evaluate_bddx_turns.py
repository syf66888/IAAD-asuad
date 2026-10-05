"""Evaluate BDDX turning accuracy from saved descriptions without a model."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evalcap.bddx_turn_accuracy import evaluate_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True, type=Path,
                        help='Merged prediction JSON, des COCO JSON, or prediction TSV.')
    parser.add_argument('--references', type=Path,
                        default=ROOT / 'datasets/BDDX_des/testing_32frames_caption_coco_format.json',
                        help='BDDX COCO action annotations or prepared caption TSV.')
    parser.add_argument('--output-dir', type=Path,
                        help='Folder for turn_accuracy.json and turn_predictions.json; defaults to the predictions folder.')
    args = parser.parse_args()
    output = args.output_dir or args.predictions.parent
    try:
        report = evaluate_files(args.predictions, args.references,
                                output / 'turn_accuracy.json', output / 'turn_predictions.json')
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    explicit = report['explicit_turns']
    score = '{:.2f}%'.format(explicit['accuracy_percent']) if explicit['accuracy'] is not None else 'N/A'
    print('BDDX turning accuracy: {} ({}/{})'.format(score, explicit['correct'], explicit['reference']))
    for direction, item in explicit['per_direction'].items():
        score = '{:.2f}%'.format(item['accuracy_percent']) if item['accuracy'] is not None else 'N/A'
        print('{}: {} ({}/{})'.format(direction, score, item['correct'], item['reference']))
    print('Results saved to ' + str(output.resolve()))


if __name__ == '__main__':
    main()
