"""Evaluate BDDX turn direction agreement using only action/des text."""
from collections import Counter
import json
from pathlib import Path

from .turn_semantics import classify

DIRECTIONS = ('left', 'right')


def _insert(rows, key, caption):
    if key is None:
        raise ValueError('Every caption must have a sample ID.')
    key = str(key)
    if not key or key in rows:
        raise ValueError('Sample IDs must be nonempty and unique: ' + key)
    if not isinstance(caption, str):
        raise ValueError('Expected caption text for sample ' + key)
    rows[key] = caption


def read_references(path):
    """Read COCO action annotations or the prepared two-column caption TSV."""
    path = Path(path)
    rows = {}
    if path.suffix.lower() == '.tsv':
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():
                continue
            key, raw = line.split('\t', 1)
            values = json.loads(raw)
            if not isinstance(values, list) or len(values) != 1:
                raise ValueError('Expected one action reference per sample: ' + key)
            _insert(rows, key, values[0].get('action', values[0].get('caption')))
    else:
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(data, dict) or not isinstance(data.get('annotations'), list):
            raise ValueError('References must be a COCO annotation JSON.')
        for row in data['annotations']:
            # Main BDDX captions concatenate action and explanation. Use action.
            _insert(rows, row.get('image_id'), row.get('action', row.get('caption')))
        if 'images' in data:
            image_ids = [str(row['id']) for row in data['images']]
            if len(image_ids) != len(set(image_ids)) or set(image_ids) != set(rows):
                raise ValueError('Image IDs and action annotation IDs must match exactly.')
    if not rows:
        raise ValueError('No action references were found.')
    return rows


def read_predictions(path):
    """Read merged description JSON, description COCO JSON, or prediction TSV."""
    path = Path(path)
    rows = {}
    if path.suffix.lower() == '.tsv':
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():
                continue
            columns = line.split('\t')
            if len(columns) not in (2, 3):
                raise ValueError('Prediction TSV must contain ID and description columns.')
            values = json.loads(columns[1])
            if not isinstance(values, list) or len(values) != 1:
                raise ValueError('Expected one generated description per sample: ' + columns[0])
            _insert(rows, columns[0], values[0].get('caption'))
    else:
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(data, list):
            raise ValueError('Predictions must be a list of description records.')
        for row in data:
            _insert(rows, row.get('image_id'), row.get('description', row.get('caption')))
    return rows


def _fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def _is_turn(label):
    return label['kind'] == 'turn' and label['direction'] in DIRECTIONS


def _label_name(label):
    return label['kind'] + ('_' + label['direction'] if label['direction'] else '')


def _subset_report(records):
    outcomes = Counter(row['outcome'] for row in records)
    accuracy = _fraction(outcomes['correct'], len(records))
    result = dict(reference=len(records), correct=outcomes['correct'], accuracy=accuracy,
                  accuracy_percent=100 * accuracy if accuracy is not None else None,
                  outcomes=dict(sorted(outcomes.items())), per_direction={})
    for direction in DIRECTIONS:
        selected = [row for row in records if row['reference_label']['direction'] == direction]
        counts = Counter(row['outcome'] for row in selected)
        accuracy = _fraction(counts['correct'], len(selected))
        result['per_direction'][direction] = dict(
            reference=len(selected), correct=counts['correct'], accuracy=accuracy,
            accuracy_percent=100 * accuracy if accuracy is not None else None,
            outcomes=dict(sorted(counts.items())))
    return result


def evaluate(predictions, references):
    """Return summary and per-sample decisions; missing/extra IDs are errors."""
    if not references or set(predictions) != set(references):
        missing = len(set(references) - set(predictions))
        extra = len(set(predictions) - set(references))
        raise ValueError('Predictions must cover every reference exactly once '
                         '(missing={}, extra={}).'.format(missing, extra))
    records = []
    for key, action in references.items():
        gold, predicted = classify(action), classify(predictions[key])
        if _is_turn(gold):
            if _is_turn(predicted):
                outcome = 'correct' if gold['direction'] == predicted['direction'] else 'opposite_direction'
            elif predicted['kind'] == 'lane':
                outcome = ('lane_substitution_same_direction'
                           if gold['direction'] == predicted['direction'] else 'lane_substitution_other')
            else:
                outcome = 'omitted_turn'
        elif _is_turn(predicted):
            outcome = 'turn_not_supported_by_reference'
        else:
            outcome = 'outside_turn_target'
        records.append(dict(image_id=key, reference=action, prediction=predictions[key],
                            reference_label=gold, predicted_label=predicted, outcome=outcome))

    turns = [row for row in records if _is_turn(row['reference_label'])]
    explicit = [row for row in turns if row['reference_label']['strict_turn']]
    generated_turns = [row for row in records if _is_turn(row['predicted_label'])]
    broad = _subset_report(turns)
    broad['predicted'] = len(generated_turns)
    broad['recall'] = broad['accuracy']
    broad['precision'] = _fraction(broad['correct'], len(generated_turns))
    broad['direction_accuracy_given_both_turn'] = _fraction(
        broad['correct'], broad['correct'] + broad['outcomes'].get('opposite_direction', 0))
    f1s = []
    for direction in DIRECTIONS:
        item = broad['per_direction'][direction]
        generated = sum(row['predicted_label']['direction'] == direction for row in generated_turns)
        item.update(predicted=generated, precision=_fraction(item['correct'], generated),
                    recall=item['accuracy'], f1=_fraction(2 * item['correct'], item['reference'] + generated))
        if item['f1'] is not None:
            f1s.append(item['f1'])
    broad['macro_f1'] = sum(f1s) / len(f1s) if f1s else None
    explicit_report = _subset_report(explicit)
    result = dict(dataset='BDDX', scope='Description agreement with annotated action',
                  total_samples=len(records), turning_accuracy=explicit_report['accuracy'],
                  explicit_turns=explicit_report, broad_steering=broad,
                  reference_label_counts=dict(Counter(_label_name(row['reference_label']) for row in records)),
                  predicted_label_counts=dict(Counter(_label_name(row['predicted_label']) for row in records)))
    return result, records


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def evaluate_files(predictions_file, references_file, output_file=None, details_file=None):
    result, records = evaluate(read_predictions(predictions_file), read_references(references_file))
    if output_file is not None:
        _write_json(output_file, result)
    if details_file is not None:
        _write_json(details_file, records)
    return result
