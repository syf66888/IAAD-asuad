"""A fixed, scale-balanced selection rule for the user's four BDDX metrics."""
import json
import math
import os
from pathlib import Path
from src.tasks.dataset_protocol import dataset_name, split_yaml


BASELINE = {
    'des': {'Bleu_4': 0.352760723579056, 'CIDEr': 2.608003661952311},
    'exp': {'Bleu_4': 0.10773384956087212, 'CIDEr': 0.9774888915237454},
}
SELECTION_NAME = 'geometric_mean_baseline_normalized_des_exp_B4_CIDEr'


def selection_name(dataset='BDDX'):
    return (SELECTION_NAME if dataset_name(dataset) == 'BDDX'
            else 'geometric_mean_des_exp_B4_CIDEr')


def joint_score(metrics, dataset='BDDX'):
    """Equal log weight for des/exp and B4/CIDEr; historical baseline = 1."""
    dataset = dataset_name(dataset)
    reference_metrics = BASELINE if dataset == 'BDDX' else {
        'des': {'Bleu_4': 1.0, 'CIDEr': 1.0},
        'exp': {'Bleu_4': 1.0, 'CIDEr': 1.0}}
    ratios = []
    for kind, targets in reference_metrics.items():
        for name, reference in targets.items():
            value = float(metrics[kind][name])
            if not math.isfinite(value) or value < 0:
                raise ValueError('Invalid {} {}: {}'.format(kind, name, value))
            ratios.append(value / reference)
    if any(value == 0 for value in ratios):
        return 0.0
    return math.exp(sum(math.log(value) for value in ratios) / len(ratios))


def selection_record(record, **extra):
    dataset = dataset_name(record.get('dataset_name', 'BDDX'))
    if record.get('validation_yaml', record.get('evaluation_yaml')) != split_yaml(dataset, 'testing'):
        raise ValueError('Selection must use the explicitly configured dataset testing protocol.')
    result = dict(record)
    result.update(dataset_name=dataset,
                  selection_metric=selection_name(dataset),
                  selection_score=joint_score(record['metrics'], dataset))
    result.update(extra)
    return result


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))
