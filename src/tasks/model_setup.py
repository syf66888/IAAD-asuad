"""Initialization and checkpoint selection for the full asuad model."""
import math

def checkpoint_score(metrics, selection, dataset='BDDX'):
    if selection == 'sum_CIDEr':
        return sum(result['CIDEr'] for result in metrics.values())
    if selection != 'joint_b4_cider':
        raise ValueError('Unknown checkpoint selection: ' + selection)
    if dataset == 'MMAU':
        from src.tasks.checkpoint_selection import joint_score
        return joint_score(metrics, dataset)
    baseline = {'des': {'Bleu_4': .352760723579056, 'CIDEr': 2.608003661952311},
                'exp': {'Bleu_4': .10773384956087212, 'CIDEr': .9774888915237454}}
    ratios = [float(metrics[task][metric]) / reference
              for task, refs in baseline.items() for metric, reference in refs.items()]
    if any(not math.isfinite(value) or value < 0 for value in ratios):
        raise ValueError('Joint checkpoint selection requires finite, nonnegative metrics.')
    return 0.0 if any(value == 0 for value in ratios) else math.exp(sum(math.log(x) for x in ratios) / 4)

def validate_model_initialization(args, model):
    report = dict(total_parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    expected = int(getattr(args, 'expected_model_parameters', 233764716))
    if report['total_parameters'] != expected:
        raise RuntimeError('asuad full-model parameter mismatch: %s != %s' % (report['total_parameters'], expected))
    return report
