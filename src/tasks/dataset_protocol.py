"""Explicit identities for the two audited dual-caption datasets.

MMAU preserves the supplied action/justification columns. Its second column
contains accident prevention advice, not BDDX driving explanations.
"""
SUPPORTED_DATASETS = ('BDDX', 'MMAU')


def dataset_name(value='BDDX'):
    if value not in SUPPORTED_DATASETS:
        raise ValueError('Expected an explicitly supported dataset: BDDX or MMAU.')
    return value


def split_yaml(dataset, split):
    dataset = dataset_name(dataset)
    if split not in ('training', 'testing'):
        raise ValueError('Only audited training/testing splits are supported.')
    return dataset + '/' + split + '_32frames.yaml'


def validate_splits(dataset, training, testing):
    dataset = dataset_name(dataset)
    if training != split_yaml(dataset, 'training') or testing != split_yaml(dataset, 'testing'):
        raise ValueError('Dataset identity and configured training/testing YAMLs disagree.')
    return dataset


def task_semantics(dataset):
    return ({'des': 'accident description', 'exp': 'accident prevention advice'}
            if dataset_name(dataset) == 'MMAU'
            else {'des': 'driving description', 'exp': 'driving explanation'})
