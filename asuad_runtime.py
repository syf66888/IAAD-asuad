"""Portable launch configuration for the full asuad model.

This module uses only Python's standard library. --help and --dry-run do not
import PyTorch or start a model. Learned module attributes and checkpoint keys
are kept exactly as in the audited experiment implementation.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
SWIN_NAME = 'swin_base_patch244_window877_kinetics600_22k.pth'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def checkpoint_link(source, target):
    """Expose an immutable external weight beside derived inference settings."""
    source = Path(source).resolve(strict=True)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.exists() or not os.path.samefile(source, target):
            raise FileExistsError('A different checkpoint already occupies ' + str(target))
        return
    try:
        os.link(source, target)
    except OSError:
        try:
            target.symlink_to(source)
        except OSError as error:
            raise RuntimeError('This filesystem needs file hard-link or symlink support for checkpoint aliases.') from error


def require_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Required file is missing: ' + str(path))
    return path


def make_parser(mode):
    description = {
        'train': 'Train the full asuad model independently on BDDX or MMAU.',
        'inference': 'Generate both captions and evaluate the full asuad checkpoint on a prepared split.',
        'scst': 'Continue CE training with SCST on BDDX or MMAU.',
    }[mode]
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--dataset', choices=['bddx', 'mmau'], default='mmau' if mode == 'scst' else 'bddx')
    parser.add_argument('--data-root', required=True, type=Path, help='Parent directory containing BDDX/ or MMAU/.')
    parser.add_argument('--flow-cache', required=True, type=Path, help='Matching completed offline LK cache.')
    parser.add_argument('--weights-root', type=Path, default=Path(os.environ.get('ASUAD_WEIGHTS', str(ROOT.parent / 'weights'))))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--config', type=Path, help='Optional JSON template; runtime paths are relocated below.')
    parser.add_argument('--checkpoint', type=Path, help='Inference or SCST checkpoint; optional explicit CE initializer. CE training starts fresh by default.')
    parser.add_argument('--native-precision', choices=['bf16', 'fp16', 'fp32'], default='bf16')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--batch-size', type=int, help='Per-device microbatch; default is the archived setting.')
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--dry-run', action='store_true', help='Write relocated settings and print the command without running it.')
    if mode == 'inference':
        parser.add_argument('--split-yaml', help='Prepared split YAML, relative to data-root or absolute.')
    else:
        parser.add_argument('--epochs', type=int)
        parser.add_argument('--max-train-steps', type=int, default=0, help='Smoke-test optimizer steps; 0 means the full run.')
        parser.add_argument('--resume-state', type=Path, help='Complete optimizer/scheduler/RNG state from this implementation.')
    if mode == 'scst':
        parser.add_argument('--reward-cache', required=True, type=Path, help='Training-only SCST reward cache JSON.gz.')
        parser.add_argument('--ce-output-dir', type=Path, help='Completed CE output directory containing checkpoint-best/ and best_checkpoint.json.')
        parser.add_argument('--base-record', type=Path, help='CE evaluation record when selecting an explicit --checkpoint.')
    return parser


def scst_initializer(options, dataset):
    """Use the selected model and evaluation record from the caller's CE run."""
    if options.ce_output_dir:
        if options.checkpoint or options.base_record:
            raise ValueError('Use --ce-output-dir or an explicit --checkpoint with --base-record.')
        ce_output = options.ce_output_dir.expanduser().resolve()
        record_path = require_file(ce_output / 'best_checkpoint.json')
        checkpoint = require_file(ce_output / 'checkpoint-best/model.bin')
    else:
        if options.checkpoint is None or options.base_record is None:
            raise ValueError('SCST needs a completed CE run: pass --ce-output-dir, or --checkpoint with --base-record.')
        checkpoint = require_file(options.checkpoint.expanduser().resolve())
        record_path = require_file(options.base_record.expanduser().resolve())
    base = copy.deepcopy(read_json(record_path))
    if base.get('dataset_name', 'BDDX') != dataset:
        raise ValueError('The CE evaluation record belongs to a different dataset.')
    if base.get('validation_yaml') != f'{dataset}/testing_32frames.yaml':
        raise ValueError('The CE evaluation record must match the configured evaluation YAML.')
    for task in ('des', 'exp'):
        for metric in ('Bleu_4', 'CIDEr'):
            value = base['metrics'][task][metric]
            if not math.isfinite(value) or value < 0:
                raise ValueError('The CE evaluation record contains invalid caption metrics.')
    actual = sha256_file(checkpoint)
    if base.get('model_sha256') and base['model_sha256'] != actual:
        raise ValueError('The selected CE checkpoint does not match its evaluation record.')
    base.update(model_path=str(checkpoint), checkpoint=str(checkpoint.parent),
                model_sha256=actual, dataset_name=dataset,
                validation_yaml=f'{dataset}/testing_32frames.yaml', model_saved=True)
    return checkpoint, base


def prepare(options, mode):
    dataset = options.dataset.upper()
    if options.device == 'cpu' and options.native_precision != 'fp32':
        raise ValueError('Use --native-precision fp32 for CPU execution.')
    for key in ('batch_size', 'epochs'):
        value = getattr(options, key, None)
        if value is not None and value < 1:
            raise ValueError('--' + key.replace('_', '-') + ' must be positive.')
    if options.num_workers is not None and options.num_workers < 0:
        raise ValueError('--num-workers must be nonnegative.')
    if getattr(options, 'max_train_steps', 0) < 0:
        raise ValueError('--max-train-steps must be nonnegative.')
    weights = options.weights_root.expanduser().resolve()
    data = options.data_root.expanduser().resolve()
    flow = options.flow_cache.expanduser().resolve()
    output = (options.output_dir or ROOT / 'outputs' / options.dataset / mode).expanduser().resolve()
    template_name = f'{options.dataset}_' + ('inference' if mode == 'inference' else 'scst' if mode == 'scst' else 'train') + '.json'
    config = copy.deepcopy(read_json(options.config or ROOT / 'configs' / template_name))
    # Only the full-model architecture from the released run is supported.
    if (config.get('vidswin_size') != 'base' or config.get('video_swin_depths') not in (None, '', [])
            or config.get('num_hidden_layers', -1) not in (-1, 12)
            or config.get('hidden_size', -1) not in (-1, 768)
            or config.get('distill_teacher_checkpoint') or config.get('tie_weights', False)
            or any(config.get(k, False) for k in ('multitask', 'use_car_sensor', 'only_signal', 'use_asr',
                                                   'object_relation_enabled', 'spatial_motion_enabled'))):
        raise ValueError('The release expects the archived 0.2338B purely visual architecture.')
    config.update(dataset_name=dataset, data_dir=str(data), flow_cache_dir=str(flow),
                  train_yaml=f'{dataset}/training_32frames.yaml',
                  test_yaml=getattr(options, 'split_yaml', None) or f'{dataset}/testing_32frames.yaml',
                  val_yaml=getattr(options, 'split_yaml', None) or f'{dataset}/testing_32frames.yaml',
                  model_name_or_path=str(ROOT / 'models/captioning/bert-base-uncased'),
                  config_name='', tokenizer_name='',
                  vision_yolo_weights=str(require_file(weights / 'pretrained/yolov5su.pt')),
                  video_swin_pretrained_path=str(require_file(weights / 'pretrained' / SWIN_NAME)),
                  output_dir=str(output), device=options.device, local_rank=-1,
                  mixed_precision_method='native', native_precision=options.native_precision,
                  do_train=mode != 'inference', do_eval=mode == 'inference', do_test=mode == 'inference',
                  do_signal_eval=False, scst=mode == 'scst', resume_checkpoint='None', eval_model_dir='',
                  pretrained_checkpoint='', resume_training_state='', scst_resume='')
    require_file(ROOT / 'models/captioning/bert-base-uncased/config.json')
    require_file(ROOT / 'models/captioning/bert-base-uncased/vocab.txt')
    if options.num_workers is not None:
        config['num_workers'] = options.num_workers
    if options.batch_size is not None:
        config['per_gpu_eval_batch_size' if mode == 'inference' else 'per_gpu_train_batch_size'] = options.batch_size
    if mode != 'inference':
        if options.epochs is not None:
            config['num_train_epochs'] = options.epochs
        config['max_train_steps'] = options.max_train_steps
        world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('OMPI_COMM_WORLD_SIZE', '1')))
        config['effective_batch_size'] = (config['per_gpu_train_batch_size']
                                           * config['gradient_accumulation_steps'] * world_size)
        if options.resume_state:
            config['scst_resume' if mode == 'scst' else 'resume_training_state'] = str(require_file(options.resume_state.resolve()))
    checkpoint = options.checkpoint
    base = None
    if mode == 'scst':
        checkpoint, base = scst_initializer(options, dataset)
    if checkpoint is None:
        if mode == 'inference':
            checkpoint = weights / options.dataset / 'model.bin'
    if checkpoint is not None:
        checkpoint = require_file(checkpoint.expanduser().resolve())
    if mode == 'inference':
        # The archived loader rereads ../log/args.json; keep it beside a linked
        # weight, using only derived settings and the original checkpoint bytes.
        alias = output / 'runtime/checkpoint/model.bin'
        checkpoint_link(checkpoint, alias)
        config.update(eval_model_dir=str(alias.parent), resume_checkpoint=str(alias))
        write_json(output / 'runtime/log/args.json', config)
    elif mode == 'train' and checkpoint is not None:
        alias = output / 'initialization/model.bin'
        checkpoint_link(checkpoint, alias)
        config['pretrained_checkpoint'] = str(alias.parent)
    elif mode == 'scst':
        reward = require_file(options.reward_cache.expanduser().resolve())
        config.update(scst_checkpoint=str(checkpoint), scst_base_record=base, scst_reward_cache=str(reward))
    config_path = output / 'launch.json'
    write_json(config_path, config)
    module = 'src.tasks.train_scst' if mode == 'scst' else 'src.tasks.train'
    argv = [sys.executable, '-m', module, '--config', str(config_path)]
    return dict(mode=mode, dataset=dataset, config=config, config_path=str(config_path),
                module=module, argv=argv, working_directory=str(ROOT),
                checkpoint=str(checkpoint) if checkpoint is not None else None,
                prepared_only=options.dry_run,
                prediction_directory=str(output / 'runtime/checkpoint') if mode == 'inference' else None)


def launch(mode):
    parser = make_parser(mode)
    options = parser.parse_args()
    try:
        receipt = prepare(options, mode)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    printable = {k: v for k, v in receipt.items() if k != 'config'}
    print(json.dumps(printable, indent=2), flush=True)
    if options.dry_run:
        return
    # Both data and weights must be provided explicitly before model execution.
    if not options.data_root.is_dir() or not options.flow_cache.is_dir():
        parser.error('data-root and flow-cache must exist before executing the model.')
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    sys.argv = [receipt['module'], '--config', receipt['config_path']]
    runpy.run_module(receipt['module'], run_name='__main__')
