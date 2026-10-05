from __future__ import absolute_import, division, print_function

import os
import sys
pythonpath = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
print(pythonpath)
sys.path.insert(0, pythonpath)
import os.path as op
import json
import math
from contextlib import nullcontext
import time
import datetime
import torch
import torch.distributed as dist
import gc
import numpy as np
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
except ImportError:
    amp = DDP = None
from tqdm import tqdm
from src.configs.config import (basic_check_arguments, shared_configs, restore_training_settings)
from src.datasets.vl_dataloader import make_data_loader
from src.evalcap.utils_caption_evaluate import evaluate_on_coco_caption, two_cap_evaluate_on_coco_caption, caption_result_path
from src.evalcap.bddx_turn_accuracy import evaluate_files as evaluate_bddx_turns
from src.utils.logger import LOGGER as logger
from src.utils.logger import (TB_LOGGER, RunningMeter, add_log_to_file)
from src.utils.load_save import TrainingRestorer, TrainingSaver
from src.utils.comm import (is_main_process,
                            get_rank, get_world_size, dist_init)
from src.utils.miscellaneous import (NoOp, mkdir, set_seed, str_to_bool,
                                    delete_tsv_files, concat_tsv_files)
from src.utils.metric_logger import MetricLogger
from src.utils.tsv_file_ops import tsv_writer, double_tsv_writer, reorder_tsv_keys
from src.utils.deepspeed import get_deepspeed_config, fp32_to_fp16
from src.modeling.asuad_model import AsuadModel
from src.modeling.load_swin import get_swin_model, reload_pretrained_swin
from src.modeling.load_bert import get_bert_model
from src.tasks.model_setup import checkpoint_score, validate_model_initialization
from src.solver import AdamW, WarmupLinearLR
import global_params

try:
    from azureml.core.run import Run
    aml_run = Run.get_context()
except ImportError:
    aml_run = NoOp()

def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, -1)[1].data # argmax
    return logits == labels

def autocast_context(args):
    if args.mixed_precision_method == 'native':
        precision = getattr(args, 'native_precision', 'fp32')
        if precision != 'fp32':
            dtype = torch.bfloat16 if precision == 'bf16' else torch.float16
            return torch.autocast(device_type=args.device.type, dtype=dtype)
    if args.mixed_precision_method == 'fairscale':
        return torch.autocast(device_type='cuda', dtype=torch.float16)
    return nullcontext()


def accumulation_window(iteration, max_iter, iters_per_epoch, grad_accum_steps):
    """Return actual microbatch count and flush flag, including short epoch tails."""
    offset = (iteration - 1) % iters_per_epoch
    window_start = iteration - 1 - offset % grad_accum_steps
    epoch_end = min(max_iter, (iteration - 1) // iters_per_epoch * iters_per_epoch + iters_per_epoch)
    window_end = min(window_start + grad_accum_steps, epoch_end)
    return window_end - window_start, iteration == window_end


def native_grad_scaler(enabled):
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        return torch.amp.GradScaler('cuda', enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def capture_training_rng(device):
    import random
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)


def restore_training_rng(state, device):
    import random
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if device.type == 'cuda' and state.get('cuda') is not None:
        torch.cuda.set_rng_state(state['cuda'], device=device)


def add_cached_motion(args, inputs, batch, is_train):
    spatial = bool(getattr(args, 'spatial_motion_enabled', False))
    if spatial and not getattr(args, 'flow_cache_dir', ''):
        raise ValueError('Spatial motion requires an offline dense flow cache.')
    if getattr(args, 'flow_cache_dir', ''):
        expected_length = (8 if is_train else 7) + int(spatial)
        if len(batch) != expected_length:
            raise ValueError(f'Cached flow expects {expected_length} batch tensors, received {len(batch)}.')
        motion = batch[-2] if spatial else batch[-1]
        if motion.ndim != 3 or motion.shape[1:] != (args.max_num_frames, 4):
            raise ValueError(f'Expected cached motion [B,{args.max_num_frames},4], got {tuple(motion.shape)}')
        inputs['motion_features'] = motion
        if spatial:
            dense = batch[-1]
            size = int(getattr(args, 'spatial_motion_size', 32))
            if dense.ndim != 5 or dense.shape[1:] != (args.max_num_frames, 2, size, size):
                raise ValueError(f'Expected dense motion [B,{args.max_num_frames},2,{size},{size}], got {tuple(dense.shape)}')
            inputs['dense_motion_features'] = dense
    return inputs


def mixed_precision_init(args, model):
    # Construct every parameter once; projection/fusion parameters use the LM LR.
    grouped = [[], [], [], []]
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        no_decay = parameter.ndim <= 1 or name.endswith('.bias')
        index = (0 if name.startswith('swin.') else 1) + (2 if no_decay else 0)
        grouped[index].append(parameter)
    groups = [dict(params=params, weight_decay=0.0 if i >= 2 else args.weight_decay,
                   lr=args.learning_rate * (args.backbone_coef_lr if i % 2 == 0 else 1.0))
              for i, params in enumerate(grouped)]
    method = args.mixed_precision_method
    if method == 'fairscale':
        raise ValueError('FairScale training was incomplete. Use native AMP or DeepSpeed.')
    optimizer_cls = torch.optim.AdamW if method == 'native' else AdamW
    optimizer = optimizer_cls(groups, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.scheduler == 'warmup_linear':
        scheduler = WarmupLinearLR(optimizer, args.max_global_step, warmup_ratio=args.warmup_ratio)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.max_global_step // 2), gamma=0.1)
    if method == 'deepspeed':
        # Avoid loading optional DeepSpeed/Triton extensions for native PPU AMP.
        import deepspeed
        model, optimizer, _, _ = deepspeed.initialize(
            config_params=get_deepspeed_config(args), model=model,
            optimizer=optimizer, lr_scheduler=scheduler)
    elif method == 'native':
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.local_rank] if args.device.type == 'cuda' else None,
                static_graph=True, gradient_as_bucket_view=True)
    elif method == 'apex':
        if amp is None:
            raise ImportError('Apex is unavailable; choose --mixed_precision_method native.')
        model, optimizer = amp.initialize(model, optimizer, enabled=True, opt_level=f'O{args.amp_opt_level}')
        if args.distributed:
            model = DDP(model)
    else:
        raise ValueError(f'Unknown mixed_precision_method: {method}')
    return args, model, optimizer, scheduler


def resume_evaluation_selection(state, args, resume_path):
    """Keep checkpoint selection comparable when resuming on another split."""
    history = state.get('eval_log', [])
    previous_yaml = state.get('evaluation_yaml')
    if previous_yaml is None and history:
        previous_yaml = history[-1].get('validation_yaml')
    current_yaml = getattr(args, 'val_yaml', None)
    if previous_yaml is None or previous_yaml == current_yaml:
        return state.get('best_score', float('-inf')), history

    # The model, optimizer, scheduler, iteration and RNG are restored unchanged.
    # Only scores measured against different references must not compete.
    transition = dict(
        resume_training_state=op.abspath(resume_path),
        previous_evaluation_yaml=previous_yaml,
        evaluation_yaml=current_yaml,
        resumed_global_step=int(state['step']),
        resumed_iteration=int(state['iteration']),
        previous_best_score=state.get('best_score'),
        previous_eval_log=history,
        selection_reset=True,
        training_state_preserved=True,
    )
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        path = op.join(args.output_dir, 'evaluation_split_transition.json')
        with open(path + '.tmp', 'w') as stream:
            json.dump(transition, stream, indent=2)
        os.replace(path + '.tmp', path)
    logger.info(f'Evaluation split changed from {previous_yaml} to {current_yaml}; '
                'resetting checkpoint selection history while preserving all training state.')
    return float('-inf'), []


def atomic_json_save(value, path):
    """Keep selection metadata intact if writing is interrupted."""
    temporary = path + '.tmp'
    with open(temporary, 'w') as stream:
        json.dump(value, stream, indent=2)
    os.replace(temporary, path)


def save_latest_training_state(args, model, training_saver, optimizer, scheduler, scaler,
                               iteration, global_step, best_score, eval_log, evaluation_pending):
    """Save a recoverable epoch before evaluation can fail independently."""
    rank_rng = capture_training_rng(args.device)
    if get_world_size() > 1:
        rank_rng_states = [None] * get_world_size()
        dist.all_gather_object(rank_rng_states, rank_rng)
    else:
        rank_rng_states = [rank_rng]
    if is_main_process():
        training_saver.save_model(
            op.join(args.output_dir, 'checkpoint-latest'), global_step, model,
            optimizer if args.mixed_precision_method != 'deepspeed' else None,
            scheduler=scheduler, scaler=scaler, iteration=iteration,
            extra_state=dict(
                epoch_boundary=iteration % args.iters_per_epoch == 0,
                evaluation_pending=bool(evaluation_pending), best_score=best_score,
                dataset_name=getattr(args, 'dataset_name', 'BDDX'),
                eval_log=eval_log, evaluation_yaml=getattr(args, 'val_yaml', None),
                world_size=get_world_size(), rank_rng_states=rank_rng_states,
                training_config={key: getattr(args, key) for key in
                    ('gradient_accumulation_steps', 'per_gpu_train_batch_size',
                     'max_num_frames', 'iters_per_epoch', 'num_train_epochs',
                     'max_global_step', 'checkpoint_selection', 'distill_teacher_sha256',
                     'distill_temperature', 'distill_token_weight', 'distill_visual_weight',
                     'distill_warmup_epochs')}))
    if get_world_size() > 1:
        dist.barrier()


def evaluate_and_select_checkpoint(args, val_dataloader, model, tokenizer, training_saver,
                                   iteration, global_step, best_score, eval_log):
    """Evaluate this exact model, including a pending evaluation on resume."""
    epoch = iteration / args.iters_per_epoch
    checkpoint_dir = op.join(args.output_dir, f'checkpoint-{math.ceil(epoch)}-{global_step}')
    evaluate_file = evaluate(args, val_dataloader, model, tokenizer, checkpoint_dir)
    if is_main_process():
        files = ([caption_result_path(evaluate_file, 'des'), caption_result_path(evaluate_file, 'exp')]
                 if args.use_sep_cap else [evaluate_file])
        names = ['des', 'exp'] if args.use_sep_cap else ['caption']
        metrics = {}
        for name, path in zip(names, files):
            with open(path) as stream:
                metrics[name] = json.load(stream)
        metric_name = getattr(args, 'checkpoint_selection', 'sum_CIDEr')
        score = checkpoint_score(metrics, metric_name, getattr(args, 'dataset_name', 'BDDX'))
        if not math.isfinite(score):
            raise FloatingPointError('Non-finite evaluation score; refusing checkpoint selection.')
        record = dict(epoch=epoch, iteration=iteration, global_step=global_step,
                      dataset_name=getattr(args, 'dataset_name', 'BDDX'),
                      selection_metric=metric_name, selection_score=score, metrics=metrics,
                      validation_yaml=args.val_yaml, prediction_dir=checkpoint_dir)
        eval_log.append(record)
        metric_prefix = 'testing' if op.basename(args.val_yaml).startswith('testing') else 'valid'
        for name, result in metrics.items():
            TB_LOGGER.log_scalar_dict({f'{metric_prefix}/{name}_{key}': value for key, value in result.items()})
        if score > best_score:
            best_score = score
            best_dir = op.join(args.output_dir, 'checkpoint-best')
            training_saver.save_model(best_dir, global_step, model)
            atomic_json_save(dict(record, checkpoint=best_dir), op.join(args.output_dir, 'best_checkpoint.json'))
        atomic_json_save(eval_log, op.join(args.output_dir, 'eval_logs.json'))
    return best_score, eval_log


def train(args, train_dataloader, val_dataloader, model, tokenizer, training_saver, optimizer, scheduler):
    max_iter = args.max_iter
    grad_accum_steps = max(1, args.gradient_accumulation_steps)
    iters_per_epoch = args.iters_per_epoch
    global_step = 0
    start_iteration = 0
    best_score = float('-inf')
    eval_log = []
    pending_evaluation = False
    method = args.mixed_precision_method
    scaler = native_grad_scaler(enabled=(method == 'native' and
                                getattr(args, 'native_precision', 'fp32') == 'fp16' and args.device.type == 'cuda'))
    if args.restore_ratio > 0:
        raise ValueError('Legacy restore_ratio does not restore sampler/scheduler state reliably; use a fresh output directory.')
    resume_path = getattr(args, 'resume_training_state', '')
    if resume_path:
        if method != 'native':
            raise ValueError('Portable training-state resume currently supports native training.')
        if op.isdir(resume_path):
            resume_path = op.join(resume_path, 'training_state.bin')
        state = torch.load(resume_path, map_location='cpu', weights_only=False)
        if state.get('dataset_name', 'BDDX') != getattr(args, 'dataset_name', 'BDDX'):
            raise ValueError('Cannot resume optimizer/scheduler state from another dataset; use pretrained weights.')
        if state.get('world_size', 1) != get_world_size():
            raise ValueError('Resume requires the same distributed world size as the saved checkpoint.')
        start_iteration = int(state['iteration'])
        if not state.get('epoch_boundary', False) or start_iteration % iters_per_epoch:
            raise ValueError('Resume requires an epoch-boundary latest checkpoint; a smoke-test partial epoch cannot be resumed.')
        saved_config = state.get('training_config', {})
        for key in ('gradient_accumulation_steps', 'per_gpu_train_batch_size', 'max_num_frames',
                    'iters_per_epoch', 'num_train_epochs', 'max_global_step', 'checkpoint_selection',
                    'distill_teacher_sha256', 'distill_temperature', 'distill_token_weight',
                    'distill_visual_weight', 'distill_warmup_epochs'):
            if key in saved_config and saved_config[key] != getattr(args, key):
                raise ValueError(f'Resume configuration mismatch for {key}: {saved_config[key]} != {getattr(args, key)}')
        (model.module if hasattr(model, 'module') else model).load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        if state.get('scaler'):
            scaler.load_state_dict(state['scaler'])
        global_step = int(state['step'])
        pending_evaluation = bool(state.get('evaluation_pending', False))
        best_score, eval_log = resume_evaluation_selection(state, args, resume_path)
        rank_states = state.get('rank_rng_states')
        if rank_states is not None:
            if len(rank_states) != get_world_size():
                raise ValueError('Checkpoint is missing a rank RNG state.')
            restore_training_rng(rank_states[get_rank()], args.device)
        else:
            legacy_cuda = state.get('rng_cuda')
            restore_training_rng(dict(python=state['rng_python'], numpy=state['rng_numpy'],
                                      torch=state['rng_torch'], cuda=legacy_cuda[0] if legacy_cuda else None),
                                 args.device)
        train_dataloader.batch_sampler.start_iter = start_iteration
        logger.info(f'Resuming complete training state from step {global_step}, microbatch {start_iteration}.')
        del state
    TB_LOGGER.global_step = global_step
    optimizer.zero_grad()
    training_saver.save_args(args)
    training_saver.save_tokenizer(tokenizer)
    meters = MetricLogger(delimiter='  ')
    training_start = time.time()
    log_start = training_start
    logged_samples = 0
    sync_after_step = global_step
    if pending_evaluation:
        logger.info(f'Completing pending evaluation at step {global_step} before another training update.')
        best_score, eval_log = evaluate_and_select_checkpoint(
            args, val_dataloader, model, tokenizer, training_saver,
            start_iteration, global_step, best_score, eval_log)
        save_latest_training_state(args, model, training_saver, optimizer, scheduler, scaler,
                                   start_iteration, global_step, best_score, eval_log, False)
    for iteration, (img_keys, batch, meta_data) in enumerate(train_dataloader, start=start_iteration + 1):
        if iteration > max_iter:
            break
        model.train()
        batch = tuple(t.to(args.device, non_blocking=True) for t in batch)
        inputs = dict(input_ids=batch[0], attention_mask=batch[1], token_type_ids=batch[2],
                      img_feats=batch[3], masked_pos=batch[4], masked_ids=batch[5], car_info=batch[6])
        inputs = add_cached_motion(args, inputs, batch, is_train=True)
        if method == 'deepspeed' and args.deepspeed_fp16:
            inputs = fp32_to_fp16(inputs)
        if iteration == 1:
            logger.info('Input shapes: ' + str({k: tuple(v.shape) for k, v in inputs.items()}))
        divisor, update_now = accumulation_window(iteration, max_iter, iters_per_epoch, grad_accum_steps)
        # Torch 1.13 static_graph needs a fully synchronized first accumulation window.
        # Later microbatches defer communication until the accumulation boundary.
        defer_sync = (method == 'native' and getattr(args, 'distributed', False)
                      and not update_now and global_step > sync_after_step)
        sync_context = model.no_sync() if defer_sync else nullcontext()
        with sync_context:
            with autocast_context(args):
                outputs = model(**inputs)
                loss, logits = outputs[:2]
                ce_loss = loss
                if args.multitask:
                    loss = loss + outputs[-3] * args.loss_sensor_w
                if args.learn_mask_enabled:
                    loss = loss + outputs[-1] * args.loss_sparse_w
            if not torch.isfinite(loss).all():
                raise FloatingPointError(f'Non-finite loss at microbatch {iteration}, keys={list(img_keys)[:4]}')
            labels = inputs['masked_ids']
            labels = labels[labels != -1]
            acc = compute_score_with_logits(logits, labels).float().mean()
            meters.update(loss=loss.item(), acc=acc.item())
            backward_loss = loss / divisor
            if method == 'deepspeed':
                # Engine GAS is explicitly 1; accumulation belongs only to this loop.
                model.backward(backward_loss)
            elif method == 'native':
                scaler.scale(backward_loss).backward()
            else:
                with amp.scale_loss(backward_loss, optimizer, delay_unscale=not update_now) as scaled_loss:
                    scaled_loss.backward()
        logged_samples += batch[0].shape[0] * get_world_size()
        if not update_now:
            continue
        global_step += 1
        if method == 'native':
            scaler.unscale_(optimizer)
        if global_step <= 2 and is_main_process():
            unwrapped = model.module if hasattr(model, 'module') else model
            for name, branch in (
                ('object_relations', getattr(getattr(unwrapped, 'visual_references', None), 'object_relation_pool', None)),
                ('spatial_motion', getattr(unwrapped, 'spatial_motion', None)),
            ):
                if branch is not None:
                    norms = {key: float(value.grad.detach().float().norm()) if value.grad is not None else None
                             for key, value in branch.named_parameters()}
                    logger.info(f'Branch gradients at step {global_step}, {name}: {norms}')
        if method != 'deepspeed' and args.max_grad_norm > 0:
            parameters = model.parameters() if method == 'native' else amp.master_params(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            if not torch.isfinite(norm) and not scaler.is_enabled():
                raise FloatingPointError(f'Non-finite gradient at step {global_step}')
            TB_LOGGER.add_scalar('train/grad_norm', float(norm), global_step)
        if method == 'deepspeed':
            model.step()
        elif method == 'native':
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
            else:
                logger.warning(f'AMP skipped update {global_step} because gradients overflowed.')
            optimizer.zero_grad(set_to_none=True)
        else:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        TB_LOGGER.add_scalar('train/loss', loss.item(), global_step)
        TB_LOGGER.add_scalar('train/lr_lm', optimizer.param_groups[1]['lr'], global_step)
        TB_LOGGER.step()
        if global_step == 1 or global_step % args.logging_steps == 0 or iteration == max_iter:
            speed = logged_samples / max(time.time() - log_start, 1e-8)
            logger.info(f'iter: {iteration}/{max_iter} global_step: {global_step}/{args.max_global_step} '
                        f'speed: {speed:.2f} images/sec {meters} lr: {optimizer.param_groups[1]["lr"]:.3e}')
            logged_samples, log_start = 0, time.time()
        epoch_end = iteration % iters_per_epoch == 0 or iteration == max_iter
        if not (epoch_end or (getattr(args, 'eval_on_first_step', False) and global_step == 1)):
            continue
        # Persist optimizer/scheduler/RNG before caption scoring or tokenization
        # can fail. A pending marker forces this evaluation to run on resume.
        save_latest_training_state(args, model, training_saver, optimizer, scheduler, scaler,
                                   iteration, global_step, best_score, eval_log,
                                   args.evaluate_during_training)
        if args.evaluate_during_training:
            best_score, eval_log = evaluate_and_select_checkpoint(
                args, val_dataloader, model, tokenizer, training_saver,
                iteration, global_step, best_score, eval_log)
            save_latest_training_state(args, model, training_saver, optimizer, scheduler, scaler,
                                       iteration, global_step, best_score, eval_log, False)
        if iteration == max_iter:
            break
    final_dir = op.join(args.output_dir, f'checkpoint-final-{global_step}')
    if is_main_process():
        training_saver.save_model(final_dir, global_step, model)
    logger.info(f'Training finished: {global_step} accumulation windows, {time.time()-training_start:.1f}s; {final_dir}')
    return final_dir


def get_predict_file(output_dir, args, data_yaml_file):
    cc = ['pred']
    # example data_yaml_file: datasets/coco_caption/test.yaml
    data = data_yaml_file.split('/')[-2]
    if data != 'coco_caption':
        cc.append(data)
    cc.append(op.splitext(op.basename(data_yaml_file))[0])
    cc.append('beam{}'.format(args.num_beams))
    cc.append('max{}'.format(args.max_gen_length))
    if args.num_keep_best != 1:
        cc.append('best{}'.format(args.num_keep_best))
    if args.output_hidden_states:
        cc.append('hidden')
    return op.join(output_dir, '{}.tsv'.format('.'.join(cc)))

def get_evaluate_file(predict_file):
    assert predict_file.endswith('.tsv')
    return op.splitext(predict_file)[0] + '.eval.json'

def evaluate(args, val_dataloader, model, tokenizer, output_dir):
    predict_file = get_predict_file(output_dir, args,
            val_dataloader.dataset.yaml_file)
    test(args, val_dataloader, model, tokenizer, predict_file)

    if get_world_size() > 1:
        dist.barrier()
    evaluate_file = get_evaluate_file(predict_file)
    if is_main_process():
        caption_file = val_dataloader.dataset.get_caption_file_in_coco_format()
        with open(predict_file) as stream:
            prediction_ids = [line.split('\t', 1)[0] for line in stream if line.strip()]
        with open(caption_file) as stream:
            reference_ids = {str(item['id']) for item in json.load(stream)['images']}
        if len(prediction_ids) != len(set(prediction_ids)) or set(prediction_ids) != reference_ids:
            raise ValueError('Prediction IDs must cover the full reference split exactly once.')
        data = val_dataloader.dataset.yaml_file.split('/')[-2]
        metrics = ('Bleu', 'ROUGE_L', 'CIDEr') if getattr(args, 'caption_metrics', 'basic') == 'basic' else None
        if args.use_sep_cap:
            result = two_cap_evaluate_on_coco_caption(predict_file, caption_file, outfile=evaluate_file, metrics=metrics)
        else:
            result = evaluate_on_coco_caption(predict_file, caption_file, outfile=evaluate_file, metrics=metrics)
        logger.info(f'evaluation result: {str(result)}')
        logger.info(f'evaluation result saved to {evaluate_file}')
        if data == 'BDDX':
            turn_file = op.splitext(predict_file)[0] + '.turn_accuracy.json'
            turn_details = op.splitext(predict_file)[0] + '.turn_predictions.json'
            turn_result = evaluate_bddx_turns(predict_file, caption_file, turn_file, turn_details)
            explicit = turn_result['explicit_turns']
            turn_score = ('{:.2f}%'.format(explicit['accuracy_percent'])
                          if explicit['accuracy'] is not None else 'N/A')
            logger.info('BDDX turning accuracy: {} ({}/{})'.format(
                turn_score, explicit['correct'], explicit['reference']))
            logger.info('turn evaluation saved to ' + turn_file)
    if get_world_size() > 1:
        dist.barrier()
    return evaluate_file

def test(args, test_dataloader, model, tokenizer, predict_file):

    cls_token_id, sep_token_id, pad_token_id, mask_token_id, period_token_id = \
        tokenizer.convert_tokens_to_ids([tokenizer.cls_token, tokenizer.sep_token,
        tokenizer.pad_token, tokenizer.mask_token, '.'])
    world_size = get_world_size()
    if world_size == 1:
        cache_file = predict_file
    else:
        # local_rank would not work for cross-node distributed training
        cache_file = op.splitext(predict_file)[0] + '_{}_{}'.format(get_rank(),
                world_size) + op.splitext(predict_file)[1]

    model.eval()
    def gen_rows():
        time_meter = 0
        # restore existing results for long running inference tasks
        exist_key2pred = {}
        tmp_file = cache_file + '.tmp.copy'
        if op.isfile(tmp_file):
            with open(tmp_file, 'r') as fp:
                for line in fp:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        exist_key2pred[parts[0]] = parts[1]

        with torch.no_grad():
            for step, (img_keys, batch, meta_data) in tqdm(enumerate(test_dataloader)):
                # torch.cuda.empty_cache()
                # is_exist = True
                # for k in img_keys:
                #     if k not in exist_key2pred:
                #         is_exist = False
                #         break
                # if is_exist:
                #     for k in img_keys:
                #         yield k, exist_key2pred[k]
                #         # return k, exist_key2pred[k]
                #     continue
                # if step > 4:
                #     break
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {'is_decode': True,
                    'input_ids': batch[0], 'attention_mask': batch[1],
                    'token_type_ids': batch[2], 'img_feats': batch[3],
                    'masked_pos': batch[4],
                    'car_info': batch[5],
                    'do_sample': False,
                    'bos_token_id': cls_token_id,
                    'pad_token_id': pad_token_id,
                    'eos_token_ids': [sep_token_id],
                    'mask_token_id': mask_token_id,
                    # for adding od labels
                    'add_od_labels': args.add_od_labels, 'od_labels_start_posid': args.max_seq_a_length,
                    # hyperparameters of beam search
                    'max_length': args.max_gen_length if not args.use_sep_cap else args.max_gen_length*2,
                    'use_sep_cap': args.use_sep_cap,
                    'num_beams': args.num_beams,
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "top_p": args.top_p,
                    "repetition_penalty": args.repetition_penalty,
                    "length_penalty": args.length_penalty,
                    "num_return_sequences": args.num_return_sequences,
                    "num_keep_best": args.num_keep_best,
                }

                inputs = add_cached_motion(args, inputs, batch, is_train=False)
                tic = time.time()
                # captions, logprobs
                
                if args.mixed_precision_method == 'deepspeed' and args.deepspeed_fp16:
                    # deepspeed does not auto cast inputs.
                    inputs = fp32_to_fp16(inputs)

                with autocast_context(args):
                    outputs = model(**inputs)
                time_meter += time.time() - tic
                all_caps = outputs[0]  # batch_size * num_keep_best * max_len
                all_confs = torch.exp(outputs[1])

                if not args.use_sep_cap:
                    for img_key, caps, confs in zip(img_keys, all_caps, all_confs):
                        res = []
                        for cap, conf in zip(caps, confs):
                            cap = tokenizer.decode(cap.tolist(), skip_special_tokens=True)
                            res.append({'caption': cap, 'conf': conf.item()})
                        if isinstance(img_key, torch.Tensor):
                            img_key = img_key.item()
                        yield img_key, json.dumps(res)
                        # return img_key, json.dumps(res)
                else:
                    for img_key, caps, confs in zip(img_keys, all_caps, all_confs):
                        all_cap_a = []
                        all_cap_b = []
                        sep_place = args.max_gen_length
                        for cap, conf in zip(caps, confs):
                            cap_1 = tokenizer.decode(cap.tolist()[:sep_place], skip_special_tokens=True)
                            cap_2 = tokenizer.decode(cap.tolist()[sep_place:], skip_special_tokens=True)
                            all_cap_a.append({'caption': cap_1, 'conf': conf.item()})
                            all_cap_b.append({'caption': cap_2, 'conf': conf.item()})
                        if isinstance(img_key, torch.Tensor):
                            img_key = img_key.item()
                        if args.use_swap_cap:
                            yield img_key, json.dumps(all_cap_b), json.dumps(all_cap_a)
                        else:
                            yield img_key, json.dumps(all_cap_a), json.dumps(all_cap_b)
                        # return img_key, json.dumps(all_cap_a), json.dumps(all_cap_b)

        logger.info(f"Inference model computing time: {(time_meter / (step+1))} seconds per batch")

    # a = gen_rows()
    if args.use_sep_cap:
        double_tsv_writer(gen_rows(), cache_file)
    else:
        tsv_writer(gen_rows(), cache_file)
    if world_size > 1:
        dist.barrier()
    if world_size > 1 and is_main_process():
        cache_files = [op.splitext(predict_file)[0] + '_{}_{}'.format(i, world_size) + \
            op.splitext(predict_file)[1] for i in range(world_size)]
        concat_tsv_files(cache_files, predict_file)
        delete_tsv_files(cache_files)
        reorder_tsv_keys(predict_file, test_dataloader.dataset.image_keys, predict_file)
    if world_size > 1:
        dist.barrier()

def signal_evaluate(args, val_dataloader, model, tokenizer, output_dir):
    predict_file = get_predict_file(output_dir, args,
            val_dataloader.dataset.yaml_file)

    cls_token_id, sep_token_id, pad_token_id, mask_token_id, period_token_id = \
        tokenizer.convert_tokens_to_ids([tokenizer.cls_token, tokenizer.sep_token,
        tokenizer.pad_token, tokenizer.mask_token, '.'])
    world_size = get_world_size()
    
    cache_file = predict_file

    model.eval()
    def gen_rows():
        time_meter = 0
        # restore existing results for long running inference tasks
        exist_key2pred = {}
        tmp_file = cache_file + '.tmp.copy'
        if op.isfile(tmp_file):
            with open(tmp_file, 'r') as fp:
                for line in fp:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        exist_key2pred[parts[0]] = parts[1]

        gt_signals = []
        pred_signals = []

        with torch.no_grad():
            for step, (img_keys, batch, meta_data) in tqdm(enumerate(val_dataloader)):

                # if step > 4:
                #     break

                batch = tuple(t.to(args.device) for t in batch)
                inputs = {'is_decode': True,
                    'input_ids': batch[0], 'attention_mask': batch[1],
                    'token_type_ids': batch[2], 'img_feats': batch[3],
                    'masked_pos': batch[4],
                    'car_info': batch[5],
                    'do_sample': False,
                    'bos_token_id': cls_token_id,
                    'pad_token_id': pad_token_id,
                    'eos_token_ids': [sep_token_id],
                    'mask_token_id': mask_token_id,
                    # for adding od labels
                    'add_od_labels': args.add_od_labels, 'od_labels_start_posid': args.max_seq_a_length,
                    # hyperparameters of beam search
                    'max_length': args.max_gen_length if not args.use_sep_cap else args.max_gen_length*2,
                    'use_sep_cap': args.use_sep_cap,
                    'num_beams': args.num_beams,
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "top_p": args.top_p,
                    "repetition_penalty": args.repetition_penalty,
                    "length_penalty": args.length_penalty,
                    "num_return_sequences": args.num_return_sequences,
                    "num_keep_best": args.num_keep_best,
                }

                inputs = add_cached_motion(args, inputs, batch, is_train=False)
                if args.mixed_precision_method == 'deepspeed' and args.deepspeed_fp16:
                    # deepspeed does not auto cast inputs.
                    inputs = fp32_to_fp16(inputs)

                with autocast_context(args):
                    outputs = model(**inputs)
                outputs = outputs
                for b in range(len(batch[0])):
                    # if all of the control signal is -1, then we know the info file is missed
                    # however, this missed info doesn't affect the results of captions due to our multi-task architecture
                    if not (batch[5][b]==-1).all():
                        gt_signals.append(batch[5][b])
                        pred_signals.append(outputs[-2][b])

        return torch.stack(gt_signals, dim=0), torch.stack(pred_signals, dim=0)

    gt_signals, pred_signals = gen_rows()

    if world_size > 1:
        dist.barrier()

    if is_main_process():
        print("computing signal prediction score")
        sigma_1 = 0.1
        sigma_2 = 0.5
        sigma_3 = 1
        sigma_4 = 5
        sigma_5 = 10

        sig1_acc = 0
        sig2_acc = 0
        sig3_acc = 0
        sig4_acc = 0
        sig5_acc = 0
        assert len(gt_signals) == len(pred_signals)

        for signal_order in range(len(args.signal_types)):
            signal_name = args.signal_types[signal_order]
            gt_signal = gt_signals[:, signal_order, :].cpu()
            pred_signal = pred_signals[:, :, signal_order].cpu()
            import numpy as np
            from sklearn.metrics import mean_squared_error
            rmse_signal = np.sqrt(mean_squared_error(gt_signal, pred_signal))

            print(f"{signal_name} \t rmse:{rmse_signal}")
            all_num = gt_signal.shape[0] * gt_signal.shape[1]   # B*frame_num
            sig1_acc = (np.count_nonzero(abs(gt_signal-pred_signal)<sigma_1)/all_num,)
            print(f"sig1_acc \t {sig1_acc}")
            sig2_acc = (np.count_nonzero(abs(gt_signal-pred_signal)<sigma_2)/all_num,)
            print(f"sig1_acc \t {sig2_acc}")
            sig3_acc = (np.count_nonzero(abs(gt_signal-pred_signal)<sigma_3)/all_num,)
            print(f"sig1_acc \t {sig3_acc}")
            sig4_acc = (np.count_nonzero(abs(gt_signal-pred_signal)<sigma_4)/all_num,)
            print(f"sig1_acc \t {sig4_acc}")
            sig5_acc = (np.count_nonzero(abs(gt_signal-pred_signal)<sigma_5)/all_num,)
            print(f"sig1_acc \t {sig5_acc}")
            print(all_num)
            if not os.path.exists(op.dirname(predict_file)):
                os.makedirs(op.dirname(predict_file))
            with open(op.dirname(predict_file) +f'/{signal_name}_test_data.json', 'w') as json_file:
                json_file.write(str({f"rmse_{signal_name}":rmse_signal,
                        # "rmse_speed":rmse_speed,
                        "sig1_acc":sig1_acc,
                        "sig2_acc":sig2_acc,
                        "sig3_acc":sig3_acc,
                        "sig4_acc":sig4_acc,
                        "sig5_acc":sig5_acc,
                        }))
    if world_size > 1:
        dist.barrier()
    if get_world_size() > 1:
        dist.barrier()
    return

def check_arguments(args):
    # shared basic checks
    basic_check_arguments(args)
    if getattr(args, 'dataset_name', 'BDDX') == 'MMAU' and args.do_train:
        from src.tasks.dataset_protocol import validate_splits
        validate_splits(args.dataset_name, args.train_yaml, args.val_yaml)
        if not args.use_sep_cap:
            raise ValueError('The audited MMAU data has two caption fields.')
    if getattr(args, 'scst', False):
        raise NotImplementedError('SCST is not implemented in this training loop; the flag cannot silently train cross entropy.')
    if (args.do_train and 'testing' in op.basename(args.val_yaml)
            and not getattr(args, 'allow_test_evaluation', False)):
        raise ValueError('Testing evaluation during training requires explicit --allow_test_evaluation true.')
    if args.num_keep_best != 1:
        raise ValueError('COCO evaluation requires exactly one generated caption per example.')
    # additional sanity check:
    args.max_img_seq_length = int((args.max_num_frames/2)*(int(args.img_res)/32)*(int(args.img_res)/32))
    
    if args.freeze_backbone or args.backbone_coef_lr == 0:
        args.backbone_coef_lr = 0
        args.freeze_backbone = True
    
    if 'reload_pretrained_swin' not in args.keys():
        args.reload_pretrained_swin = False

    if not len(args.pretrained_checkpoint) and args.reload_pretrained_swin:
        logger.info("No pretrained_checkpoint to be loaded, disable --reload_pretrained_swin")
        args.reload_pretrained_swin = False

    if args.learn_mask_enabled==True and args.attn_mask_type != 'learn_without_crossattn' and args.attn_mask_type != 'learn_with_swap_crossattn': 
        args.attn_mask_type = 'learn_vid_att'

def explicit_inference_override_keys(argv):
    keys = {token[2:].split('=', 1)[0] for token in argv if token.startswith('--')}
    config_path = None
    for index, token in enumerate(argv):
        if token == '--config' and index + 1 < len(argv):
            config_path = argv[index + 1]
        elif token.startswith('--config='):
            config_path = token.split('=', 1)[1]
    if config_path:
        with open(config_path) as stream:
            keys.update(json.load(stream))
    return keys


def update_existing_config_for_inference(args):
    ''' load asuad args for evaluation and inference 
    '''
    assert args.do_test or args.do_eval
    checkpoint = args.eval_model_dir
    try:
        json_path = op.join(checkpoint, os.pardir, 'log', 'args.json')
        f = open(json_path,'r')
        json_data = json.load(f)

        from easydict import EasyDict
        train_args = EasyDict(json_data)
    except Exception as e:
        train_args = torch.load(op.join(checkpoint, 'training_args.bin'), map_location='cpu', weights_only=False)

    train_args.eval_model_dir = args.eval_model_dir
    train_args.resume_checkpoint = op.join(args.eval_model_dir, 'model.bin')
    train_args.do_train = False
    train_args.do_eval = True
    train_args.do_signal_eval = True if hasattr(args, 'do_signal_eval') and args.do_signal_eval else False
    train_args.do_test = True
    # Keep checkpoint settings unless a CLI argument/config explicitly overrides runtime values.
    overrides = explicit_inference_override_keys(sys.argv[1:])
    for key in ('device', 'local_rank', 'mixed_precision_method', 'native_precision',
                'num_beams', 'length_penalty', 'per_gpu_eval_batch_size', 'num_workers', 'caption_metrics',
                'val_yaml', 'test_video_fname', 'signal_types', 'data_dir', 'model_name_or_path',
                'config_name', 'tokenizer_name', 'vision_yolo_weights', 'flow_cache_dir'):
        if hasattr(args, key) and (key in overrides or not hasattr(train_args, key)):
            setattr(train_args, key, getattr(args, key))
    asset_dir = train_args.model_name_or_path
    if not asset_dir or not all(op.isfile(op.join(asset_dir, filename)) for filename in ('config.json', 'vocab.txt')):
        fallback_dir = 'models/captioning/bert-base-uncased/'
        if 'model_name_or_path' in overrides or not all(op.isfile(op.join(fallback_dir, filename)) for filename in ('config.json', 'vocab.txt')):
            raise FileNotFoundError(f'Caption model assets require config.json and vocab.txt: {asset_dir}')
        logger.warning(f'Legacy saved model asset path {asset_dir!r} is incomplete; using {fallback_dir!r}.')
        train_args.model_name_or_path = fallback_dir
    return train_args

def get_custom_args(base_config):
    parser = base_config.parser
    parser.add_argument('--dataset_name', choices=['BDDX', 'MMAU'], default='BDDX')
    parser.add_argument('--max_num_frames', type=int, default=32)
    parser.add_argument('--img_res', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument("--grid_feat", type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument("--kinetics", type=str, default='400', help="400 or 600")
    parser.add_argument("--pretrained_2d", type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument("--vidswin_size", type=str, default='base')
    parser.add_argument('--freeze_backbone', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--use_checkpoint', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--backbone_coef_lr', type=float, default=0.001)
    parser.add_argument("--reload_pretrained_swin", type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--learn_mask_enabled', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--loss_sparse_w', type=float, default=0)
    parser.add_argument('--loss_sensor_w', type=float, default=0)
    parser.add_argument('--sparse_mask_soft2hard', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--transfer_method', type=int, default=-1,
                        help="0: load all asuad pre-trained weights, 1: load only pre-trained sparse mask")
    parser.add_argument('--att_mask_expansion', type=int, default=-1,
                        help="-1: random init, 0: random init and then diag-based copy, 1: interpolation")
    parser.add_argument('--resume_checkpoint', type=str, default='None')
    parser.add_argument('--test_video_fname', type=str, default='None')
    parser.add_argument('--caption_metrics', choices=['basic', 'full'], default='basic')
    parser.add_argument('--resume_training_state', default='', help='Resume a native epoch-boundary checkpoint with the same world size.')
    parser.add_argument('--native_precision', choices=['fp32', 'fp16', 'bf16'], default='fp32')
    parser.add_argument('--max_train_steps', type=int, default=0, help='Limit optimizer windows for smoke tests; 0 means full training.')
    parser.add_argument('--eval_on_first_step', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--allow_test_evaluation', type=str_to_bool, nargs='?', const=True, default=False,
                        help='Explicitly allow evaluating testing at every training epoch.')
    parser.add_argument('--checkpoint_selection', choices=['sum_CIDEr', 'joint_b4_cider'], default='sum_CIDEr')
    args = base_config.parse_args()
    return args

def load_model_weights(model, state, allow_missing_mask=False, strict=False):
    if strict and allow_missing_mask:
        raise ValueError('Strict checkpoint initialization cannot omit the video mask.')
    result = model.load_state_dict(state, strict=strict)
    missing_core = [key for key in result.missing_keys
                    if not key.startswith(('visual_references.detector.', 'reference_fusion.'))
                    and not (allow_missing_mask and key == 'learn_vid_att.weight')]
    if missing_core:
        raise RuntimeError(f'Checkpoint is missing {len(missing_core)} required core weights: {missing_core[:16]}')
    logger.info(f'Checkpoint loaded: {len(state)} keys, missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}')
    if result.missing_keys:
        logger.warning(f'Missing checkpoint keys (first 16; new modules initialize normally): {result.missing_keys[:16]}')
    if result.unexpected_keys:
        logger.warning(f'Unused checkpoint keys (first 16): {result.unexpected_keys[:16]}')
    return result


def main(args):
    if args.do_train==False or args.do_eval==True:
        args = update_existing_config_for_inference(args) 

    args.device = torch.device(args.device)

    dist_init(args)
    logger.info("Setup CUDA, GPU & distributed training")
    
    check_arguments(args)
    logger.info("Check arguments")

    mkdir(args.output_dir)
    logger.info(f"creating output_dir at: {args.output_dir}")

    set_seed(args.seed, args.num_gpus)
    global_params.global_args = args
    
    if args.mixed_precision_method == "apex":
        precision_description = f"apex O{args.amp_opt_level}"
    elif args.mixed_precision_method == "deepspeed":
        amp_info = '' if args.deepspeed_fp16 else f'amp, {args.amp_opt_level}'
        fp16_info = '' if not args.deepspeed_fp16 else f'fp16, {args.zero_opt_stage}'
        precision_description = f"deepspeed, {amp_info}{fp16_info}"
    elif args.mixed_precision_method == "fairscale":
        assert args.distributed, "fairscale can only be used for distributed training"
        precision_description = f"fairscale, fp16: {args.fairscale_fp16}, default zero_opt 2"
    else:
        precision_description = 'native ' + args.native_precision

    logger.info(
        "device: {}, n_gpu: {}, rank: {}, "
        "16-bits training: {}".format(
            args.device, args.num_gpus, get_rank(), precision_description))

    if not is_main_process():
        logger.disabled = True
        training_saver = NoOp()
    else:
        training_saver = TrainingSaver(args.output_dir)
        TB_LOGGER.create(op.join(args.output_dir, 'log'))
        add_log_to_file(op.join(args.output_dir, 'log', "log.txt"))

    logger.info(f"Pytorch version is: {torch.__version__}")
    logger.info(f"Cuda version is: {torch.version.cuda}")
    logger.info(f"cuDNN version is : {torch.backends.cudnn.version()}" )

    # Get Video Swin backbone 
    swin_model = get_swin_model(args)

    # Get BERT and tokenizer for DCG (Driving Caption Generation) 
    bert_model, config, tokenizer = get_bert_model(args)

    # build asuad based on training configs
    if args.multitask:
        raise ValueError('The asuad release is purely visual.')
    asuad_model = AsuadModel(args, config, swin_model, bert_model)
    total_params = sum(p.numel() for p in asuad_model.parameters())
    if getattr(args, 'dataset_name', 'BDDX') == 'MMAU':
        expected = int(getattr(args, 'expected_model_parameters', 233764716))
        if total_params != expected:
            raise RuntimeError(f'MMAU large-model parameter mismatch: {total_params} != {expected}')
    trainable_params = sum(p.numel() for p in asuad_model.parameters() if p.requires_grad)
    print(f"参数量: {total_params / 1e9:.2f}B")
    asuad_model.freeze_backbone(freeze=args.freeze_backbone)
    if args.do_train and not args.do_eval:
        initialization = validate_model_initialization(args, asuad_model)
        logger.info('Model initialization verification: %s', initialization)
        if is_main_process():
            with open(op.join(args.output_dir, 'initialization_verified.json'), 'w') as stream:
                json.dump(initialization, stream, indent=2)

    if args.do_eval:
        # load weights for eval/inference
        logger.info(f"Loading state dict from checkpoint {args.resume_checkpoint}")
        cpu_device = torch.device('cpu')
        pretrained_model = torch.load(args.resume_checkpoint, map_location=cpu_device, weights_only=False)

        if isinstance(pretrained_model, dict):
            load_model_weights(asuad_model, pretrained_model, strict=getattr(args, 'dataset_name', 'BDDX') == 'MMAU')
        else:
            load_model_weights(asuad_model, pretrained_model.state_dict(), strict=getattr(args, 'dataset_name', 'BDDX') == 'MMAU')

    elif args.do_train and args.pretrained_checkpoint != '':
        ckpt_path = op.join(args.pretrained_checkpoint, 'model.bin')
        assert op.exists(ckpt_path), f"{ckpt_path} does not exist"
        logger.info(f"Loading state dict from checkpoint {ckpt_path}")
        cpu_device = torch.device('cpu')
        pretrained_model = torch.load(ckpt_path, map_location=cpu_device, weights_only=False)

        if args.learn_mask_enabled == False:
            if isinstance(pretrained_model, dict):
                load_model_weights(asuad_model, pretrained_model, strict=getattr(args, 'dataset_name', 'BDDX') == 'MMAU')
            else:
                load_model_weights(asuad_model, pretrained_model.state_dict(), strict=getattr(args, 'dataset_name', 'BDDX') == 'MMAU')

        elif args.learn_mask_enabled == True:
            pretrained_mask_shape = pretrained_model['learn_vid_att.weight'].shape
            init_mask_shape = asuad_model.learn_vid_att.weight.shape

            #-------------------------------------------------------------
            # transfer at the same frame rate
            if pretrained_mask_shape==init_mask_shape: 
                # init using entire pre-trained asuad weights
                if args.transfer_method==0:
                    if isinstance(pretrained_model, dict):
                        load_model_weights(asuad_model, pretrained_model, strict=getattr(args, 'dataset_name', 'BDDX') == 'MMAU')
                    else:
                        load_model_weights(asuad_model, pretrained_model.state_dict(), strict=getattr(args, 'dataset_name', 'BDDX') == 'MMAU')
                # init using only pre-trained sparse att mask weights
                else:
                    asuad_model.reload_attn_mask(pretrained_model['learn_vid_att.weight'])
            #-------------------------------------------------------------
            # transfer across different frame rates
            else:  
                # init using entire pre-trained asuad weights, except sparse attn mask
                if args.transfer_method==0:
                    if isinstance(pretrained_model, dict):
                        new_state_dict={}
                        for k,v in zip(pretrained_model.keys(), pretrained_model.values()):
                            if k!='learn_vid_att.weight' or k=='learn_vid_att.weight' and pretrained_mask_shape==init_mask_shape:
                                new_state_dict[k] = v
                        load_model_weights(asuad_model, new_state_dict, allow_missing_mask=True, strict=getattr(args, 'dataset_name', 'BDDX') == 'MMAU')
                        del new_state_dict
                    else:
                        pretrained_model_state_dict = pretrained_model.state_dict()
                        new_state_dict={}
                        for k,v in zip(pretrained_model_state_dict.keys(), pretrained_model_state_dict.values()):
                            if k!='learn_vid_att.weight' or k=='learn_vid_att.weight' and pretrained_mask_shape==init_mask_shape:
                                new_state_dict[k] = v
                        load_model_weights(asuad_model, new_state_dict, allow_missing_mask=True, strict=getattr(args, 'dataset_name', 'BDDX') == 'MMAU')
                        del new_state_dict

                # expand pre-trained sparse att mask to the desired size          
                if args.att_mask_expansion >= 0:
                    asuad_model.reload_attn_mask(pretrained_model['learn_vid_att.weight'])
                # Otherwise retain the newly initialized mask.

        del pretrained_model
        gc.collect()
        torch.cuda.empty_cache()

        args.eval_model_dir = args.pretrained_checkpoint
        checkpoint = args.eval_model_dir
        assert op.isdir(checkpoint)
        asuad_model.max_img_seq_length = int(args.max_img_seq_length)
        asuad_model.config.num_visual_tokens = int(args.max_img_seq_length)
        # Preserve the caption asset directory; checkpoints contain model.bin, not config/vocabulary.
        if args.reload_pretrained_swin:
            asuad_model.swin = reload_pretrained_swin(asuad_model.swin, args)

    asuad_model.to(args.device)
    
    if args.do_train:
        args = restore_training_settings(args)
        train_dataloader = make_data_loader(args, args.train_yaml, tokenizer, args.distributed, is_train=True)
        val_dataloader = make_data_loader(args, args.val_yaml, tokenizer, args.distributed, is_train=False)

        args.iters_per_epoch = len(train_dataloader.batch_sampler.batch_sampler)
        args.max_iter = len(train_dataloader)
        grad_accum = max(1, args.gradient_accumulation_steps)
        if args.max_train_steps > 0:
            args.max_iter = min(args.max_iter, args.max_train_steps * grad_accum)
        full_epochs, tail = divmod(args.max_iter, args.iters_per_epoch)
        args.global_iters_per_epoch = math.ceil(args.iters_per_epoch / grad_accum)
        args.max_global_step = full_epochs * args.global_iters_per_epoch + math.ceil(tail / grad_accum)
        args.save_steps = args.global_iters_per_epoch
        # args.save_steps = 10

        args, asuad_model, optimizer, scheduler = mixed_precision_init(args, asuad_model)
        train(args, train_dataloader, val_dataloader, asuad_model, tokenizer, training_saver, optimizer, scheduler)

    elif args.do_eval:
        val_dataloader = make_data_loader(args, args.val_yaml, tokenizer, args.distributed, is_train=False)
        if args.mixed_precision_method == 'deepspeed' and args.deepspeed_fp16:
            asuad_model.half()
        if args.do_signal_eval:
            signal_evaluate(args, val_dataloader, asuad_model, tokenizer, args.eval_model_dir)
        else:
            evaluate_file = evaluate(args, val_dataloader, asuad_model, tokenizer, args.eval_model_dir)
    
    if args.distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    shared_configs.shared_video_captioning_config(cbs=True, scst=True)
    args = get_custom_args(shared_configs)
    main(args)
