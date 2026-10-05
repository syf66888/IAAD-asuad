"""Audited dual-caption SCST, started only after a verified supervised checkpoint.

Sampling and likelihood recomputation share visual features inside one DDP
forward. The reward corpus is training-only; model selection follows the user's
explicit per-epoch testing protocol and fixed four-metric selection rule.
"""
import argparse
import copy
import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.distributed as dist
from easydict import EasyDict

import global_params
from src.datasets.vl_dataloader import make_data_loader
from src.modeling.load_bert import get_bert_model
from src.modeling.load_swin import get_swin_model
from src.modeling.asuad_model import AsuadModel
from src.solver import WarmupLinearLR
from src.tasks.checkpoint_selection import atomic_json, joint_score, selection_record, SELECTION_NAME
from src.tasks.dataset_protocol import validate_splits, task_semantics
from src.tasks.train import (autocast_context, add_cached_motion, accumulation_window,
    native_grad_scaler, evaluate, caption_result_path, capture_training_rng, restore_training_rng)
from src.utils.comm import dist_init, get_world_size, get_rank, is_main_process
from src.utils.load_save import TrainingSaver, atomic_torch_save, cpu_tree
from src.utils.logger import LOGGER as logger, TB_LOGGER, add_log_to_file
from src.utils.miscellaneous import set_seed


def policy_loss(log_probs, mask, des_advantage, exp_advantage, des_weight=0.5, exp_weight=0.5):
    """Causal credit: description actions affect both rewards; exp affects exp.

Advantages are fixed (training CIDEr-scaled) rewards, not differentiable
metrics. Sum valid action log probabilities, including a naturally sampled EOS.
"""
    if log_probs.ndim != 3 or log_probs.shape[1] != 2 or mask.shape != log_probs.shape:
        raise ValueError('Expected [B*samples,2,slot_length] probabilities and mask.')
    des = torch.as_tensor(des_advantage, device=log_probs.device, dtype=torch.float32).detach()
    exp = torch.as_tensor(exp_advantage, device=log_probs.device, dtype=torch.float32).detach()
    if des.shape != (log_probs.shape[0],) or exp.shape != des.shape:
        raise ValueError('Reward/sample ordering mismatch.')
    # Mask before summing: 0 * -inf would otherwise make a padded action NaN.
    sequence_log_probs = log_probs.float().masked_fill(~mask.bool(), 0.0).sum(-1)
    return -((des_weight * des + exp_weight * exp) * sequence_log_probs[:, 0]
             + exp_weight * exp * sequence_log_probs[:, 1]).mean()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def special_token_ids(tokenizer):
    """The bundled legacy tokenizer supports string conversion, not *_id properties."""
    result = {name: tokenizer.convert_tokens_to_ids(token) for name, token in (
        ('bos_token_id', tokenizer.cls_token), ('pad_token_id', tokenizer.pad_token),
        ('mask_token_id', tokenizer.mask_token), ('eos_token_id', tokenizer.sep_token))}
    if (any(not isinstance(value, int) or value < 0 for value in result.values())
            or len(set(result.values())) != 4):
        raise ValueError('BERT BOS/PAD/MASK/SEP must resolve to distinct vocabulary IDs.')
    result['eos_token_ids'] = [result.pop('eos_token_id')]
    return result


def accumulation_scale(iteration, max_iter, per_epoch, accumulation, batch_size,
                       samples_per_rank, actual_batch_size):
    """Weight microbatch means by samples, including a shorter final batch.

    DistributedSampler gives every rank the same sample count. Averaging those
    per-rank gradients therefore preserves the global example mean as well.
    """
    divisor, flush = accumulation_window(iteration, max_iter, per_epoch, accumulation)
    offset = (iteration - 1) % per_epoch
    window_start = offset - offset % accumulation
    window_samples = sum(min(batch_size, samples_per_rank - index * batch_size)
                         for index in range(window_start, window_start + divisor))
    expected = min(batch_size, samples_per_rank - offset * batch_size)
    if expected <= 0 or window_samples <= 0 or actual_batch_size != expected:
        raise ValueError('SCST sampler length and actual microbatch size disagree.')
    return float(actual_batch_size) / window_samples, flush


def resume_signature(args, base_record):
    keys = ('learning_rate', 'seed', 'scst_sample_n', 'per_gpu_train_batch_size',
            'scst_baseline_type', 'scst_ce_weight', 'num_beams', 'length_penalty',
            'gradient_accumulation_steps', 'num_train_epochs', 'backbone_coef_lr',
            'scst_weight_decay', 'scst_warmup_ratio', 'scst_cider_weight', 'scst_bleu4_weight',
            'loss_sparse_w', 'max_grad_norm', 'adam_epsilon', 'native_precision',
            'dataset_name', 'train_yaml', 'val_yaml', 'data_dir', 'flow_cache_dir', 'freeze_backbone',
            'feature_fusion', 'object_relation_enabled', 'spatial_motion_enabled',
            'learn_mask_enabled', 'learn_mask_log_bias', 'sparse_mask_soft2hard',
            'max_gen_length', 'max_seq_length', 'max_num_frames', 'img_res',
            'num_workers', 'early_stopping_patience', 'scst_min_epochs', 'use_swap_cap',
            'vision_yolo_weights', 'vision_yolo_frames', 'vision_yolo_imgsz', 'vision_yolo_batch_size',
            'vision_flow_backend', 'vision_flow_size', 'fusion_hidden_dim', 'fusion_dropout',
            'spatial_motion_size', 'object_relation_hidden_dim', 'object_relation_top_k',
            'spatial_motion_hidden_dim', 'vision_yolo_conf', 'vision_yolo_max_det')
    return dict(version=1, config={key: args.get(key) for key in keys},
                reward_cache_sha256=file_sha256(args.scst_reward_cache),
                supervised_model_sha256=base_record['model_sha256'])


def validate_resume_state(state, signature, per_epoch, max_iter, world_size):
    iteration = int(state['iteration'])
    if not state.get('epoch_boundary') or iteration < 0 or iteration > max_iter or iteration % per_epoch:
        raise ValueError('SCST can resume only from an epoch boundary within this trial.')
    if state.get('resume_signature') != signature:
        raise ValueError('SCST resume reward/model/optimizer/input configuration mismatch.')
    if state.get('world_size') != world_size or len(state.get('rank_rng_states', [])) != world_size:
        raise ValueError('SCST resume world size or per-rank RNG states differ.')
    history = state['eval_log']
    if len(history) != iteration // per_epoch:
        raise ValueError('SCST resume evaluation history does not match completed epochs.')
    for epoch, record in enumerate(history, 1):
        if record['epoch'] != epoch or record['iteration'] != epoch * per_epoch:
            raise ValueError('SCST resume has a missing or mislabeled epoch evaluation.')
        selection_record(record)
    best = state['best_record']
    if not math.isclose(selection_record(best)['selection_score'], best['selection_score'], rel_tol=1e-10):
        raise ValueError('SCST resume selected score differs from its metric record.')
    if not Path(best['model_path']).is_file() or file_sha256(best['model_path']) != best['model_sha256']:
        raise ValueError('SCST resume selected checkpoint is missing or changed.')


def termination_reason(completed_epochs, max_epochs, stale_epochs, min_epochs, patience):
    if completed_epochs >= max_epochs:
        return 'epochs_complete'
    if completed_epochs >= min_epochs and stale_epochs >= patience:
        return 'early_stopping'
    return None


def validate_completion(iteration, max_iter, per_epoch, history, stopped_early):
    if not history or iteration % per_epoch or len(history) != iteration // per_epoch:
        raise RuntimeError('SCST cannot finish without evaluation of every completed epoch.')
    if iteration != max_iter and not stopped_early:
        raise RuntimeError('SCST training loader ended before the planned final iteration.')


def configure(config):
    args = EasyDict(config)
    args.dataset_name = validate_splits(args.get('dataset_name', 'BDDX'), args.train_yaml, args.val_yaml)
    if any(args.get(key, False) for key in ('use_car_sensor', 'multitask', 'only_signal', 'use_asr')):
        raise ValueError('This SCST implementation is pure vision only.')
    if not args.use_sep_cap or args.max_seq_length != 2 * args.max_gen_length or args.add_od_labels:
        raise ValueError('SCST requires the existing two fixed caption slots without extra text inputs.')
    if args.get('use_swap_cap', False):
        raise ValueError('SCST rewards require description followed by explanation; swapped captions are unsupported.')
    if args.get('mixed_precision_method', 'native') != 'native':
        raise ValueError('SCST supports native AMP, including the PPU torch backend.')
    if not Path(args.scst_checkpoint).is_file():
        raise FileNotFoundError(args.scst_checkpoint)
    args.update(scst=True, do_train=True, do_eval=False, do_test=False,
                num_keep_best=1, num_return_sequences=1,
                allow_test_evaluation=True, save_model=True, mixed_precision_method='native')
    for key, value in dict(scst_sample_n=2, scst_cider_weight=1.0, scst_bleu4_weight=0.0,
                           scst_baseline_type=args.get('sc_baseline_type', 'greedy'),
                           scst_ce_weight=0.0, num_beams=1,
                           early_stopping_patience=2, scst_min_epochs=2, max_train_steps=0,
                           scst_resume='', smoke=False, logging_steps=10,
                           scst_weight_decay=0.05, scst_warmup_ratio=0.05).items():
        # Legacy EasyDict inherits dict.setdefault, which bypasses its attribute
        # synchronization. __setitem__ delegates to __setattr__ in this version.
        if key not in args:
            args[key] = value
    if (isinstance(args.scst_sample_n, bool) or not 1 <= args.scst_sample_n <= 16
            or int(args.scst_sample_n) != args.scst_sample_n):
        raise ValueError('Use an integer number of 1-16 SCST samples to bound memory.')
    args.scst_sample_n = int(args.scst_sample_n)
    if args.scst_baseline_type not in ('greedy', 'leave_one_out'):
        raise ValueError('SCST baseline must be greedy or leave_one_out.')
    if args.scst_baseline_type == 'leave_one_out' and args.scst_sample_n < 2:
        raise ValueError('leave_one_out requires at least two independent samples.')
    # The canonical field takes precedence over metadata inherited from older runs.
    args.sc_baseline_type = args.scst_baseline_type
    if (not math.isfinite(args.scst_ce_weight) or args.scst_ce_weight < 0
            or isinstance(args.num_beams, bool) or args.num_beams < 1
            or int(args.num_beams) != args.num_beams):
        raise ValueError('Invalid SCST CE weight or evaluation beam count.')
    if (args.num_train_epochs < 1 or args.scst_min_epochs < 1
            or (not args.smoke and args.scst_min_epochs > args.num_train_epochs) or args.early_stopping_patience < 1
            or args.gradient_accumulation_steps < 1 or args.per_gpu_train_batch_size < 1
            or args.logging_steps < 1 or args.max_train_steps < 0
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        raise ValueError('Invalid SCST epoch, accumulation, learning-rate or stopping configuration.')
    args.device = torch.device(args.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    args.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    dist_init(args)
    return args


def read_base_record(args):
    value = args.get('scst_base_record')
    if isinstance(value, str):
        with open(value) as stream:
            value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError('scst_base_record must identify the exact selected supervised model and metrics.')
    if value.get('dataset_name', 'BDDX') != args.get('dataset_name', 'BDDX'):
        raise ValueError('SCST baseline metrics belong to a different dataset.')
    checkpoint = Path(args.scst_checkpoint).resolve()
    if (not value.get('model_path') or Path(value['model_path']).resolve() != checkpoint
            or not value.get('model_sha256') or file_sha256(checkpoint) != value['model_sha256']):
        raise ValueError('The supervised metrics record is not bound to this checkpoint file/hash.')
    return selection_record(value)


def optimizer_for(args, model, updates):
    groups = [[], [], [], []]
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        no_decay = parameter.ndim <= 1 or name.endswith('.bias')
        groups[(0 if name.startswith('swin.') else 1) + (2 if no_decay else 0)].append(parameter)
    groups = [dict(params=parameters,
                   weight_decay=0.0 if index >= 2 else args.scst_weight_decay,
                   lr=args.learning_rate * (args.backbone_coef_lr if index % 2 == 0 else 1.0))
              for index, parameters in enumerate(groups)]
    optimizer = torch.optim.AdamW(groups, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearLR(optimizer, updates, warmup_ratio=args.scst_warmup_ratio)
    return optimizer, scheduler


def decode_captions(tokenizer, token_ids):
    ids = token_ids.detach().cpu().tolist()
    return [[tokenizer.decode(row[part], skip_special_tokens=True) for row in ids] for part in range(2)]


def run(config):
    args = configure(config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / 'run_status.json'
    status = dict(pid=os.getpid(), started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  phase='initializing', stage='scst', supervised_checkpoint=args.scst_checkpoint,
                  learning_rate=args.learning_rate, seed=args.seed, evaluation_yaml=args.val_yaml)

    def update_status(phase, **extra):
        if is_main_process():
            status.update(phase=phase, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)
            atomic_json(status_path, status)

    update_status('initializing')
    try:
        if is_main_process():
            (output / 'log').mkdir(exist_ok=True)
            add_log_to_file(str(output / 'log/log.txt'))
            TB_LOGGER.create(str(output / 'log'))
        else:
            logger.disabled = True
        set_seed(args.seed, args.num_gpus)
        global_params.global_args = args
        base_record = read_base_record(args)
        signature = resume_signature(args, base_record)
        from src.tasks.scst_reward import BDDXScstReward
        reward = BDDXScstReward(args.scst_reward_cache,
            cider_weight=args.scst_cider_weight, bleu4_weight=args.scst_bleu4_weight,
            dataset_name=args.dataset_name)
        swin = get_swin_model(args)
        captioner, model_config, tokenizer = get_bert_model(args)
        model = AsuadModel(args, model_config, swin, captioner)
        if args.dataset_name == 'MMAU':
            total = sum(p.numel() for p in model.parameters())
            expected = int(args.get('expected_model_parameters', 233764716))
            if total != expected:
                raise RuntimeError(f'MMAU SCST parameter mismatch: {total} != {expected}')
        weights = torch.load(args.scst_checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(weights, strict=True)
        del weights
        gc.collect()
        model.freeze_backbone(args.freeze_backbone)
        model.to(args.device)
        train_loader = make_data_loader(args, args.train_yaml, tokenizer, args.distributed, is_train=True)
        eval_loader = make_data_loader(args, args.val_yaml, tokenizer, args.distributed, is_train=False)
        per_epoch = len(train_loader.batch_sampler.batch_sampler)
        samples_per_rank = len(train_loader.batch_sampler.batch_sampler.sampler)
        max_iter = len(train_loader)
        if per_epoch < 1 or max_iter < 1:
            raise ValueError('SCST requires a nonempty training loader.')
        accumulation = max(1, args.gradient_accumulation_steps)
        if args.max_train_steps:
            if not args.smoke:
                raise ValueError('A truncated run must be explicitly marked smoke, never a ranked trial.')
            max_iter = min(max_iter, args.max_train_steps * accumulation)
        epochs, tail = divmod(max_iter, per_epoch)
        updates = epochs * math.ceil(per_epoch / accumulation) + math.ceil(tail / accumulation)
        args.update(iters_per_epoch=per_epoch, max_iter=max_iter, max_global_step=updates)
        optimizer, scheduler = optimizer_for(args, model, updates)
        scaler = native_grad_scaler(args.native_precision == 'fp16' and args.device.type == 'cuda')
        saver = TrainingSaver(str(output))
        best_score = base_record['selection_score']
        best = dict(base_record, stage='supervised_baseline', model_path=args.scst_checkpoint,
                    checkpoint=str(Path(args.scst_checkpoint).parent), model_saved=True)
        history, stale_epochs, step, start_iteration = [], 0, 0, 0
        nonzero_advantages, finite_updates = 0, 0
        if args.scst_resume:
            if args.smoke:
                raise ValueError('SCST smoke tests cannot resume an existing training state.')
            state = torch.load(args.scst_resume, map_location='cpu', weights_only=False)
            validate_resume_state(state, signature, per_epoch, max_iter, get_world_size())
            model.load_state_dict(state['model'], strict=True)
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            scaler.load_state_dict(state['scaler'])
            history, best = state['eval_log'], state['best_record']
            best_score, stale_epochs = best['selection_score'], state['stale_epochs']
            start_iteration, step = state['iteration'], state['step']
            nonzero_advantages = state.get('nonzero_advantages', 0)
            finite_updates = state.get('finite_updates', 0)
            restore_training_rng(state['rank_rng_states'][get_rank()], args.device)
            train_loader.batch_sampler.start_iter = start_iteration
            del state
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model,
                device_ids=[args.local_rank] if args.device.type == 'cuda' else None,
                static_graph=True, gradient_as_bucket_view=True)
        if is_main_process():
            saver.save_args(args)
            saver.save_tokenizer(tokenizer)
            atomic_json(output / 'best_checkpoint.json', best)
            atomic_json(output / 'selection_protocol.json', dict(name=base_record['selection_metric'],
                dataset_name=args.dataset_name, task_semantics=task_semantics(args.dataset_name),
                baseline_included=True, training_reward_split=args.train_yaml,
                evaluation_split=args.val_yaml, decoder_dropout='disabled for sampling and likelihood',
                description_credit='des reward plus exp reward', explanation_credit='exp reward',
                scst_baseline_type=args.scst_baseline_type, scst_sample_n=args.scst_sample_n,
                scst_ce_weight=args.scst_ce_weight, num_beams=args.num_beams))
        special_ids = special_token_ids(tokenizer)
        optimizer.zero_grad(set_to_none=True)
        update_status('training', step=step, completed_epochs=len(history))
        started, last_log, seen_samples = time.time(), time.time(), 0
        completed_reason = termination_reason(len(history), args.num_train_epochs, stale_epochs,
                                              args.scst_min_epochs, args.early_stopping_patience)
        stopped_early = completed_reason == 'early_stopping'
        last_iteration = start_iteration
        # A committed terminal epoch may have been saved just before process failure.
        # Resume finalization without taking another training step in that case.
        pending_batches = () if completed_reason else train_loader
        for iteration, (keys, batch, metadata) in enumerate(pending_batches, start=start_iteration + 1):
            if iteration > max_iter:
                break
            last_iteration = iteration
            model.train()
            batch = tuple(value.to(args.device, non_blocking=True) for value in batch)
            inputs = dict(input_ids=batch[0], attention_mask=batch[1], token_type_ids=batch[2],
                          img_feats=batch[3], masked_pos=batch[4], masked_ids=batch[5], car_info=batch[6])
            inputs = add_cached_motion(args, inputs, batch, is_train=True)
            inputs.update(scst=True, scst_options=dict(special_ids=special_ids,
                sample_n=args.scst_sample_n, max_length=args.max_gen_length,
                baseline_type=args.scst_baseline_type, ce_weight=args.scst_ce_weight))
            weight, update_now = accumulation_scale(iteration, max_iter, per_epoch, accumulation,
                args.per_gpu_train_batch_size, samples_per_rank, batch[0].shape[0])
            # Always synchronized: reliable with reentrant visual checkpointing on torch 1.13.
            with autocast_context(args):
                result = model(**inputs)
            sampled = decode_captions(tokenizer, result['sampled_ids'])
            greedy = (decode_captions(tokenizer, result['greedy_ids'])
                      if result['greedy_ids'] is not None else (None, None))
            rewards = reward.score(list(keys), sampled[0], sampled[1], greedy[0], greedy[1],
                                   baseline_type=args.scst_baseline_type)
            des_adv = rewards['des']['advantage']
            exp_adv = rewards['exp']['advantage']
            rl_loss = policy_loss(result['sampled_log_probs'], result['sampled_mask'], des_adv, exp_adv)
            sparse_loss = result['sparse_loss'].float() * args.loss_sparse_w
            ce_loss = result['ce_loss'].float()
            loss = rl_loss + sparse_loss + args.scst_ce_weight * ce_loss
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite SCST loss for training keys ' + str(list(keys)))
            nonzero_advantages += int(np.count_nonzero(des_adv) + np.count_nonzero(exp_adv))
            scaler.scale(loss * weight).backward()
            seen_samples += batch[0].shape[0] * get_world_size()
            metric_values = (float(loss.detach()), float(rl_loss.detach()), float(sparse_loss.detach()),
                             float(ce_loss.detach()))
            # Release the feature/decoder graph before a new forward or evaluation,
            # including the early continue used by gradient accumulation.
            del result, sampled, greedy, rewards, loss, rl_loss, sparse_loss, ce_loss, inputs, batch, metadata
            if not update_now:
                continue
            step += 1
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if not torch.isfinite(grad_norm) and not scaler.is_enabled():
                raise FloatingPointError('Nonfinite SCST gradient at step {}'.format(step))
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
                finite_updates += 1
            optimizer.zero_grad(set_to_none=True)
            if step == 1 or step % args.logging_steps == 0 or iteration == max_iter:
                speed = seen_samples / max(time.time() - last_log, 1e-6)
                logger.info('SCST iter %d/%d step %d/%d loss %.6f RL %.6f sparse %.6f CE %.6f '
                            'des_adv %.6f exp_adv %.6f grad_norm %.6f speed %.3f clips/s',
                            iteration, max_iter, step, updates, *metric_values,
                            float(np.mean(des_adv)), float(np.mean(exp_adv)), float(grad_norm), speed)
                TB_LOGGER.add_scalar('scst/rl_loss', metric_values[1], step)
                TB_LOGGER.add_scalar('scst/ce_loss', metric_values[3], step)
                TB_LOGGER.add_scalar('scst/total_loss', metric_values[0], step)
                update_status('training', step=step, iteration=iteration, loss=metric_values[0],
                              nonzero_advantages=nonzero_advantages, finite_updates=finite_updates)
                last_log, seen_samples = time.time(), 0
            if args.smoke and iteration == max_iter:
                if finite_updates < 1 or nonzero_advantages < 1:
                    raise RuntimeError('Smoke must exercise finite updates and a nonzero reward advantage.')
                update_status('smoke_complete', step=step, nonzero_advantages=nonzero_advantages,
                              finite_updates=finite_updates, elapsed_seconds=time.time() - started)
                break
            if iteration % per_epoch:
                continue
            epoch = iteration // per_epoch
            prediction_dir = output / 'epoch-{}'.format(epoch)
            update_status('evaluating', step=step, epoch=epoch)
            evaluate_file = evaluate(args, eval_loader, model, tokenizer, str(prediction_dir))
            stop = False
            if is_main_process():
                metrics = {}
                for kind in ('des', 'exp'):
                    with open(caption_result_path(evaluate_file, kind)) as stream:
                        metrics[kind] = json.load(stream)
                record = selection_record(dict(epoch=epoch, iteration=iteration, global_step=step,
                    dataset_name=args.dataset_name,
                    metrics=metrics, validation_yaml=args.val_yaml, prediction_dir=str(prediction_dir),
                    stage='scst', learning_rate=args.learning_rate, seed=args.seed,
                    scst_sample_n=args.scst_sample_n, scst_baseline_type=args.scst_baseline_type,
                    scst_ce_weight=args.scst_ce_weight))
                history.append(record)
                if record['selection_score'] > best_score + 1e-8:
                    best_score, stale_epochs = record['selection_score'], 0
                    # Keep published winners immutable so a crash between best
                    # and latest commits cannot invalidate the older resume state.
                    best_dir = output / 'checkpoint-best-epoch-{}'.format(epoch)
                    saver.save_model(str(best_dir), step, model)
                    best_path = best_dir / 'model.bin'
                    best = dict(record, checkpoint=str(best_dir), model_path=str(best_path),
                                model_sha256=file_sha256(best_path), model_saved=True)
                    atomic_json(output / 'best_checkpoint.json', best)
                else:
                    stale_epochs += 1
                atomic_json(output / 'eval_logs.json', history)
                logger.info('SCST epoch %d joint_score %.6f best %.6f stale %d',
                            epoch, record['selection_score'], best_score, stale_epochs)
                stop = epoch >= args.scst_min_epochs and stale_epochs >= args.early_stopping_patience
            decision = [stop, best, stale_epochs, history]
            if args.distributed:
                dist.broadcast_object_list(decision, src=0)
                stop, best, stale_epochs, history = decision
                best_score = best['selection_score']
            rank_rng = capture_training_rng(args.device)
            rank_rng_states = [None] * get_world_size()
            if args.distributed:
                dist.all_gather_object(rank_rng_states, rank_rng)
            else:
                rank_rng_states = [rank_rng]
            if is_main_process():
                saver.save_model(str(output / 'checkpoint-latest'), step, model, optimizer,
                    scheduler=scheduler, scaler=scaler, iteration=iteration,
                    extra_state=dict(epoch_boundary=True, world_size=get_world_size(),
                        rank_rng_states=rank_rng_states, eval_log=history, best_record=best,
                        stale_epochs=stale_epochs, resume_signature=signature,
                        nonzero_advantages=nonzero_advantages, finite_updates=finite_updates,
                        training_config={key: args[key] for key in
                            ('learning_rate', 'seed', 'scst_sample_n', 'per_gpu_train_batch_size',
                             'gradient_accumulation_steps', 'num_train_epochs')}))
            if args.distributed:
                dist.barrier()
            update_status('training', completed_epochs=epoch, step=step, best_score=best_score)
            if stop:
                stopped_early = True
                logger.info('SCST early stopping after epoch %d', epoch)
                break
        if not args.smoke:
            validate_completion(last_iteration, max_iter, per_epoch, history, stopped_early)
            completed_reason = termination_reason(len(history), args.num_train_epochs, stale_epochs,
                                                  args.scst_min_epochs, args.early_stopping_patience)
            if is_main_process():
                atomic_json(output / 'comparison.json', dict(supervised_baseline=base_record,
                    selected=best, improvement=best['selection_score'] - base_record['selection_score'],
                    completed_epochs=len(history), selection_metric=base_record['selection_metric'],
                    termination_reason=completed_reason))
                update_status('complete', completed_epochs=len(history), best_score=best_score,
                              selected_checkpoint=best['model_path'], termination_reason=completed_reason,
                              elapsed_seconds=time.time() - started)
            if args.distributed:
                dist.barrier()
        elif last_iteration != max_iter or finite_updates < 1 or nonzero_advantages < 1:
            raise RuntimeError('SCST smoke ended without all requested finite training updates.')
    except Exception as error:
        update_status('failed', error=repr(error))
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    cli = parser.parse_args()
    with open(cli.config) as stream:
        configuration = json.load(stream)
    os.chdir(str(ROOT))
    run(configuration)
