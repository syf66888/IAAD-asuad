"""Gradient-safe SCST decoding for the fixed-width BDDX caption pair.

The discrete rollouts reuse the existing inference prefix/cache path under
``no_grad``. A single deterministic, parallel BERT forward then recomputes
their token probabilities with gradients. Content and masked query streams
have separate causal masks, so a query cannot see its target or later words.
No model parameters are added; the caller encodes the video exactly once.
"""

from contextlib import contextmanager

import torch
from torch.nn import functional as F


@contextmanager
def _deterministic_decoder(decoder):
    modes = [(module, module.training) for module in decoder.modules()]
    decoder.eval()
    try:
        yield
    finally:
        # Preserve deliberately frozen submodules as well as the root mode.
        for module, training in modes:
            module.training = training


def generation_attention_mask(attention_mask, visual_token_count, max_length=35):
    """Keep learned visual attention, replacing all GT-dependent text masks."""
    if max_length < 2:
        raise ValueError('A caption slot needs BOS and EOS positions.')
    if attention_mask.ndim != 3 or attention_mask.shape[-1] != attention_mask.shape[-2]:
        raise ValueError('Expected a square [B,T+V,T+V] attention mask.')
    if visual_token_count < 1 or visual_token_count > attention_mask.shape[-1]:
        raise ValueError('Invalid visual token count.')
    text_length = 2 * max_length
    visual = attention_mask[:, -visual_token_count:, -visual_token_count:]
    mask = visual.new_zeros((visual.shape[0], text_length + visual_token_count,
                             text_length + visual_token_count))
    mask[:, :text_length, :text_length] = torch.ones(
        text_length, text_length, device=visual.device, dtype=visual.dtype).tril()
    mask[:, :text_length, text_length:] = 1
    mask[:, text_length:, text_length:] = visual
    return mask


def caption_token_mask(tokens, eos_token_ids, pad_token_id, forced_eos=None):
    """Keep sampled actions; exclude BOS, forced EOS and post-EOS padding.

    Production generation can sample PAD *before* EOS and then continue.
    That PAD is a policy action too, even though the tokenizer removes it;
    silently discarding its log probability would bias the policy gradient.
    Every rollout ends with natural or forced EOS, so all actual padding is
    unambiguously after that first EOS.
    """
    if pad_token_id in eos_token_ids:
        raise ValueError('PAD and EOS must be different for the dual-caption policy.')
    eos = torch.zeros_like(tokens, dtype=torch.bool)
    for token_id in eos_token_ids:
        eos |= tokens.eq(token_id)
    previous_eos = eos.long().cumsum(dim=-1) - eos.long()
    valid = previous_eos.eq(0)
    valid[..., 0] = False
    if forced_eos is not None:
        valid &= ~forced_eos
    return valid


def _types(batch_size, max_length, device):
    result = torch.zeros(batch_size, 2 * max_length, dtype=torch.long, device=device)
    result[:, max_length:] = 1
    return result


def _rollout(decoder, features, attention_mask, special_ids, max_length,
             do_sample, top_k, temperature):
    """Production prefix/cache semantics, retaining forced-EOS provenance.

    Unlike the old generation utility, sampling/softmax use FP32 under AMP.
    Natural EOS remains a policy action; a length-limit substitution does not.
    """
    batch_size = features.shape[0]
    total_length = 2 * max_length
    state = dict(
        img_seq_len=features.shape[1], max_seq_len=total_length,
        mask_token_id=special_ids['mask_token_id'], prev_encoded_layers=None,
        add_od_labels=False, od_labels_len=0, od_label_ids=None,
        img_feats=features, full_attention_mask=attention_mask,
        full_masked_pos=torch.ones(batch_size, total_length, dtype=torch.long, device=features.device),
        full_token_type_ids=_types(batch_size, max_length, features.device),
        full_position_ids=torch.arange(total_length, device=features.device)[None].expand(batch_size, -1))
    absent = object()
    old_state = {key: getattr(decoder, key, absent) for key in state}
    encoder = decoder.bert.encoder
    old_hidden = encoder.output_hidden_states
    old_attentions = encoder.output_attentions
    encoder.output_hidden_states = True
    encoder.output_attentions = False
    for key, value in state.items():
        setattr(decoder, key, value)

    def decode(prefix, limit):
        active = torch.ones(batch_size, device=features.device, dtype=torch.bool)
        forced = torch.zeros(batch_size, limit, device=features.device, dtype=torch.bool)
        past = None
        while prefix.shape[1] < limit:
            inputs = decoder.prepare_inputs_for_generation(prefix, past=past)
            outputs = decoder(**inputs)
            logits = outputs[0][:, -1].float()
            if decoder._do_output_past(outputs):
                past = outputs[1]
            if do_sample:
                logits = logits / temperature
                if top_k:
                    cutoff = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < cutoff, float('-inf'))
                next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)[:, 0]
            else:
                next_token = logits.argmax(dim=-1)
            next_token = torch.where(active, next_token, next_token.new_full((), special_ids['pad_token_id']))
            is_eos = torch.zeros_like(active)
            for eos_id in special_ids['eos_token_ids']:
                is_eos |= next_token.eq(eos_id)
            if prefix.shape[1] == limit - 1:
                replaced = active & ~is_eos
                forced[:, -1] = replaced
                next_token = torch.where(replaced, next_token.new_full((), special_ids['eos_token_ids'][0]), next_token)
                is_eos |= replaced
            prefix = torch.cat((prefix, next_token[:, None]), dim=1)
            active &= ~is_eos
            if not active.any():
                break
        if prefix.shape[1] < limit:
            prefix = torch.cat((prefix, prefix.new_full(
                (batch_size, limit - prefix.shape[1]), special_ids['pad_token_id'])), dim=1)
        return prefix, forced

    try:
        bos = torch.full((batch_size, 1), special_ids['bos_token_id'], device=features.device, dtype=torch.long)
        action, action_forced = decode(bos, max_length)
        decoder.prev_encoded_layers = None
        decoder.full_attention_mask = attention_mask.clone()
        decoder.full_attention_mask[:, :, :max_length] *= action.ne(
            special_ids['pad_token_id'])[:, None].to(attention_mask.dtype)
        pair, pair_forced = decode(torch.cat((action, bos), dim=1), total_length)
        pair_forced[:, :max_length] = action_forced
        return pair.reshape(batch_size, 2, max_length), pair_forced.reshape(batch_size, 2, max_length)
    finally:
        encoder.output_hidden_states = old_hidden
        encoder.output_attentions = old_attentions
        for key, old_value in old_state.items():
            if old_value is absent:
                delattr(decoder, key)
            else:
                setattr(decoder, key, old_value)


def _parallel_query_logits(decoder, features, generation_mask, tokens, special_ids):
    """Score every sampled word under its exact generated prefix in one pass.

    Layout is [des content, pair content for exp, des queries, exp queries,
    visual]. The extra copy of des content is necessary: production inference
    re-encodes des with its PAD columns removed before starting exp. A PAD
    sampled *within* des must still remain visible to later des words.

    Each query uses the target position/type embedding and a MASK input token;
    it attends only to its real prefix, itself, and visual features. Content
    cannot attend to queries, and visual features cannot attend to text.
    """
    batch_size, segments, slot_length = tokens.shape
    if segments != 2 or slot_length < 2:
        raise ValueError('Expected sampled tokens [B,2,L], with L >= 2.')
    visual_length = features.shape[1]
    pair_length = 2 * slot_length
    content_length = 3 * slot_length
    query_length = 2 * (slot_length - 1)
    text_length = content_length + query_length
    flat_tokens = tokens.reshape(batch_size, pair_length)
    input_ids = torch.cat((tokens[:, 0], flat_tokens,
                          tokens.new_full((batch_size, query_length), special_ids['mask_token_id'])), dim=1)
    positions = torch.cat((torch.arange(slot_length, device=tokens.device),
                           torch.arange(pair_length, device=tokens.device),
                           torch.arange(1, slot_length, device=tokens.device),
                           torch.arange(slot_length + 1, pair_length, device=tokens.device)))
    token_types = torch.cat((tokens.new_zeros(batch_size, slot_length),
                             _types(batch_size, slot_length, tokens.device),
                             tokens.new_zeros(batch_size, slot_length - 1),
                             tokens.new_ones(batch_size, slot_length - 1)), dim=1)
    mask = generation_mask.new_zeros((batch_size, text_length + visual_length,
                                      text_length + visual_length))
    des_attention = generation_mask[:, :slot_length, :slot_length]
    exp_attention = generation_mask[:, :pair_length, :pair_length].clone()
    action_valid = tokens[:, 0].ne(special_ids['pad_token_id'])
    exp_attention[:, :, :slot_length] = (
        exp_attention[:, :, :slot_length] * action_valid[:, None].to(exp_attention.dtype))
    mask[:, :slot_length, :slot_length] = des_attention
    mask[:, slot_length:content_length, slot_length:content_length] = exp_attention
    # The query cannot attend to the content token at its own target position.
    des_prefix = torch.arange(slot_length, device=tokens.device)[None] < torch.arange(
        1, slot_length, device=tokens.device)[:, None]
    exp_prefix = torch.arange(pair_length, device=tokens.device)[None] < torch.arange(
        slot_length + 1, pair_length, device=tokens.device)[:, None]
    query_split = content_length + slot_length - 1
    mask[:, content_length:query_split, :slot_length] = des_attention[:, 1:] * des_prefix
    mask[:, query_split:text_length, slot_length:content_length] = (
        exp_attention[:, slot_length + 1:] * exp_prefix)
    mask[:, content_length:text_length, content_length:text_length] = torch.eye(
        query_length, device=tokens.device, dtype=mask.dtype)
    mask[:, :text_length, text_length:] = 1
    mask[:, text_length:, text_length:] = generation_mask[:, pair_length:, pair_length:]
    hidden = decoder.bert(
        input_ids, img_feats=features, attention_mask=mask,
        position_ids=positions[None].expand(batch_size, -1), token_type_ids=token_types)[0]
    # Apply the large vocabulary head only to the queries, not content/video.
    return decoder.cls(hidden[:, content_length:text_length]).reshape(
        batch_size, 2, slot_length - 1, -1).float()


def sampled_token_log_probs(decoder, visual_features, generation_mask, tokens,
                            special_ids, top_k=0, temperature=1.0, forced_eos=None):
    """Differentiable rescoring; tokens/masks/log_probs have [B,2,L] shape."""
    if temperature <= 0 or top_k < 0:
        raise ValueError('temperature must be positive and top_k nonnegative.')
    with _deterministic_decoder(decoder):
        logits = _parallel_query_logits(decoder, visual_features, generation_mask, tokens, special_ids)
        logits = logits / temperature
        if top_k:
            cutoff = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < cutoff, float('-inf'))
        scores = F.log_softmax(logits, dim=-1).gather(-1, tokens[:, :, 1:, None]).squeeze(-1)
        scores = torch.cat((scores.new_zeros(scores.shape[0], 2, 1), scores), dim=-1)
        valid = caption_token_mask(tokens, special_ids['eos_token_ids'], special_ids['pad_token_id'], forced_eos)
        # Forced terminal EOS can be outside top-k. masked_fill avoids 0 * -inf.
        return scores.masked_fill(~valid, 0.0), valid


def scst_decode(decoder, visual_features, attention_mask, special_ids,
                sample_n=2, max_length=35, top_k=0, temperature=1.0,
                baseline_type='greedy', teacher_forcing=None):
    """Independent sampled pairs with optional greedy and masked CE forwards.

    Returns a dictionary with ``greedy_ids`` [B,2,L], and ``sampled_ids``,
    ``sampled_log_probs``, ``sampled_mask`` [B*N,2,L]. Batch order is
    ``b0s0,b0s1,...,b1s0,b1s1,...``. Both captions condition only on sampled
    words; exp conditions on the corresponding sampled des, never the GT.

    Call *inside* the outer AsuadModel/DDP forward after its one visual
    encoding; gradients flow through the shared visual features and decoder.
    Pure vision, no OD labels or sensor tokens. ``max_length`` is per caption.
    """
    if isinstance(sample_n, bool) or not 1 <= sample_n <= 16 or int(sample_n) != sample_n:
        raise ValueError('sample_n must be an integer in [1, 16].')
    sample_n = int(sample_n)
    if baseline_type not in ('greedy', 'leave_one_out'):
        raise ValueError('baseline_type must be greedy or leave_one_out.')
    if baseline_type == 'leave_one_out' and sample_n < 2:
        raise ValueError('leave_one_out requires at least two independent samples.')
    if temperature <= 0 or top_k < 0:
        raise ValueError('temperature must be positive and top_k nonnegative.')
    required = {'bos_token_id', 'pad_token_id', 'eos_token_ids', 'mask_token_id'}
    if set(special_ids) != required or not special_ids['eos_token_ids']:
        raise ValueError('Provide BOS/PAD/MASK ids and a non-empty eos_token_ids list.')
    mask = generation_attention_mask(attention_mask, visual_features.shape[1], max_length)
    repeated_features = visual_features.repeat_interleave(sample_n, dim=0)
    repeated_mask = mask.repeat_interleave(sample_n, dim=0)
    greedy = None
    with _deterministic_decoder(decoder):
        with torch.no_grad():
            if baseline_type == 'greedy':
                greedy, _ = _rollout(decoder, visual_features.detach(), mask.detach(), special_ids,
                                     max_length, False, 0, 1.0)
            sampled, forced = _rollout(decoder, repeated_features.detach(), repeated_mask.detach(),
                                       special_ids, max_length, True, top_k, temperature)
        # Old AMP backends can cache parameter casts made under no_grad.
        # The likelihood pass must create casts that retain parameter gradients.
        torch.clear_autocast_cache()
        scores, valid = sampled_token_log_probs(
            decoder, repeated_features, repeated_mask, sampled, special_ids, top_k, temperature, forced)
    # Keep the original teacher-forcing mask and masked targets. In particular,
    # never replace it with the generation mask or use sampled tokens as labels.
    # The decoder's train/frozen modes have been restored before this forward.
    ce_loss = visual_features.new_zeros(())
    if teacher_forcing is not None:
        teacher = dict(teacher_forcing)
        required_teacher = {'input_ids', 'attention_mask', 'masked_pos', 'masked_ids'}
        if not required_teacher.issubset(teacher):
            raise ValueError('Masked CE requires original training inputs, mask and targets.')
        if any(teacher[name].shape[0] != visual_features.shape[0] for name in required_teacher):
            raise ValueError('Masked CE uses one original training example per visual encoding.')
        positions = teacher['masked_pos'].eq(1)
        targets = teacher['masked_ids'].ne(-1)
        if (not positions.any() or not torch.equal(positions.sum(-1), targets.sum(-1))):
            raise ValueError('Masked CE positions and nonpadding targets must match per clip.')
        teacher['img_feats'] = visual_features
        teacher['is_training'] = True
        ce_loss = decoder(**teacher)[0].float()
        if ce_loss.ndim != 0:
            raise ValueError('Caption decoder must return scalar masked CE loss.')
    return {'greedy_ids': greedy, 'sampled_ids': sampled,
            'sampled_log_probs': scores, 'sampled_mask': valid, 'sampled_forced_eos': forced,
            'ce_loss': ce_loss}
