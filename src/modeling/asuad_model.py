"""Driving caption model with correctly aligned visual reference features."""

import math
from contextlib import nullcontext

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from src.modeling.reference_fusion import ReferenceFeatureFusion, video_grid_to_tokens


class AsuadModel(torch.nn.Module):
    def __init__(self, args, config, swin, transformer_encoder):
        super().__init__()
        self.config = config
        self.use_checkpoint = args.use_checkpoint and not args.freeze_backbone
        # Native checkpoint preserves BF16/FP16 autocast dtype during replay;
        # FairScale 0.4.x replays under the default FP16 context instead.
        self.swin = swin
        self.checkpoint_offload_to_cpu = getattr(args, 'checkpoint_offload_to_cpu', True)
        self.trans_encoder = transformer_encoder
        self.img_feature_dim = int(args.img_feature_dim)
        self.use_grid_feat = args.grid_feat
        self.latent_feat_size = self.swin.backbone.norm.normalized_shape[0]
        # Retain the Eden checkpoint's C+4 projection shape in every mode.
        self.fc = torch.nn.Linear(self.latent_feat_size + 4, self.img_feature_dim)
        self.compute_mask_on_the_fly = False
        self.mask_prob = args.mask_prob
        self.mask_token_id = -1
        self.max_img_seq_length = args.max_img_seq_length
        self.max_num_frames = getattr(args, 'max_num_frames', 2)
        self.expand_car_info = torch.nn.Linear(self.max_num_frames, self.img_feature_dim)
        self.use_car_sensor = getattr(args, 'use_car_sensor', False)
        self.learn_mask_enabled = getattr(args, 'learn_mask_enabled', False)
        self.sparse_mask_soft2hard = getattr(args, 'sparse_mask_soft2hard', False)
        self.trans_encoder.bert.use_log_attention_mask = bool(getattr(args, 'learn_mask_log_bias', False))
        if self.learn_mask_enabled:
            self.learn_vid_att = torch.nn.Embedding(self.max_img_seq_length ** 2, 1)
            self.sigmoid = torch.nn.Sigmoid()

        self.feature_fusion = getattr(args, 'feature_fusion', 'legacy')
        if self.feature_fusion not in ('none', 'legacy', 'gated'):
            raise ValueError('feature_fusion must be none, legacy, or gated')
        self.visual_references = None
        if self.feature_fusion != 'none':
            from src.modeling.visual_references import VisualReferenceExtractor
            self.visual_references = VisualReferenceExtractor(args)
        if self.feature_fusion == 'gated':
            self.reference_fusion = ReferenceFeatureFusion(
                self.img_feature_dim,
                hidden_dim=getattr(args, 'fusion_hidden_dim', 128),
                reference_dim=self.latent_feat_size + 4,
                dropout=getattr(args, 'fusion_dropout', 0.1),
            )

    def encode_visual_features(self, images, reference_features=None, motion_features=None):
        """Return [B,N,D] tokens, preserving spatial and temporal ordering."""
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError(f'Expected images [B,S,3,H,W], got {images.shape}')
        if reference_features is not None and motion_features is not None:
            raise ValueError('Pass complete reference_features or per-frame motion_features, not both')
        video_inputs = images.permute(0, 2, 1, 3, 4)
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            # Images do not require gradients, but reentrant checkpoint needs
            # one differentiable input to retain parameter gradients.
            dummy = video_inputs.new_empty(0, requires_grad=True)
            offload = (torch.autograd.graph.save_on_cpu(pin_memory=video_inputs.is_cuda)
                       if self.checkpoint_offload_to_cpu else nullcontext())
            with offload:
                grid = checkpoint(lambda frames, unused: self.swin(frames),
                                  video_inputs, dummy, use_reentrant=True)
        else:
            grid = self.swin(video_inputs)
        tokens, temporal_steps = video_grid_to_tokens(grid, self.latent_feat_size)
        batch_size, token_count, channels = tokens.shape
        spatial_tokens = token_count // temporal_steps

        if self.feature_fusion == 'none':
            motion = tokens.new_zeros(batch_size, temporal_steps, 4)
            object_refs = None
        elif reference_features is not None:
            expected = (batch_size, temporal_steps, channels + 4)
            if tuple(reference_features.shape) != expected:
                raise ValueError(f'Expected reference_features {expected}, got {reference_features.shape}')
            reference_features = reference_features.to(device=tokens.device, dtype=tokens.dtype)
            object_refs, motion = reference_features[..., :-4], reference_features[..., -4:]
        else:
            if motion_features is None:
                object_refs, motion = self.visual_references(images, grid)
            else:
                object_refs, motion = self.visual_references(images, grid, motion_features=motion_features)
            if tuple(object_refs.shape) != (batch_size, temporal_steps, channels):
                raise ValueError(f'Object references must align with Swin time, got {object_refs.shape}')
            if tuple(motion.shape) != (batch_size, temporal_steps, 4):
                raise ValueError(f'Motion references must be [B,T,4], got {motion.shape}')
            object_refs = object_refs.to(device=tokens.device, dtype=tokens.dtype)
            motion = motion.to(device=tokens.device, dtype=tokens.dtype)

        if self.feature_fusion == 'legacy':
            # Corrected legacy concatenation: one region per Swin time step,
            # followed by pooling to the configured budget (formerly 784).
            time_tokens = tokens.reshape(batch_size, temporal_steps, spatial_tokens, channels)
            time_tokens = torch.cat((time_tokens, object_refs.unsqueeze(2)), dim=2)
            time_motion = motion.unsqueeze(2).expand(-1, -1, spatial_tokens + 1, -1)
            combined = torch.cat((time_tokens, time_motion), dim=-1).reshape(batch_size, -1, channels + 4)
            combined = F.adaptive_avg_pool1d(combined.transpose(1, 2), self.max_img_seq_length).transpose(1, 2)
            return self.fc(combined)

        if token_count != self.max_img_seq_length:
            raise ValueError(
                f'Swin produced {token_count} tokens from grid {tuple(grid.shape)}, '
                f'but max_img_seq_length={self.max_img_seq_length}; align the frame/grid configuration'
            )
        aligned_motion = motion.repeat_interleave(spatial_tokens, dim=1)
        features = self.fc(torch.cat((tokens, aligned_motion), dim=-1))
        if self.feature_fusion == 'gated':
            references = torch.cat((object_refs, motion), dim=-1)
            features = self.reference_fusion(features, references, temporal_steps=temporal_steps)
        return features

    def forward(self, *args, **kwargs):
        if args and isinstance(args[0], dict):
            kwargs = dict(args[0])
            args = ()
        # Auxiliary tensors belong to this module, never the caption decoder.
        reference_features = kwargs.pop('reference_features', None)
        motion_features = kwargs.pop('motion_features', None)
        return_visual_features = kwargs.pop('return_visual_features', False)
        scst = kwargs.pop('scst', False)
        scst_options = kwargs.pop('scst_options', None)
        if return_visual_features and (scst or kwargs.get('is_decode', False)):
            raise ValueError('Feature distillation is only supported by the teacher-forced caption path.')
        features = self.encode_visual_features(kwargs['img_feats'], reference_features, motion_features)
        video_token_count = features.shape[1]
        sensor_token_count = 0
        if self.use_car_sensor:
            car_info = kwargs['car_info']
            if car_info.ndim != 3 or car_info.shape[-1] != self.max_num_frames:
                raise ValueError(f'Expected car_info [B,signals,{self.max_num_frames}], got {car_info.shape}')
            car_info = car_info.to(device=features.device, dtype=features.dtype)
            sensor_tokens = self.expand_car_info(car_info)
            sensor_token_count = sensor_tokens.shape[1]
            features = torch.cat((features, sensor_tokens), dim=1)
        kwargs['img_feats'] = features
        if self.trans_encoder.bert.encoder.output_attentions:
            self.trans_encoder.bert.encoder.set_output_attentions(False)

        if self.learn_mask_enabled:
            mask = kwargs['attention_mask']
            if mask.ndim != 3 or mask.shape[-1] != mask.shape[-2]:
                raise ValueError('Learned video attention requires a square [B,L,L] attention mask')
            if video_token_count != self.max_img_seq_length:
                raise ValueError('Learned attention token count does not match video features')
            start = mask.shape[-1] - video_token_count - sensor_token_count
            if start < 0:
                raise ValueError('Attention mask is shorter than the visual and sensor tokens')
            probabilities = self.learn_vid_att.weight.float().reshape(video_token_count, video_token_count).sigmoid()
            diagonal = torch.eye(video_token_count, device=probabilities.device, dtype=probabilities.dtype)
            video_attention = probabilities * (1.0 - diagonal)
            learned = diagonal + video_attention
            if self.sparse_mask_soft2hard:
                hard = (learned >= 0.5).to(learned.dtype)
                learned = hard + learned - learned.detach() if self.training else hard
            mask = mask.float().clone()
            end = start + video_token_count
            # Leave caption padding, causality and appended sensors intact.
            allowed_video = mask[:, start:end, start:end].clone()
            mask[:, start:end, start:end] = allowed_video * learned
            kwargs['attention_mask'] = mask

        if scst:
            if self.use_car_sensor or scst_options is None:
                raise ValueError('SCST requires pure visual inputs and explicit decoding options.')
            from .scst_decode import scst_decode
            options = dict(scst_options)
            ce_weight = float(options.pop('ce_weight', 0.0))
            if not math.isfinite(ce_weight) or ce_weight < 0:
                raise ValueError('SCST CE weight must be finite and nonnegative.')
            teacher = None
            if ce_weight > 0:
                teacher = {name: kwargs[name] for name in (
                    'input_ids', 'attention_mask', 'masked_pos', 'masked_ids',
                    'token_type_ids', 'position_ids', 'head_mask') if name in kwargs}
            outputs = scst_decode(self.trans_encoder, features, kwargs['attention_mask'],
                                  teacher_forcing=teacher, **options)
            outputs['sparse_loss'] = (self.get_loss_sparsity(video_attention)
                                      if self.learn_mask_enabled else features.new_zeros(()))
            return outputs
        outputs = self.trans_encoder(*args, **kwargs)
        if self.learn_mask_enabled:
            outputs = outputs + (self.get_loss_sparsity(video_attention),)
        return (outputs, features) if return_visual_features else outputs

    def get_loss_sparsity(self, video_attention):
        return video_attention.abs().mean()

    def reload_attn_mask(self, pretrain_attn_mask):
        pretrained_tokens = math.isqrt(pretrain_attn_mask.numel())
        if pretrained_tokens ** 2 != pretrain_attn_mask.numel():
            raise ValueError('Pretrained attention mask must contain a square number of values')
        mask = pretrain_attn_mask.reshape(1, 1, pretrained_tokens, pretrained_tokens).float()
        if pretrained_tokens != self.max_img_seq_length:
            mask = F.interpolate(mask, size=(self.max_img_seq_length, self.max_img_seq_length), mode='bilinear', align_corners=False)
        with torch.no_grad():
            self.learn_vid_att.weight.copy_(mask.reshape(-1, 1).to(self.learn_vid_att.weight))

    def freeze_backbone(self, freeze=True):
        for parameter in self.swin.parameters():
            parameter.requires_grad = not freeze
