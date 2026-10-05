import itertools
import re

import torch
from src.utils.logger import LOGGER as logger
from src.modeling.video_swin.swin_transformer import SwinTransformer3D
from src.modeling.video_swin.config import Config


def resolve_swin_depths(requested, source_depths):
    """Allow structural depth reduction without changing pretrained widths."""
    if requested is None or requested == '' or requested == []:
        return list(source_depths)
    if isinstance(requested, str):
        requested = requested.split(',')
    if not isinstance(requested, (list, tuple)) or len(requested) != len(source_depths):
        raise ValueError('video_swin_depths must contain one integer per stage')
    depths = []
    for value, maximum in zip(requested, source_depths):
        if isinstance(value, bool) or (not isinstance(value, (int, str))):
            raise ValueError('Video Swin depths must be integers')
        depth = int(value)
        if depth < 1 or depth > maximum:
            raise ValueError('Video Swin depths must be positive and cannot exceed pretrained depths')
        depths.append(depth)
    return depths


def uniform_block_indices(source_depth, target_depth):
    """Spread retained blocks across a stage while preserving W/SW parity.

    Blocks alternate unshifted/shifted windows. Blind rounding can put shifted
    weights into an unshifted block, so candidates preserve the original parity.
    For the tiny model, 6->4 selects [0, 1, 4, 5], and 2->1 selects [0].
    """
    if not 1 <= target_depth <= source_depth:
        raise ValueError('Invalid source/target block counts')
    if target_depth == 1:
        return [0]
    ideals = [i * (source_depth - 1) / (target_depth - 1) for i in range(target_depth)]
    candidates = (indices for indices in itertools.combinations(range(source_depth), target_depth)
                  if all(old % 2 == new % 2 for new, old in enumerate(indices)))
    return list(min(candidates, key=lambda indices: (
        sum((old - ideal) ** 2 for old, ideal in zip(indices, ideals)), indices)))


def remap_swin_checkpoint(source_state, target_state, source_depths, target_depths):
    """Build a complete, shape-checked state for the actual reduced backbone."""
    indices = [uniform_block_indices(old, new)
               for old, new in zip(source_depths, target_depths)]
    block_pattern = re.compile(r'^backbone\.layers\.(\d+)\.blocks\.(\d+)\.(.+)$')
    mapped = {}
    for key, target in target_state.items():
        match = block_pattern.match(key)
        source_key = key
        if match:
            stage, block, suffix = int(match[1]), int(match[2]), match[3]
            source_key = 'backbone.layers.%d.blocks.%d.%s' % (stage, indices[stage][block], suffix)
        if source_key not in source_state:
            raise RuntimeError('Video Swin pretrained tensor missing: %s (target %s)' % (source_key, key))
        tensor = source_state[source_key]
        if tuple(tensor.shape) != tuple(target.shape):
            raise RuntimeError('Video Swin tensor shape mismatch for %s: source %s, target %s' % (
                key, tuple(tensor.shape), tuple(target.shape)))
        mapped[key] = tensor
    return mapped, indices


def _load_pretrained_3d(video_swin, model_path, source_depths, target_depths):
    checkpoint_3d = torch.load(model_path, map_location='cpu', weights_only=False)
    if 'state_dict' not in checkpoint_3d:
        raise RuntimeError('Expected an official Video Swin checkpoint with state_dict')
    state, indices = remap_swin_checkpoint(checkpoint_3d['state_dict'], video_swin.state_dict(),
                                          source_depths, target_depths)
    video_swin.load_state_dict(state, strict=True)
    video_swin.pretrained_block_indices = indices
    logger.info('Loaded complete Video Swin backbone from %s; retained stage blocks: %s',
                model_path, indices)


def get_swin_model(args):
    if int(args.img_res) == 384:
        assert args.vidswin_size == "large"
        config_path = 'src/modeling/video_swin/swin_%s_384_patch244_window81212_kinetics%s_22k.py'%(args.vidswin_size, args.kinetics)
        model_path = 'models/video_swin_transformer/swin_%s_384_patch244_window81212_kinetics%s_22k.pth'%(args.vidswin_size, args.kinetics)
    else:
        # in the case that args.img_res == '224'
        #用的是这个
        #config_path = 'src/modeling/video_swin/swin_%s_patch244_window877_kinetics%s_22k.py'%(args.vidswin_size, args.kinetics)
        #model_path = 'models/video_swin_transformer/swin_%s_patch244_window877_kinetics%s_22k.pth'%(args.vidswin_size, args.kinetics)
        pretraining = '1k' if args.vidswin_size == 'tiny' else '22k'
        stem = 'swin_%s_patch244_window877_kinetics%s_%s' % (args.vidswin_size, args.kinetics, pretraining)
        config_path = 'src/modeling/video_swin/%s.py' % stem
        model_path = 'models/video_swin_transformer/%s.pth' % stem
        #base,
    if args.pretrained_2d:
        config_path = 'src/modeling/video_swin/swin_base_patch244_window877_kinetics400_22k.py'
        model_path = 'models/swin_transformer/swin_base_patch4_window7_224_22k.pth'

    model_path = getattr(args, 'video_swin_pretrained_path', '') or model_path

    logger.info(f'video swin (config path): {config_path}')
    if args.pretrained_checkpoint == '':
        logger.info(f'video swin (model path): {model_path}')
    cfg = Config.fromfile(config_path)
    source_depths = list(cfg.model['backbone']['depths'])
    target_depths = resolve_swin_depths(getattr(args, 'video_swin_depths', None), source_depths)
    if args.pretrained_2d and target_depths != source_depths:
        raise ValueError('Depth reduction currently requires a 3D Video Swin pretrained checkpoint')
    pretrained_path = model_path if args.pretrained_2d else None
    backbone = SwinTransformer3D(
                    pretrained=pretrained_path,
                    pretrained2d=args.pretrained_2d,
                    patch_size=cfg.model['backbone']['patch_size'],
                    in_chans=3,
                    embed_dim=cfg.model['backbone']['embed_dim'],
                    depths=target_depths,
                    num_heads=cfg.model['backbone']['num_heads'],
                    window_size=cfg.model['backbone']['window_size'],
                    mlp_ratio=4.,
                    qkv_bias=True,
                    qk_scale=None,
                    drop_rate=0.,
                    attn_drop_rate=0.,
                    drop_path_rate=0.2,
                    norm_layer=torch.nn.LayerNorm,
                    patch_norm=cfg.model['backbone']['patch_norm'],
                    frozen_stages=-1,
                    use_checkpoint=False)

    video_swin = AsuadVideoSwin(args=args, cfg=cfg, backbone=backbone)
    video_swin.pretrained_source_depths = source_depths
    video_swin.model_depths = target_depths
    video_swin.pretrained_model_path = str(model_path)

    if not args.pretrained_2d:
        #进入这个
        _load_pretrained_3d(video_swin, model_path, source_depths, target_depths)
    else:
        video_swin.backbone.init_weights()
    return video_swin

def reload_pretrained_swin(video_swin, args):
    if not args.reload_pretrained_swin:
        return video_swin
    if hasattr(video_swin, 'pretrained_model_path') and not args.pretrained_2d:
        _load_pretrained_3d(video_swin, video_swin.pretrained_model_path,
                           video_swin.pretrained_source_depths, video_swin.model_depths)
        return video_swin
    if int(args.img_res) == 384:
        model_path = './models/video_swin_transformer/swin_%s_384_patch244_window81212_kinetics%s_22k.pth'%(args.vidswin_size, args.kinetics)
    else:
        # in the case that args.img_res == '224'
        model_path = './models/video_swin_transformer/swin_%s_patch244_window877_kinetics%s_22k.pth'%(args.vidswin_size, args.kinetics)

    checkpoint_3d = torch.load(model_path, map_location='cpu', weights_only=False)
    missing, unexpected = video_swin.load_state_dict(checkpoint_3d['state_dict'], strict=False)
    logger.info(f"re-loaded video_swin_transformer from {model_path}")

    logger.info(f"Missing keys in loaded video_swin_transformerr: {missing}")
    logger.info(f"Unexpected keys in loaded video_swin_transformer: {unexpected}")
    return video_swin

class AsuadVideoSwin(torch.nn.Module):
    def __init__(self, args, cfg, backbone):
        super(AsuadVideoSwin, self).__init__()
        #用父类 torch.nn.Module 的 __init__ 方法，即初始化父类的属性和方法。
        #确保子类 AsuadVideoSwin 在初始化时会继承父类 torch.nn.Module 的所有属性和方法，并且进行必要的初始化，以确保整个类的正确工作。
        self.backbone = backbone
        self.use_grid_feature = args.grid_feat

    def forward(self, x):
        x = self.backbone(x)
        return x
