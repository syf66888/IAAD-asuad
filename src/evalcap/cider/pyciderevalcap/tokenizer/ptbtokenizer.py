"""CIDEr adapter for the checked Stanford PTB tokenizer runtime."""

import os

from src.evalcap.coco_caption.pycocoevalcap.tokenizer.ptbtokenizer import (
    PUNCTUATIONS,
    STANFORD_CORENLP_3_4_1_JAR,
    _tokenize_captions,
)


class PTBTokenizer:
    """Preserve CIDEr's ground-truth dictionary and result-list interfaces."""

    def __init__(self, _source='gts'):
        if _source not in ('gts', 'res'):
            raise ValueError('PTBTokenizer source must be gts or res, got {!r}'.format(_source))
        self.source = _source

    def tokenize(self, captions_for_image):
        image_ids = []
        captions = []
        if self.source == 'gts':
            for image_id, image_captions in captions_for_image.items():
                for caption in image_captions:
                    image_ids.append(image_id)
                    captions.append(caption['caption'])
        else:
            for caption in captions_for_image:
                image_ids.append(caption['image_id'])
                captions.append(caption['caption'])
        lines = _tokenize_captions(captions, os.path.dirname(os.path.abspath(__file__)))
        if self.source == 'res':
            return [{'image_id': image_id, 'caption': [line]}
                    for image_id, line in zip(image_ids, lines)]
        result = {}
        for image_id, line in zip(image_ids, lines):
            result.setdefault(image_id, []).append(line)
        return result
