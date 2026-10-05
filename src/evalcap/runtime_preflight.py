"""Exercise the actual PTB/B4/CIDEr/ROUGE stack before spending GPU time."""
import hashlib
import math
from pathlib import Path


def validate_caption_metrics():
    from src.evalcap.coco_caption.pycocoevalcap.tokenizer import ptbtokenizer
    from src.evalcap.coco_caption.pycocoevalcap.bleu.bleu import Bleu
    from src.evalcap.coco_caption.pycocoevalcap.cider.cider import Cider
    from src.evalcap.coco_caption.pycocoevalcap.rouge.rouge import Rouge

    captions = {
        'left': [{'caption': 'The car is turning left.'}],
        'stop': [{'caption': 'The car slows down at a red light.'}],
        'forward': [{'caption': 'The car is driving forward.'}],
    }
    tokenized = ptbtokenizer.PTBTokenizer().tokenize(captions)
    if set(tokenized) != set(captions) or any(len(v) != 1 or not v[0] for v in tokenized.values()):
        raise RuntimeError('PTB preflight lost captions or returned empty reference tokens.')
    bleu, _ = Bleu(4).compute_score(tokenized, tokenized)
    cider, _ = Cider('corpus').compute_score(tokenized, tokenized)
    rouge, _ = Rouge().compute_score(tokenized, tokenized)
    metrics = dict(Bleu_4=float(bleu[3]), CIDEr=float(cider), ROUGE_L=float(rouge))
    if not all(math.isfinite(x) for x in metrics.values()):
        raise RuntimeError('Caption evaluator preflight returned non-finite scores.')
    if metrics['Bleu_4'] < .99 or metrics['ROUGE_L'] < .99 or metrics['CIDEr'] < 5:
        raise RuntimeError('Identity-caption evaluator preflight returned unexpected scores: %s' % metrics)
    jar = Path(ptbtokenizer.__file__).with_name(ptbtokenizer.STANFORD_CORENLP_3_4_1_JAR)
    return dict(status='passed', scope='real Java PTB and B4/CIDEr/ROUGE identity fixture; not model quality',
                fixture_captions=len(captions), metrics=metrics, jar=str(jar),
                jar_sha256=hashlib.sha256(jar.read_bytes()).hexdigest())
