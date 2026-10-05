"""Training-only, fixed-IDF rewards for audited dual-caption SCST.

The repository's evaluator calls its metric ``CIDEr``, but implements the
CIDEr-D clipping and Gaussian length penalty too. We reuse that mathematics;
only IDF differs: rewards always use the full *training* references, whereas
reported testing metrics retain their existing testing-corpus IDF. No EOS text
is appended, matching the evaluator's PTB tokenization.
"""

import argparse
from collections import OrderedDict, defaultdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np

from src.evalcap.cider.pyciderevalcap.ciderD.ciderD_scorer import CiderScorer, precook
from src.evalcap.coco_caption.pycocoevalcap.bleu.bleu_scorer import BleuScorer
from src.evalcap.coco_caption.pycocoevalcap.tokenizer import ptbtokenizer


TASK_FIELDS = {"des": "action", "exp": "justification"}
CACHE_VERSION = 1
TOKENIZATION = "stanford-ptb-3.4.1-lowercase-coco-punctuation-no-eos"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class JavaPTBTokenizer:
    """Use the evaluation tokenizer's actual Java model, with checked errors.

    References are tokenized once when building the cache. Repeated generated
    captions use a bounded in-process LRU; all misses share one Java invocation.
    Temporary files live in TMPDIR, never in the potentially quota-limited repo.
    """

    def __init__(self, jar_path=None, java="java", max_cache=20000):
        self.jar_path = Path(jar_path) if jar_path else Path(ptbtokenizer.__file__).with_name(
            ptbtokenizer.STANFORD_CORENLP_3_4_1_JAR)
        if not self.jar_path.is_file():
            raise FileNotFoundError("SCST requires the evaluator's PTB jar: %s" % self.jar_path)
        self.java = java
        self.max_cache = max_cache
        self.cache = OrderedDict()
        self.fingerprint = {"method": TOKENIZATION, "jar_sha256": _sha256(self.jar_path)}

    def __call__(self, captions):
        texts = [str(c).replace("\r", " ").replace("\n", " ") for c in captions]
        missing = list(dict.fromkeys(s for s in texts if s not in self.cache))
        computed = {}
        if missing:
            name = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as f:
                    name = f.name
                    f.write("\n".join(missing) + "\n")
                command = [self.java, "-cp", str(self.jar_path),
                           "edu.stanford.nlp.process.PTBTokenizer", "-preserveLines", "-lowerCase", name]
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        check=True, timeout=120)
                lines = result.stdout.decode("utf-8").split("\n")
                if lines and lines[-1] == "":
                    lines.pop()
                if len(lines) != len(missing):
                    raise RuntimeError("PTB returned %d lines for %d captions" % (len(lines), len(missing)))
                computed = {s: " ".join(w for w in line.strip().split(" ")
                                        if w not in ptbtokenizer.PUNCTUATIONS)
                            for s, line in zip(missing, lines)}
            finally:
                if name is not None:
                    os.unlink(name)
        output = []
        for s in texts:
            tokenized = computed[s] if s in computed else self.cache[s]
            output.append(tokenized)
            self.cache[s] = tokenized
            self.cache.move_to_end(s)
        while len(self.cache) > self.max_cache:
            self.cache.popitem(last=False)
        return output


def _training_source(path):
    path = Path(path).resolve()
    if not re.match(r"^training(?:[._]|$)", path.name):
        raise ValueError("SCST cache accepts only a training source: %s" % path)
    return path


def read_training_references(train_yaml, dataset_name='BDDX'):
    """Read exactly the training YAML's selected TSV caption rows/indices.

    Fail closed on other splits, composite datasets, mislabeled testing keys,
    missing captions and duplicated keys. Do not discover/open testing files.
    """
    import yaml
    from src.tasks.dataset_protocol import dataset_name as validate_dataset
    dataset_name = validate_dataset(dataset_name)

    train_yaml = _training_source(train_yaml)
    if train_yaml.parent.name != dataset_name:
        raise ValueError("Expected training YAML inside the explicitly configured dataset directory")
    cfg = yaml.safe_load(train_yaml.read_text(encoding="utf-8"))
    if cfg.get("composite"):
        raise ValueError("SCST reference builder supports a single caption TSV only")
    caption_path = _training_source(train_yaml.parent / cfg["caption"])
    sources = [train_yaml, caption_path]
    selected = None
    if cfg.get("caption_linelist"):
        linelist = _training_source(train_yaml.parent / cfg["caption_linelist"])
        sources.append(linelist)
        selected = defaultdict(set)
        for line in linelist.read_text(encoding="utf-8").splitlines():
            row, cap = map(int, line.split("\t"))
            if min(row, cap) < 0:
                raise ValueError("Negative training caption index")
            selected[row].add(cap)

    references = {task: {} for task in TASK_FIELDS}
    seen_rows = set()
    with caption_path.open(encoding="utf-8") as stream:
        for row_index, line in enumerate(stream):
            if selected is not None and row_index not in selected:
                continue
            seen_rows.add(row_index)
            key, payload = line.rstrip("\n").split("\t", 1)
            if not key.startswith("training_"):
                raise ValueError("Non-training key in reward source: %s" % key)
            if key in references["des"]:
                raise ValueError("Duplicate training key: %s" % key)
            rows = json.loads(payload)
            indices = sorted(selected[row_index]) if selected is not None else range(len(rows))
            if not rows or any(i >= len(rows) for i in indices):
                raise ValueError("Invalid training caption index for %s" % key)
            for task, field in TASK_FIELDS.items():
                values = [rows[i][field] for i in indices]
                if not values or any(not isinstance(v, str) or not v.strip() for v in values):
                    raise ValueError("Missing %s training caption for %s" % (task, key))
                references[task][key] = values
    if selected is not None and set(selected) != seen_rows:
        raise ValueError("Training linelist references nonexistent caption rows")
    if len(references["des"]) < 2:
        raise ValueError("CIDEr IDF requires at least two training clips")
    return references, [{"path": str(p), "sha256": _sha256(p)} for p in sources]


def build_training_cache(train_yaml, cache_path, tokenizer=None, dataset_name='BDDX'):
    """Build/reuse an atomic, auditable JSON gzip cache; never a COCO pickle."""
    raw_references, sources = read_training_references(train_yaml, dataset_name)
    tokenizer = tokenizer or JavaPTBTokenizer()
    cache_path = Path(cache_path)
    if cache_path.is_file():
        cache = load_training_cache(cache_path, dataset_name)
        if cache["sources"] != sources or cache["tokenizer"] != tokenizer.fingerprint:
            raise ValueError("Stale SCST reward cache: training sources or PTB tokenizer changed")
        return cache
    cache = {"version": CACHE_VERSION, "dataset": dataset_name, "split": "training",
             "sources": sources, "tokenizer": tokenizer.fingerprint, "tasks": {}}
    for task, refs in raw_references.items():
        keys = list(refs)
        flat = tokenizer([caption for key in keys for caption in refs[key]])
        tokenized, offset = {}, 0
        for key in keys:
            tokenized[key] = flat[offset:offset + len(refs[key])]
            offset += len(refs[key])
        df = defaultdict(float)
        for captions in tokenized.values():
            for ngram in {ngram for caption in captions for ngram in precook(caption)}:
                df[ngram] += 1.0
        cache["tasks"][task] = {"references": tokenized, "ref_len": len(tokenized),
                                "document_frequency": [[list(ngram), count] for ngram, count in sorted(df.items())]}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=str(cache_path.parent), delete=False) as temporary:
            name = temporary.name
        with gzip.open(name, "wt", encoding="utf-8") as stream:
            json.dump(cache, stream, ensure_ascii=False, sort_keys=True)
        os.replace(name, str(cache_path))
        name = None
    finally:
        if name is not None:
            os.unlink(name)
    return cache


def load_training_cache(cache_path, dataset_name='BDDX'):
    from src.tasks.dataset_protocol import dataset_name as validate_dataset
    dataset_name = validate_dataset(dataset_name)
    with gzip.open(cache_path, "rt", encoding="utf-8") as stream:
        cache = json.load(stream)
    if (cache.get("version") != CACHE_VERSION or cache.get("dataset") != dataset_name
            or cache.get("split") != "training" or not cache.get("sources")
            or cache.get("tokenizer", {}).get("method") != TOKENIZATION):
        raise ValueError("Expected a versioned dataset-matched training-only PTB reward cache")
    for task in TASK_FIELDS:
        entry = cache["tasks"][task]
        refs, count = entry["references"], entry["ref_len"]
        if count < 2 or count != len(refs) or any(not key.startswith("training_") for key in refs):
            raise ValueError("Invalid training references in SCST cache")
        if any(not values or not all(isinstance(c, str) for c in values) for values in refs.values()):
            raise ValueError("Invalid tokenized reference captions")
        if any(not 1 <= len(ngram) <= 4 or not 0 < df <= count
               for ngram, df in entry["document_frequency"]):
            raise ValueError("Invalid training document frequencies")
    if cache["tasks"]["des"]["references"].keys() != cache["tasks"]["exp"]["references"].keys():
        raise ValueError("Description and explanation training keys do not match")
    return cache


class BDDXScstReward:
    """Fixed-training-IDF per-caption rewards, not corpus BLEU optimization.

    ``score`` takes B keys and B greedy captions for each task. Sampled captions
    have length B * samples_per_clip, in repeat_interleave/batch-major order.
    It also accepts already expanded keys/greedies of that same length.

    Each task reward is ``cider_weight * CIDEr/10 + bleu4_weight * sentence_B4``.
    Equal task weights keep description and explanation equally represented.
    The default CIDEr/10 puts both metrics in comparable theoretical [0, 1]
    ranges without data-dependent normalization or statistics from testing.
    Corpus testing B4 is not the average of sentence B4 used as an RL reward.
    """

    def __init__(self, cache_path, cider_weight=1.0, bleu4_weight=0.0,
                 cider_scale=10.0, des_weight=0.5, exp_weight=0.5, tokenizer=None,
                 dataset_name='BDDX'):
        values = [cider_weight, bleu4_weight, cider_scale, des_weight, exp_weight]
        if not all(math.isfinite(v) for v in values) or min(values) < 0 or cider_scale == 0:
            raise ValueError("Reward weights must be finite, nonnegative, with positive CIDEr scale")
        if cider_weight + bleu4_weight == 0 or des_weight + exp_weight == 0:
            raise ValueError("At least one metric and one task must have positive weight")
        self.cache = load_training_cache(cache_path, dataset_name)
        self.tokenizer = tokenizer or JavaPTBTokenizer()
        if self.tokenizer.fingerprint != self.cache["tokenizer"]:
            raise ValueError("Reward tokenizer must match training cache tokenizer")
        self.cider_weight, self.bleu4_weight, self.cider_scale = cider_weight, bleu4_weight, cider_scale
        total_task_weight = des_weight + exp_weight
        self.task_weights = {"des": des_weight / total_task_weight, "exp": exp_weight / total_task_weight}
        self.df = {task: defaultdict(float, {tuple(n): df for n, df in entry["document_frequency"]})
                   for task, entry in self.cache["tasks"].items()}

    def _task_scores(self, task, keys, hypotheses):
        references = self.cache["tasks"][task]["references"]
        cider = CiderScorer(df_mode="corpus")
        cider.df_mode = "fixed_{}_training".format(self.cache['dataset'].lower())
        # Copy prevents unknown generated ngrams from growing the shared DF cache.
        cider.document_frequency = defaultdict(float, self.df[task])
        cider.ref_len = math.log(self.cache["tasks"][task]["ref_len"])
        bleu = BleuScorer(n=4) if self.bleu4_weight else None
        for key, caption in zip(keys, hypotheses):
            if key not in references:
                raise KeyError("Reward requested for a key outside configured dataset training: %s" % key)
            cider += (caption, references[key])
            if bleu is not None:
                bleu += (caption, references[key])
        _, cider_scores = cider.compute_score()
        b4 = np.asarray(bleu.compute_score(option="closest", verbose=0)[1][3]) if bleu else np.zeros(len(keys))
        reward = self.cider_weight * cider_scores / self.cider_scale + self.bleu4_weight * b4
        if not np.isfinite(reward).all():
            raise FloatingPointError("SCST produced non-finite rewards")
        return {"reward": reward.astype(np.float32), "cider": cider_scores.astype(np.float32),
                "bleu4": b4.astype(np.float32)}

    def score(self, keys, sampled_des, sampled_exp, greedy_des=None, greedy_exp=None,
              baseline_type='greedy'):
        keys = list(keys)
        batch, sample_count = len(keys), len(sampled_des)
        if baseline_type not in ('greedy', 'leave_one_out'):
            raise ValueError('Unknown SCST baseline type')
        if (not batch or not sample_count or sample_count % batch or len(sampled_exp) != sample_count):
            raise ValueError("Expected B keys and B*N batch-major samples per task")
        samples_per_clip = sample_count // batch
        if baseline_type == 'greedy':
            if greedy_des is None or greedy_exp is None or len(greedy_des) != batch or len(greedy_exp) != batch:
                raise ValueError('Greedy baseline requires one des/exp pair per clip')
            greedy_des, greedy_exp = list(greedy_des), list(greedy_exp)
        else:
            if samples_per_clip < 2:
                raise ValueError('leave_one_out requires at least two samples per clip')
            if greedy_des is not None or greedy_exp is not None:
                raise ValueError('leave_one_out must not consume a greedy baseline')
            greedy_des, greedy_exp = [], []
        expanded_keys = [key for key in keys for _ in range(samples_per_clip)]
        generated = list(sampled_des) + list(sampled_exp) + list(greedy_des) + list(greedy_exp)
        tokenized = self.tokenizer(generated)
        greedy_count = len(greedy_des)
        hypotheses = {"des": tokenized[:sample_count] + tokenized[2 * sample_count:2 * sample_count + greedy_count],
                      "exp": tokenized[sample_count:2 * sample_count] + tokenized[2 * sample_count + greedy_count:]}
        output = {"samples_per_clip": samples_per_clip, "task_weights": dict(self.task_weights),
                  "baseline_type": baseline_type}
        for task in TASK_FIELDS:
            scored = self._task_scores(task, expanded_keys + (keys if greedy_count else []), hypotheses[task])
            sample = scored["reward"][:sample_count]
            if baseline_type == 'greedy':
                baseline = np.repeat(scored["reward"][sample_count:], samples_per_clip)
            else:
                # Exclude this sample and stay within its clip; never normalize
                # by group std or include the sample itself in its baseline.
                grouped = sample.astype(np.float64).reshape(batch, samples_per_clip)
                baseline = ((grouped.sum(-1, keepdims=True) - grouped) /
                            (samples_per_clip - 1)).reshape(-1).astype(np.float32)
            output[task] = {"sample": sample, "baseline": baseline, "advantage": sample - baseline,
                            "sample_cider": scored["cider"][:sample_count],
                            "greedy_cider": scored["cider"][sample_count:],
                            "sample_bleu4": scored["bleu4"][:sample_count],
                            "greedy_bleu4": scored["bleu4"][sample_count:]}
        for name in ("sample", "baseline", "advantage"):
            output[name] = sum(self.task_weights[task] * output[task][name] for task in TASK_FIELDS)
        return output


def eos_inclusive_mask(token_ids, eos_token_id, pad_token_id, forced_eos=None):
    """Boolean action mask: include natural EOS/PAD, exclude forced/after-EOS.

    Inputs are generated actions only (no initial BOS). Sequence policy loss is
    ``-advantage.detach() * sum(mask * sampled_token_logprob)``. A positive
    advantage must increase those log probabilities; zero advantage gives zero
    policy gradient. For sequential des->exp generation, use the joint reward
    advantage for both segments, since description also affects explanation.
    An internally sampled PAD is a policy action, even though text decoding
    removes it. Only padding after EOS is excluded; forced length-limit EOS
    must be identified by the optional boolean ``forced_eos`` provenance mask.
    """
    ids = np.asarray(token_ids)
    if ids.ndim != 2 or eos_token_id == pad_token_id:
        raise ValueError("Expected [batch, length] actions and distinct EOS/PAD")
    stopped = ids == eos_token_id
    previous_stops = np.cumsum(stopped, axis=1) - stopped.astype(np.int64)
    valid = previous_stops == 0
    if forced_eos is not None:
        forced = np.asarray(forced_eos, dtype=bool)
        if forced.shape != ids.shape:
            raise ValueError('Forced EOS mask must match the generated token shape')
        valid &= ~forced
    return valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_yaml", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset_name", choices=['BDDX', 'MMAU'], default='BDDX')
    args = parser.parse_args()
    cache = build_training_cache(args.train_yaml, args.cache, dataset_name=args.dataset_name)
    print(json.dumps({"cache": args.cache, "split": cache["split"],
                      "training_clips": cache["tasks"]["des"]["ref_len"],
                      "sources": cache["sources"]}, indent=2))


if __name__ == "__main__":
    main()
