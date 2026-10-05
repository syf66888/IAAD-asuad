"""Stanford PTB tokenization with explicit runtime and alignment checks."""

import os
import subprocess
import tempfile


STANFORD_CORENLP_3_4_1_JAR = 'stanford-corenlp-3.4.1.jar'
PUNCTUATIONS = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-",
                ".", "?", "!", ",", ":", "-", "--", "...", ";"]


def _tokenize_captions(captions, jar_dir):
    """Return exactly one tokenized line per caption using the original PTB rules.

    Keep the original LF-to-space normalization. Other separators that Stanford
    treats as line breaks must fail the alignment check instead of silently
    shifting captions to another image ID.
    """
    if not captions:
        return []
    jar_path = os.path.join(jar_dir, STANFORD_CORENLP_3_4_1_JAR)
    if not os.path.isfile(jar_path):
        raise FileNotFoundError(
            'Stanford PTBTokenizer JAR is missing: {}. Restore the original '
            'Stanford CoreNLP 3.4.1 asset before caption evaluation.'.format(jar_path))

    sentences = '\n'.join(caption.replace('\n', ' ') for caption in captions)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.txt') as temporary:
            temporary_path = temporary.name
            temporary.write(sentences.encode('utf-8'))
        cmd = ['java', '-cp', jar_path, 'edu.stanford.nlp.process.PTBTokenizer',
               '-preserveLines', '-lowerCase', temporary_path]
        try:
            process = subprocess.run(cmd, cwd=jar_dir, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, check=False)
        except OSError as exc:
            raise RuntimeError('Unable to execute Java PTBTokenizer with JAR {}: {}'.format(
                jar_path, exc)) from exc

        stderr = process.stderr.decode('utf-8', errors='replace')
        if process.returncode != 0:
            raise RuntimeError('Java PTBTokenizer failed (returncode={}, JAR={}): {}'.format(
                process.returncode, jar_path, stderr.strip()))
        try:
            # Stanford preserves input line endings, including a final empty
            # caption; stripping stdout would lose that caption.
            lines = process.stdout.decode('utf-8').split('\n')
        except UnicodeDecodeError as exc:
            raise RuntimeError('Java PTBTokenizer returned invalid UTF-8; stderr: {}'.format(
                stderr.strip())) from exc
        if len(lines) != len(captions):
            raise RuntimeError(
                'Java PTBTokenizer caption alignment failure: {} input captions but {} '
                'output lines. Check embedded CR/Unicode line separators; no captions '
                'were assigned to image IDs. stderr: {}'.format(
                    len(captions), len(lines), stderr.strip()))
        return [' '.join(word for word in line.rstrip().split(' ')
                         if word not in PUNCTUATIONS) for line in lines]
    finally:
        if temporary_path is not None:
            os.remove(temporary_path)


class PTBTokenizer:
    """Python wrapper of Stanford PTBTokenizer."""

    def tokenize(self, captions_for_image):
        image_ids = []
        captions = []
        for image_id, image_captions in captions_for_image.items():
            for caption in image_captions:
                image_ids.append(image_id)
                captions.append(caption['caption'])
        lines = _tokenize_captions(captions, os.path.dirname(os.path.abspath(__file__)))
        result = {}
        for image_id, line in zip(image_ids, lines):
            result.setdefault(image_id, []).append(line)
        return result
