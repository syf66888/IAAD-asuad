"""Guard failure propagation and caption/ID alignment in the metric runtime."""

import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from src.evalcap.coco_caption.pycocoevalcap.tokenizer import ptbtokenizer as coco
from src.evalcap.cider.pyciderevalcap.tokenizer import ptbtokenizer as cider


class PTBTokenizerRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.jar_dir = self.directory.name
        self.jar = os.path.join(self.jar_dir, coco.STANFORD_CORENLP_3_4_1_JAR)
        with open(self.jar, 'wb') as stream:
            stream.write(b'test runtime placeholder')

    def run_mock(self, captions, stdout, returncode=0, stderr=b''):
        paths = []

        def invoke(cmd, **kwargs):
            paths.append(cmd[-1])
            self.assertTrue(os.path.isfile(cmd[-1]))
            self.assertEqual(cmd[2], self.jar)
            self.assertEqual(kwargs['cwd'], self.jar_dir)
            with open(cmd[-1], 'rb') as stream:
                self.assertEqual(stream.read(), '\n'.join(c.replace('\n', ' ') for c in captions).encode())
            return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)

        try:
            with mock.patch.object(coco.subprocess, 'run', side_effect=invoke):
                return coco._tokenize_captions(captions, self.jar_dir)
        finally:
            self.assertTrue(paths)
            for path in paths:
                self.assertFalse(os.path.exists(path), 'temporary input leaked')

    def test_missing_jar_fails_before_launch(self):
        os.remove(self.jar)
        with mock.patch.object(coco.subprocess, 'run') as execute:
            with self.assertRaisesRegex(FileNotFoundError, 'Stanford.*missing'):
                coco._tokenize_captions(['a caption'], self.jar_dir)
            execute.assert_not_called()

    def test_java_error_retains_returncode_and_stderr_and_cleans_input(self):
        with self.assertRaisesRegex(RuntimeError, 'returncode=1.*ClassNotFoundException'):
            self.run_mock(['a caption'], b'', 1, b'ClassNotFoundException PTBTokenizer')

    def test_missing_java_cleans_input_and_reports_cause(self):
        paths = []

        def fail(cmd, **kwargs):
            paths.append(cmd[-1])
            raise FileNotFoundError('java executable not installed')

        with mock.patch.object(coco.subprocess, 'run', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'java executable not installed'):
                coco._tokenize_captions(['caption'], self.jar_dir)
        self.assertEqual(len(paths), 1)
        self.assertFalse(os.path.exists(paths[0]))

    def test_empty_inputs_do_not_require_java_or_jar(self):
        with mock.patch.object(coco.subprocess, 'run') as execute:
            self.assertEqual(coco._tokenize_captions([], '/missing'), [])
            self.assertEqual(coco.PTBTokenizer().tokenize({}), {})
            self.assertEqual(cider.PTBTokenizer().tokenize({}), {})
            self.assertEqual(cider.PTBTokenizer('res').tokenize([]), [])
            execute.assert_not_called()

    def test_missing_or_extra_lines_fail_instead_of_silently_truncating(self):
        for output in (b'one', b'one\ntwo\nthree'):
            with self.subTest(output=output):
                with self.assertRaisesRegex(RuntimeError, 'alignment failure'):
                    self.run_mock(['one', 'two'], output)

    def test_final_empty_caption_is_preserved(self):
        self.assertEqual(self.run_mock(['hello', ''], b'hello\n'), ['hello', ''])
        self.assertEqual(self.run_mock([''], b''), [''])

    def test_lf_normalization_and_punctuation_rules_unchanged(self):
        self.assertEqual(self.run_mock(['He says "Stop!"\nThen turns.'],
                                      b"he says `` stop ! '' then turns ."),
                         ['he says stop then turns'])

    def test_invalid_utf8_is_explicit_and_cleans_input(self):
        with self.assertRaisesRegex(RuntimeError, 'invalid UTF-8'):
            self.run_mock(['caption'], b'\xff')

    def test_multi_reference_and_result_ids_remain_in_input_order(self):
        captions = {17: [{'caption': 'a'}, {'caption': 'b'}],
                    'other': [{'caption': 'c'}]}
        for module in (coco, cider):
            with mock.patch.object(module, '_tokenize_captions', return_value=['aa', 'bb', 'cc']) as call:
                self.assertEqual(module.PTBTokenizer().tokenize(captions),
                                 {17: ['aa', 'bb'], 'other': ['cc']})
                self.assertEqual(call.call_args[0][0], ['a', 'b', 'c'])
                self.assertEqual(call.call_args[0][1], os.path.dirname(os.path.abspath(module.__file__)))
        with mock.patch.object(cider, '_tokenize_captions', return_value=['aa', 'bb']):
            self.assertEqual(cider.PTBTokenizer('res').tokenize(
                [{'image_id': 3, 'caption': 'a'}, {'image_id': 3, 'caption': 'b'}]),
                [{'image_id': 3, 'caption': ['aa']}, {'image_id': 3, 'caption': ['bb']}])

    def test_invalid_cider_source_rejected(self):
        with self.assertRaisesRegex(ValueError, 'source'):
            cider.PTBTokenizer('typo')


class RealStanfordRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jar_dir = os.path.dirname(os.path.abspath(coco.__file__))
        if not shutil.which('java') or not os.path.isfile(os.path.join(
                cls.jar_dir, coco.STANFORD_CORENLP_3_4_1_JAR)):
            raise unittest.SkipTest('requires Java and the original Stanford 3.4.1 JAR')

    def test_known_ptb_output_and_empty_caption_preserved(self):
        self.assertEqual(coco._tokenize_captions(
            ['He says "Stop!"\nThen turns.', 'The car turns left.', ''], self.jar_dir),
            ['he says stop then turns', 'the car turns left', ''])

    def test_unicode_line_separator_cannot_shift_image_ids(self):
        for separator in ('\r', '\u2028', '\u2029'):
            with self.subTest(separator=repr(separator)):
                with self.assertRaisesRegex(RuntimeError, 'alignment failure'):
                    coco._tokenize_captions(['left' + separator + 'right'], self.jar_dir)


if __name__ == '__main__':
    unittest.main()
