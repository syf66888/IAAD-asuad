"""Check turn semantics, complete-sample scoring, and evaluation integration."""
import ast
import json
import os.path as op
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from src.evalcap.bddx_turn_accuracy import evaluate, evaluate_files, read_predictions, read_references
from src.evalcap.turn_semantics import classify

ROOT = Path(__file__).resolve().parents[1]


class _TurnFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)

    def write_json(self, name, value):
        path = self.folder / name
        path.write_text(json.dumps(value), encoding='utf-8')
        return path


class TurnAccuracyTests(_TurnFiles):
    def test_observed_action_boundaries(self):
        cases = [
            ('The car turns right', 'turn', 'right', True),
            ('The car starts to turn left', 'turn', 'left', True),
            ('The car steers to the left', 'turn', 'left', False),
            ('The car steers to the left and drives in the middle lane', 'turn', 'left', False),
            ('The car is steering into the right lane', 'lane', 'right', False),
            ('The car is merging into the right lane', 'lane', 'right', False),
            ('The car is moving to the right lane', 'lane', 'right', False),
            ('The car is turning to the right lane', 'lane', 'right', False),
            ('The car turns left and parks on the right side of the road', 'turn', 'left', True),
            ('The car is driving on the left turn lane', 'position', 'left', False),
            ('The car steers left and back', 'correction', 'left', False),
            ('The car is steering left and right', 'ambiguous', None, False),
            ('The car is making a U-turn', 'uturn', None, False),
            ('The car is waiting to turn right', 'planned', 'right', False),
            ('The car slows nearly to a stop before turning right', 'planned', 'right', False),
            ('The car stops because another car is turning left', 'none', None, False),
            ('The car is driving forward since it has the right of way', 'none', None, False),
        ]
        for text, kind, direction, strict in cases:
            with self.subTest(text=text):
                result = classify(text)
                self.assertEqual((result['kind'], result['direction'], result['strict_turn']),
                                 (kind, direction, strict))

    def test_lane_substitution_wrong_direction_and_omission_are_errors(self):
        references = {'correct': 'turn left', 'opposite': 'turn right',
                      'lane': 'turn left', 'omitted': 'turn right',
                      'steering': 'steer right', 'planned': 'waiting to turn left'}
        predictions = {'correct': 'steer left', 'opposite': 'turn left',
                       'lane': 'changing to the left lane', 'omitted': '',
                       'steering': 'turn right', 'planned': 'turn left'}
        result, rows = evaluate(predictions, references)
        explicit = result['explicit_turns']
        self.assertEqual((explicit['correct'], explicit['reference']), (1, 4))
        self.assertEqual(result['turning_accuracy'], 0.25)
        self.assertEqual(explicit['outcomes'], {'correct': 1, 'opposite_direction': 1,
                                              'lane_substitution_same_direction': 1, 'omitted_turn': 1})
        self.assertEqual((result['broad_steering']['correct'], result['broad_steering']['reference']), (2, 5))
        self.assertEqual(rows[-1]['outcome'], 'turn_not_supported_by_reference')

    def test_non_turn_samples_do_not_inflate_the_accuracy(self):
        references = {'left': 'turn left', 'right': 'turn right'}
        predictions = {'left': 'turn left', 'right': 'go forward'}
        for index in range(100):
            references[str(index)] = 'go forward'
            predictions[str(index)] = 'go forward'
        result, _ = evaluate(predictions, references)
        self.assertEqual(result['turning_accuracy'], 0.5)
        self.assertEqual(result['explicit_turns']['reference'], 2)

    def test_missing_and_extra_predictions_are_rejected(self):
        for predictions in ({'a': 'turn left'}, {'a': 'turn left', 'b': '', 'c': ''}):
            with self.subTest(predictions=predictions):
                with self.assertRaisesRegex(ValueError, 'cover every reference'):
                    evaluate(predictions, {'a': 'turn left', 'b': 'turn right'})

    def test_empty_turn_subset_is_reported_as_undefined(self):
        result, _ = evaluate({'a': 'go forward'}, {'a': 'go forward'})
        self.assertIsNone(result['turning_accuracy'])
        self.assertIsNone(result['explicit_turns']['accuracy_percent'])
        self.assertIsNone(result['broad_steering']['precision'])

    def test_reference_uses_action_and_prediction_uses_description(self):
        references = self.write_json('references.json', {'annotations': [
            {'image_id': 1, 'action': 'turn left', 'caption': 'another car turns right'}],
            'images': [{'id': 1}]})
        predictions = self.write_json('predictions.json', [
            {'image_id': '1', 'description': 'turn left', 'caption': 'turn right',
             'explanation': 'because another car turns right'}])
        result = evaluate_files(predictions, references)
        self.assertEqual(result['turning_accuracy'], 1.0)

    def test_prediction_formats_produce_the_same_decisions(self):
        merged = self.write_json('merged.json', [{'image_id': 'a', 'description': 'turn left'}])
        coco = self.write_json('coco.json', [{'image_id': 'a', 'caption': 'turn left'}])
        tsv = self.folder / 'predictions.tsv'
        tsv.write_text('a\t' + json.dumps([{'caption': 'turn left'}]) + '\t' +
                       json.dumps([{'caption': 'turn right'}]) + '\n', encoding='utf-8-sig')
        for path in (merged, coco, tsv):
            self.assertEqual(read_predictions(path), {'a': 'turn left'})

    def test_duplicate_samples_and_multiple_references_are_rejected(self):
        duplicate = [{'image_id': 'a', 'caption': 'turn left'}] * 2
        predictions = self.write_json('duplicate_predictions.json', duplicate)
        references = self.write_json('duplicate_references.json', {'annotations': duplicate})
        with self.assertRaisesRegex(ValueError, 'unique'):
            read_predictions(predictions)
        with self.assertRaisesRegex(ValueError, 'unique'):
            read_references(references)
        tsv = self.folder / 'references.tsv'
        tsv.write_text('a\t' + json.dumps([{'action': 'turn left'}, {'action': 'turn right'}]) + '\n')
        with self.assertRaisesRegex(ValueError, 'one action reference'):
            read_references(tsv)

    def test_action_tsv_and_description_coco_reference_are_supported(self):
        tsv = self.folder / 'references.tsv'
        tsv.write_text('a\t' + json.dumps([{'action': 'turn left', 'justification': 'turn right'}]) + '\n')
        coco = self.write_json('coco_references.json', {'annotations': [
            {'image_id': 'a', 'caption': 'turn left'}], 'images': [{'id': 'a'}]})
        self.assertEqual(read_references(tsv), read_references(coco))

    def test_reference_images_must_cover_the_annotations(self):
        path = self.write_json('missing_image.json', {'annotations': [
            {'image_id': 'a', 'action': 'turn left'}], 'images': [{'id': 'b'}]})
        with self.assertRaisesRegex(ValueError, 'match exactly'):
            read_references(path)

    def test_best_model_matches_the_recorded_turn_counts(self):
        result = evaluate_files(ROOT / 'results/bddx/testing_predictions.json',
                                ROOT / 'datasets/BDDX_des/testing_32frames_caption_coco_format.json',
                                self.folder / 'turn_accuracy.json', self.folder / 'turn_predictions.json')
        explicit = result['explicit_turns']
        self.assertEqual((result['total_samples'], explicit['reference'], explicit['correct']), (2859, 178, 154))
        self.assertEqual((explicit['per_direction']['left']['correct'],
                          explicit['per_direction']['right']['correct']), (70, 84))
        self.assertAlmostEqual(result['turning_accuracy'], 154 / 178)
        self.assertEqual(len(json.loads((self.folder / 'turn_predictions.json').read_text())), 2859)


class TrainingEvaluationIntegrationTests(_TurnFiles):
    # Execute the actual evaluation function without importing the GPU model.
    # Caption metrics and model generation are mocked; turn evaluation is real.
    def run_evaluation(self, dataset='BDDX', main_process=True):
        reference = self.write_json('annotations.json', {'annotations': [
            {'image_id': 'a', 'action': 'turn left'}, {'image_id': 'b', 'action': 'turn right'}],
            'images': [{'id': 'a'}, {'id': 'b'}]})
        prediction = self.folder / ('pred.' + dataset + '.testing.tsv')
        prediction.write_text(''.join(key + '\t' + json.dumps([{'caption': text}]) + '\t' +
                              json.dumps([{'caption': 'because of traffic'}]) + '\n'
                              for key, text in [('a', 'turn left'), ('b', 'go forward')]))
        source = ast.parse((ROOT / 'src/tasks/train.py').read_text(encoding='utf-8'))
        function = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == 'evaluate')
        scope = dict(get_predict_file=Mock(return_value=str(prediction)), test=Mock(),
                     get_world_size=Mock(return_value=1), is_main_process=Mock(return_value=main_process),
                     get_evaluate_file=lambda path: op.splitext(path)[0] + '.eval.json',
                     json=json, op=op, logger=SimpleNamespace(info=Mock()),
                     two_cap_evaluate_on_coco_caption=Mock(return_value=[{'Bleu_4': 0.3}, {'Bleu_4': 0.1}]),
                     evaluate_on_coco_caption=Mock(), evaluate_bddx_turns=Mock(wraps=evaluate_files))
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'train_evaluation', 'exec'), scope)
        data = SimpleNamespace(yaml_file=dataset + '/testing_32frames.yaml',
                               get_caption_file_in_coco_format=lambda: str(reference))
        value = scope['evaluate'](SimpleNamespace(use_sep_cap=True, caption_metrics='basic'),
                                  SimpleNamespace(dataset=data), None, None, str(self.folder))
        return prediction, scope, value

    def test_bddx_evaluation_automatically_writes_turn_results(self):
        prediction, scope, result = self.run_evaluation()
        scope['evaluate_bddx_turns'].assert_called_once()
        self.assertEqual(result, str(prediction.with_suffix('.eval.json')))
        report = json.loads(prediction.with_suffix('.turn_accuracy.json').read_text())
        self.assertEqual(report['turning_accuracy'], 0.5)
        self.assertEqual(len(json.loads(prediction.with_suffix('.turn_predictions.json').read_text())), 2)

    def test_mmau_evaluation_does_not_run_bddx_turn_scoring(self):
        prediction, scope, _ = self.run_evaluation('MMAU')
        scope['evaluate_bddx_turns'].assert_not_called()
        self.assertFalse(prediction.with_suffix('.turn_accuracy.json').exists())

    def test_non_main_process_does_not_write_turn_results(self):
        prediction, scope, _ = self.run_evaluation(main_process=False)
        scope['evaluate_bddx_turns'].assert_not_called()
        self.assertFalse(prediction.with_suffix('.turn_accuracy.json').exists())


if __name__ == '__main__':
    unittest.main()
