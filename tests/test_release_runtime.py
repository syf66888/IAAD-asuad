"""Check relocated real checkpoint configurations without importing PyTorch."""
import json
from pathlib import Path
import tempfile
import unittest

import asuad_runtime as runtime


class ReleaseRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.weights = self.folder / 'weights'
        for rel in ('pretrained/yolov5su.pt', 'pretrained/' + runtime.SWIN_NAME,
                    'bddx/model.bin', 'mmau/model.bin'):
            path = self.weights / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(rel.encode())

    def options(self, mode, dataset='bddx', extra=()):
        return runtime.make_parser(mode).parse_args([
            '--dataset', dataset, '--data-root', str(self.folder / 'data'),
            '--flow-cache', str(self.folder / 'lk'), '--weights-root', str(self.weights),
            '--output-dir', str(self.folder / mode / dataset), '--dry-run', *extra])

    def test_bddx_train_preserves_batch_and_caption_slots(self):
        cfg = runtime.prepare(self.options('train'), 'train')['config']
        self.assertEqual((cfg['per_gpu_train_batch_size'], cfg['gradient_accumulation_steps'],
                          cfg['effective_batch_size']), (4, 4, 16))
        self.assertEqual((cfg['max_seq_a_length'], cfg['max_seq_length'], cfg['max_gen_length']), (35, 70, 35))
        self.assertFalse(cfg['pretrained_checkpoint'])

    def test_mmau_train_starts_fresh_without_task_weights(self):
        # Neither dataset should depend on a previously trained task checkpoint.
        for dataset in ('bddx', 'mmau'):
            (self.weights / dataset / 'model.bin').unlink()
        receipt = runtime.prepare(self.options('train', 'mmau'), 'train')
        cfg = receipt['config']
        self.assertFalse(cfg['pretrained_checkpoint'])
        self.assertIsNone(receipt['checkpoint'])
        self.assertEqual((cfg['max_seq_length'], cfg['max_gen_length']), (224, 112))

    def test_explicit_ce_initializer_is_optional_and_respected(self):
        source = self.folder / 'my_ce/model.bin'
        source.parent.mkdir()
        source.write_bytes(b'my CE model')
        cfg = runtime.prepare(self.options('train', 'mmau', ['--checkpoint', str(source)]), 'train')['config']
        self.assertTrue(Path(cfg['pretrained_checkpoint'], 'model.bin').samefile(source))

    def test_best_model_inference_loads_derived_inference_args(self):
        result = runtime.prepare(self.options('inference', 'mmau'), 'inference')
        config = result['config']
        saved = runtime.read_json(Path(config['eval_model_dir']).parent / 'log/args.json')
        self.assertFalse(saved['scst'])
        self.assertFalse(saved['do_train'])
        self.assertEqual(saved['data_dir'], str((self.folder / 'data').resolve()))
        self.assertTrue(Path(saved['resume_checkpoint']).samefile(self.weights / 'mmau/model.bin'))
        self.assertEqual(saved['max_gen_length'], 112)

    def test_different_checkpoint_alias_is_not_overwritten(self):
        source = self.weights / 'bddx/model.bin'
        target = self.folder / 'alias/model.bin'
        runtime.checkpoint_link(source, target)
        with self.assertRaises(FileExistsError):
            runtime.checkpoint_link(self.weights / 'mmau/model.bin', target)

    def test_microbatch_updates_actual_effective_batch(self):
        cfg = runtime.prepare(self.options('train', extra=['--batch-size', '2']), 'train')['config']
        self.assertEqual(cfg['effective_batch_size'], 8)

    def test_cpu_requires_fp32(self):
        with self.assertRaises(ValueError):
            runtime.prepare(self.options('inference', extra=['--device', 'cpu']), 'inference')

    def test_does_not_accept_compact_model_template(self):
        cfg = runtime.read_json(runtime.ROOT / 'configs/bddx_train.json')
        cfg['vidswin_size'] = 'tiny'
        template = self.folder / 'compact.json'
        runtime.write_json(template, cfg)
        with self.assertRaises(ValueError):
            runtime.prepare(self.options('train', extra=['--config', str(template)]), 'train')

    def ce_output(self, dataset):
        folder = self.folder / ('ce_' + dataset)
        model = folder / 'checkpoint-best/model.bin'
        model.parent.mkdir(parents=True)
        model.write_bytes(b'my selected CE checkpoint')
        runtime.write_json(folder / 'best_checkpoint.json', {
            'dataset_name': dataset.upper(), 'validation_yaml': dataset.upper() + '/testing_32frames.yaml',
            'epoch': 1, 'metrics': {'des': {'Bleu_4': .2, 'CIDEr': 1.}, 'exp': {'Bleu_4': .1, 'CIDEr': .8}}})
        return folder, model

    def test_scst_continues_ce_for_both_datasets(self):
        cache = self.folder / 'reward.json.gz'
        cache.write_bytes(b'fixture')
        for dataset, slot in [('bddx', 35), ('mmau', 112)]:
            folder, model = self.ce_output(dataset)
            receipt = runtime.prepare(self.options('scst', dataset, [
                '--ce-output-dir', str(folder), '--reward-cache', str(cache)]), 'scst')
            cfg = receipt['config']
            self.assertEqual(cfg['dataset_name'], dataset.upper())
            self.assertEqual(cfg['max_gen_length'], slot)
            self.assertEqual(cfg['max_seq_length'], 2 * slot)
            self.assertEqual(Path(cfg['scst_checkpoint']), model.resolve())
            self.assertEqual(cfg['scst_base_record']['model_sha256'], runtime.sha256_file(model))
            self.assertEqual(cfg['scst_base_record']['dataset_name'], dataset.upper())
            self.assertTrue(cfg['scst'])

    def test_scst_requires_user_ce_run(self):
        with self.assertRaisesRegex(ValueError, 'completed CE run'):
            runtime.prepare(self.options('scst', 'mmau', ['--reward-cache', 'unused']), 'scst')

    def test_scst_rejects_dataset_mismatch(self):
        folder, model = self.ce_output('bddx')
        with self.assertRaisesRegex(ValueError, 'different dataset'):
            runtime.prepare(self.options('scst', 'mmau', [
                '--ce-output-dir', str(folder), '--reward-cache', 'unused']), 'scst')

    def test_scst_rejects_mismatched_base_record(self):
        cache = self.folder / 'reward.json.gz'
        cache.write_bytes(b'fixture')
        folder, model = self.ce_output('mmau')
        record = folder / 'best_checkpoint.json'
        base = runtime.read_json(record)
        base['model_sha256'] = 'incorrect'
        runtime.write_json(record, base)
        with self.assertRaisesRegex(ValueError, 'does not match'):
            runtime.prepare(self.options('scst', 'mmau', [
                '--checkpoint', str(model), '--base-record', str(record), '--reward-cache', str(cache)]), 'scst')


if __name__ == '__main__':
    unittest.main()
