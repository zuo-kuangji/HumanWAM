"""CPU-only whole-body adapter tests using synthetic data, never robot datasets."""
from collections import OrderedDict
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / 'src/imagewam/datasets/wholebody.py'
SPEC = importlib.util.spec_from_file_location('wholebody_dataset_under_test', MODULE_PATH)
md = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(md)


def raw_columns(n=100, bias=0.0):
    time = np.linspace(0, 1, n, dtype=np.float32)[:, None]
    def field(dim):
        return time + np.arange(dim, dtype=np.float32)[None] * 0.01 + bias
    left = time * np.linspace(0.2, 0.7, 6, dtype=np.float32)[None] + 0.1
    right = 1.0 - left * 0.8
    return {"observation.state": field(43), "action.motion_token": field(64),
            "observation.inspire_left_hand_state": left,
            "observation.inspire_right_hand_state": right,
            "observation.projected_gravity": field(3),
            "teleop.left_hand_joints": field(7) + 10000,
            "teleop.right_hand_joints": field(7) - 10000,
            "teleop.inspire_left_hand_cmd": np.full((n, 6), 10000),
            "teleop.inspire_right_hand_cmd": np.full((n, 6), -10000)}


def mapped(n=100, bias=0.0):
    return md.map_numeric(raw_columns(n, bias))


class WholeBodyNumericTests(unittest.TestCase):
    def test_exact_mapping_and_no_command_targets(self):
        raw = raw_columns()
        state, action = md.map_numeric(raw)
        self.assertEqual(state.shape, (100, 44))
        self.assertEqual(action.shape, (100, 76))
        np.testing.assert_array_equal(state[:, :22], raw['observation.state'][:, :22])
        np.testing.assert_array_equal(state[:, 22:29], raw['observation.state'][:, 29:36])
        np.testing.assert_array_equal(state[:, 29:35], raw['observation.inspire_left_hand_state'])
        np.testing.assert_array_equal(state[:, 35:41], raw['observation.inspire_right_hand_state'])
        np.testing.assert_array_equal(state[:, 41:44], raw['observation.projected_gravity'])
        np.testing.assert_array_equal(action[:, :64], raw['action.motion_token'])
        np.testing.assert_array_equal(action[:, 64:76], state[:, 29:41])
        self.assertFalse(any(k.startswith('teleop.') for k in md.NUMERIC_COLUMNS))

    def test_invalid_numeric_inputs_fail(self):
        for feature, value in [('action.motion_token', np.nan),
                               ('observation.inspire_left_hand_state', -0.001),
                               ('observation.inspire_right_hand_state', 1.001)]:
            raw = raw_columns()
            raw[feature][0, 0] = value
            with self.assertRaises(ValueError): md.map_numeric(raw)
        raw = raw_columns(); raw['observation.state'] = raw['observation.state'][:, :42]
        with self.assertRaises(ValueError): md.map_numeric(raw)

    def test_fresh_statistics_include_every_episode_without_parent(self):
        arrays = {0: mapped(), 1: mapped(bias=100)}
        stats = md.fit_statistics(arrays, [0, 1])
        values = np.concatenate([arrays[0][1], arrays[1][1]])
        np.testing.assert_array_equal(stats['action'][md.ACTION_KEY]['global_q01'],
                                      np.quantile(values.astype(np.float64), 0.01, axis=0))
        self.assertGreater(stats['action'][md.ACTION_KEY]['global_max'][0], 100)
        self.assertEqual(stats['normalization_reference_episode_ids'], [0, 1])
        self.assertEqual(stats['train_frames'], 200)
        self.assertEqual(stats['heldout_episode_ids'], [])
        self.assertEqual(stats['hand_target_time_offset'], 0)
        self.assertNotIn('parent_statistics_path', stats)
        for ids in [[0], [1, 0], [0, 0, 1]]:
            with self.assertRaises(ValueError): md.fit_statistics(arrays, ids)

    def test_fixed_hand_bounds_and_constant_dimensions(self):
        state, action = mapped()
        action[:, 0] = 0.25
        action[:, 64:76] = state[:, 29:41] = 0.25
        normalizer = md.WholeBodyNormalizer(md.fit_statistics({0: (state, action)}, [0]))
        normalized = normalizer.forward(action, 'action')
        self.assertTrue(np.isfinite(normalized).all())
        np.testing.assert_array_equal(normalized[:, 0], 0)
        np.testing.assert_array_equal(normalized[:, 64:76], -0.5)
        np.testing.assert_array_equal(normalizer.forward(state, 'state')[:, 29:41], -0.5)
        np.testing.assert_allclose(normalizer.backward(normalized), action, atol=1e-6)
        self.assertTrue((normalizer.forward(action + 1e9, 'action') == 5).all())

    def test_normalizer_rejects_old_schema_and_bad_statistics(self):
        stats = md.fit_statistics({0: mapped()}, [0])
        for key, value in [('schema', 'sonic-inspire-state44-action76-sameframe-all-episodes-parentnorm-v2'),
                           ('hand_target_time_offset', 1), ('train_split_mode', 'train_only')]:
            wrong = deepcopy(stats); wrong[key] = value
            with self.assertRaises(ValueError): md.WholeBodyNormalizer(wrong)
        wrong = deepcopy(stats); wrong['action'][md.ACTION_KEY]['global_q01'][0] = float('nan')
        with self.assertRaises(ValueError): md.WholeBodyNormalizer(wrong)

    def test_window_boundaries_strict_history_and_same_episode(self):
        middle = md.window_indices(50, 100)
        np.testing.assert_array_equal(middle['past_action'], np.arange(10, 50))
        np.testing.assert_array_equal(middle['action'], np.arange(50, 90))
        np.testing.assert_array_equal(middle['image'], [50, 90])
        self.assertTrue(md.window_indices(0, 100)['past_action_is_pad'].all())
        self.assertEqual(int(md.window_indices(2, 100)['past_action_is_pad'].sum()), 38)
        tail = md.window_indices(99, 100)
        self.assertEqual(int(tail['action_is_pad'].sum()), 39)
        self.assertTrue((tail['action'] == 99).all())
        np.testing.assert_array_equal(tail['image_is_pad'], [False, True])

    def test_resize_white_roundoff_clamped(self):
        result = md.resize_rgb_frame_uint8(np.full((480, 640, 3), 255, np.uint8))
        self.assertEqual(tuple(result.shape), (3, 224, 224))
        self.assertGreaterEqual(float(result.min()), 0)
        self.assertLessEqual(float(result.max()), 1)
        torch.testing.assert_close(result, torch.ones_like(result), rtol=0, atol=5e-7)


HAS_ARROW = importlib.util.find_spec('pyarrow') is not None


@unittest.skipUnless(HAS_ARROW, 'PyArrow is required for on-disk synthetic fixtures')
class WholeBodyFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / 'raw'
        self.output = self.base / 'statistics'
        self.cache = self.base / 'text'
        self.write_fixture()

    def write_fixture(self, episode_count=2, length=100,
                      instruction='put the object into the container', resolution=(480, 640)):
        import pyarrow as pa
        import pyarrow.parquet as pq
        self.instruction = instruction
        (self.root / 'meta').mkdir(parents=True, exist_ok=True)
        (self.root / 'data/chunk-000').mkdir(parents=True, exist_ok=True)
        self.cache.mkdir(exist_ok=True)
        self.arrays = {}
        features = {k: {'shape': [v], 'dtype': 'float32'} for k, v in {
            'observation.state': 43, 'action.motion_token': 64,
            'observation.inspire_left_hand_state': 6, 'observation.inspire_right_hand_state': 6,
            'observation.projected_gravity': 3}.items()}
        for hand in ('observation.inspire_left_hand_state', 'observation.inspire_right_hand_state'):
            features[hand]['names'] = list(md.HAND_FIELD_ORDER)
        features.update({cam: {'shape': [*resolution, 3], 'dtype': 'video'} for cam in md.CAMERAS})
        info = dict(codebase_version='v2.1', fps=30, total_episodes=episode_count,
                    total_frames=episode_count * length, total_videos=episode_count * 2,
                    chunks_size=1000, splits={'train': f'0:{episode_count}'}, features=features)
        (self.root / 'meta/info.json').write_text(json.dumps(info))
        episodes, stats = [], []
        for eid in range(episode_count):
            raw = raw_columns(n=length, bias=eid * 2)
            self.arrays[eid] = md.map_numeric(raw)
            table = {k: raw[k].tolist() for k in md.NUMERIC_COLUMNS if k in raw}
            table.update(timestamp=(np.arange(length) / 30).astype(np.float32).tolist(),
                         frame_index=list(range(length)), episode_index=[eid] * length,
                         index=list(range(eid * length, (eid + 1) * length)), task_index=[0] * length)
            pq.write_table(pa.Table.from_pydict(table), self.root / f'data/chunk-000/episode_{eid:06d}.parquet')
            episodes.append(dict(episode_index=eid, length=length, tasks=[instruction]))
            stats.append(dict(episode_index=eid, stats={k: {'count': [length]} for k in table}))
            for camera in md.CAMERAS:
                directory = self.root / 'videos/chunk-000' / camera
                directory.mkdir(parents=True, exist_ok=True)
                (directory / f'episode_{eid:06d}.mp4').write_bytes(b'Synthetic decoder fixture; not real video')
        for name, rows in [('episodes.jsonl', episodes), ('episodes_stats.jsonl', stats),
                           ('tasks.jsonl', [dict(task_index=0, task=instruction)])]:
            (self.root / 'meta' / name).write_text(''.join(json.dumps(row) + '\n' for row in rows))
        prompt = md.DEFAULT_PROMPT.format(task=instruction)
        filename = hashlib.sha256(prompt.encode()).hexdigest() + '.qwen3_flux2_len128.pt'
        torch.save({'text_hidden_states': torch.zeros((128, 7680), dtype=torch.bfloat16),
                    'text_attention_mask': torch.ones(128, dtype=torch.bool)}, self.cache / filename)

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self):
        return md.prepare_statistics(str(self.root), str(self.output))

    def dataset(self, **kwargs):
        def decoder(path, indices):
            offset = 0.5 if 'chest' in str(path) else 0
            return {i: torch.full((3, 224, 224), i / 200 + offset) for i in indices}
        return md.WholeBodyLeRobotDataset(str(self.root), str(self.output / 'dataset_stats.json'),
            str(self.cache), decoder=decoder, **kwargs)

    def test_no_training_split_or_modality_required_and_source_frozen(self):
        before = {str(p.relative_to(self.root)): md.sha256(p) for p in self.root.rglob('*') if p.is_file()}
        meta = md.load_metadata(self.root)
        self.assertEqual(meta['train'], [0, 1]); self.assertEqual(meta['validation'], [])
        audit = self.prepare()
        after = {str(p.relative_to(self.root)): md.sha256(p) for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        self.assertFalse((self.root / 'training_split.json').exists())
        self.assertEqual(len(audit['metadata_sha256']), 4)
        self.assertEqual(len(audit['parquet_sha256']), 2)
        self.assertFalse(audit['normalization_inherited_from_other_task'])
        self.assertEqual(audit['train_frames'], 200)
        with self.assertRaises(FileExistsError): self.prepare()
        with self.assertRaises(ValueError): md.prepare_statistics(str(self.root), str(self.root / 'cache'))

    def test_optional_modality_hashed_and_invalid_task_rejected(self):
        (self.root / 'meta/modality.json').write_text('{}')
        self.assertEqual(len(md.metadata_paths(self.root)), 5)
        taskfile = self.root / 'meta/tasks.jsonl'
        for rows in [[dict(task_index=0, task='')], [dict(task_index=1, task='valid task')],
                     [dict(task_index=0, task='one'), dict(task_index=1, task='two')]]:
            taskfile.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            with self.assertRaises(ValueError): md.load_metadata(self.root)

    def test_dataset_counts_instruction_and_camera_resolution_are_portable(self):
        for count, length, instruction, resolution in [
                (1, 53, 'fold the cloth', (720, 1280)),
                (3, 81, 'close the drawer', (240, 320))]:
            with self.subTest(episodes=count):
                self.root = self.base / f'raw_{count}'
                self.output = self.base / f'statistics_{count}'
                self.cache = self.base / f'text_{count}'
                self.write_fixture(count, length, instruction, resolution)
                audit = self.prepare()
                self.assertEqual(audit['episodes'], count)
                self.assertEqual(audit['total_frames'], count * length)
                self.assertEqual(audit['instruction'], instruction)
                self.assertEqual(audit['raw_video_resolutions'][md.CAMERAS[0]], list(resolution))
                ds = self.dataset()
                self.assertEqual(len(ds), count * length)
                self.assertEqual(ds.episode_ids, list(range(count)))
                self.assertIn(instruction, ds[0]['prompt'])
                self.assertEqual(ds[len(ds) - 1]['episode_index'], count - 1)

    def test_wrong_feature_order_counts_fps_and_rgb_shape_are_rejected(self):
        path = self.root / 'meta/info.json'
        original = json.loads(path.read_text())
        cases = []
        wrong = deepcopy(original); wrong['features']['observation.inspire_left_hand_state']['names'].reverse()
        cases.append(wrong)
        for key, value in [('total_episodes', 3), ('total_frames', 199), ('fps', 15), ('total_videos', 3)]:
            wrong = deepcopy(original); wrong[key] = value; cases.append(wrong)
        for shape in [[0, 640, 3], [480, 640, 4], [480, 640], [480, -1, 3]]:
            wrong = deepcopy(original); wrong['features'][md.CAMERAS[0]]['shape'] = shape; cases.append(wrong)
        for wrong in cases:
            path.write_text(json.dumps(wrong))
            with self.assertRaises(ValueError): md.load_metadata(self.root)
        path.write_text(json.dumps(original))
        episodes = self.root / 'meta/episodes.jsonl'
        rows = [json.loads(line) for line in episodes.read_text().splitlines()]
        rows[1]['episode_index'] = 2
        episodes.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        with self.assertRaises(ValueError): md.load_metadata(self.root)

    def test_actual_side_prefixed_hand_names_are_accepted(self):
        path = self.root / 'meta/info.json'
        info = json.loads(path.read_text())
        for side in ('left', 'right'):
            info['features'][f'observation.inspire_{side}_hand_state']['names'] = [
                f'{side}_{name}' for name in md.HAND_FIELD_ORDER]
        path.write_text(json.dumps(info))
        md.load_metadata(self.root)
        self.prepare()
        sample = self.dataset()[50]
        self.assertEqual(tuple(sample['action'].shape), (40, 76))
        self.assertEqual(tuple(sample['proprio'].shape), (40, 44))

    def test_wrong_side_mixed_duplicate_and_short_hand_names_are_rejected(self):
        path = self.root / 'meta/info.json'
        original = json.loads(path.read_text())
        for side in ('left', 'right'):
            other_side = 'right' if side == 'left' else 'left'
            correct = [f'{side}_{name}' for name in md.HAND_FIELD_ORDER]
            mixed = correct.copy(); mixed[2] = f'{other_side}_{md.HAND_FIELD_ORDER[2]}'
            duplicate = correct.copy(); duplicate[2] = duplicate[1]
            for names in ([f'{other_side}_{name}' for name in md.HAND_FIELD_ORDER],
                          list(reversed(correct)), mixed, duplicate, correct[:-1], correct + [correct[-1]]):
                with self.subTest(side=side, names=names):
                    wrong = deepcopy(original)
                    wrong['features'][f'observation.inspire_{side}_hand_state']['names'] = names
                    path.write_text(json.dumps(wrong))
                    with self.assertRaises(ValueError): md.load_metadata(self.root)
        path.write_text(json.dumps(original))

    def test_complete_sample_contract_and_current_state_only(self):
        self.prepare(); ds = self.dataset()
        item = ds[50]
        expected = {'video': (3, 2, 224, 448), 'action': (40, 76), 'past_action': (40, 76),
                    'proprio': (40, 44), 'action_is_pad': (40,), 'past_action_is_pad': (40,),
                    'image_is_pad': (2,), 'action_dim_is_pad': (76,), 'proprio_dim_is_pad': (44,),
                    'text_hidden_states': (128, 7680), 'text_attention_mask': (128,)}
        for name, shape in expected.items():
            self.assertEqual(tuple(item[name].shape), shape, name)
        state, action = self.arrays[0]
        np.testing.assert_array_equal(item['proprio'][0], ds.normalizer.forward(state[50], 'state'))
        self.assertTrue(torch.equal(item['proprio'], item['proprio'][:1].expand_as(item['proprio'])))
        np.testing.assert_array_equal(item['action'][0], ds.normalizer.forward(action[50], 'action'))
        np.testing.assert_array_equal(item['past_action'][-1], ds.normalizer.forward(action[49], 'action'))
        torch.testing.assert_close(item['action'][0, 64:76], item['proprio'][0, 29:41], rtol=0, atol=0)
        self.assertFalse(torch.equal(item['action'][0, 64:76], item['action'][1, 64:76]))
        self.assertTrue(torch.equal(item['text_hidden_states'], ds[51]['text_hidden_states']))
        self.assertEqual(item['video'][0, 0, 0, 0], -0.5)
        self.assertEqual(item['video'][0, 0, 0, 224], 0.5)
        self.assertAlmostEqual(item['video'][0, 1, 0, 0].item(), -0.1, places=6)
        self.assertIn(self.instruction, item['prompt'])
        self.assertNotIn('cloth', item['prompt'])

    def test_episode_start_end_padding_and_collation(self):
        self.prepare(); ds = self.dataset(video_cache_frames=4)
        self.assertEqual(len(ds), 200)
        for index in [0, 100]:
            item = ds[index]
            self.assertTrue(item['past_action_is_pad'].all())
            self.assertTrue((item['past_action'] == 0).all())
        tail = ds[99]
        self.assertEqual(int(tail['action_is_pad'].sum()), 39)
        self.assertTrue(tail['image_is_pad'][1])
        self.assertEqual(ds[100]['episode_index'], 1)
        for i in range(5): ds[i]
        self.assertLessEqual(len(ds.frame_cache), 4)
        with self.assertRaises(IndexError): ds[200]
        with self.assertRaises(ValueError): self.dataset(split='validation')
        batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=2, num_workers=0)))
        self.assertEqual(tuple(batch['video'].shape), (2, 3, 2, 224, 448))
        self.assertEqual(tuple(batch['action'].shape), (2, 40, 76))

    def test_metadata_or_parquet_mutation_rejected(self):
        self.prepare()
        metadata = self.root / 'meta/episodes.jsonl'
        old = metadata.read_text(); metadata.write_text(old + '\n')
        with self.assertRaises(ValueError): self.dataset()
        metadata.write_text(old)
        parquet = self.root / 'data/chunk-000/episode_000001.parquet'
        with parquet.open('ab') as stream: stream.write(b'changed')
        with self.assertRaises(ValueError): self.dataset()

    def test_partial_statistics_or_wrong_cache_rejected(self):
        self.prepare(); path = self.output / 'dataset_stats.json'
        original = json.loads(path.read_text())
        for key, bad in [('train_episode_ids', [0]), ('heldout_episode_ids', [1]),
                         ('normalization_reference_episode_ids', [0]), ('train_frames', 100),
                         ('instruction', 'get the cloth and place it into the blanket'), ('parquet_sha256', {})]:
            changed = deepcopy(original); changed[key] = bad; path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError): self.dataset()
        path.write_text(json.dumps(original))
        wrong_cache = self.base / 'empty_text_cache'; wrong_cache.mkdir()
        with self.assertRaises(FileNotFoundError):
            md.WholeBodyLeRobotDataset(str(self.root), str(path), str(wrong_cache))


class WholeBodyFrameCacheTests(unittest.TestCase):
    def dataset(self, capacity):
        ds = object.__new__(md.WholeBodyLeRobotDataset)
        ds.root = Path('/unused-by-fake-decoder'); ds.video_cache_frames = capacity
        ds.frame_cache = OrderedDict(); ds.decode_calls = []
        def decoder(path, indices):
            ds.decode_calls.append(list(indices))
            return {i: self.frame(i) for i in indices}
        ds.decoder = decoder
        return ds

    @staticmethod
    def frame(index):
        return torch.tensor(index / 1000, dtype=torch.float32).view(1, 1, 1).expand(3, 224, 224)

    def test_full64_miss_before_oldest_hit_regression(self):
        ds = self.dataset(64); camera = md.CAMERAS[0]
        ds.frame_cache[(0, camera, 219)] = self.frame(219)
        for i in range(1000, 1063): ds.frame_cache[(0, camera, i)] = self.frame(i)
        result = ds._frames(0, camera, [179, 219])
        torch.testing.assert_close(result, torch.stack([self.frame(179), self.frame(219)]), rtol=0, atol=0)
        self.assertEqual(ds.decode_calls, [[179]])
        self.assertEqual(len(ds.frame_cache), 64)

    def test_capacity_one_duplicate_mixed_hits_resolved_before_eviction(self):
        ds = self.dataset(1); camera = md.CAMERAS[0]
        ds.frame_cache[(0, camera, 10)] = self.frame(10)
        result = ds._frames(0, camera, [20, 10, 20, 10])
        torch.testing.assert_close(result, torch.stack([self.frame(i) for i in [20, 10, 20, 10]]), rtol=0, atol=0)
        self.assertEqual(ds.decode_calls, [[20]])
        self.assertEqual(list(ds.frame_cache), [(0, camera, 10)])

    def test_disabled_cache_deduplicates_each_decode_request(self):
        ds = self.dataset(0)
        for _ in range(2): ds._frames(0, md.CAMERAS[0], [99, 99])
        self.assertEqual(ds.decode_calls, [[99], [99]])
        self.assertEqual(len(ds.frame_cache), 0)

    def test_invalid_frame_does_not_mutate_cache(self):
        ds = self.dataset(1); camera = md.CAMERAS[0]
        original = self.frame(219); ds.frame_cache[(0, camera, 219)] = original
        ds.decoder = lambda path, indices: {179: torch.zeros(3, 1, 1)}
        with self.assertRaises(ValueError): ds._frames(0, camera, [219, 179])
        self.assertEqual(list(ds.frame_cache), [(0, camera, 219)])
        self.assertIs(ds.frame_cache[(0, camera, 219)], original)


if __name__ == '__main__':
    unittest.main()
