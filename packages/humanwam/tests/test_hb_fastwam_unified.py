"""Exercise HB's FastWAM patch with real Torch workers and boundary fixtures."""

import importlib.util
import multiprocessing
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from humanwam.augmentation import SampleKey, UnifiedAugmentation

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "hb_training/fastwam_tools"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def native(monkeypatch):
    class Processor:
        is_train = True

        def preprocess(self, data):
            return {"idx": data["idx"], "proprio": data["state"]["default"],
                    "action": data["action"], "pixel_values": data["images"]}

        def action_state_transform(self, data):
            return data

    class BaseDataset:
        def __init__(self, processor):
            self.processor = processor

        def __getitem__(self, index):
            # Simulate native read retry: request0 actually loads sample2.
            actual = 2 if index == 0 else index
            return self.processor.preprocess({
                "idx": actual, "episode_index": np.int64(7), "frame_index": np.int64(actual + 11),
                "state": {"default": torch.zeros(17, 44)},
                "action": torch.full((16, 76), 0.2),
                "images": torch.linspace(0.1, 0.9, 2 * 17 * 3 * 32 * 32).reshape(2, 17, 3, 32, 32),
            })

    class VideoDataset:
        def __init__(self, processor):
            self.base = BaseDataset(processor)

        def __len__(self):
            return 4

        def _get(self, index):
            return self.base[index]

        def __getitem__(self, index):
            return self._get(index)

    sampler_source = ROOT / "third_party/FastWAM/src/fastwam/utils/samplers.py"
    if not sampler_source.is_file():
        pytest.skip("Pinned FastWAM sampler source absent")
    native_sampler = load_module("_hb_fastwam_native_sampler", sampler_source).ResumableEpochSampler

    class Sampler(native_sampler):
        def __init__(self, dataset):
            super().__init__(dataset, seed=42, batch_size=1, num_processes=1)

    modules = {
        "fastwam.datasets.lerobot.base_lerobot_dataset": ("BaseLerobotDataset", BaseDataset),
        "fastwam.datasets.lerobot.processors.fastwam_processor": ("FastWAMProcessor", Processor),
        "fastwam.datasets.lerobot.robot_video_dataset": ("RobotVideoDataset", VideoDataset),
        "fastwam.utils.samplers": ("ResumableEpochSampler", Sampler),
    }
    for name, (attribute, value) in modules.items():
        for count in range(1, len(name.split("."))):
            parent = ".".join(name.split(".")[:count])
            if parent not in sys.modules:
                package = ModuleType(parent)
                package.__path__ = []
                monkeypatch.setitem(sys.modules, parent, package)
        module = ModuleType(name)
        setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("HB_AUG_DIR", str(ROOT / "hb_training/hb_groot"))
    patch = load_module("hb_fastwam_aug_test", TOOLS / "hb_fastwam_aug.py")
    return patch, Processor, VideoDataset, Sampler


def test_unified_uses_actual_episode_frame_for_both_modalities_and_leaves_targets(native):
    patch, Processor, Dataset, _ = native
    patch.install("unified_v3", seed=42)
    dataset = Dataset(Processor())
    item = dataset[(0, 3)]  # Actual native sample is2, not requested0.
    expected = UnifiedAugmentation.from_hb_profile("unified_v3", seed=42)
    sample = SampleKey(7, 13, 3)
    np.testing.assert_array_equal(item["proprio"], expected.apply_state(np.zeros((17, 44), np.float32), body_dims=29, sample=sample))
    pixels = torch.linspace(0.1, 0.9, 2 * 17 * 3 * 32 * 32).reshape(2, 17, 3, 32, 32)
    mask = torch.zeros(17, dtype=torch.bool)
    mask[0] = True
    rgb = expected.apply_rgb(pixels.permute(1, 0, 2, 3, 4), sample=sample, observation_mask=mask)
    torch.testing.assert_close(item["pixel_values"], rgb.permute(1, 0, 2, 3, 4), rtol=0, atol=0)
    torch.testing.assert_close(item["action"], torch.full((16, 76), 0.2))
    assert "_hb_aug_sample" not in item
    assert set(item) == {"idx", "proprio", "action", "pixel_values"}
    assert not item["proprio"][:, 29:].count_nonzero()


def test_eval_and_stats_do_not_augment(native):
    patch, Processor, Dataset, _ = native
    patch.install("unified_v3", seed=42)
    processor = Processor()
    stats = {"state": {"default": torch.zeros(17, 44)}, "action": torch.zeros(16, 76)}
    processor.action_state_transform(stats)
    assert not stats["state"]["default"].count_nonzero()
    processor.is_train = False
    item = Dataset(processor)[(1, 5)]
    assert not item["proprio"].count_nonzero()
    expected = torch.linspace(0.1, 0.9, 2 * 17 * 3 * 32 * 32).reshape(2, 17, 3, 32, 32)
    torch.testing.assert_close(item["pixel_values"], expected, rtol=0, atol=0)


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="Native HB FastWAM uses Linux fork workers")
def test_prefetched_persistent_workers_replay_ids_and_resume_epoch(native):
    patch, Processor, Dataset, Sampler = native
    torch.set_num_threads(1)
    patch.install("unified_v3", seed=42)
    dataset = Dataset(Processor())
    sampler = Sampler(dataset)
    sampler.epoch_offset = 4  # Native resumed epoch offset.
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=2,
                                        prefetch_factor=2, persistent_workers=True, multiprocessing_context="fork")
    first = list(loader)
    sampler.epoch = 1
    second = list(loader)
    engine = UnifiedAugmentation.from_hb_profile("unified_v3", seed=42)
    for epoch, batches in ((4, first), (5, second)):
        for item in batches:
            actual = int(item["idx"][0])
            expected = engine.apply_state(np.zeros((17, 44), np.float32), body_dims=29,
                                          sample=SampleKey(7, actual + 11, epoch))
            np.testing.assert_array_equal(item["proprio"][0], expected)
    assert not torch.equal(first[0]["proprio"], second[0]["proprio"])
    del loader


def test_legacy_remains_default_and_profile_switch_needs_a_new_process(native, monkeypatch):
    patch, Processor, Dataset, _ = native
    monkeypatch.delenv("HB_AUG_PROFILE", raising=False)
    patch.install()
    assert Processor._hb_aug_profile == "hb_v2"
    item = Dataset(Processor())[1]
    assert item["proprio"][:, :29].count_nonzero()
    assert not item["proprio"][:, 29:].count_nonzero()
    torch.testing.assert_close(item["action"], torch.full((16, 76), 0.2))
    with pytest.raises(ValueError, match="fresh process"):
        patch.install("unified")


def test_missing_or_invalid_sample_identity_fails_without_random_retry(native):
    patch, Processor, _, _ = native
    patch.install("unified_v3", seed=42)
    processor = Processor()
    with pytest.raises(ValueError, match="episode_index"):
        processor.preprocess({"state": {"default": torch.zeros(17, 44)}})
    with pytest.raises(ValueError, match="nonnegative integer IDs"):
        processor.preprocess({"episode_index": 1.5, "frame_index": 2})


def test_train_selection_is_explicit_and_disable_does_not_import_augmentation(monkeypatch, tmp_path):
    monkeypatch.setenv("FASTWAM_SRC", str(tmp_path))
    module = load_module("hb_fastwam_train_test", TOOLS / "hb_fastwam_train.py")
    monkeypatch.delenv("HB_AUG_PROFILE", raising=False)
    assert module.parse(["hb_aug_profile=unified", "hb_aug_seed=51", "batch_size=8"]) == {"batch_size": 8}
    assert __import__("os").environ["HB_AUG_PROFILE"] == "unified"
    monkeypatch.setenv("HB_FW_AUG", "0")
    monkeypatch.delitem(sys.modules, "hb_fastwam_aug", raising=False)
    assert module.install_augmentation() == "disabled"
    assert "hb_fastwam_aug" not in sys.modules


def test_native_resume_epoch_indices_survive_accelerate_rank_sharding(native):
    pytest.importorskip("accelerate")
    from accelerate.data_loader import BatchSamplerShard

    patch, Processor, Dataset, Sampler = native
    patch.install("unified_v3", seed=42)
    dataset = Dataset(Processor())
    sampler = Sampler(dataset)
    sampler.epoch_offset = 4
    sampler.num_processes = 2
    sampler.set_resume_batch_offset(1)
    engine = UnifiedAugmentation.from_hb_profile("unified_v3", seed=42)
    count = 0
    for rank in range(2):
        batches = torch.utils.data.BatchSampler(sampler, batch_size=1, drop_last=False)
        shard = BatchSamplerShard(batches, num_processes=2, process_index=rank, even_batches=False)
        loader = torch.utils.data.DataLoader(dataset, batch_sampler=shard, num_workers=0)
        for item in loader:
            actual = int(item["idx"][0])
            expected = engine.apply_state(np.zeros((17, 44), np.float32), body_dims=29,
                                          sample=SampleKey(7, actual + 11, 4))
            np.testing.assert_array_equal(item["proprio"][0], expected)
            count += 1
    assert count == 2  # Four anchors minus the two globally consumed by resume.


@pytest.mark.parametrize("selector,version", [
    ("unified", "humanwam_unified_augmentation_v3"),
    ("unified_v3", "humanwam_unified_augmentation_v3"),
    ("unified_v2", "humanwam_unified_augmentation_v2"),
    ("unified_v1", "humanwam_unified_augmentation_v1"),
])
def test_versioned_profiles_report_effective_version_and_use_the_selected_engine(native, selector, version):
    patch, Processor, Dataset, _ = native
    assert patch.install(selector, seed=42) == version
    assert patch.install(selector, seed=42) == version
    assert Processor._hb_aug_version == version
    engine = UnifiedAugmentation.from_hb_profile(selector, seed=42)
    pixels = torch.linspace(0.1, 0.9, 2 * 17 * 3 * 32 * 32).reshape(2, 17, 3, 32, 32)
    mask = torch.zeros(17, dtype=torch.bool)
    mask[0] = True
    expected = engine.apply_rgb(pixels.permute(1, 0, 2, 3, 4), sample=SampleKey(7, 13, 3), observation_mask=mask)
    item = Dataset(Processor())[(0, 3)]
    torch.testing.assert_close(item["pixel_values"], expected.permute(1, 0, 2, 3, 4), rtol=0, atol=0)


@pytest.mark.parametrize("selector", ["unified", "unified_v1", "unified_v2", "unified_v3"])
def test_training_cli_preserves_explicit_version_and_returns_effective_version(monkeypatch, tmp_path, selector):
    monkeypatch.setenv("FASTWAM_SRC", str(tmp_path))
    monkeypatch.setenv("HB_FW_AUG", "1")
    module = load_module("hb_fastwam_train_versions", TOOLS / "hb_fastwam_train.py")
    assert module.parse(["hb_aug_profile=" + selector, "hb_aug_seed=42", "batch_size=8"]) == {"batch_size": 8}
    expected = UnifiedAugmentation.from_hb_profile(selector, seed=42).profile
    fake = ModuleType("hb_fastwam_aug")
    fake.install = lambda: expected
    monkeypatch.setitem(sys.modules, "hb_fastwam_aug", fake)
    assert module.install_augmentation() == expected
