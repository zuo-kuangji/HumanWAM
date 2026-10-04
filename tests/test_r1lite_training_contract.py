"""CPU contracts for the public real-robot configuration and input adapter."""
import importlib.util
from pathlib import Path

from hydra import compose, initialize_config_dir
import numpy as np
from PIL import Image
import pytest
import torch

from imagewam.models.backbones.imagewam import ImageWAM

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/r1lite/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("task", ["r1lite_cups_a2ai2i_20ep", "r1lite_insert_flower_20ep"])
def test_protocol(task):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=[f"task={task}"])
    assert cfg.batch_size == 10 and cfg.gradient_accumulation_steps == 1
    assert cfg.num_epochs == 20 and cfg.keep_latest_weights == 4
    assert cfg.keep_latest_state_only and cfg.model.proprio_dim == 7
    assert list(cfg.data.train.video_size) == [224, 448]
    assert cfg.data.train.past_action_size == 16
    assert cfg.data.train.skip_padding_as_possible is False
    assert cfg.data.train.processor.delta_action_dim_mask is None
    assert cfg.model.flux2_action_image_context.knowledge_insulation
    assert cfg.model.flux2_action_image_context.include_future_tokens
    assert cfg.model.loss.image_i2i.current_residual_scale == 1
    assert cfg.model.loss.image_i2i.variance_normalize
    assert cfg.model.loss.image_i2i.ic_weight == 1
    assert cfg.model.loss.action_a2a.ic_weight == .5
    assert list(cfg.model.loss.action_a2a.source_noise_dim_mask) == [True] * 7
    assert cfg.model.action_head_init_scale == 1 and not cfg.model.zero_init_action_head
    assert cfg.model.qwen_context_len == cfg.data.train.qwen_context_len == 128
    if "flower" in task:
        assert "pick up the flower" in cfg.data.train.override_instruction


def test_epoch_schedule():
    train = load_script("train")
    assert train.schedule(61068, 8, 10) == (764, 15280)
    assert train.schedule(45684, 8, 10) == (572, 11440)
    with pytest.raises(ValueError):
        train.schedule(45684, 0, 10)


def test_camera_layout_and_pixel_range(tmp_path):
    infer = load_script("infer")
    for name, color in [("head", 255), ("wrist", 0)]:
        Image.fromarray(np.full((30, 60, 3), color, dtype=np.uint8)).save(tmp_path / f"{name}.png")
    image = infer.pack_images(tmp_path / "head.png", tmp_path / "wrist.png")
    assert image.shape == (1, 3, 224, 448)
    assert torch.all(image[..., :224] == 1)
    assert torch.all(image[..., 224:] == -1)


def test_noise_affects_all_seven_dimensions():
    model = ImageWAM.__new__(ImageWAM)
    torch.nn.Module.__init__(model)
    model.action_a2a_source_noise_std = .5
    model.action_a2a_source_noise_dim_mask = [True] * 7
    source = torch.ones(2, 16, 7)
    torch.manual_seed(42)
    expected = source + torch.randn_like(source) * .5
    torch.manual_seed(42)
    result = model._apply_action_a2a_source_noise(source)
    torch.testing.assert_close(result, expected)
    assert not torch.equal(result[..., 6], source[..., 6])


def test_current_proprio_is_one_token_not_the_future_sequence():
    model = ImageWAM.__new__(ImageWAM)
    torch.nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.proprio_dim = 7
    model.proprio_encoder = torch.nn.Linear(7, 12)
    model.pack_proprio_after_text = True
    context, mask = torch.zeros(1, 128, 12), torch.zeros(1, 128, dtype=torch.bool)
    mask[:, :55] = True
    proprio = torch.randn(1, 16, 7)
    result, result_mask = model._append_proprio_to_context_if_enabled(context, mask, proprio, source="test")
    assert result.shape == (1, 129, 12) and result_mask.sum() == 56
    torch.testing.assert_close(result[:, 55], model.proprio_encoder(proprio[:, 0]))


def test_inference_requires_finite_matching_shapes():
    infer = load_script("infer")
    class Identity:
        def forward(self, x):
            return x
    past, state = infer.normalize_inputs(Identity(), np.zeros((16, 7)), np.ones(7))
    assert past.shape == (1, 16, 7) and state.shape == (1, 7)
    with pytest.raises(ValueError):
        infer.normalize_inputs(Identity(), np.zeros((15, 7)), np.ones(7))
    with pytest.raises(ValueError):
        infer.normalize_inputs(Identity(), np.full((16, 7), np.nan), np.ones(7))
