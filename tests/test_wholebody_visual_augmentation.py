"""CPU RGB augmentation checks; no model weights, dataset mutation or CUDA."""
import copy
import colorsys
import itertools
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from imagewam.datasets.wholebody_augmentation import (
    DEFAULT_VISUAL_AUGMENTATION, VISUAL_OPERATIONS, WholeBodyVisualAugmenter,
)


def generator(seed=42):
    return torch.Generator(device="cpu").manual_seed(seed)


def camera_pair():
    y,x=torch.meshgrid(torch.linspace(0,1,32),torch.linspace(0,1,32),indexing="ij")
    frame=torch.stack((x,y,(x+y)*.5))
    return torch.stack((frame,frame))


def only(operation,**overrides):
    cfg=copy.deepcopy(DEFAULT_VISUAL_AUGMENTATION)
    cfg["probability"]=1
    for key in VISUAL_OPERATIONS:cfg[key]["probability"]=0
    cfg[operation].update(probability=1,**overrides)
    return cfg


def all_operations():
    cfg=copy.deepcopy(DEFAULT_VISUAL_AUGMENTATION)
    cfg["probability"]=1
    for key in VISUAL_OPERATIONS:cfg[key]["probability"]=1
    return cfg


@pytest.fixture(autouse=True)
def one_cpu_thread():
    before=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_current_and_future_share_every_sampled_visual_parameter():
    original=camera_pair();aug=WholeBodyVisualAugmenter(all_operations())
    changed=aug.apply(original,generator=generator())
    assert torch.equal(changed[0],changed[1])
    assert not torch.equal(changed,original)
    assert bool(torch.isfinite(changed).all()) and 0<=float(changed.min())<=float(changed.max())<=1


def test_no_future_image_leaks_into_current_via_contrast_mean_or_blur():
    first=camera_pair();second=first.clone();second[1]=1-first[1]
    aug=WholeBodyVisualAugmenter(all_operations())
    a=aug.apply(first,generator=generator());b=aug.apply(second,generator=generator())
    assert torch.equal(a[0],b[0])
    assert not torch.equal(a[1],b[1])


def test_affine_uses_replicated_boundary_instead_of_black_triangles():
    aug=WholeBodyVisualAugmenter(only("jitter"))
    white=torch.ones(2,3,32,32)
    output=aug.apply(white,generator=generator())
    torch.testing.assert_close(output,white,rtol=0,atol=1e-6)


def test_small_geometric_jitter_does_not_mirror_left_and_right():
    original=camera_pair();aug=WholeBodyVisualAugmenter(only("jitter"))
    for seed in (1,42,571):
        output=aug.apply(original,generator=generator(seed))
        assert output[0,0,:,:4].mean()<output[0,0,:,-4:].mean()
        assert float((output-original).abs().max())<.3


def test_controlled_overexposure_really_clips_highlights():
    original=torch.full((2,3,32,32),.75)
    aug=WholeBodyVisualAugmenter(only("exposure",gain=[1.5,1.5],gamma=[1.,1.]))
    output=aug.apply(original,generator=generator())
    assert torch.equal(output,torch.ones_like(output))
    assert torch.equal(original,torch.full_like(original,.75))


def test_separate_rare_overexposure_branch_is_independent_of_general_exposure():
    original=torch.full((2,3,16,16),.7)
    cfg=only("overexposure",gain=[1.8,1.8])
    output=WholeBodyVisualAugmenter(cfg).apply(original,generator=generator())
    assert torch.equal(output,torch.ones_like(output))
    assert cfg["exposure"]["probability"]==0


def test_blur_attenuates_high_frequency_detail_without_temporal_mixing():
    row=torch.arange(32);checker=((row[:,None]+row[None,:])%2).float()
    original=checker[None,None].expand(2,3,-1,-1).clone()
    aug=WholeBodyVisualAugmenter(only("blur",sigma=[1.,1.],kernel_size=5))
    output=aug.apply(original,generator=generator())
    before=(original[...,1:]-original[...,:-1]).abs().mean()
    after=(output[...,1:]-output[...,:-1]).abs().mean()
    assert after<before*.25
    assert torch.equal(output[0],output[1])


def test_hue_rotation_uses_rgb_channel_semantics():
    red=torch.zeros(2,3,8,8);red[:,0]=1
    params={"exposure":None,"overexposure":None,"jitter":None,"blur":None,
            "color":{"contrast":1.,"saturation":1.,"hue":1/3}}
    output=WholeBodyVisualAugmenter().apply(red,generator=generator(),parameters=params)
    green=torch.zeros_like(red);green[:,1]=1
    torch.testing.assert_close(output,green,rtol=0,atol=1e-6)


@pytest.mark.parametrize("contrast,saturation,hue", list(itertools.product(
    (0.85, 1.15), (0.8, 1.2), (-5 / 360, 5 / 360),
)))
def test_primary_colors_keep_dominant_channel_and_hue_within_five_degrees(
    contrast, saturation, hue,
):
    # This checks RGB/HSV numerical behavior, not a real classifier's decisions.
    original = torch.eye(3).reshape(3, 3, 1, 1).expand(-1, -1, 8, 8).clone()
    parameters = {key: None for key in ("exposure", "overexposure", "jitter", "blur")}
    parameters["color"] = {"contrast": contrast, "saturation": saturation, "hue": hue}
    output = WholeBodyVisualAugmenter(only("color")).apply(
        original, generator=generator(), parameters=parameters,
    )
    assert torch.equal(output.argmax(dim=1), original.argmax(dim=1))
    assert bool(torch.isfinite(output).all())
    for color in range(3):
        before = colorsys.rgb_to_hsv(*original[color, :, 0, 0].tolist())[0]
        after = colorsys.rgb_to_hsv(*output[color, :, 0, 0].tolist())[0]
        difference = abs(after - before)
        circular_degrees = min(difference, 1 - difference) * 360
        assert circular_degrees <= 5 + 1e-4


def test_maximum_updated_gaussian_blur_smooths_more_than_previous_maximum_and_preserves_pair():
    row = torch.arange(32)
    checker = ((row[:, None] + row[None, :]) % 2).float()
    original = checker[None, None].expand(2, 3, -1, -1).clone()
    saved = original.clone()
    old = WholeBodyVisualAugmenter(only("blur", sigma=[0.8, 0.8], kernel_size=5)).apply(
        original, generator=generator(),
    )
    updated = WholeBodyVisualAugmenter(only("blur", sigma=[1.2, 1.2], kernel_size=9)).apply(
        original, generator=generator(),
    )
    def spatial_variation(image):
        return ((image[..., 1:] - image[..., :-1]).abs().mean()
                + (image[..., 1:, :] - image[..., :-1, :]).abs().mean())
    assert spatial_variation(updated) < spatial_variation(old)
    assert torch.equal(updated[0], updated[1])
    assert bool(torch.isfinite(updated).all())
    assert 0 < float(updated.min()) <= float(updated.max()) <= 1
    assert float(updated.mean()) > 0.1
    torch.testing.assert_close(original, saved, rtol=0, atol=0)


def test_seed_reproducibility_and_advancing_worker_generator_change_parameters():
    original=camera_pair();aug=WholeBodyVisualAugmenter(all_operations())
    a=aug.apply(original,generator=generator(42));b=aug.apply(original,generator=generator(42))
    c=aug.apply(original,generator=generator(43))
    assert torch.equal(a,b) and not torch.equal(a,c)
    advancing=generator(42)
    assert not torch.equal(aug.apply(original,generator=advancing),aug.apply(original,generator=advancing))


def test_no_global_rng_input_cache_or_config_mutation():
    config=all_operations();before_config=copy.deepcopy(config)
    original=camera_pair();before_frame=original.clone()
    aug=WholeBodyVisualAugmenter(config);global_state=torch.get_rng_state().clone()
    output=aug.apply(original,generator=generator())
    assert output.data_ptr()!=original.data_ptr()
    assert torch.equal(original,before_frame) and config==before_config
    assert torch.equal(torch.get_rng_state(),global_state)


def test_disabled_and_all_zero_probabilities_are_exact_copies_without_rng_draws():
    original=camera_pair().double()
    zero=copy.deepcopy(DEFAULT_VISUAL_AUGMENTATION)
    zero["probability"]=1
    for key in VISUAL_OPERATIONS:zero[key]["probability"]=0
    for cfg in ({"enabled":False},zero):
        g=generator();before=g.get_state().clone()
        output=WholeBodyVisualAugmenter(cfg).apply(original,generator=g)
        assert torch.equal(output,original) and output.dtype==original.dtype
        assert output.data_ptr()!=original.data_ptr() and torch.equal(before,g.get_state())


def test_overall_visual_probability_controls_the_entire_pipeline():
    original=camera_pair()
    config=all_operations();config["probability"]=0
    g=generator();before=g.get_state().clone()
    result=WholeBodyVisualAugmenter(config).apply(original,generator=g)
    assert torch.equal(result,original) and torch.equal(before,g.get_state())
    config["probability"]=.5
    augmenter=WholeBodyVisualAugmenter(config)
    # First uniform draws: seed0=.4963 (keep), seed1=.7576 (skip).
    assert not torch.equal(augmenter.apply(original,generator=generator(0)),original)
    assert torch.equal(augmenter.apply(original,generator=generator(1)),original)


def test_controller_can_apply_a_preselected_global_gate_without_resampling_it():
    original=camera_pair();config=all_operations();config["probability"]=0
    augmenter=WholeBodyVisualAugmenter(config)
    assert not torch.equal(augmenter.apply(original,generator=generator(),apply_probability=False),original)


def test_explicit_parameters_can_be_shared_across_cameras_without_new_rng_draws():
    original=camera_pair();aug=WholeBodyVisualAugmenter(all_operations())
    params=aug.sample_parameters(generator());g=generator(71);before=g.get_state().clone()
    a=aug.apply(original,generator=g,parameters=params)
    b=aug.apply(original,generator=generator(99),parameters=params)
    assert torch.equal(a,b) and torch.equal(before,g.get_state())
    assert WholeBodyVisualAugmenter({"camera_mode":"shared"}).camera_mode=="shared"


@pytest.mark.parametrize("dtype",[torch.float16,torch.float32,torch.float64])
def test_output_preserves_clip_shape_dtype_and_bounds(dtype):
    original=camera_pair().to(dtype)
    output=WholeBodyVisualAugmenter(all_operations()).apply(original,generator=generator())
    assert output.shape==original.shape and output.dtype==dtype and bool(torch.isfinite(output).all())
    assert bool((output>=0).all()) and bool((output<=1).all())


@pytest.mark.parametrize("invalid",[torch.ones(2,3,8,8,dtype=torch.uint8),torch.zeros(2,8,8,3),
    torch.zeros(0,3,8,8),torch.full((2,3,8,8),1.1),torch.full((2,3,8,8),float("nan"))])
def test_invalid_image_contracts_are_rejected(invalid):
    with pytest.raises(ValueError,match="finite float RGB"):
        WholeBodyVisualAugmenter().apply(invalid,generator=generator())


@pytest.mark.parametrize("config",[{"enabled":"true"},{"unknown":1},{"exposure":{"probability":1.1}},
    {"exposure":{"gamma":[0,1]}},{"color":{"hue":.7}},{"jitter":{"scale":[1,.5]}},
    {"blur":{"kernel_size":4}},{"blur":{"sigma":[float("nan"),1]}},{"camera_mode":"unknown"}])
def test_invalid_configuration_is_rejected_early(config):
    with pytest.raises(ValueError):WholeBodyVisualAugmenter(config)


def test_pixel_noise_replays_private_seed_across_pair_without_rng_or_cache_mutation():
    original = torch.full((2, 3, 64, 64), 0.5)
    saved = original.clone()
    aug = WholeBodyVisualAugmenter(only("pixel_noise", std=[8 / 255, 8 / 255]))
    params = aug.sample_parameters(generator(12))
    supplied = copy.deepcopy(params)
    g = generator(55)
    before = g.get_state().clone()
    global_before = torch.get_rng_state().clone()
    output = aug.apply(original, generator=g, parameters=params)
    repeated = aug.apply(original, generator=generator(88), parameters=params)
    assert torch.equal(output, repeated) and torch.equal(output[0], output[1])
    torch.testing.assert_close(output[0].sub(0.5).std(), torch.tensor(8 / 255), rtol=0.04, atol=0)
    assert not torch.equal(output[0, 0], output[0, 1])
    assert torch.equal(g.get_state(), before) and torch.equal(torch.get_rng_state(), global_before)
    assert torch.equal(original, saved) and params == supplied
    next_params = aug.sample_parameters(generator(13))
    assert not torch.equal(output, aug.apply(original, generator=g, parameters=next_params))


def test_jpeg_matches_per_frame_codec_with_shared_quality_and_no_future_dependence():
    from io import BytesIO
    import numpy as np
    from PIL import Image

    original = torch.rand((2, 3, 32, 32), generator=generator(17))
    original[1] = original[0]
    aug = WholeBodyVisualAugmenter(only("jpeg", quality=[50, 50]))
    params = aug.sample_parameters(generator(18))
    output = aug.apply(original, generator=generator(), parameters=params)
    with BytesIO() as stream:
        pixels = (original[0] * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
        Image.fromarray(pixels).save(stream, format="JPEG", quality=50, subsampling=0)
        stream.seek(0)
        with Image.open(stream) as decoded:
            reference = torch.from_numpy(np.array(decoded.convert("RGB"), copy=True)).permute(2, 0, 1).float() / 255
    torch.testing.assert_close(output[0], reference, rtol=0, atol=0)
    assert torch.equal(output[0], output[1]) and not torch.equal(output, original)
    changed_future = original.clone()
    changed_future[1] = 1 - original[1]
    other = aug.apply(changed_future, generator=generator(), parameters=params)
    assert torch.equal(other[0], output[0]) and not torch.equal(other[1], output[1])


def test_new_operation_ranges_and_actual_event_frequencies_include_global_gate():
    aug = WholeBodyVisualAugmenter()
    g = generator(33)
    noise_stds, qualities, jitter = [], [], []
    for _ in range(5000):
        params = aug.sample_parameters(g)
        if params["pixel_noise"] is not None:
            noise_stds.append(params["pixel_noise"]["std"])
        if params["jpeg"] is not None:
            qualities.append(params["jpeg"]["quality"])
        if params["jitter"] is not None:
            jitter.append(params["jitter"])
    assert 500 < len(noise_stds) < 750  # 50% global * 25% child
    assert 500 < len(qualities) < 750
    assert 850 < len(jitter) < 1150  # 50% global * 40% child
    assert all(4 / 255 <= std <= 8 / 255 for std in noise_stds)
    assert set(qualities) == set(range(50, 81))
    assert all(type(quality) is int for quality in qualities)
    assert all(abs(p["translation_x"]) <= 0.04 and abs(p["translation_y"]) <= 0.04
               and abs(p["rotation_degrees"]) <= 3 and 0.96 <= p["scale"] <= 1.04 for p in jitter)


def test_previous_five_operation_parameters_never_implicitly_enable_new_effects():
    original = camera_pair()
    params = dict.fromkeys(VISUAL_OPERATIONS[:5])
    g = generator(12)
    before = g.get_state().clone()
    output = WholeBodyVisualAugmenter().apply(original, generator=g, parameters=params)
    assert torch.equal(output, original) and torch.equal(before, g.get_state())


@pytest.mark.parametrize("operation,value", [
    ("pixel_noise", {"std": float("nan"), "seed": 0}),
    ("pixel_noise", {"std": 0.2, "seed": 0}),
    ("pixel_noise", {"std": 0.02, "seed": True}),
    ("pixel_noise", {"std": 0.02, "seed": -1}),
    ("jpeg", {"quality": True}), ("jpeg", {"quality": 50.0}),
    ("jpeg", {"quality": 0}), ("jpeg", {"quality": 101}),
])
def test_invalid_explicit_corruption_parameters_fail(operation, value):
    params = WholeBodyVisualAugmenter.empty_parameters()
    params[operation] = value
    with pytest.raises(ValueError, match="noise|JPEG"):
        WholeBodyVisualAugmenter().apply(camera_pair(), generator=generator(), parameters=params)
