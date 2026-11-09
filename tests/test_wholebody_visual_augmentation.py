"""CPU RGB augmentation checks; no model weights, dataset mutation or CUDA."""
import copy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from imagewam.datasets.wholebody_augmentation import (
    DEFAULT_VISUAL_AUGMENTATION, WholeBodyVisualAugmenter,
)


def generator(seed=42):
    return torch.Generator(device="cpu").manual_seed(seed)


def camera_pair():
    y,x=torch.meshgrid(torch.linspace(0,1,32),torch.linspace(0,1,32),indexing="ij")
    frame=torch.stack((x,y,(x+y)*.5))
    return torch.stack((frame,frame))


def only(operation,**overrides):
    cfg=copy.deepcopy(DEFAULT_VISUAL_AUGMENTATION)
    for key in ("exposure","overexposure","color","jitter","blur"):cfg[key]["probability"]=0
    cfg[operation].update(probability=1,**overrides)
    return cfg


def all_operations():
    cfg=copy.deepcopy(DEFAULT_VISUAL_AUGMENTATION)
    for key in ("exposure","overexposure","color","jitter","blur"):cfg[key]["probability"]=1
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
        assert float((output-original).abs().max())<.15


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
    for key in ("exposure","overexposure","color","jitter","blur"):zero[key]["probability"]=0
    for cfg in ({"enabled":False},zero):
        g=generator();before=g.get_state().clone()
        output=WholeBodyVisualAugmenter(cfg).apply(original,generator=g)
        assert torch.equal(output,original) and output.dtype==original.dtype
        assert output.data_ptr()!=original.data_ptr() and torch.equal(before,g.get_state())


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
