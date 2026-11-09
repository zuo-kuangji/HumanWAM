# DiT4DiT backend

Official upstream: <https://github.com/Mondo-Robotics/DiT4DiT>

Pinned commit: `66a6f3a12e2c8740157c1e478795952c040d31dd`.
The official project page <https://dit4dit.github.io/> links this repository.
Code license is MIT. Cosmos model weights and other dependencies retain their
own licenses and access requirements.

This component calls the actual upstream APIs. It does not reimplement the
video/action networks. GPU dependencies remain in a separate DiT4DiT environment,
because its CUDA/Torch/Diffusers/Cosmos stack differs from GR00T, OpenPI and
ImageWAM. The upstream README uses Python>=3.10, Torch2.7.0 and CUDA12.8;
`requirements.txt` is the upstream lock-style package list. No GPU dependencies,
weights or robot drivers were installed during integration.

## Inference contract

`build_policy(options)` constructs the native
`DiT4DiT.model.framework.base_framework.baseframework.from_pretrained(checkpoint)`.
The checkpoint must be a local `RUN/checkpoints/*.pt` with its original
`RUN/config.yaml` and `RUN/dataset_statistics.json`, as required by upstream.
The native loader strictly matches the model state dict.

Required options are `upstream_root` and `checkpoint`. Optional `action_dim`,
`state_dim` and `horizon` are assertions against the checkpoint configuration,
never overrides of learned architecture. `device` defaults to`cuda`;
`use_bf16` follows upstream's opt-in behavior.

`predict(observation)` takes:

- `images`: exactly one current uint8 HWC RGB image in a list or a dense
  `[1,H,W,3]` NumPy array from the common NPZ CLI. It is resized using the
  checkpoint training resolution. Native Cosmos interprets image1 onward as
  **future video frames**, not extra cameras; additional cameras are rejected.
- `prompt`: a nonempty instruction string.
- `state_normalized`: finite float `[state_dim]` or`[1,state_dim]`, already
  encoded, normalized and padded by the checkpoint-specific data pipeline.

The returned dictionary contains `actions`, `normalized:true`,
`action_space:dit4dit_native_normalized`, `horizon`, and native shape/provenance.
It is suitable for offline prediction or a service that explicitly understands
these normalized units. It is **not** a raw robot command. No padding removal,
gripper reinterpretation, action unnormalization or SONIC conversion occurs.
Raw state, action history, wrong schema/dimensions and nonfinite values fail
before reaching the native backend. `reset()` is a no-op because this pinned
model has no recurrent inference state.

The official realG1 config uses action32/state64, while its decoupledWBC config
uses action36/state64. These are different from our action76
`motion64,left_measured6,right_measured6` contract. The nativeDiT4DiT adapter
must not be connected to the SONIC76 deployment chain until a separately
implemented and validated training/data/controller mapping exists.

## Native launch commands

`humanwam.backends.dit4dit_commands.build_train_command(options, source_dir)` returns argv/cwd/env for
`python -m accelerate.commands.launch ... DiT4DiT/training/train.py --config_yaml`.
The `native_config` option must name an existing native YAML. Optional
`overrides` maps actual upstream dot-keys to scalar values; structured values
belong in the YAML. `num_processes` defaults to8. Working directory is the
upstream root because the native trainer reads relative DeepSpeed paths.
No shell interpolation is used.

Apply `patches/dit4dit-cosmos-log.patch` to this pinned checkout during
bootstrap. It corrects one upstream training logging statement that reads the
absent`framework.qwenvl.base_vlm` key even though the shipped configs contain
`framework.cosmos25.base_model`. The change only fixes that log message;
model/optimizer/loss definitions are untouched. `git apply --check` passed.

`humanwam.backends.dit4dit_commands.build_serve_command(options, source_dir)` calls the real native
`python -m deployment.model_server.server_policy --ckpt_path ...` WebSocket
server. Port defaults to10093 and optional`use_bf16` maps to its actual flag.
That native server binds all interfaces and speaks upstream's own protocol;
it returns normalized actions. It is separate from a generic HumanWAM server.
Starting the service does not command the robot or enable a SONIC controller.

## Verification

Run the component boundary tests with a Python containing NumPy and Pillow:

```bash
python -m pytest packages/humanwam/tests/test_dit4dit_adapter.py
```

The CPU tests inject a native-shaped backend. They check the actual adapter
boundary, units, shape/schema mismatches, checkpoint companions, camera
semantics, native command construction, and preservation of a path containing
spaces. They are not GPU numerical, checkpoint loading or robot rollout tests.
No trained weights or task-success claim is included in this integration.

### Actual release metadata and GPU prerequisites

The official public `mondo-robotics/dit4dit-model` LIBERO checkpoint at
revision `46237aebd3df427fcfb6a8ddc5ba5a3ab7b04a44` declares action8,
state16 and horizon8. Its `config.yaml` omits `datasets.vla_data.image_size`;
the adapter preserves the supplied current-image resolution when that key
is absent, matching native inference. It resizes only when a checkpoint
explicitly records `image_size`. A small copy of this public config is kept
as a regression fixture; no model weight is included.

That checkpoint is 21.28GB. Native model construction also loads the
21.23GB `nvidia/Cosmos-Predict2.5-2B` pipeline at revision
`diffusers/base/post-trained`; the unmodified path therefore needs about
42.51GB of downloads. The published config's `/path/to/Cosmos-Predict2.5-2B`
must be relocated in a private runtime copy to the real local backbone.
The existing account received HTTP403 for this gated backbone during the
2026-11-09 access check; terms/access were not changed and no large model
downloads were started. Actual GPU inference for DiT4DiT remains unverified.

The existing GR00T `diffusers0.35.1` does not provide
`Cosmos2_5_PredictBasePipeline`. Use a separate DiT4DiT environment with its
pinned diffusers commit `3996788b602eaae4da41a1d45726b62e662b73cf`, rather
than replacing GR00T's dependency. The inference path additionally needs the
normal PyTorch/torchvision, transformers, accelerate, OmegaConf, NumPy and
Pillow dependencies declared upstream; full training requirements are broader.
