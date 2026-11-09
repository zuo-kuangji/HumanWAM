# GR00T N1.7 + SONIC integration component

Both upstream repositories were actually cloned for this integration:

| Repository | Pinned commit |
|---|---|
| `NVIDIA/Isaac-GR00T` | `7aa0bca50f1097f8b17f5c274c98264b72a1419f` |
| `NVlabs/GR00T-WholeBodyControl` | `1e1ec4f5f27e6c3e03048f693bd7c60cce04afda` |

Use those git revisions as submodules in HumanWAM. Preserve each upstream
license. Source code is Apache2; SONIC trained weights carry NVIDIA's model
license. This component does not download or publish trained weights.

## Native policy and protocol

`humanwam.backends.groot.build_policy(config)` imports the native
`gr00t.policy.gr00t_policy.Gr00tPolicy`. It requires `model_path`, accepts
`device`, and defaults to the `UNITREE_G1_SONIC` embodiment. The trained
checkpoint's processor supplies its own modality schema, normalization and
horizon. It must support the requested embodiment.

`predict(observation)` returns `{"actions": native_action_dict, "info": info}`.
It preserves native batched observations:

- `video`: camera-key dictionaries of uint8 RGB `[B,T,H,W,3]` arrays
- `state`: modality-key dictionaries of float32 `[B,T,D]` arrays
- `language`: native-key dictionaries of nested `list[list[str]]`

The adapter also implements native `get_action(observation, options=None)`
returning `(action_dict, info_dict)`, `reset(options=None)`, and
`get_modality_config()`. Thus `gr00t.policy.server_client.PolicyServer` may
serve the adapter using the actual SONIC ZeroMQ/msgpack-numpy protocol.
Do not label an HTTP endpoint as SONIC-compatible.

`build_train_command` emits the real native `examples/finetune.sh` argv and
environment. `build_serve_command` invokes the actual native
`gr00t/eval/run_gr00t_server.py`, binding localhost by default. Both are pure
command construction, never automatic training or robot execution.

## Stock SONIC and our measured-hand recipe are distinct

Stock `UNITREE_G1_SONIC` uses the upstream action keys `motion_token`,
`left_hand_joints`, `right_hand_joints` and horizon40. The stock hand interface
is 7D per side, so its common action space is78D and body/hand/gravity state46D.
These are not the measured Inspire6+6 values in the ImageWAM feedback76 recipe.

`configs/humanwam/groot/feedback76_h16.py` defines a separate native training configuration under
`NEW_EMBODIMENT`, with two cameras, H16, action64+6+6 and state29+6+6+3.
`configs/humanwam/groot/feedback76_modality.json` maps the raw LeRobot fields explicitly, including
same-row `observation.inspire_{left,right}_hand_state` action labels. It does
not reinterpret recorded 7D binary hand commands as six measured actuators.
Use this metadata in an independent dataset copy; do not overwrite raw data.
Training computes the corresponding native processor normalization for that
copy. The custom modality file alone is not a trained checkpoint.

Optional `output_format=feedback76` requires that separately trained
`NEW_EMBODIMENT` checkpoint, H16 metadata, and actual finite float32 outputs
with widths64/6/6. The adapter then also returns unbatched `[16,76]` under
`action`. It rejects stock 78D outputs. Passing this dense array to a physical
controller still needs the correct motion decoder and explicit hand mapping.
The stock SONIC7D command consumer must not receive feedback6D implicitly.

## Environment isolation

At the pinned revision, GR00T's native pyproject requires Python3.12,
torch2.9.0, torchvision0.24.0 and transformers4.57.3. ImageWAM's verified
training environment uses Python3.11, torch2.7.1 and transformers4.56.1;
keep separate per-policy environments rather than installing both into one.
SONIC training additionally has its own Isaac Lab environment; controller
deployment uses its own C++/Python toolchain. Keep upstream optional heavy
dependencies, FlashAttention and export packages out of HumanWAM's lightweight
registry import path.

## Existing experimental full-head FM / ODE packages

The workspace retains independent H16 GR00T FM and ODE experiments under
`eval_jobs/g1_groot_baseline_fm_h16_20261101_v1/code` and the corresponding ODE
directories. They include custom checkpoint protocols, cached VLM features
and strict source/weight manifests. They are not standard Hugging Face full
checkpoints accepted by `Gr00tPolicy`, and must not be silently loaded by this
native adapter. Their exact compatibility metadata and native cached-policy
loader must accompany any later migration. This integration preserves these
experiments read-only and does not replace a validated legacy deployment.

## Verification

### Existing local backbone and offline loading

The actual N1.7 GPU check exposed a transformers4.57.3 tokenizer metadata
lookup even with downloaded files and offline flags. Use a separate view to
point exactly the two native `model_name` fields at the existing local asset:

```bash
python scripts/humanwam/prepare_gr00t_offline.py \
  --checkpoint /path/to/GR00T-N1.7-3B-snapshot \
  --backbone /path/to/Cosmos-Reason2-2B-snapshot \
  --output /path/to/new-gr00t-offline-view
```

The HumanWAM package must be installed in the chosen runtime. The output must
be fresh and outside both caches. The helper creates a local alias under
`assets/nvidia/Cosmos-Reason2-2B`, copies only `config.json` and the native
processor config with those locator fields rewritten, and symlinks the
original weights, statistics and remaining files. Its JSON receipt records
both metadata SHA256 values and exact rewritten field paths. It performs no
downloads, weight conversion, parameter initialization or cache modification.

Set the policy's `model_path` to the new view. The resulting real GPU test
still loads the original trained weights and retains native strict checks;
the helper is not a test flag that skips model weights. The requested
embodiment must still exist in that checkpoint's processor metadata.

For local checkpoints, the adapter reads `processor_config.json` (or the native
`processor/processor_config.json` fallback) before importing or allocating the
model. The requested tag must exist in `processor_kwargs.modality_configs`.
Native aliases such as `REAL_G1` are resolved to their exact pinned enum values;
an absent `UNITREE_G1_SONIC` tag raises an error and never falls back to an EEF
profile. The generic N1.7 base can therefore be tested on a supported EEF
embodiment, but that is not evidence of a SONIC-trained policy.
HF repository identifiers are left to native loading without a metadata
network request. Checkpoints without local processor metadata also continue
to the native loader, which owns the remaining processor validation.

`packages/humanwam/tests/test_groot_adapter.py` exercises native construction,
observation and info round-trip, native wire method forwarding, rejected
horizon/hand-width conflicts, and command construction against synthetic
entrypoint fixtures. The emitted commands were also checked against the actual
pinned upstream scripts. The native engine is substituted at the boundary, so
these are not GPU checkpoint loads or robot rollout tests.

```bash
python -m pytest packages/humanwam/tests/test_groot_adapter.py
```
