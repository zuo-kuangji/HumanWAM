# Psi-Zero without neck channels — 2026-11-10

HumanWAM defaults to `g1_neckless43_action78` in both Psi-Zero training and
serving templates. This native layout uses body joint positions29 + hands14
as state43, and motion token64 + hands14 as action78. It has no neck channels
or gravity3 input. It is separate from ImageWAM's state44/action76 recipe.

## Public checkpoint evidence

The checked repository is `USC-PSI-Lab/psi-model`, revision
`74d7abaa61b51dfe4e95f539a83738c276ce27b2`, run
`psi0/simple-checkpoints/sonic-wbcbox.neckle.flow1000.cosine.lr1.0e-04.b256.gpus8.2608260223`.
Its `argv.txt` names the same native `finetune_sonic_psi0_config` module used by
the maintained SONIC server. The saved configuration declares state43,
action78, H30, observation horizon1, one egocentric camera, 240x320 resize/crop,
and embeds matching state/action normalization arrays. This is an official
carry-box task model, not a newly trained model for HumanWAM's other tasks.

| Public metadata | Bytes | SHA256 of downloaded metadata |
|---|---:|---|
| `run_config.json` | 15154 | `075149ab6417a5a3bfab20d7b7d481a2c43eef7a4d908c7632bf52eac245e02e` |
| `argv.txt` | 2109 | `076fca5c6dbc8ca64edc01e59fec7ca1cdab1dcd3fee073b966e6e640aecec57` |
| `dataset_statistics.json` | 17395 | `c70582a8896caf40751bc6dfbed4a91ae97ffc999c865c811ea858dca040cbbc` |

A range request read the safetensors JSON header without downloading the
6,254,227,440-byte weight file. It confirms:

| Tensor | Shape |
|---|---|
| `action_header.obs_proj._obs_proc.1.weight` | `[1536,43]` |
| `action_header.action_proj_out.linear.weight` | `[78,1536]` |
| `action_header.action_proj_in.dec_pos` | `[30,1536]` |

The repository declares the complete weight SHA256 as
`b60c11f84887fc8b6307851c23dc9169cef4c7e292fcaaed706d6254952bc711`.
This is a remote LFS declaration, not an independently verified full-file hash.
The reduced regression fixture preserves actual public values and records
the source revision and complete run-config digest.

The example uses SONIC v1 `action.body_token`. It must not be presented as a
SONIC v1.1 token model. Its saved `model.rtc=false`; enabling server RTC uses
test-time guidance rather than establishing RTC training. The native loader
and metadata paths were inspected at Psi0 commit
`cda3fdfafe47cb6fa75400d013de1bd36e3cea05`.

## Software checks

Regression checks cover default and explicit no-neck profiles, actual public
43/78 metadata, rejected 45/80 checkpoints and neck fields, normalization widths,
loaded model state width, and preprocessed `[1,1,43]` state. Explicit
`checkpoint_native` keeps historical native loading separate from the default.

The training builder binds model dimensions, both repack/field dimensions,
matching modality/statistics keys, and prediction/execution windows. It rejects
conflicting or repeated managed flags, including legacy neck keys, and accepts
an explicitly declared already assembled78D action column. Tests preserve the
original native-argument file and do not start training or robot control.

The native CPU environment passed 172 policy/runtime tests without skips.
The focused Psi-Zero suite passed 22 tests, including the shortened-horizon
execution-window check. Local targeted lint, compilation and diff checks passed.
Native logs are under
`/mnt/data0/aaa/humanwam_augmentation_qa_20261110/psi0_neckless_native`.

These checks do not load complete Psi-Zero weights on a GPU or establish
closed-loop robot performance. Existing training jobs, checkpoints and raw
datasets were not modified. Download and initialization instructions are in
[the Psi-Zero guide](PSI0.md) and [model initialization](../MODEL_INITIALIZATION.md).
