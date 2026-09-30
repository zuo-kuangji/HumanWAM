# π₀.₅ A2A (JAX LoRA)

This directory contains the A2A-only OpenPI patch used for π₀.₅ experiments on LIBERO. It adds a past-action flow source, endpoint consistency, a five-epoch JAX LoRA configuration, and one-step evaluation support. It does **not** include RTC, checkpoints, datasets, or credentials. The ImageWAM code on `main` is unchanged.

## Apply to OpenPI

The patch is based on [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) commit `215abfb` (`docs(droid): fix config search instruction (#1023)`). Apply it to a clean checkout of that commit:

```bash
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi
git checkout 215abfb
curl -L https://raw.githubusercontent.com/zuo-kuangji/HumanWAM/pi05-a2a/experiments/pi05_a2a/openpi-a2a.patch -o openpi-a2a.patch
git apply --check openpi-a2a.patch
git apply openpi-a2a.patch
uv sync
```

The patch adds `pi05_libero_a2a_jax_lora_warmup_cosine`. It uses LoRA in both the vision-language model and action expert, a 10-action horizon, global batch size 384, five epochs, warmup-cosine learning rate, a past-action source with noise standard deviation 0.5 on the six continuous coordinates, and endpoint-consistency weight 0.5. The gripper coordinate is not noised. The action loss masks padding and dimensions beyond the seven-dimensional LIBERO action space.

## Train

Prepare the `physical-intelligence/libero` LeRobot dataset according to upstream OpenPI's [LIBERO instructions](https://github.com/Physical-Intelligence/openpi/tree/main/examples/libero), then compute normalization statistics for the A2A configuration:

```bash
uv run scripts/compute_libero_norm_stats_fast.py --config-name pi05_libero_a2a_jax_lora_warmup_cosine --dataset-root /path/to/lerobot/physical-intelligence/libero
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_a2a_jax_lora_warmup_cosine --exp-name=my_a2a_run
```

The stats are written under `assets/pi05_libero_a2a/physical-intelligence/libero/` and must accompany a released checkpoint. Training uses the official π₀.₅ base weights specified in the OpenPI config. The run saves an epoch checkpoint (normally 712, 1424, 2136, 2848, and 3560 optimizer steps for this dataset and batch size); check the actual loader length if the dataset differs.

## Evaluate on standard LIBERO

Start a policy server with one flow step, then evaluate each of the four standard suites with the same checkpoint. In a second terminal, for example:

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero_a2a_jax_lora_warmup_cosine --policy.dir=/path/to/checkpoint/3560 --policy.num-steps=1
```

```bash
cd examples/libero
python main.py --task-suite-name=libero_spatial --num-trials-per-task=25 --use-past-actions --seed=42 --result-out-path=/path/to/results/spatial.json
```

Repeat with `libero_object`, `libero_goal`, and `libero_10` (the Long suite). The four standard suites each contain 10 tasks; 25 trials per task gives 1,000 rollouts overall. Do not report a partial suite as a 1,000-rollout result. The client sends the last 10 executed actions; the policy normalizes them with the same action statistics used in training. The policy splits its JAX RNG on each call, so source noise is newly sampled even when the evaluation seed is fixed.

## Tests and provenance

```bash
JAX_PLATFORMS=cpu uv run pytest -q tests/transforms_a2a_test.py tests/policy_a2a_norm_test.py tests/test_pi05_jax_lora.py
```

This patch was extracted from the NTU-ZUO A2A experimental workspace, with unrelated RTC modifications excluded. It was checked to apply cleanly to the pinned upstream source, and the nine tests above passed on that server. This code-only branch does not publish the LoRA weights; point `--policy.dir` to a separately distributed checkpoint that also contains its normalization assets.

OpenPI's upstream license remains applicable to the patched OpenPI source.
