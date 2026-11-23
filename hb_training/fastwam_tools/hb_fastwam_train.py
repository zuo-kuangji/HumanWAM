"""2026-11-16: native FastWAM Gaussian-FM training on our benchmark interface, with the HB augmentation v2.
Profiles (HB_FW_PROFILE): g1 = state44 / action76 / cameras ego_view + chest (HumanWAM fastwam_sonic recipe);
h2 = state78 / action75 / head + chest; h2p = state78 / action82 (+ planner 7) / head + chest. Fixed recipe of HumanWAM
fastwam_sonic_training.compose_sonic_config: 17 raw samples = 16 actions + 5 video frames (stride 4), 224 x 448 side by
side, min/max statistics, absolute labels, action scheduler shift 1.0, original Wan files.
Arguments: key=value Hydra overrides (JSON values parsed). Env: FASTWAM_SRC, DIFFSYNTH_MODEL_BASE_PATH, HB_AUG_DIR.
HB_MODEL_YAML=<path> writes the resolved model config; HB_COMPOSE_ONLY=1 prints the config; HB_DATA_SAMPLE=1 reads one
augmented training sample on CPU; HB_FW_AUG=0 disables the augmentation (benchmark only)."""
import json, os, sys
from pathlib import Path

ROOT = Path(os.environ["FASTWAM_SRC"]).resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
TASK = "libero_uncond_2cam224_1e-4"
PROFILES = {"g1": (("ego_view", "chest"), 44, 76, 29), "h2": (("head", "chest"), 78, 75, 31),
            "h2p": (("head", "chest"), 78, 82, 31)}
CAMS, SDIM, ADIM, NJ = PROFILES[os.environ.get("HB_FW_PROFILE", "g1")]
os.environ.setdefault("HB_FW_JOINTS", str(NJ))
# 2026-11-23 user (REPRODUCIBILITY.md section 3): seed the augmentation stream (hb_fastwam_aug._rng) with the training seed
os.environ.setdefault("HB_AUG_SEED", next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("seed=")), "42"))
print(f"HB_AUG_SEED={os.environ['HB_AUG_SEED']} (seeded augmentation stream, REPRODUCIBILITY.md section 3)", file=sys.stderr, flush=True)
SHAPE_META = {
    "images": [{"key": n, "raw_shape": [3, 224, 224], "shape": [3, 224, 224]} for n in CAMS],
    "action": [{"key": "default", "raw_shape": ADIM, "shape": ADIM}],
    "state": [{"key": "default", "raw_shape": SDIM, "shape": SDIM}],
}
FIXED = {
    "data.train.shape_meta": SHAPE_META, "data.train.num_frames": 17, "data.train.global_sample_stride": 1,
    "data.train.action_video_freq_ratio": 4, "data.train.video_size": [224, 448], "data.train.concat_multi_camera": "horizontal",
    "data.train.processor.action_output_dim": ADIM, "data.train.processor.proprio_output_dim": SDIM,
    "data.train.processor.delta_action_dim_mask": {"default": [False] * ADIM}, "data.train.processor.action_state_transforms": None,
    "data.train.processor.use_stepwise_action_norm": False, "data.train.processor.norm_default_mode": "min/max",
    "data.train.processor.norm_exception_mode": None, "model.redirect_common_files": False,
    "model.action_scheduler.train_shift": 1.0, "model.action_scheduler.infer_shift": 1.0,
}


def build(overrides):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from fastwam.utils.config_resolvers import register_default_resolvers
    register_default_resolvers()
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=["task=" + TASK])
    for k, v in {**FIXED, **overrides}.items():
        OmegaConf.update(cfg, k, v, merge=False)
    for k, v in FIXED.items():
        a = OmegaConf.select(cfg, k)
        if OmegaConf.is_config(a):
            a = OmegaConf.to_container(a, resolve=True)
        assert a == v, (k, a, v)
    return cfg


def parse(argv):
    out = {}
    for a in argv:
        k, v = a.split("=", 1)
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


if __name__ == "__main__":
    from omegaconf import OmegaConf
    cfg = build(parse(sys.argv[1:]))
    if os.environ.get("HB_MODEL_YAML"):
        Path(os.environ["HB_MODEL_YAML"]).write_text(OmegaConf.to_yaml(cfg.model, resolve=True))
        print("HBFW_MODEL_YAML", os.environ["HB_MODEL_YAML"], flush=True)
        raise SystemExit(0)
    if os.environ.get("HB_COMPOSE_ONLY"):
        print(OmegaConf.to_yaml(cfg, resolve=True))
        raise SystemExit(0)
    import fastwam.runtime as native
    assert Path(native.__file__).resolve() == ROOT / "src/fastwam/runtime.py", native.__file__
    if os.environ.get("HB_FW_AUG", "1") != "0":
        import hb_fastwam_aug
        hb_fastwam_aug.install()
    print("HBFW_RECIPE", json.dumps(dict(profile=os.environ.get("HB_FW_PROFILE", "g1"), cameras=CAMS, state_dim=SDIM,
                                         action_dim=ADIM, perturbed_joints=int(os.environ["HB_FW_JOINTS"]),
                                         aug=os.environ.get("HB_FW_AUG", "1") != "0",
                                         batch_size=cfg.batch_size, max_steps=cfg.max_steps, lr=cfg.learning_rate)), flush=True)
    if os.environ.get("HB_DATA_SAMPLE"):  # CPU check: build the dataset and read augmented samples, no model
        from omegaconf import open_dict
        from fastwam.utils import misc
        misc.register_work_dir(os.environ.get("HB_SAMPLE_WORKDIR", "/tmp/hbfw-sample"))
        tr, _ = native.build_datasets(cfg.data)
        s = tr[len(tr) // 2]

        def desc(x):
            if isinstance(x, dict):
                return {k: desc(v) for k, v in x.items()}
            if hasattr(x, "shape"):
                return [tuple(x.shape), str(x.dtype), round(float(x.float().min()), 3), round(float(x.float().max()), 3)]
            return type(x).__name__
        print("HBFW_SAMPLE", len(tr), json.dumps(desc(s), default=str), flush=True)
        raise SystemExit(0)
    if os.environ.get("HB_INLOOP_EVERY", "250") != "0":  # 2026-11-16: benchmark action nMSE every 250 steps
        try:
            import hb_inloop_fastwam
            hb_inloop_fastwam.install()
        except Exception as e:  # noqa: BLE001 - never block training
            print("HB_INLOOP fastwam install failed:", repr(e), flush=True)
    native.run_training(cfg)
