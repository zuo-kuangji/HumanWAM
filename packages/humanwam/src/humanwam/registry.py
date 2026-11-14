"""Backend families, with explicit input/output semantics and separate runtimes."""

import importlib
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Backend:
    name: str
    module: str
    commands: str
    source: str
    input_schema: str
    output_schema: str
    environment: str


BACKENDS = {
    "fastwam_sonic": Backend(
        "Fast-WAM G1 feedback76 + SONIC",
        "humanwam.backends.fastwam_sonic",
        "humanwam.backends.fastwam_sonic",
        "third_party/FastWAM",
        "GR00T feedback76 nested current video/state/language; state44, ego/chest",
        "separately trained Gaussian FM H16 motion64 + measured Inspire6/6; native SONIC wire",
        "Separate native Fast-WAM environment plus humanwam[sonic]",
    ),
    "fastwam": Backend(
        "Fast-WAM (native Gaussian)",
        "humanwam.backends.fastwam",
        "humanwam.backends.fastwam_commands",
        "third_party/FastWAM",
        "native LIBERO/RoboTwin current RGB views + state + instruction",
        "official task checkpoint action7/14; no implicit SONIC conversion",
        "Separate native Fast-WAM Wan2.2 environment",
    ),
    "imagewam_joint_reference": Backend(
        "ImageWAM Delta-flow joint reference",
        "humanwam.backends.imagewam_joint_reference",
        "humanwam.backends.imagewam_joint_reference",
        ".",
        "two current RGB + current state44 + past stride2 history16; v2 also needs current root and observation clock",
        "joint47 v2: prediction40/execution16 with root rotation6d; legacy joint41/47 via explicit config; SONIC v1.1",
        "ImageWAM FLUX.2 BF16 environment; distinct from motion76 checkpoints",
    ),
    "psi0": Backend(
        "Psi-Zero + SONIC",
        "humanwam.backends.psi0",
        "humanwam.backends.psi0_commands",
        "third_party/Psi0",
        "one current RGB + explicit raw state layout + instruction",
        "checkpoint-owned SONIC actions and normalization; not feedback76 by default",
        "Psi0 Python3.11/Torch2.7 environment; dedicated PSI SONIC fork",
    ),
    "groot_n17": Backend(
        "GR00T N1.7 + SONIC",
        "humanwam.backends.groot",
        "humanwam.backends.groot",
        "third_party/Isaac-GR00T",
        "native GR00T nested video/state/language",
        "checkpoint modality config; stock SONIC78 differs from measured-feedback76",
        "GR00T environment; see pinned upstream pyproject",
    ),
    "pi05": Backend(
        "pi0.5 / pi0.5 A2A",
        "humanwam.backends.pi05",
        "humanwam.backends.pi05_commands",
        "third_party/openpi",
        "native OpenPI; raw past_actions for A2A",
        "checkpoint action space; released LIBERO A2A patch is 7D/H10, not SONIC76",
        "OpenPI JAX or PyTorch environment, separate from GR00T/ImageWAM",
    ),
    "imagewam_delta": Backend(
        "ImageWAM Delta-flow",
        "humanwam.backends.imagewam_delta",
        "humanwam.backends.imagewam_delta",
        ".",
        "two RGB views + current measured state44 + strictly-past raw action76",
        "motion64 + measured Inspire6/6; new H16, historical H40 requires explicit config",
        "ImageWAM FLUX.2 BF16 CUDA environment",
    ),
    "dit4dit": Backend(
        "DiT4DiT",
        "humanwam.backends.dit4dit",
        "humanwam.backends.dit4dit_commands",
        "third_party/DiT4DiT",
        "native single current camera + language + normalized state",
        "native checkpoint-normalized actions; no implicit SONIC conversion",
        "DiT4DiT Cosmos environment, separate from other model stacks",
    ),
}


def describe_backends():
    return {key: asdict(value) for key, value in BACKENDS.items()}


def backend(name):
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend {name!r}; choose one of {', '.join(BACKENDS)}")
    return BACKENDS[name]


def build_policy(config):
    from .config import repository_root

    spec = backend(config["backend"])
    options = dict(config.get("options", {}))
    root = repository_root()
    source = Path(config.get("source_dir", root / spec.source))
    source = (root / source if not source.is_absolute() else source).resolve()
    if config["backend"] == "dit4dit":
        options.setdefault("upstream_root", str(source))
    options.setdefault("source_dir", str(source))
    return importlib.import_module(spec.module).build_policy(options)


def build_command(config, action, root):
    if action not in ("train", "serve"):
        raise ValueError("Only train and serve are native process commands")
    spec = backend(config["backend"])
    module = importlib.import_module(spec.commands)
    options = dict(config.get("options", {}))
    options.setdefault("python", config.get("python", "python3"))
    if config["backend"] in ("imagewam_delta", "imagewam_joint_reference", "fastwam", "fastwam_sonic"):
        options.setdefault("launcher_config", config.get("_config_path"))
    source = Path(config.get("source_dir", root / spec.source))
    if not source.is_absolute():
        source = root / source
    return getattr(module, f"build_{action}_command")(options, source)
