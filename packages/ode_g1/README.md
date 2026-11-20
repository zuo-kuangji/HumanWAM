# HumanWAM GR00T ODE for Unitree G1

This package contains HumanWAM's local GR00T ODE implementation: task data
preparation, feature caching, the ODE head, training, offline prediction and the
native G1 deployment loop. Import it as `ode_g1`; changes are maintained and
committed in this repository.

The initial code came from ODE Policy's public `real-g1` component. See
[UPSTREAM.md](UPSTREAM.md) for the original revision and attribution,
[Apache-2.0](LICENSE-APACHE-2.0) for the license, and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for retained source notices.

See the [HumanWAM training and deployment guide](../../docs/policies/GROOT_ODE.md)
for environment setup, data preparation, training, inference and SONIC integration.
The ODE implementation is included here. The compatible NVIDIA Isaac-GR00T
framework, model weights and robot dependencies are installed separately as
specified in that guide.

After preparing the compatible environment, install from the HumanWAM root:

```bash
python -m pip install --no-deps -e './packages/ode_g1[data,vision]'
```

`--no-deps` assumes the platform dependencies are already installed. Python callers
can use the package directly:

```python
from ode_g1.loader import load_policy
from ode_g1.model import SonicODEConfig, install_sonic_ode
from ode_g1.runtime import ODEPolicy, RGBEncoder
```

The `ode-g1` command provides `init-config`, `inspect`, `prepare`, `features`,
`train`, `infer` and `deploy`. The HumanWAM `groot_ode` wrapper supplies its
configuration and GR00T policy RPC interface. Native `deploy` connects to robot
services; use `deploy --check` for configuration validation without those
connections.

Run the CPU tests from the HumanWAM root with a compatible Torch/NumPy environment:

```bash
PYTHONPATH=packages/ode_g1/src python -m pytest packages/ode_g1/tests -q
```

These tests use toy models and synthetic data. They do not establish pretrained
backbone, GPU or robot performance.
