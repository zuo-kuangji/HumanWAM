# Origin and local maintenance

The ODE G1 implementation is maintained as source code in HumanWAM under
`packages/ode_g1`. It is not an ODE-POLICY Git submodule or a runtime checkout of
that project. Subsequent implementation changes are versioned with HumanWAM.

## Initial source

The initial import copied the tracked files from the `groot/` component of:

- Project: [NTUMARS/ODE-POLICY](https://github.com/NTUMARS/ODE-POLICY)
- Source branch: `real-g1`
- Source commit: `fd8bec470faf7793702533358867db02819e8f27`
- Source tree: [groot/ at that commit](https://github.com/NTUMARS/ODE-POLICY/tree/fd8bec470faf7793702533358867db02819e8f27/groot)
- Original training and deployment guide: [fixed upstream guide](https://github.com/NTUMARS/ODE-POLICY/blob/fd8bec470faf7793702533358867db02819e8f27/docs/deploy_groot/README.md)

This source revision records attribution and the starting point for comparison.
It does not constrain later local modifications or require an external
ODE-POLICY checkout when running HumanWAM. Runtime source, tests and original
license notices were included in the initial import. Documentation paths were
updated for the HumanWAM repository.

## Local changes after import

The HumanWAM import updates the package README and documentation links, adds this
source record, and describes the package as a locally maintained implementation.
Package metadata and distribution manifests include this provenance document
alongside the retained license and third-party notices. The missing-GR00T-runtime
error in `src/ode_g1/loader.py` points to HumanWAM's ODE guide instead of the
original repository's documentation path; that file identifies the local change.
These documentation, packaging and error-message changes do not alter model
computation. The HumanWAM adapter and command builders import this local package
and keep the compatible Isaac-GR00T framework as a separate dependency.

## Attribution and licensing

The local implementation is adapted from Jindou Jia et al.'s ODE Policy G1
component and builds on NVIDIA Isaac-GR00T and SONIC. Preserve the original
copyright and source notices when modifying or redistributing these files.
The original [Apache-2.0 license](LICENSE-APACHE-2.0) and
[third-party notices](THIRD_PARTY_NOTICES.md) are retained in this directory.
Use the linked ODE Policy project for the authors' publication and citation
information; the local integration does not claim authorship of that work.

## Separate framework and assets

NVIDIA Isaac-GR00T remains a framework dependency, as with HumanWAM's other
model integrations. This ODE implementation uses the compatible source revision
`23ace64f17aa5015259b8609d371eb61a357c776`, prepared separately at
`third_party/Isaac-GR00T-ODE`. Its source revision is distinct from the ODE code's
origin and from the GR00T/Cosmos model revisions.

Model weights, processors, SONIC controller assets, datasets, camera services
and hardware drivers are separate artifacts with their own terms. They are not
included in this package. Installation and local operation are documented in
[HumanWAM's ODE guide](../../docs/policies/GROOT_ODE.md).
