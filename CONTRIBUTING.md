# Maintaining HumanWAM backends

Each family has one registry entry and one lazy adapter. Keep native model code
in a pinned submodule or the original ImageWAM package, with the applicable
upstream license. Do not copy caches, datasets, credentials, optimizer states or
model weights into Git.

When updating an upstream, change `backends.lock.json`, its gitlink and the
relevant patch together. Run bootstrap twice, run `humanwam doctor`, then the
lightweight interface suite. An expected patched submodule is locally dirty;
do not commit its working tree as a new undocumented upstream revision.

Every new adapter must document its cameras, current-state indexing, action
space, units, normalization owner, history padding and checkpoint horizon.
Check invalid input and mismatched checkpoints before copying parameters or
returning robot actions. A generic array width is not a robot embodiment.

For ImageWAM changes, run `tests/test_wholebody*.py` in the native ImageWAM
environment. For native model changes, run that upstream's relevant tests and
an actual-weight GPU smoke before claiming GPU inference compatibility.
Keep CPU adapter tests and actual model/robot validation separate in reports.

Use an integration branch and pull request for shared changes. Keep legacy
checkpoint formats readable through their original runtime; do not relabel
formats or silently change an existing trained horizon.
