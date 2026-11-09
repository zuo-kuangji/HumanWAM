"""Safe arrays plus JSON metadata, including native GR00T nested modalities."""

import numpy as np


def observation_from_npz(file, metadata=None):
    with np.load(file, allow_pickle=False) as arrays:
        if len(set(arrays.files)) != len(arrays.files):
            raise ValueError("Duplicate array names in NPZ archive")
        result = {}
        for key in arrays.files:
            if key.startswith(("video.", "state.", "language.")):
                group, name = key.split(".", 1)
                if group in result and not isinstance(result[group], dict):
                    raise ValueError("Conflicting native modality keys")
                result.setdefault(group, {})[name] = arrays[key]
            else:
                if key in result:
                    raise ValueError("Conflicting observation keys")
                result[key] = arrays[key]
    metadata = metadata or {}
    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a dict")
    for key, value in metadata.items():
        if key in result:
            raise ValueError("Duplicate observation metadata key: " + key)
        result[key] = value
    return result
