"""Explicit measured-Inspire recipe, separate from stock SONIC78/H40.

Load this only in an isolated GR00T training process with --embodiment-tag
NEW_EMBODIMENT. The raw dataset needs the matching modality.json mapping.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

feedback76_h16 = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["ego_view", "chest"]),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_leg",
            "right_leg",
            "waist",
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
            "projected_gravity",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["motion_token", "left_hand_joints", "right_hand_joints"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
            for _ in range(3)
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0], modality_keys=["annotation.human.task_description"]
    ),
}

register_modality_config(feedback76_h16, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
