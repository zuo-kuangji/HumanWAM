"""G1 joint47 / H40 stride-2 GR00T modality config (2026-11-18 user: joint-angle action experiment, aligned with
HumanWAM joint47 v2): action rows at source offsets 0,2,...,78 of body29 | Inspire left6 | right6 | root rotation6D
(current-heading relative, hb_j47.py). State44 and cameras as feedback76."""
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

g1_joint47_h40 = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["ego_view", "chest"]),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["left_leg", "right_leg", "waist", "left_arm", "right_arm", "left_hand", "right_hand",
                       "projected_gravity"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 80, 2)),
        modality_keys=["body29", "left_hand_target", "right_hand_target", "root_rot6d"],
        action_configs=[ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF,
                                     format=ActionFormat.DEFAULT) for _ in range(4)],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
}

register_modality_config(g1_joint47_h40, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
