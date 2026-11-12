"""A14 H2 (Sharpa hands) GR00T modality config with THREE cameras head + chest + wrist (user 2026-11-12; the third-person camera is never a model input). Otherwise identical to h2_body31_hand44_h16.py:, same protocol as the A18 Delta-flow run
(schema a18-h2-sharpa-body31-hand44-state78-chunk16-v1): cameras head+chest, state = 75 joints + projected
gravity, action = action.body_target (31) + action.hands (44), horizon 16, absolute targets."""
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

h2w3_body31_hand44_h16 = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["head", "chest", "wrist"]),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["left_leg", "right_leg", "waist", "left_arm", "right_arm", "head", "left_hand", "right_hand",
                       "projected_gravity"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["body_target", "left_hand_target", "right_hand_target"],
        action_configs=[ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF,
                                     format=ActionFormat.DEFAULT) for _ in range(3)],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
}

register_modality_config(h2w3_body31_hand44_h16, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
