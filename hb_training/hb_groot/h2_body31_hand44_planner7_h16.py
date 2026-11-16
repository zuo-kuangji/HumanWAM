"""A14 v2 H2 (Sharpa hands) GR00T modality config (H2 training guide hb_training/docs/h2/a14_training_guide.md 4.1):
cameras head + chest, state78 = 75 joints + projected gravity, action82 = action.body_target (31) + action.hands (44)
+ action.planner (7, the SONIC planner command), horizon 16, absolute targets. Identical to h2_body31_hand44_h16.py
except the added `planner` action modality."""
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

h2_body31_hand44_planner7_h16 = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["head", "chest"]),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["left_leg", "right_leg", "waist", "left_arm", "right_arm", "head", "left_hand", "right_hand",
                       "projected_gravity"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["body_target", "left_hand_target", "right_hand_target", "planner"],
        action_configs=[ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF,
                                     format=ActionFormat.DEFAULT) for _ in range(4)],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
}

register_modality_config(h2_body31_hand44_planner7_h16, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
