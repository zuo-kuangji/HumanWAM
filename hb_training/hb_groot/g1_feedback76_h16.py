"""G1 feedback76 / H16 GR00T modality config (copy of HumanWAM configs/humanwam/groot/feedback76_h16.py @ c76527b).

The matching modality.json is generated per dataset from the recorded joint names (prepare_groot_data.py);
HumanWAM's feedback76_modality.json has right_arm at observation.state[29:36] (left-hand fingers), which is wrong
for our G1 exports (right arm is [22:29]).
"""
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

feedback76_h16 = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["ego_view", "chest"]),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["left_leg", "right_leg", "waist", "left_arm", "right_arm", "left_hand", "right_hand",
                       "projected_gravity"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["motion_token", "left_hand_joints", "right_hand_joints"],
        action_configs=[ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF,
                                     format=ActionFormat.DEFAULT) for _ in range(3)],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
}

register_modality_config(feedback76_h16, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
