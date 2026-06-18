"""HumanoidGen benchmark data config, embodiment tags, and mixtures."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform


class HumanoidGenGR1DataConfig:
    embodiment_tag = EmbodimentTag.GR1
    video_keys = ["video.head_camera", "video.left_wrist_camera", "video.right_wrist_camera"]
    state_keys = ["state.joints"]
    action_keys = ["action.joints"]
    tactile_keys = ["tactile.left_hand", "tactile.right_hand"]
    tactile_fingertip_indices = [2, 4, 6, 8, 10]
    state_key_dims = {"state.joints": 26}
    action_key_dims = {"action.joints": 26}
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
            "tactile": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.tactile_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
        ])


class HumanoidGenGR1CoarseActionDataConfig(HumanoidGenGR1DataConfig):
    language_keys = ["annotation.human.coarse_action"]


ROBOT_TYPE_CONFIG_MAP = {
    "humanoidgen_gr1": HumanoidGenGR1DataConfig(),
    "humanoidgen_gr1_coarse": HumanoidGenGR1CoarseActionDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {}

DATASET_NAMED_MIXTURES = {
    "humanoidgen_all": [
        (".", 1.0, "humanoidgen_gr1"),
    ],
    "humanoidgen_all_coarse": [
        (".", 1.0, "humanoidgen_gr1_coarse"),
    ],
}
