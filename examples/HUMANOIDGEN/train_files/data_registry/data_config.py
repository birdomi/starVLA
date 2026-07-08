"""HumanoidGen benchmark data config, embodiment tags, and mixtures."""

import json

import numpy as np

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except Exception:
            pass
    return getattr(cfg, key, default)


class HumanoidGenTaskFilteredDataset(LeRobotSingleDataset):
    def __init__(self, *args, task_index=None, task_name=None, **kwargs):
        self._filter_task_index = None if task_index in (None, "") else int(task_index)
        self._filter_task_name = None if task_name in (None, "") else str(task_name)
        super().__init__(*args, **kwargs)
        self._apply_task_filter()

    def _resolve_task_names(self):
        if self._filter_task_name:
            return {self._filter_task_name}
        if self._filter_task_index is None:
            return None
        task_row = self.tasks.loc[self._filter_task_index]
        return {str(task_row["task"])}

    def _apply_task_filter(self):
        wanted_tasks = self._resolve_task_names()
        if not wanted_tasks:
            return

        episodes_path = self.dataset_path / "meta" / "episodes.jsonl"
        kept_episode_ids = []
        with open(episodes_path, "r", encoding="utf-8") as f:
            for line in f:
                episode = json.loads(line)
                tasks = {str(task) for task in episode.get("tasks", [])}
                if tasks & wanted_tasks:
                    kept_episode_ids.append(int(episode["episode_index"]))

        kept = set(kept_episode_ids)
        keep_mask = np.array([int(ep) in kept for ep in self._trajectory_ids], dtype=bool)
        self._trajectory_ids = self._trajectory_ids[keep_mask]
        self._trajectory_lengths = self._trajectory_lengths[keep_mask]
        self._all_steps = [step for step in self._all_steps if int(step[0]) in kept]

        if not self._all_steps:
            raise ValueError(f"HumanoidGen task filter matched no data: {sorted(wanted_tasks)}")

        print(
            f"[HumanoidGen] task filter kept episodes={len(self._trajectory_ids)} "
            f"steps={len(self._all_steps)} tasks={sorted(wanted_tasks)}"
        )


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

    def make_dataset(self, dataset_path, modality_configs, transforms, embodiment_tag, video_backend,
                     delete_pause_frame=False, data_cfg=None, **kwargs):
        return HumanoidGenTaskFilteredDataset(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
            delete_pause_frame=delete_pause_frame,
            data_cfg=data_cfg,
            robot_data_config=self,
            task_index=_cfg_get(data_cfg, "task_index", None),
            task_name=_cfg_get(data_cfg, "task_name", None),
        )


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
