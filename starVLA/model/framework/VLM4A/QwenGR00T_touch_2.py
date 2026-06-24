# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
QwenGR00T_touch_2 Framework

Qwen-VL hidden states + TouchAngleEncoder tokens are concatenated as
cross-attention condition for the GR00T flow-matching action head.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.framework.VLM4A.QwenGR00T_touch import (
    QwenGR00TTouchDefaultConfig,
    Qwen_GR00T_touch,
)
from starVLA.model.modules.action_model.GR00T_ActionHeader import (
    FlowmatchingActionHead,
    get_action_model,
)
from starVLA.model.modules.action_model.GR00T_ActionHeader_touch import TouchAngleEncoder
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass
class QwenGR00TTouch2DefaultConfig(QwenGR00TTouchDefaultConfig):
    name: str = "QwenGR00T_touch_2"


@FRAMEWORK_REGISTRY.register("QwenGR00T_touch_2")
class Qwen_GR00T_touch_2(Qwen_GR00T_touch):
    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        baseframework.__init__(self)
        self.config = merge_framework_config(QwenGR00TTouch2DefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        vl_hidden_dim = self.qwen_vl_interface.model.config.hidden_size
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = vl_hidden_dim

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

        touch_cfg = self.config.framework.action_model.get("touch", {})
        enabled = bool(touch_cfg.get("enabled", False))
        checkpoint = touch_cfg.get("checkpoint_encoder", None)
        self.require_touch = bool(touch_cfg.get("require_touch", enabled or checkpoint))
        self.touch_encoder = (
            TouchAngleEncoder(touch_cfg, target_dim=vl_hidden_dim)
            if enabled or checkpoint
            else None
        )

    def _encode_touch_condition(
        self,
        touch: dict[str, torch.Tensor] | None,
        target_dtype: torch.dtype,
        target_device: torch.device,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self.touch_encoder is None:
            if self.require_touch:
                raise RuntimeError("QwenGR00T_touch_2 requires touch, but touch encoder is disabled.")
            return None
        if touch is None:
            if self.require_touch:
                raise KeyError("QwenGR00T_touch_2 requires touch dict with `joint_contact`.")
            return None
        return self.touch_encoder(touch, target_dtype=target_dtype, state=state).to(device=target_device)

    @staticmethod
    def _concat_touch_to_vlm(vl_embs, touch_features, attention_mask=None):
        if touch_features is None:
            return vl_embs, attention_mask

        vl_embs = torch.cat((vl_embs, touch_features), dim=1)
        if attention_mask is not None:
            touch_mask = torch.ones(
                touch_features.shape[:2],
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat((attention_mask, touch_mask), dim=1)
        return vl_embs, attention_mask

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        backbone_attention_mask = qwen_inputs.get("attention_mask", None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            state_tensor = None
            if state is not None:
                state_tensor = torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype)

            touch = self._extract_touch_inputs(examples, device=last_hidden.device)
            touch_features = self._encode_touch_condition(
                touch,
                target_dtype=last_hidden.dtype,
                target_device=last_hidden.device,
                state=state_tensor,
            )
            last_hidden, backbone_attention_mask = self._concat_touch_to_vlm(
                last_hidden,
                touch_features,
                backbone_attention_mask,
            )

            actions = torch.tensor(np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype)
            actions_target = actions[:, -self.action_horizon :, :]

            repeated_diffusion_steps = (
                self.config.framework.action_model.get("repeated_diffusion_steps", 4)
                if self.config and hasattr(self.config, "framework")
                else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
            if backbone_attention_mask is not None:
                backbone_attention_mask = backbone_attention_mask.repeat(repeated_diffusion_steps, 1).to(
                    dtype=torch.bool
                )

            state_repeated = state_tensor.repeat(repeated_diffusion_steps, 1, 1) if state_tensor is not None else None
            action_loss = self.action_model(
                last_hidden_repeated,
                actions_target_repeated,
                state_repeated,
                encoder_attention_mask=backbone_attention_mask,
            )

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        backbone_attention_mask = qwen_inputs.get("attention_mask", None)
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]

        state_tensor = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None
            else None
        )

        with torch.autocast("cuda", dtype=torch.float32):
            touch = self._extract_touch_inputs(examples, device=last_hidden.device)
            touch_features = self._encode_touch_condition(
                touch,
                target_dtype=last_hidden.dtype,
                target_device=last_hidden.device,
                state=state_tensor,
            )
            last_hidden, backbone_attention_mask = self._concat_touch_to_vlm(
                last_hidden,
                touch_features,
                backbone_attention_mask,
            )
            pred_actions = self.action_model.predict_action(
                last_hidden,
                state_tensor,
                encoder_attention_mask=backbone_attention_mask,
            )

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}
