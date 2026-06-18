# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025].
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-GR00T Touch Framework
Qwen-VL + AngleTransformer touch tokens + Flow-matching head
Flow-matching header is copyright from GR00T N1.5,
"""

import sys
from pathlib import Path

# Add workspace root to Python path if not already there
_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_ActionHeader_touch import (
    FlowmatchingActionHeadTouch as FlowmatchingActionHead,
    get_action_model,
)
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenGR00T_touch
#  - Documents every framework-level parameter with type + description
#  - YAML values override these defaults; extra YAML keys are preserved
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenGR00TTouchDefaultConfig:
    """QwenGR00T_touch framework default parameters.

    All fields can be overridden by the corresponding key in the YAML
    ``framework:`` section.  Extra YAML keys not listed here are kept
    as-is (Config-as-API flexibility).
    """

    # --- Registry identifier ---
    name: str = "QwenGR00T_touch"

    # === VLM backbone (Qwen2.5-VL / Qwen3-VL) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            # Path to base VLM checkpoint (local or HF hub id)
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            # Attention implementation: "flash_attention_2" | "eager" | "sdpa"
            "attn_implementation": "flash_attention_2",
            # VLM hidden dimension (used for cross-attention alignment)
            "vl_hidden_dim": 2048,
        }
    )

    # # === DINO encoder (optional multi-view spatial tokens) === Dino is not used in this QwenGR00T version, we can add it later when we want to use it
    # dino: dict = field(default_factory=lambda: {
    #     # DINO backbone variant: "dinov2_vits14" | "dinov2_vitb14" | ...
    #     "dino_backbone": "dinov2_vits14",
    # })

    # === Action head (Flow-matching / DiT diffusion) ===
    action_model: dict = field(
        default_factory=lambda: {
            # DiT model size: "DiT-B" | "DiT-L" | "DiT-XL"
            "action_model_type": "DiT-B",
            # Hidden dim for action model (auto-aligned at runtime)
            "action_hidden_dim": 1024,
            "hidden_size": 1024,
            # Whether to add positional embeddings in the action head
            "add_pos_embed": True,
            "max_seq_len": 1024,
            # Dimensionality of each action vector (e.g., 7 for 6-DoF + gripper)
            "action_dim": 7,
            # State dimension (proprioception input)
            "state_dim": 7,
            # Canonical chunk length (number of action steps the head predicts).
            # Legacy YAMLs may use future_action_window_size = action_horizon - 1;
            # apply_config_compat normalises both directions.
            "action_horizon": 8,
            # Repeat factor for flow-matching loss (more noise samples per batch)
            "repeated_diffusion_steps": 4,
            # Beta distribution params for noise schedule
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            # Inference denoising steps
            "num_inference_timesteps": 4,
            # Number of vision tokens fed to action head
            "num_target_vision_tokens": 32,
            # Optional AngleTransformer tactile/angle encoder.
            "touch": {
                "enabled": True,
                "repo_path": "../touch-rep",
                "model_size": "tiny",
                "checkpoint_encoder": None,
                "train_encoder": True,
                "joint_contact_key": "joint_contact",
                "finger_angles_key": "finger_angles",
                "require_touch": True,
                "derive_finger_angles_from_state": True,
                "finger_angle_state_indices": [
                    [7, 8, 8, 8],
                    [9, 9, 9, 9],
                    [10, 10, 10, 10],
                    [11, 11, 11, 11],
                    [12, 12, 12, 12],
                    [20, 21, 21, 21],
                    [22, 22, 22, 22],
                    [23, 23, 23, 23],
                    [24, 24, 24, 24],
                    [25, 25, 25, 25],
                ],
                "in_dim": 10,
                "in_chans": 3,
                "pos_in_dim": 10,
                "pos_in_chans": 4,
                "sequence_length": 1,
                "time_chunk_size": 1,
                "num_register_tokens": 1,
                "token_source": "x_tokens",
            },
            # === DiT Transformer sub-config ===
            "diffusion_model_cfg": {
                # Cross-attention dim (aligned to VLM hidden_size at runtime)
                "cross_attention_dim": 2048,
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "num_layers": 16,
                "output_dim": 1024,
                "positional_embeddings": None,
            },
        }
    )

    # # === Training precision flag === This is unnecessary, unused parameter
    # reduce_in_full_precision: bool = True


@FRAMEWORK_REGISTRY.register("QwenGR00T_touch")
class Qwen_GR00T_touch(baseframework):
    """
    Multimodal vision-language-action model (GR00T variant).

    Components:
      - Qwen2.5-VL / Qwen3-VL backbone for fused language/vision token embeddings
      - Flow-matching (DiT) diffusion head for continuous action sequence modeling

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        # Merge framework defaults with YAML config (YAML wins on conflicts)
        self.config = merge_framework_config(QwenGR00TTouchDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # align dims --> we should put them to config or no?
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        # `action_horizon` is the single source of truth for chunk length.
        # Legacy aliases (`future_action_window_size`, `past_action_window_size`)
        # are normalised upstream by `share_tools.apply_config_compat`, so we
        # only ever read `action_horizon` here.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

    def _read_key(self, obj, key: str):
        if not key:
            return None
        if isinstance(obj, dict) and key in obj:
            return obj[key]
        cur = obj
        for part in key.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return None
        return cur

    def _read_touch_key(self, example: dict, key: str):
        value = self._read_key(example, key)
        if value is not None:
            return value
        for container_key in ("touch", "tactile"):
            container = example.get(container_key)
            if isinstance(container, dict):
                value = self._read_key(container, key)
                if value is not None:
                    return value
        return None

    def _stack_touch_values(self, values, device, dtype):
        if torch.is_tensor(values[0]):
            return torch.stack([v.to(device=device, dtype=dtype) for v in values], dim=0)
        return torch.as_tensor(np.array(values), device=device, dtype=dtype)

    def _extract_touch_inputs(self, examples: List[dict], device, dtype=torch.float32):
        touch_cfg = self.config.framework.action_model.get("touch", {})
        if not touch_cfg.get("enabled", False) and not touch_cfg.get("checkpoint_encoder", None):
            return None

        contact_key = touch_cfg.get("joint_contact_key", "joint_contact")
        angle_key = touch_cfg.get("finger_angles_key", "finger_angles")
        contacts = [self._read_touch_key(example, contact_key) for example in examples]
        angles = [self._read_touch_key(example, angle_key) for example in examples] if angle_key else [None] * len(examples)

        missing_contact = any(v is None for v in contacts)
        missing_angles = any(v is None for v in angles)
        derive_angles = touch_cfg.get("derive_finger_angles_from_state", True)

        if missing_contact or (missing_angles and not derive_angles):
            if touch_cfg.get("require_touch", False):
                sample_keys = sorted(examples[0].keys())
                touch_keys = sorted(examples[0].get("touch", {}).keys()) if isinstance(examples[0].get("touch"), dict) else []
                tactile_keys = (
                    sorted(examples[0].get("tactile", {}).keys()) if isinstance(examples[0].get("tactile"), dict) else []
                )
                need = f"`{contact_key}`"
                if not derive_angles:
                    need += f" and `{angle_key}`"
                raise KeyError(
                    "QwenGR00T_touch requires touch inputs, but batch sample is missing them. "
                    f"Need {need} at sample root, sample['touch'], or sample['tactile']. "
                    f"Sample keys={sample_keys}, touch keys={touch_keys}, tactile keys={tactile_keys}."
                )
            return None

        touch = {"joint_contact": self._stack_touch_values(contacts, device=device, dtype=dtype)}
        if not missing_angles:
            touch["finger_angles"] = self._stack_touch_values(angles, device=device, dtype=dtype)
        return touch

    def _repeat_touch_inputs(self, touch, repeats: int):
        if touch is None:
            return None
        return {
            key: value.repeat((repeats,) + (1,) * (value.dim() - 1))
            for key, value in touch.items()
        }

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """ """
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]

        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        backbone_attention_mask = qwen_inputs.get("attention_mask", None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.bfloat16):
            touch = self._extract_touch_inputs(examples, device=last_hidden.device)
            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -self.action_horizon :, :]  # (B, action_horizon, action_dim)

            repeated_diffusion_steps = (
                self.config.framework.action_model.get("repeated_diffusion_steps", 4)
                if self.config and hasattr(self.config, "framework")
                else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
            touch_repeated = self._repeat_touch_inputs(touch, repeated_diffusion_steps)
            if backbone_attention_mask is not None:
                backbone_attention_mask = backbone_attention_mask.repeat(repeated_diffusion_steps, 1).to(
                    dtype=torch.bool
                )

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(
                last_hidden_repeated, actions_target_repeated, state_repeated,
                encoder_attention_mask=backbone_attention_mask,
                touch=touch_repeated,
            )  # (B, chunk_len, action_dim)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        """
        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory
        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]

        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # Step 1: QWenVL input format
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

            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        state = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None
            else None
        )

        # Step 4: Action Expert Forward
        with torch.autocast("cuda", dtype=torch.float32):
            touch = self._extract_touch_inputs(examples, device=last_hidden.device)
            pred_actions = self.action_model.predict_action(
                last_hidden, state, encoder_attention_mask=backbone_attention_mask, touch=touch
            )  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    model: Qwen_GR00T_touch = Qwen_GR00T_touch(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image],
        "lang": "This is a fake instruction for testing.",
    }
    sample2 = sample.copy()
    sample2["lang"] = "Another fake instruction for testing."

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"Action Loss: {action_loss.item()}")

    predict_output = model.predict_action(examples=[sample])
    normalized_actions = predict_output["normalized_actions"]
    print(f"Unnormalized Action: {normalized_actions}")

    print("Finished")
