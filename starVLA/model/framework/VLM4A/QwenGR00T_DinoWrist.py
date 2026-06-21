# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
QwenGR00T_DinoWrist Framework

Head camera goes to Qwen-VL. Wrist cameras go to DINO and are concatenated
with state features before the GR00T DiT action model.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_ActionHeader_dinowrist import (
    FlowmatchingActionHeadDinoWrist,
    get_action_model,
)
from starVLA.model.modules.dino_model.dino_any import get_dino_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)


class WristDinoQueryAdapter(nn.Module):
    def __init__(self, dino_dim, output_dim, num_query_tokens=16, num_heads=8):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.empty(1, num_query_tokens, output_dim))
        self.kv_norm = nn.LayerNorm(dino_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            kdim=dino_dim,
            vdim=dino_dim,
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(output_dim)
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)

    def forward(self, dino_tokens):
        queries = self.query_tokens.expand(dino_tokens.shape[0], -1, -1)
        kv_tokens = self.kv_norm(dino_tokens)
        attn_out, _ = self.cross_attn(queries, kv_tokens, kv_tokens, need_weights=False)
        return self.out_norm(queries + attn_out)


@dataclass
class QwenGR00TDinoWristDefaultConfig:
    name: str = "QwenGR00T_DinoWrist"

    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            "attn_implementation": "flash_attention_2",
            "vl_hidden_dim": 2048,
        }
    )

    dino: dict = field(
        default_factory=lambda: {
            "dino_backbone": "dinov3_vits16",
            "checkpoint_path": "playground/Pretrained_models/DINOv3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
            "repo_dir": None,
            "head_view_indices": [0],
            "wrist_view_indices": [1, 2],
            "require_wrist": True,
            "train_encoder": False,
            "num_wrist_query_tokens": 16,
            "wrist_query_num_heads": 8,
        }
    )

    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "DiT-B",
            "action_hidden_dim": 1024,
            "hidden_size": 1024,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "action_dim": 7,
            "state_dim": 7,
            "action_horizon": 8,
            "repeated_diffusion_steps": 4,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "num_inference_timesteps": 4,
            "num_target_vision_tokens": 32,
            "diffusion_model_cfg": {
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


@FRAMEWORK_REGISTRY.register("QwenGR00T_DinoWrist")
class Qwen_GR00T_DinoWrist(baseframework):
    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = merge_framework_config(QwenGR00TDinoWristDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )

        self.action_model: FlowmatchingActionHeadDinoWrist = get_action_model(config=self.config)
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

        dino_cfg = self.config.framework.dino
        self.dino_encoder = get_dino_model(
            backone_name=dino_cfg.get("dino_backbone", "dinov3_vits16"),
            checkpoint_path=dino_cfg.get("checkpoint_path", None),
            repo_dir=dino_cfg.get("repo_dir", None),
        )
        self.wrist_query_adapter = WristDinoQueryAdapter(
            dino_dim=self.dino_encoder.num_channels,
            output_dim=self.action_model.input_embedding_dim,
            num_query_tokens=int(dino_cfg.get("num_wrist_query_tokens", 16)),
            num_heads=int(dino_cfg.get("wrist_query_num_heads", 8)),
        ).to(dtype=torch.bfloat16)

        self.train_dino_encoder = bool(dino_cfg.get("train_encoder", False))
        if not self.train_dino_encoder:
            self.dino_encoder.requires_grad_(False)

    @staticmethod
    def _ensure_view_list(images):
        images = to_pil_preserve(images)
        return images if isinstance(images, list) else [images]

    def _split_views(self, examples: List[dict]):
        dino_cfg = self.config.framework.dino
        head_indices = list(dino_cfg.get("head_view_indices", [0]))
        wrist_indices = list(dino_cfg.get("wrist_view_indices", [1, 2]))
        require_wrist = bool(dino_cfg.get("require_wrist", True))

        batch_head_images = []
        batch_wrist_images = []
        for example in examples:
            images = self._ensure_view_list(example["image"])
            head_views = [images[idx] for idx in head_indices if idx < len(images)]
            if not head_views:
                raise ValueError(f"No head camera found. image view count={len(images)}, head_indices={head_indices}")

            if "wrist_views" in example:
                wrist_views = self._ensure_view_list(example["wrist_views"])
            else:
                wrist_views = [images[idx] for idx in wrist_indices if idx < len(images)]

            if require_wrist and not wrist_views:
                raise ValueError(
                    f"No wrist camera found. image view count={len(images)}, wrist_indices={wrist_indices}"
                )

            batch_head_images.append(head_views)
            batch_wrist_images.append(wrist_views)

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_head_images = resize_images(batch_head_images, target_size=train_obs_image_size)
            batch_wrist_images = resize_images(batch_wrist_images, target_size=train_obs_image_size)

        return batch_head_images, batch_wrist_images

    def _encode_head_vlm(self, head_images, instructions):
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=head_images, instructions=instructions)
        attention_mask = qwen_inputs.get("attention_mask", None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]
        return last_hidden, attention_mask

    def _encode_wrist_dino(self, wrist_images, target_device, target_dtype):
        if not wrist_images or not wrist_images[0]:
            return None

        image_tensors = self.dino_encoder.prepare_dino_input(wrist_images)

        grad_ctx = torch.enable_grad() if self.train_dino_encoder else torch.no_grad()
        with grad_ctx:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                dino_features = self.dino_encoder(image_tensors, include_cls=True)

        batch_size = len(wrist_images)
        dino_features = dino_features.reshape(batch_size, -1, dino_features.shape[-1]).to(
            device=target_device,
            dtype=self.wrist_query_adapter.query_tokens.dtype,
        )
        wrist_features = self.wrist_query_adapter(dino_features)
        return wrist_features.to(dtype=target_dtype)

    @staticmethod
    def _repeat_optional_tensor(value, repeats: int):
        return value.repeat(repeats, 1, 1) if value is not None else None

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        head_images, wrist_images = self._split_views(examples)
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        last_hidden, backbone_attention_mask = self._encode_head_vlm(head_images, instructions)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            wrist_features = self._encode_wrist_dino(
                wrist_images,
                target_device=last_hidden.device,
                target_dtype=last_hidden.dtype,
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
            wrist_features_repeated = self._repeat_optional_tensor(wrist_features, repeated_diffusion_steps)

            if backbone_attention_mask is not None:
                backbone_attention_mask = backbone_attention_mask.repeat(repeated_diffusion_steps, 1).to(
                    dtype=torch.bool
                )

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(
                last_hidden_repeated,
                actions_target_repeated,
                state_repeated,
                encoder_attention_mask=backbone_attention_mask,
                wrist_features=wrist_features_repeated,
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

        head_images, wrist_images = self._split_views(examples)
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        last_hidden, backbone_attention_mask = self._encode_head_vlm(head_images, instructions)
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)

        wrist_features = self._encode_wrist_dino(
            wrist_images,
            target_device=last_hidden.device,
            target_dtype=last_hidden.dtype,
        )
        state = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None
            else None
        )

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                last_hidden,
                state,
                encoder_attention_mask=backbone_attention_mask,
                wrist_features=wrist_features,
            )

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}
