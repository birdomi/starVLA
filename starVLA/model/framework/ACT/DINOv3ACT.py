# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""DINOv3 + ACT action chunking framework."""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.dino_model.dino_any import get_dino_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


@dataclass
class DINOv3ACTDefaultConfig:
    name: str = "DINOv3ACT"

    dino: dict = field(
        default_factory=lambda: {
            "dino_backbone": "dinov3_vits16",
            "checkpoint_path": None,
            "repo_dir": None,
            "train_encoder": False,
            "include_cls": True,
            "max_views": 4,
        }
    )

    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "ACT",
            "action_dim": 7,
            "state_dim": 7,
            "action_horizon": 8,
            "hidden_dim": 512,
            "num_encoder_layers": 2,
            "num_decoder_layers": 4,
            "num_heads": 8,
            "dim_feedforward": 2048,
            "dropout": 0.1,
            "loss_type": "l1",
        }
    )


class ACTDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        action_horizon: int,
        state_dim: int = 0,
        hidden_dim: int = 512,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.state_dim = int(state_dim or 0)

        self.vision_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
        )
        self.state_proj = nn.Linear(self.state_dim, hidden_dim) if self.state_dim > 0 else None
        self.view_embed = nn.Embedding(16, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.action_queries = nn.Parameter(torch.empty(1, self.action_horizon, hidden_dim))
        self.action_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.action_dim),
        )
        nn.init.normal_(self.action_queries, mean=0.0, std=0.02)

    def _add_view_embed(self, tokens: torch.Tensor, batch_size: int, num_views: int) -> torch.Tensor:
        tokens_per_view = tokens.shape[1] // max(num_views, 1)
        view_ids = torch.arange(num_views, device=tokens.device).repeat_interleave(tokens_per_view)
        view_ids = view_ids[: tokens.shape[1]].clamp(max=self.view_embed.num_embeddings - 1)
        return tokens + self.view_embed(view_ids).unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, dino_tokens: torch.Tensor, num_views: int, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = dino_tokens.shape[0]
        memory = self.vision_proj(dino_tokens)
        memory = self._add_view_embed(memory, batch_size=batch_size, num_views=num_views)

        if state is not None and self.state_proj is not None:
            if state.dim() == 3:
                state = state[:, -1, :]
            state_token = self.state_proj(state).unsqueeze(1)
            memory = torch.cat((state_token, memory), dim=1)

        memory = self.encoder(memory)
        queries = self.action_queries.expand(batch_size, -1, -1)
        decoded = self.decoder(tgt=queries, memory=memory)
        return self.action_head(decoded)


@FRAMEWORK_REGISTRY.register("DINOv3ACT")
class DINOv3ACT(baseframework):
    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(DINOv3ACTDefaultConfig, config)

        dino_cfg = self.config.framework.dino
        self.dino_encoder = get_dino_model(
            backone_name=dino_cfg.get("dino_backbone", "dinov3_vits16"),
            checkpoint_path=dino_cfg.get("checkpoint_path", None),
            repo_dir=dino_cfg.get("repo_dir", None),
        )
        self.train_dino_encoder = bool(dino_cfg.get("train_encoder", False))
        if not self.train_dino_encoder:
            self.dino_encoder.requires_grad_(False)

        act_cfg = self.config.framework.action_model
        self.action_horizon = int(act_cfg.action_horizon)
        self.action_dim = int(act_cfg.action_dim)
        self.loss_type = str(act_cfg.get("loss_type", "l1")).lower()

        self.action_model = ACTDecoder(
            input_dim=self.dino_encoder.num_channels,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            state_dim=int(act_cfg.get("state_dim", 0)),
            hidden_dim=int(act_cfg.get("hidden_dim", 512)),
            num_encoder_layers=int(act_cfg.get("num_encoder_layers", 2)),
            num_decoder_layers=int(act_cfg.get("num_decoder_layers", 4)),
            num_heads=int(act_cfg.get("num_heads", 8)),
            dim_feedforward=int(act_cfg.get("dim_feedforward", 2048)),
            dropout=float(act_cfg.get("dropout", 0.1)),
        )

    @staticmethod
    def _ensure_view_list(images):
        images = to_pil_preserve(images)
        return images if isinstance(images, list) else [images]

    def _images_from_examples(self, examples: List[dict]):
        max_views = int(self.config.framework.dino.get("max_views", 4))
        batch_images = []
        for example in examples:
            views = self._ensure_view_list(example["image"])
            if max_views > 0:
                views = views[:max_views]
            batch_images.append(views)
        num_views = len(batch_images[0]) if batch_images else 0
        if num_views == 0:
            raise ValueError("DINOv3ACT needs at least one image view")
        if any(len(views) != num_views for views in batch_images):
            raise ValueError("DINOv3ACT needs same view count per sample in a batch")
        return batch_images, num_views

    def _encode_images(self, batch_images: List[list], num_views: int) -> torch.Tensor:
        image_tensors = self.dino_encoder.prepare_dino_input(batch_images)
        grad_ctx = torch.enable_grad() if self.train_dino_encoder else torch.no_grad()
        device_type = "cuda" if image_tensors.is_cuda else "cpu"
        with grad_ctx:
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=image_tensors.is_cuda):
                tokens = self.dino_encoder(
                    image_tensors,
                    include_cls=bool(self.config.framework.dino.get("include_cls", True)),
                )
        batch_size = len(batch_images)
        return tokens.reshape(batch_size, num_views * tokens.shape[1], tokens.shape[2])

    def _action_model_dtype(self):
        return next(self.action_model.parameters()).dtype

    def _target_actions(self, examples: List[dict], device, dtype) -> torch.Tensor:
        actions = torch.as_tensor(np.array([example["action"] for example in examples]), device=device, dtype=dtype)
        return actions[:, -self.action_horizon :, : self.action_dim]

    def _state_tensor(self, examples: List[dict], device, dtype) -> Optional[torch.Tensor]:
        if "state" not in examples[0]:
            return None
        return torch.as_tensor(np.array([example["state"] for example in examples]), device=device, dtype=dtype)

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        batch_images, num_views = self._images_from_examples(examples)
        dino_tokens = self._encode_images(batch_images, num_views=num_views)
        act_dtype = self._action_model_dtype()
        dino_tokens = dino_tokens.to(dtype=act_dtype)
        state = self._state_tensor(examples, device=dino_tokens.device, dtype=act_dtype)
        pred_actions = self.action_model(dino_tokens, num_views=num_views, state=state)
        target_actions = self._target_actions(examples, device=pred_actions.device, dtype=pred_actions.dtype)

        if self.loss_type == "mse":
            action_loss = F.mse_loss(pred_actions.float(), target_actions.float())
        else:
            action_loss = F.l1_loss(pred_actions.float(), target_actions.float())
        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        if type(examples) is not list:
            examples = [examples]
        batch_images, num_views = self._images_from_examples(examples)
        dino_tokens = self._encode_images(batch_images, num_views=num_views)
        act_dtype = self._action_model_dtype()
        dino_tokens = dino_tokens.to(dtype=act_dtype)
        state = self._state_tensor(examples, device=dino_tokens.device, dtype=act_dtype)
        pred_actions = self.action_model(dino_tokens, num_views=num_views, state=state)
        return {"normalized_actions": pred_actions.detach().float().cpu().numpy()}
