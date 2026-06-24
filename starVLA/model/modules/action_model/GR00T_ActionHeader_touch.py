from __future__ import annotations

from pathlib import Path
import sys
import types
from typing import Any

import torch
from torch import nn

from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    try:
        return cfg.get(key, default)
    except AttributeError:
        return getattr(cfg, key, default)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve_path(path: str | None) -> Path | None:
    if not path:
        return None
    out = Path(path).expanduser()
    if not out.is_absolute():
        out = (_repo_root() / out).resolve()
    return out


def _install_tactile_ssl_checkpoint_stubs() -> None:
    """Allow loading tactile_ssl checkpoints while only using their tensor state."""
    module_name = "tactile_ssl.model.custom_scheduler"
    tactile_ssl = sys.modules.setdefault("tactile_ssl", types.ModuleType("tactile_ssl"))
    tactile_model = sys.modules.setdefault("tactile_ssl.model", types.ModuleType("tactile_ssl.model"))
    custom_scheduler = sys.modules.setdefault(module_name, types.ModuleType(module_name))

    for class_name in ("WarmupCosineScheduler", "CosineWDSchedule"):
        if not hasattr(custom_scheduler, class_name):
            setattr(custom_scheduler, class_name, type(class_name, (), {"__module__": module_name}))

    setattr(tactile_ssl, "model", tactile_model)
    setattr(tactile_model, "custom_scheduler", custom_scheduler)


def _load_checkpoint_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    _install_tactile_ssl_checkpoint_stubs()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model") or checkpoint.get("state_dict") or checkpoint
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected checkpoint state_dict to be a dict, got {type(state_dict).__name__}.")
    return state_dict


def _strip_encoder_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = (
        "model_encoder.",
        "teacher_encoder.backbone.",
        "student_encoder.backbone.",
        "encoder.",
    )
    for prefix in prefixes:
        stripped = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        if stripped:
            return stripped
    return state_dict


def _infer_angle_transformer_kwargs(raw: dict[str, torch.Tensor]) -> dict[str, Any]:
    inferred: dict[str, Any] = {}
    if "contact_pos_embed" in raw:
        inferred["in_dim"] = int(raw["contact_pos_embed"].shape[1] * 2)
        inferred["embed_dim"] = int(raw["contact_pos_embed"].shape[-1])
    if "angle_pos_embed" in raw:
        inferred["pos_in_dim"] = int(raw["angle_pos_embed"].shape[1] * 2)
    if "sensor_embed.proj.weight" in raw:
        inferred["in_chans"] = int(raw["sensor_embed.proj.weight"].shape[1])
        inferred.setdefault("embed_dim", int(raw["sensor_embed.proj.weight"].shape[0]))
        inferred["time_chunk_size"] = int(raw["sensor_embed.proj.weight"].shape[-1])
    if "angle_embed.proj.weight" in raw:
        inferred["pos_in_chans"] = int(raw["angle_embed.proj.weight"].shape[1])
    if "register_tokens" in raw:
        inferred["num_register_tokens"] = int(raw["register_tokens"].shape[1])
    if "mask_token" in raw:
        inferred["with_masktoken"] = True
    block_indices = {
        int(k.split(".", 2)[1])
        for k in raw
        if k.startswith("blocks.") and len(k.split(".", 2)) > 2 and k.split(".", 2)[1].isdigit()
    }
    if block_indices:
        inferred["depth"] = max(block_indices) + 1
    sensor_block_indices = {
        int(k.split(".", 2)[1])
        for k in raw
        if k.startswith("sensor_block.") and len(k.split(".", 2)) > 2 and k.split(".", 2)[1].isdigit()
    }
    if sensor_block_indices:
        inferred["pre_fusion_depth"] = max(sensor_block_indices) + 1
    if "pos_embed" in raw and "in_dim" in inferred:
        num_chunks = int(raw["pos_embed"].shape[1] // inferred["in_dim"])
        inferred["sequence_length"] = num_chunks * int(inferred.get("time_chunk_size", 1))
    return inferred


class LocalPatchEmbed1d(nn.Module):
    def __init__(self, modal_chans: int, modal_lens: int, chunk_size: int, embed_dim: int, padding: int = 0):
        super().__init__()
        self.modal_chans = int(modal_chans)
        self.modal_lens = int(modal_lens)
        self.num_chunks = int(modal_lens // chunk_size)
        self.chunk_size = int(chunk_size)
        self.embed_dim = int(embed_dim)
        self.proj = nn.Conv1d(
            in_channels=self.modal_chans,
            out_channels=self.embed_dim,
            kernel_size=self.chunk_size,
            stride=self.chunk_size,
            padding=int(padding),
        )
        self.layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        return x.transpose(1, 2)


class LocalMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, act_layer=nn.GELU, drop: float = 0.0, bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class LocalAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, attn_bias=None, return_attn: bool = False):
        batch_size, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, tokens, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)
        if attn_bias is not None:
            attn = attn + attn_bias
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(batch_size, tokens, channels)
        x = self.proj_drop(self.proj(x))
        return attn if return_attn else x


class LocalTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = LocalAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = nn.Identity()
        self.drop_path1 = nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = LocalMlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = nn.Identity()
        self.drop_path2 = nn.Identity()

    def forward(self, x: torch.Tensor, attn_bias=None, return_attn: bool = False) -> torch.Tensor:
        if return_attn:
            return self.attn(self.norm1(x), attn_bias=attn_bias, return_attn=True)
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


FULL_SKELETON_SIZE = 42
TACTILE_SENSOR_IDXS = [4, 8, 12, 16, 20, 25, 29, 33, 37, 41]


class LocalAngleTransformer(nn.Module):
    """Self-contained AngleTransformer compatible with touch-rep checkpoint keys."""

    def __init__(
        self,
        in_dim: int = 42,
        in_chans: int = 1,
        pos_in_dim: int = 10,
        pos_in_chans: int = 4,
        sequence_length: int = 1,
        time_chunk_size: int = 1,
        num_register_tokens: int = 1,
        embed_dim: int = 192,
        depth: int = 8,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        with_masktoken: bool = True,
        use_null_token: bool = True,
        pre_fusion_depth: int | None = None,
        normalization: Any = None,
        fine_tune_sensor: bool = False,
        fine_tune_sensor_shallow_blocks: int | None = 0,
        **_: Any,
    ):
        super().__init__()
        if sequence_length % time_chunk_size != 0:
            raise ValueError(
                f"sequence_length({sequence_length}) must be divisible by time_chunk_size({time_chunk_size})."
            )
        if in_dim % 2 != 0:
            raise ValueError(f"in_dim({in_dim}) must be even.")
        if pos_in_dim % 2 != 0:
            raise ValueError(f"pos_in_dim({pos_in_dim}) must be even.")

        self.in_dim = int(in_dim)
        self.in_chans = int(in_chans)
        self.pos_in_dim = int(pos_in_dim)
        self.pos_in_chans = int(pos_in_chans)
        self.embed_dim = int(embed_dim)
        self.sequence_length = int(sequence_length)
        self.time_chunk_size = int(time_chunk_size)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.num_register_tokens = int(num_register_tokens)
        self.use_null_token = bool(use_null_token)
        self.pre_fusion_depth = int(pre_fusion_depth if pre_fusion_depth is not None else self.depth // 4)
        self.fine_tune_sensor_shallow_blocks = (
            self.pre_fusion_depth if fine_tune_sensor_shallow_blocks is None else int(fine_tune_sensor_shallow_blocks)
        )

        self.register_tokens = (
            nn.Parameter(torch.zeros(1, self.num_register_tokens, self.embed_dim))
            if self.num_register_tokens > 0
            else None
        )
        num_chunks = self.sequence_length // self.time_chunk_size
        self.pos_embed = nn.Parameter(torch.zeros(1, num_chunks * self.in_dim, self.embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim)) if with_masktoken else None

        self.blocks = nn.ModuleList(
            [
                LocalTransformerBlock(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    drop=dropout,
                )
                for _ in range(self.depth)
            ]
        )
        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-6)

        self.contact_pos_embed = nn.Parameter(torch.zeros(2, self.in_dim // 2, self.embed_dim))
        self.angle_pos_embed = nn.Parameter(torch.zeros(2, self.pos_in_dim // 2, self.embed_dim))
        self.hand_embed = nn.Parameter(torch.zeros(2, self.embed_dim))
        self.sensor_embed = LocalPatchEmbed1d(self.in_chans, self.sequence_length, self.time_chunk_size, self.embed_dim)
        self.angle_embed = LocalPatchEmbed1d(
            self.pos_in_chans,
            self.sequence_length,
            self.time_chunk_size,
            self.embed_dim,
        )
        self.sensor_block = nn.ModuleList(
            [
                LocalTransformerBlock(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    drop=0.0,
                )
                for _ in range(self.pre_fusion_depth)
            ]
        )

        if normalization is not None:
            mean = torch.tensor(_cfg_get(normalization, "mean", [0.0]), dtype=torch.float32)
            std = torch.tensor(_cfg_get(normalization, "std", [1.0]), dtype=torch.float32)
        else:
            mean = torch.zeros(self.in_chans, dtype=torch.float32)
            std = torch.ones(self.in_chans, dtype=torch.float32)
        self.register_buffer("signal_mean", mean)
        self.register_buffer("signal_std", std)

        self.init_weights()
        self.fine_tune_sensor = bool(fine_tune_sensor)
        if self.fine_tune_sensor:
            self._apply_fine_tune_sensor()

    def init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.register_tokens is not None:
            nn.init.trunc_normal_(self.register_tokens, std=1e-6)
        if self.mask_token is not None:
            nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.hand_embed, std=0.02)
        nn.init.trunc_normal_(self.contact_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.angle_pos_embed, std=0.02)

    def _apply_fine_tune_sensor(self) -> None:
        shallow_blocks = min(self.fine_tune_sensor_shallow_blocks, len(self.blocks))
        for name, param in self.named_parameters():
            top = name.split(".", 1)[0]
            is_shallow_block = False
            if top == "blocks":
                parts = name.split(".", 2)
                is_shallow_block = len(parts) > 1 and parts[1].isdigit() and int(parts[1]) < shallow_blocks
            param.requires_grad = top in {"sensor_embed", "sensor_block"} or name == "contact_pos_embed" or is_shallow_block

    def _full_embed(self, per_hand: torch.Tensor) -> torch.Tensor:
        hand_embed = self.hand_embed.float()
        left = per_hand[0].float() + hand_embed[0]
        right = per_hand[1].float() + hand_embed[1]
        return torch.cat([left, right], dim=0)

    def _adapt_contact_channels(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.in_chans:
            return x
        if self.in_chans == 1:
            return torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        if x.shape[-1] == 1:
            return x.expand(*x.shape[:-1], self.in_chans)
        raise ValueError(f"Contact channel dim {x.shape[-1]} does not match encoder in_chans {self.in_chans}.")

    def expand_to_skeleton(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, steps, sensors, chans = x.shape
        if sensors == self.in_dim:
            return x, None
        if self.in_dim == FULL_SKELETON_SIZE and sensors == len(TACTILE_SENSOR_IDXS):
            x_full = torch.zeros(batch_size, steps, self.in_dim, chans, device=x.device, dtype=x.dtype)
            x_full[:, :, TACTILE_SENSOR_IDXS, :] = x
            null_mask = torch.ones(batch_size, self.in_dim, dtype=torch.bool, device=x.device)
            null_mask[:, TACTILE_SENSOR_IDXS] = False
            return x_full, null_mask.unsqueeze(0)
        raise ValueError(f"Cannot map {sensors} contact sensors to encoder in_dim {self.in_dim}.")

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.signal_mean.to(device=x.device, dtype=x.dtype)
        std = self.signal_std.to(device=x.device, dtype=x.dtype).clamp(min=1e-6)
        return (x - mean) / std

    def _apply_embed1d(self, x: torch.Tensor, embed: LocalPatchEmbed1d) -> torch.Tensor:
        batch_size, _, sensors, _ = x.shape
        x = x.permute(0, 2, 3, 1).reshape(batch_size * sensors, x.shape[-1], x.shape[1])
        x = embed(x)
        return x.permute(0, 2, 1).reshape(batch_size, sensors, x.shape[-1], self.embed_dim).transpose(1, 2)

    def pre_sensor_embed(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply_embed1d(self.normalize(x), self.sensor_embed)

    def pre_pos_embed(self, pos: torch.Tensor) -> torch.Tensor:
        return self._apply_embed1d(pos, self.angle_embed)

    @staticmethod
    def _repeat_joint_embed(joint_embed: torch.Tensor, token_count: int) -> torch.Tensor:
        if token_count == joint_embed.shape[0]:
            return joint_embed
        if token_count % joint_embed.shape[0] != 0:
            raise ValueError(f"Cannot repeat joint embedding of length {joint_embed.shape[0]} to {token_count}.")
        return joint_embed.repeat(token_count // joint_embed.shape[0], 1)

    def prepare_tokens_with_mask(
        self,
        x: torch.Tensor,
        masktoken_masks: torch.Tensor | None,
        joint_embed: torch.Tensor,
        skip_register: bool,
    ) -> torch.Tensor:
        token_count = x.shape[1] * x.shape[2]
        x = x + self._repeat_joint_embed(joint_embed, token_count).view(1, x.shape[1], x.shape[2], -1)
        if masktoken_masks is not None:
            if self.mask_token is None:
                raise RuntimeError("masktoken_masks provided, but encoder has no mask_token.")
            mask = masktoken_masks.flatten(0, 1)
            mask = mask[:, None, :, None].expand(-1, x.shape[1], -1, x.shape[-1])
            x = torch.where(mask, self.mask_token.to(dtype=x.dtype), x)
        x = x.reshape(x.shape[0], token_count, x.shape[-1])
        if self.register_tokens is not None and not skip_register:
            x = torch.cat([self.register_tokens.to(dtype=x.dtype).expand(x.shape[0], -1, -1), x], dim=1)
        return x

    def sensor_transform(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.sensor_block:
            x = block(x, None)
        return x

    def transform_concat(self, sen: torch.Tensor, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sen_pe = self._repeat_joint_embed(self._full_embed(self.contact_pos_embed), sen.shape[1]).to(
            device=sen.device,
            dtype=sen.dtype,
        )
        angle_tokens = pos.shape[1] - self.num_register_tokens
        ang_pe = self._repeat_joint_embed(self._full_embed(self.angle_pos_embed), angle_tokens).to(
            device=pos.device,
            dtype=pos.dtype,
        )
        sen = sen + sen_pe
        if angle_tokens > 0:
            pos[:, self.num_register_tokens:] = pos[:, self.num_register_tokens:] + ang_pe
        fused = torch.cat([pos, sen], dim=1)
        for block in self.blocks:
            fused = block(fused, None)
        fused = self.norm(fused)
        return fused, fused

    def _align_angles(self, finger_angles: torch.Tensor, target_steps: int) -> torch.Tensor:
        if finger_angles.shape[1] == 1 and target_steps != 1:
            finger_angles = finger_angles.expand(-1, target_steps, -1, -1)
        if finger_angles.shape[2] < self.pos_in_dim:
            pad = finger_angles[:, :, -1:, :].expand(-1, -1, self.pos_in_dim - finger_angles.shape[2], -1)
            finger_angles = torch.cat((finger_angles, pad), dim=2)
        elif finger_angles.shape[2] > self.pos_in_dim:
            finger_angles = finger_angles[:, :, : self.pos_in_dim, :]
        if finger_angles.shape[-1] != self.pos_in_chans:
            raise ValueError(
                f"Finger angle channel dim {finger_angles.shape[-1]} does not match encoder pos_in_chans {self.pos_in_chans}."
            )
        return finger_angles

    def forward_features(self, joint_contact: torch.Tensor, finger_angles: torch.Tensor) -> dict[str, torch.Tensor]:
        if joint_contact.dim() != 4:
            raise ValueError(f"Expected joint_contact [B, T, N, C], got {joint_contact.shape}.")
        if finger_angles.dim() != 4:
            raise ValueError(f"Expected finger_angles [B, T, N, C], got {finger_angles.shape}.")

        joint_contact = self._adapt_contact_channels(joint_contact)
        masktoken_masks = None
        if self.use_null_token:
            joint_contact, masktoken_masks = self.expand_to_skeleton(joint_contact)
        elif joint_contact.shape[2] != self.in_dim:
            raise ValueError(f"Expected {self.in_dim} contact sensors, got {joint_contact.shape[2]}.")

        finger_angles = self._align_angles(finger_angles, target_steps=joint_contact.shape[1])
        sensor_tokens = self.pre_sensor_embed(joint_contact)
        angle_tokens = self.pre_pos_embed(finger_angles)

        sensor_embed = self._full_embed(self.contact_pos_embed).to(device=sensor_tokens.device, dtype=sensor_tokens.dtype)
        angle_embed = self._full_embed(self.angle_pos_embed).to(device=angle_tokens.device, dtype=angle_tokens.dtype)
        sensor_tokens = self.prepare_tokens_with_mask(
            sensor_tokens,
            masktoken_masks=masktoken_masks,
            joint_embed=sensor_embed,
            skip_register=True,
        )
        angle_tokens = self.prepare_tokens_with_mask(
            angle_tokens,
            masktoken_masks=None,
            joint_embed=angle_embed,
            skip_register=False,
        )
        sensor_tokens = self.sensor_transform(sensor_tokens)
        x_prenorm, x_tokens = self.transform_concat(sensor_tokens, angle_tokens)
        reg = self.num_register_tokens
        return {
            "x_norm_regtokens": x_tokens[:, :reg],
            "x_norm_patchtokens": x_tokens[:, reg:],
            "x_prenorm": x_prenorm[:, reg:],
            "x_tokens": x_tokens,
            "patch_tokens": x_tokens[:, reg:],
            "register_tokens": x_tokens[:, :reg],
        }

    def forward(self, joint_contact: torch.Tensor, finger_angles: torch.Tensor) -> torch.Tensor:
        return self.forward_features(joint_contact, finger_angles)["x_tokens"]


def _angle_factory(touch_cfg: Any):
    model_size = _cfg_get(touch_cfg, "model_size", "tiny")
    presets = {
        "tiny": {"embed_dim": 192, "depth": 8, "num_heads": 3},
        "small": {"embed_dim": 384, "depth": 12, "num_heads": 6},
    }
    preset = presets.get(model_size, {})

    def factory(**kwargs):
        kwargs = {**preset, **kwargs}
        kwargs["embed_dim"] = int(_cfg_get(touch_cfg, "embed_dim", kwargs.get("embed_dim", 192)))
        kwargs["depth"] = int(_cfg_get(touch_cfg, "depth", kwargs.get("depth", 4)))
        kwargs["num_heads"] = int(_cfg_get(touch_cfg, "num_heads", kwargs.get("num_heads", 3)))
        return LocalAngleTransformer(**kwargs)

    return factory


class TouchAngleEncoder(nn.Module):
    """Local touch encoder wrapper. Returns tokens aligned to VLM hidden dim."""

    def __init__(self, touch_cfg: Any, target_dim: int):
        super().__init__()
        factory = _angle_factory(touch_cfg)
        checkpoint = _resolve_path(_cfg_get(touch_cfg, "checkpoint_encoder", None))
        checkpoint_raw = None
        inferred_kwargs = {}
        if checkpoint is not None:
            checkpoint_raw = _strip_encoder_state_dict(_load_checkpoint_state_dict(checkpoint))
            inferred_kwargs = _infer_angle_transformer_kwargs(checkpoint_raw)
            if inferred_kwargs.get("in_dim") == FULL_SKELETON_SIZE:
                inferred_kwargs["use_null_token"] = True

        kwargs = {
            "in_dim": _cfg_get(touch_cfg, "in_dim", 10),
            "in_chans": _cfg_get(touch_cfg, "in_chans", 3),
            "pos_in_dim": _cfg_get(touch_cfg, "pos_in_dim", 10),
            "pos_in_chans": _cfg_get(touch_cfg, "pos_in_chans", 4),
            "sequence_length": _cfg_get(touch_cfg, "sequence_length", 1),
            "time_chunk_size": _cfg_get(touch_cfg, "time_chunk_size", 1),
            "num_register_tokens": _cfg_get(touch_cfg, "num_register_tokens", 1),
            "pos_embed_fn": _cfg_get(touch_cfg, "pos_embed_fn", "learned"),
            "with_masktoken": _cfg_get(touch_cfg, "with_masktoken", False),
            "use_null_token": _cfg_get(touch_cfg, "use_null_token", False),
            "fine_tune_sensor": _cfg_get(touch_cfg, "fine_tune_sensor", False),
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        kwargs.update(inferred_kwargs)
        self.encoder = factory(**kwargs)
        self.token_source = _cfg_get(touch_cfg, "token_source", "x_tokens")
        self.train_encoder = bool(_cfg_get(touch_cfg, "train_encoder", True))
        self.derive_finger_angles_from_state = bool(_cfg_get(touch_cfg, "derive_finger_angles_from_state", True))
        self.finger_angle_state_indices = _cfg_get(
            touch_cfg,
            "finger_angle_state_indices",
            [
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
        )

        embed_dim = int(getattr(self.encoder, "embed_dim"))
        self.projector = nn.Identity() if embed_dim == target_dim else nn.Linear(embed_dim, target_dim)

        if checkpoint is not None:
            self.load_encoder(checkpoint, raw_state_dict=checkpoint_raw)

        if not self.train_encoder:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

    def load_encoder(self, checkpoint_path: Path, raw_state_dict: dict[str, torch.Tensor] | None = None) -> None:
        raw = raw_state_dict
        if raw is None:
            raw = _strip_encoder_state_dict(_load_checkpoint_state_dict(checkpoint_path))

        model_state = self.encoder.state_dict()
        filtered = {k: v for k, v in raw.items() if k in model_state and v.shape == model_state[k].shape}
        self.encoder.load_state_dict(filtered, strict=False)
        print(f"Loaded local touch encoder keys: {len(filtered)}/{len(raw)} from {checkpoint_path}")

    def _encode_once(self, joint_contact: torch.Tensor, finger_angles: torch.Tensor) -> torch.Tensor:
        ctx = torch.enable_grad() if self.train_encoder else torch.no_grad()
        with ctx:
            if self.token_source == "forward":
                tokens = self.encoder(joint_contact, finger_angles)
            else:
                features = self.encoder.forward_features(joint_contact, finger_angles)
                tokens = features[self.token_source]
        return self.projector(tokens)

    def _finger_angles_from_state(self, state: torch.Tensor, target_steps: int) -> torch.Tensor:
        if state is None:
            raise KeyError("Missing `finger_angles`, and state is None. Cannot derive finger angles from state.")

        if state.dim() == 2:
            state = state.unsqueeze(1)
        if state.dim() != 3:
            raise ValueError(f"Expected state shape [B, T, D] or [B, D], got {state.shape}.")
        if state.shape[-1] <= max(max(row) for row in self.finger_angle_state_indices):
            raise ValueError(
                f"State dim {state.shape[-1]} too small for finger angle indices {self.finger_angle_state_indices}."
            )

        values = []
        for finger_indices in self.finger_angle_state_indices:
            zero_angle = torch.zeros_like(state[..., finger_indices[0]])
            finger_values = [zero_angle, *[state[..., idx] for idx in finger_indices[1:]]]
            values.append(torch.stack(finger_values, dim=-1))
        finger_angles = torch.stack(values, dim=-2)

        if finger_angles.shape[1] == 1 and target_steps != 1:
            finger_angles = finger_angles.expand(-1, target_steps, -1, -1)
        elif finger_angles.shape[1] != target_steps:
            raise ValueError(
                f"Derived finger angle steps {finger_angles.shape[1]} do not match touch steps {target_steps}."
            )
        return finger_angles

    def forward(
        self,
        touch: dict[str, torch.Tensor],
        target_dtype: torch.dtype,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        joint_contact = touch["joint_contact"]

        param = next(self.parameters())
        joint_contact = joint_contact.to(device=param.device, dtype=param.dtype)

        if joint_contact.dim() == 3:
            joint_contact = joint_contact.unsqueeze(1)

        finger_angles = touch.get("finger_angles", None)
        if finger_angles is None:
            if not self.derive_finger_angles_from_state:
                raise KeyError("Missing `finger_angles` in touch input.")
            state = state.to(device=param.device, dtype=param.dtype) if state is not None else None
            finger_angles = self._finger_angles_from_state(state, target_steps=joint_contact.shape[1])
        else:
            finger_angles = finger_angles.to(device=param.device, dtype=param.dtype)

        if finger_angles.dim() == 3:
            finger_angles = finger_angles.unsqueeze(1)

        if joint_contact.dim() == 4:
            batch_size, steps, num_joints, chans = joint_contact.shape
            seq_len = int(getattr(self.encoder, "sequence_length", steps))
            if steps == seq_len:
                tokens = self._encode_once(joint_contact, finger_angles)
            elif seq_len == 1:
                flat_contact = joint_contact.reshape(batch_size * steps, 1, num_joints, chans)
                flat_angles = finger_angles.reshape(
                    batch_size * steps, 1, finger_angles.shape[-2], finger_angles.shape[-1]
                )
                tokens = self._encode_once(flat_contact, flat_angles)
                tokens = tokens.reshape(batch_size, steps * tokens.shape[1], tokens.shape[-1])
            else:
                raise ValueError(f"Touch sequence length {steps} does not match local touch encoder length {seq_len}.")
        elif joint_contact.dim() == 5:
            batch_size, windows, steps, num_joints, chans = joint_contact.shape
            flat_contact = joint_contact.reshape(batch_size * windows, steps, num_joints, chans)
            flat_angles = finger_angles.reshape(
                batch_size * windows, steps, finger_angles.shape[-2], finger_angles.shape[-1]
            )
            tokens = self._encode_once(flat_contact, flat_angles)
            tokens = tokens.reshape(batch_size, windows * tokens.shape[1], tokens.shape[-1])
        else:
            raise ValueError(f"Expected joint_contact dim 3/4/5, got {joint_contact.shape}.")

        return tokens.to(dtype=target_dtype)


class FlowmatchingActionHeadTouch(FlowmatchingActionHead):
    def __init__(self, full_config):
        super().__init__(full_config)
        touch_cfg = _cfg_get(full_config.framework.action_model, "touch", {})
        enabled = bool(_cfg_get(touch_cfg, "enabled", False))
        checkpoint = _cfg_get(touch_cfg, "checkpoint_encoder", None)
        self.require_touch = bool(_cfg_get(touch_cfg, "require_touch", enabled or checkpoint))
        self.touch_encoder = (
            TouchAngleEncoder(touch_cfg, target_dim=self.input_embedding_dim)
            if enabled or checkpoint
            else None
        )

    def _encode_touch_features(self, touch, target_dtype, target_device, state=None):
        if self.touch_encoder is None:
            if self.require_touch:
                raise RuntimeError("QwenGR00T_touch requires touch, but touch encoder is disabled.")
            return None
        if touch is None:
            if self.require_touch:
                raise KeyError("QwenGR00T_touch requires touch dict with `joint_contact`.")
            return None

        return self.touch_encoder(touch, target_dtype=target_dtype, state=state).to(device=target_device)

    @staticmethod
    def _concat_state_touch_features(state_features, touch_features):
        # print(state_features.shape, touch_features.shape)
        if state_features is None:
            return touch_features
        if touch_features is None:
            return state_features
        return torch.cat((state_features, touch_features), dim=1)

    def forward(
        self,
        vl_embs: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor = None,
        encoder_attention_mask=None,
        touch: dict[str, torch.Tensor] | None = None,
    ):
        device = vl_embs.device

        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)

        state_features = self.state_encoder(state) if state is not None else None
        touch_features = self._encode_touch_features(
            touch,
            target_dtype=action_features.dtype,
            target_device=device,
            state=state,
        )
        prefix_features = self._concat_state_touch_features(state_features, touch_features)

        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = (
            torch.cat((prefix_features, future_tokens, action_features), dim=1)
            if prefix_features is not None
            else torch.cat((future_tokens, action_features), dim=1)
        )

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=encoder_attention_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,
        )
        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1] :]

        return ((pred_actions - velocity) ** 2).mean()

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        state: torch.Tensor = None,
        encoder_attention_mask=None,
        touch: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        state_features = self.state_encoder(state) if state is not None else None
        touch_features = self._encode_touch_features(
            touch,
            target_dtype=vl_embs.dtype,
            target_device=device,
            state=state,
        )
        prefix_features = self._concat_state_touch_features(state_features, touch_features)

        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)

            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
            action_features = self.action_encoder(actions, timesteps_tensor)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = (
                torch.cat((prefix_features, future_tokens, action_features), dim=1)
                if prefix_features is not None
                else torch.cat((future_tokens, action_features), dim=1)
            )

            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output)
            pred_velocity = pred[:, -self.action_horizon :]
            actions = actions + dt * pred_velocity

        return actions


def get_action_model(config=None):
    return FlowmatchingActionHeadTouch(full_config=config)
