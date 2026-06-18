from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
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


def _ensure_lightning_rank_zero_stub() -> None:
    try:
        importlib.import_module("lightning.fabric.utilities")
        return
    except ModuleNotFoundError:
        pass

    lightning = sys.modules.setdefault("lightning", types.ModuleType("lightning"))
    fabric = sys.modules.setdefault("lightning.fabric", types.ModuleType("lightning.fabric"))
    utilities = types.ModuleType("lightning.fabric.utilities")

    def rank_zero_only(fn=None, *args, **kwargs):
        if fn is None:
            return lambda wrapped: wrapped
        return fn

    utilities.rank_zero_only = rank_zero_only
    lightning.fabric = fabric
    fabric.utilities = utilities
    sys.modules["lightning.fabric.utilities"] = utilities


def _import_angle_factory(touch_cfg: Any):
    repo_path = _resolve_path(_cfg_get(touch_cfg, "repo_path", "../touch-rep"))
    if repo_path and repo_path.is_dir() and str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    _ensure_lightning_rank_zero_stub()
    module = importlib.import_module("tactile_ssl.model.angle_transformer")
    model_size = _cfg_get(touch_cfg, "model_size", "tiny")
    factory_name = {"tiny": "angle_tiny", "small": "angle_small"}.get(model_size, model_size)
    return getattr(module, factory_name)


class TouchAngleEncoder(nn.Module):
    """AngleTransformer wrapper. Returns tokens aligned to VLM hidden dim."""

    def __init__(self, touch_cfg: Any, target_dim: int):
        super().__init__()
        factory = _import_angle_factory(touch_cfg)

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

        checkpoint = _resolve_path(_cfg_get(touch_cfg, "checkpoint_encoder", None))
        if checkpoint is not None:
            self.load_encoder(checkpoint)

        if not self.train_encoder:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

    def load_encoder(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model") or checkpoint.get("state_dict") or checkpoint

        prefixes = (
            "model_encoder.",
            "teacher_encoder.backbone.",
            "student_encoder.backbone.",
            "encoder.",
        )
        raw = None
        for prefix in prefixes:
            stripped = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
            if stripped:
                raw = stripped
                break
        if raw is None:
            raw = state_dict

        model_state = self.encoder.state_dict()
        filtered = {k: v for k, v in raw.items() if k in model_state and v.shape == model_state[k].shape}
        self.encoder.load_state_dict(filtered, strict=False)
        print(f"Loaded AngleTransformer encoder keys: {len(filtered)}/{len(raw)} from {checkpoint_path}")

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
                raise ValueError(f"Touch sequence length {steps} does not match AngleTransformer length {seq_len}.")
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
