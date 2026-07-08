# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""
GR00T N1.7 wrapper.

Loads same-name, same-shape tensors from a GR00T N1.7 style checkpoint into
the starVLA GR00T action head. NVIDIA checkpoints commonly use
`action_head.*`; starVLA uses `action_model.*`, so the loader remaps that
prefix before matching.
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from starVLA.model.framework.VLM4A.CosmosGR00T import Cosmos_GR00T
from starVLA.model.tools import FRAMEWORK_REGISTRY
from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)


def _rank0_print(message):
    if int(os.environ.get("RANK", "0")) == 0:
        print(message, flush=True)


def _print_sample(title, rows, limit):
    if not rows:
        return
    shown = rows if limit < 0 else rows[:limit]
    _rank0_print(f"[GR00T_N1_7] {title}: showing {len(shown)}/{len(rows)}")
    for row in shown:
        _rank0_print(f"  {row}")


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except Exception:
            pass
    return getattr(cfg, key, default)


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_prefixes(value, default=("action_model",)):
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        value = value.strip()
        if value.lower() in {"", "all", "*", "none"}:
            return tuple()
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(part).strip() for part in value if str(part).strip())


def _normalize_key(key):
    for prefix in ("module.", "model.", "policy."):
        if key.startswith(prefix):
            key = key[len(prefix) :]
    remaps = (
        ("action_head.", "action_model."),
        ("action_head_diffusion_model.", "action_model.model."),
        ("backbone.model.", "qwen_vl_interface.model."),
    )
    for old, new in remaps:
        if key.startswith(old):
            key = new + key[len(old) :]
    return key


def _prefix_ok(key, prefixes):
    return not prefixes or any(key.startswith(prefix) for prefix in prefixes)


def _files_from_index(root, prefixes):
    files = []
    for index_path in sorted(root.glob("*.index.json")):
        try:
            index = json.loads(index_path.read_text())
        except Exception:
            continue
        weight_map = index.get("weight_map", {})
        selected = {
            root / filename
            for key, filename in weight_map.items()
            if _prefix_ok(_normalize_key(key), prefixes)
        }
        files.extend(sorted(path for path in selected if path.exists()))
    return files


def _checkpoint_files(path, prefixes):
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"GR00T checkpoint not found: {path}")

    indexed = _files_from_index(path, prefixes)
    if indexed:
        return indexed

    patterns = ("*.safetensors", "*.bin", "*.pt", "*.pth")
    files = []
    for pattern in patterns:
        files.extend(sorted(path.glob(pattern)))
    return files


def _load_state_file(path):
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")

    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "module"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


@FRAMEWORK_REGISTRY.register("GR00T_N1_7")
class GR00T_N1_7(Cosmos_GR00T):
    """
    Cosmos GR00T with GR00T N1.7 partial init.

    Config keys:
      framework.gr00t_n1_7.checkpoint_path
      framework.gr00t_n1_7.load_prefixes
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        load_cfg = _cfg_get(self.config.framework, "gr00t_n1_7", None)
        self.vlm_hidden_state_index = int(_cfg_get(load_cfg, "vlm_hidden_state_index", 16))
        self._load_gr00t_n1_7_pretrained()

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
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
            last_hidden = qwenvl_outputs.hidden_states[self.vlm_hidden_state_index]
            if not torch.isfinite(last_hidden).all():
                raise FloatingPointError("GR00T_N1_7 last_hidden has NaN or Inf")

        with torch.autocast("cuda", enabled=False):
            actions = torch.tensor(np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype)
            if not torch.isfinite(actions).all():
                raise FloatingPointError("GR00T_N1_7 actions has NaN or Inf")
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

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype)
                if not torch.isfinite(state).all():
                    raise FloatingPointError("GR00T_N1_7 state has NaN or Inf")
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(
                last_hidden_repeated, actions_target_repeated, state_repeated,
                encoder_attention_mask=backbone_attention_mask,
            )
            if not torch.isfinite(action_loss).all():
                raise FloatingPointError("GR00T_N1_7 action_loss has NaN or Inf")

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> np.ndarray:
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
            last_hidden = qwenvl_outputs.hidden_states[self.vlm_hidden_state_index]

        state = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None
            else None
        )

        with torch.autocast("cuda", enabled=False):
            pred_actions = self.action_model.predict_action(
                last_hidden, state, encoder_attention_mask=backbone_attention_mask
            )

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    def _load_gr00t_n1_7_pretrained(self):
        load_cfg = _cfg_get(self.config.framework, "gr00t_n1_7", None)
        checkpoint_path = _cfg_get(load_cfg, "checkpoint_path", None)
        if not checkpoint_path:
            logger.warning("[GR00T_N1_7] no GR00T N1.7 checkpoint path set")
            return

        prefixes = _as_prefixes(_cfg_get(load_cfg, "load_prefixes", None))
        files = _checkpoint_files(checkpoint_path, prefixes)
        if not files:
            raise FileNotFoundError(f"No checkpoint tensors found under {checkpoint_path}")

        model_state = self.state_dict()
        load_state = {}
        bad_weights = []
        loaded_rows = []
        skipped_prefix_rows = []
        seen = 0
        skipped_missing = 0
        skipped_shape = 0
        skipped_prefix = 0
        skipped_non_tensor = 0
        log_limit = int(os.environ.get("GR00T_LOAD_LOG_LIMIT", "80"))

        _rank0_print(
            "[GR00T_N1_7] strict load start: "
            f"checkpoint={checkpoint_path}, prefixes={prefixes}, files={len(files)}"
        )
        for file_path in files:
            _rank0_print(f"[GR00T_N1_7] load file: {file_path}")

        for file_path in files:
            state = _load_state_file(file_path)
            if not isinstance(state, dict):
                bad_weights.append(f"{file_path}: checkpoint object is {type(state).__name__}, not dict")
                continue

            for raw_key, tensor in state.items():
                seen += 1
                key = _normalize_key(raw_key)
                if not _prefix_ok(key, prefixes):
                    skipped_prefix += 1
                    skipped_prefix_rows.append(f"{raw_key} -> {key}")
                    continue
                if not torch.is_tensor(tensor):
                    skipped_non_tensor += 1
                    bad_weights.append(f"{raw_key}: non-tensor")
                    continue
                target = model_state.get(key)
                if target is None:
                    skipped_missing += 1
                    bad_weights.append(f"{raw_key} -> {key}: target key missing")
                    continue
                if target.shape != tensor.shape:
                    skipped_shape += 1
                    bad_weights.append(
                        f"{raw_key} -> {key}: shape {tuple(tensor.shape)} != {tuple(target.shape)}"
                    )
                    continue
                load_state[key] = tensor
                loaded_rows.append(f"{raw_key} -> {key}: shape={tuple(tensor.shape)}")

        _rank0_print(
            "[GR00T_N1_7] strict load summary: "
            f"seen={seen}, loaded={len(load_state)}, skipped_prefix={skipped_prefix}, "
            f"missing={skipped_missing}, shape_mismatch={skipped_shape}, non_tensor={skipped_non_tensor}"
        )
        _print_sample("loaded", loaded_rows, log_limit)
        _print_sample("skipped by prefix", skipped_prefix_rows, log_limit)
        _print_sample("bad weights", bad_weights, log_limit)

        if bad_weights:
            sample = "\n  ".join(bad_weights[:20])
            raise RuntimeError(
                "GR00T N1.7 strict weight load failed. "
                f"bad={len(bad_weights)}, loaded={len(load_state)}, files={len(files)}\n  {sample}"
            )

        if not load_state:
            raise RuntimeError(
                "GR00T N1.7 checkpoint loaded zero tensors. "
                "Check checkpoint_path, load_prefixes, and DiT shape."
            )

        self.load_state_dict(load_state, strict=False)
        logger.info(
            "[GR00T_N1_7] loaded %d tensors from %d files "
            "(seen=%d, skipped_prefix=%d, skipped_missing=%d, "
            "skipped_shape=%d, skipped_non_tensor=%d)",
            len(load_state),
            len(files),
            seen,
            skipped_prefix,
            skipped_missing,
            skipped_shape,
            skipped_non_tensor,
        )
