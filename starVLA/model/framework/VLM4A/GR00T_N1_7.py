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
from pathlib import Path
from typing import Optional

import torch

from starVLA.model.framework.VLM4A.CosmosGR00T import Cosmos_GR00T
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


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
    Cosmos-Reason2/Qwen3-VL GR00T with GR00T N1.7 partial init.

    Config keys:
      framework.gr00t_n1_7.checkpoint_path
      framework.gr00t_n1_7.load_prefixes
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self._load_gr00t_n1_7_pretrained()

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
        seen = 0
        skipped_missing = 0
        skipped_shape = 0
        skipped_prefix = 0
        skipped_non_tensor = 0

        for file_path in files:
            state = _load_state_file(file_path)
            if not isinstance(state, dict):
                continue

            for raw_key, tensor in state.items():
                seen += 1
                key = _normalize_key(raw_key)
                if not _prefix_ok(key, prefixes):
                    skipped_prefix += 1
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

        expected_keys = {key for key in model_state if _prefix_ok(key, prefixes)}
        missing_from_checkpoint = sorted(expected_keys - set(load_state))
        if missing_from_checkpoint:
            bad_weights.extend(f"{key}: missing from checkpoint" for key in missing_from_checkpoint[:20])

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
