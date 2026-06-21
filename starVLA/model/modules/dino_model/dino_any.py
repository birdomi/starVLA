"""
DINO wrapper for DINOv2/DINOv3 torch hub models.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import torch
from torch import nn
from torchvision import transforms


def apply_transform(view, transform):
    return transform(view)


class DINOBackBone(nn.Module):
    def __init__(self, backbone_name="dinov3_vits16", checkpoint_path=None, repo_dir=None) -> None:
        super().__init__()
        hub_repo = "facebookresearch/dinov3" if backbone_name.startswith("dinov3") else "facebookresearch/dinov2"
        local_repo = "facebookresearch_dinov3_main" if backbone_name.startswith("dinov3") else "facebookresearch_dinov2_main"
        torch_home = os.environ.get("TORCH_HOME", "~/.cache/torch/")
        default_code_path = os.path.expanduser(f"{torch_home}/hub/{local_repo}")
        checkpoint_path = os.path.expanduser(checkpoint_path) if checkpoint_path else None
        repo_dir = os.path.expanduser(repo_dir) if repo_dir else default_code_path
        try:
            if checkpoint_path:
                self.body = torch.hub.load(repo_dir, backbone_name, source="local", weights=checkpoint_path)
            else:
                self.body = torch.hub.load(hub_repo, backbone_name)
        except Exception:
            import traceback

            traceback.print_exc()
            print(f"Failed to load {backbone_name} from torch hub, loading from local")
            weights_path = checkpoint_path or os.path.expanduser(f"{torch_home}/hub/checkpoints/{backbone_name}_pretrain.pth")
            if backbone_name.startswith("dinov3"):
                self.body = torch.hub.load(repo_dir, backbone_name, source="local", weights=weights_path)
            else:
                self.body = torch.hub.load(repo_dir, backbone_name, source="local", pretrained=False)
                self.body.load_state_dict(torch.load(weights_path))

        channel_map = {
            "dinov2_vits14": 384,
            "dinov2_vitb14": 768,
            "dinov2_vitl14": 1024,
            "dinov2_vitg14": 1408,
            "dinov3_vits16": 384,
            "dinov3_vitb16": 768,
            "dinov3_vitl16": 1024,
            "dinov3_vith16": 1280,
            "dinov3_vit7b16": 4096,
        }
        self.num_channels = channel_map.get(backbone_name)
        if self.num_channels is None:
            self.num_channels = getattr(self.body, "num_features", None) or getattr(self.body, "embed_dim", None)
        if self.num_channels is None:
            raise NotImplementedError(f"DINO backbone {backbone_name} channel dim is unknown")

        self.dino_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def forward(self, tensor, include_cls=False):
        features = self.body.forward_features(tensor)
        if isinstance(features, dict):
            xs = features.get("x_norm_patchtokens")
            if xs is None:
                xs = features.get("patch_tokens")
            if xs is None:
                xs = features.get("x_prenorm")
            if xs is None:
                raise KeyError(f"DINO forward_features keys do not include patch tokens: {sorted(features.keys())}")
            cls = features.get("x_norm_clstoken")
            if cls is None:
                cls = features.get("cls_token")
            if include_cls and cls is not None:
                if cls.dim() == 2:
                    cls = cls.unsqueeze(1)
                xs = torch.cat((cls, xs), dim=1)
        else:
            xs = features
            if xs.dim() == 2:
                xs = xs.unsqueeze(1)
        return xs

    def prepare_dino_input(self, img_list):
        with ThreadPoolExecutor() as executor:
            image_tensors = torch.stack(
                [
                    torch.stack(list(executor.map(lambda view: apply_transform(view, self.dino_transform), views)))
                    for views in img_list
                ]
            )

        batch_size, num_view, channels, height, width = image_tensors.shape
        image_tensors = image_tensors.view(batch_size * num_view, channels, height, width)
        return image_tensors.to(next(self.parameters()).device)


def get_dino_model(backone_name="dinov3_vits16", checkpoint_path=None, repo_dir=None) -> DINOBackBone:
    return DINOBackBone(backbone_name=backone_name, checkpoint_path=checkpoint_path, repo_dir=repo_dir)
