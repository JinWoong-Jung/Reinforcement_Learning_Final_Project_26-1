from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError("PyTorch is required for DPO training and inference.")


class DPOPolicyNetwork(nn.Module if nn is not None else object):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Sequence[int]) -> None:
        _require_torch()
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = int(obs_dim)
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, int(h)))
            layers.append(nn.ReLU())
            in_dim = int(h)
        layers.append(nn.Linear(in_dim, int(action_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)


@dataclass
class DPOModelMetadata:
    obs_dim: int
    num_problems: int
    action_dim: int
    hidden_sizes: list[int]


class DPOPolicyModel:
    """Lightweight preference-trained policy with an SB3-like predict() API."""

    def __init__(
        self,
        obs_dim: int,
        num_problems: int,
        hidden_sizes: Sequence[int] = (256, 256),
        obs_mean: np.ndarray | None = None,
        obs_std: np.ndarray | None = None,
        state_dict: dict | None = None,
        device: str = "cpu",
    ) -> None:
        _require_torch()
        self.metadata = DPOModelMetadata(
            obs_dim=int(obs_dim),
            num_problems=int(num_problems),
            action_dim=int(num_problems) + 1,
            hidden_sizes=[int(x) for x in hidden_sizes],
        )
        self.device = torch.device(device)
        self.policy = DPOPolicyNetwork(
            obs_dim=self.metadata.obs_dim,
            action_dim=self.metadata.action_dim,
            hidden_sizes=self.metadata.hidden_sizes,
        ).to(self.device)
        if state_dict is not None:
            self.policy.load_state_dict(state_dict)
        self.policy.eval()
        self.obs_mean = (
            np.asarray(obs_mean, dtype=np.float32)
            if obs_mean is not None
            else np.zeros(self.metadata.obs_dim, dtype=np.float32)
        )
        self.obs_std = (
            np.asarray(obs_std, dtype=np.float32)
            if obs_std is not None
            else np.ones(self.metadata.obs_dim, dtype=np.float32)
        )
        self.obs_std = np.maximum(self.obs_std, 1e-6)

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        return (obs - self.obs_mean) / self.obs_std

    def _decode_action(self, action_idx: int) -> np.ndarray:
        if int(action_idx) <= 0:
            return np.array([0, 0], dtype=np.int64)
        target_idx = int(action_idx) - 1
        return np.array([1, target_idx], dtype=np.int64)

    def predict(self, obs, deterministic: bool = True):
        _require_torch()
        obs_np = self._normalize_obs(np.asarray(obs, dtype=np.float32))
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.policy(obs_t)[0]
            if deterministic:
                action_idx = int(torch.argmax(logits).item())
            else:
                probs = torch.softmax(logits, dim=-1)
                action_idx = int(torch.multinomial(probs, num_samples=1).item())
        return self._decode_action(action_idx), None

    def save(self, path: str) -> None:
        _require_torch()
        payload = {
            "metadata": {
                "obs_dim": self.metadata.obs_dim,
                "num_problems": self.metadata.num_problems,
                "action_dim": self.metadata.action_dim,
                "hidden_sizes": self.metadata.hidden_sizes,
            },
            "state_dict": self.policy.state_dict(),
            "obs_mean": self.obs_mean,
            "obs_std": self.obs_std,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "DPOPolicyModel":
        _require_torch()
        payload = torch.load(path, map_location=device, weights_only=False)
        meta = payload["metadata"]
        return cls(
            obs_dim=int(meta["obs_dim"]),
            num_problems=int(meta["num_problems"]),
            hidden_sizes=list(meta["hidden_sizes"]),
            obs_mean=np.asarray(payload.get("obs_mean"), dtype=np.float32),
            obs_std=np.asarray(payload.get("obs_std"), dtype=np.float32),
            state_dict=payload["state_dict"],
            device=device,
        )
