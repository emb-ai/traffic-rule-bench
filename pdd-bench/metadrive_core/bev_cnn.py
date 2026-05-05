import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomBEVCNN(BaseFeaturesExtractor):
    """
     BEV feature extractor for 3/5/6 channel BEV + state dict observations.

    - 3 channels: branch + attention fusion (legacy pdd-bench setup)
    - 5/6 channels: shared CNN stack (MetaDrive setups)
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        if not isinstance(observation_space, gym.spaces.Dict):
            raise TypeError(
                "CustomBEVCNN expects a Dict observation space with 'image' and 'state' entries. "
                f"Got {type(observation_space)}"
            )

        if not hasattr(observation_space, 'spaces') or "image" not in observation_space.spaces or "state" not in observation_space.spaces:
            available_keys = list(observation_space.spaces.keys()) if hasattr(observation_space, 'spaces') else 'N/A'
            raise KeyError(
                "Observation space must contain 'image' and 'state' keys. "
                f"Available: {available_keys}"
            )

        image_space = observation_space["image"]
        state_space = observation_space["state"]
        if len(image_space.shape) != 3:
            raise ValueError("Image observation must have shape (H, W, C).")

        self.H, self.W, self.C = image_space.shape
        self.state_dim = state_space.shape[0]

        if self.C == 3:
            self._arch = "branch_attn"
            self._build_branch_attn()
        elif self.C in (5, 6):
            self._arch = "conv"
            self._build_conv(self.C)
        else:
            raise ValueError(
                "CustomBEVCNN expects 3, 5, or 6 channels. "
                f"Got {self.C} channels."
            )

        # debug if necessary
        self._forward_count = 0
        self._debug = False

    def _build_branch_attn(self):
        def make_branch():
            return nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )

        self.branch1 = make_branch()
        self.branch2 = make_branch()
        self.branch3 = make_branch()

        with torch.no_grad():
            sample_input = torch.zeros((1, 1, self.H, self.W))
            n_flatten = self.branch1(sample_input).shape[1]

        self._init_state_and_output(n_flatten)

        attn_hidden = max(1, n_flatten // 16)
        self.attn = nn.Sequential(
            nn.Linear(n_flatten, attn_hidden),
            nn.ReLU(),
            nn.Linear(attn_hidden, n_flatten),
            nn.Sigmoid(),
        )

    def _build_conv(self, n_input_channels: int):
        # output_size = (input_size - kernel_size) // stride + 1
        def conv_output_size(input_size, kernel_size, stride):
            return (input_size - kernel_size) // stride + 1

        h1 = conv_output_size(self.H, 5, 2)
        w1 = conv_output_size(self.W, 5, 2)
        h2 = conv_output_size(h1, 5, 2)
        w2 = conv_output_size(w1, 5, 2)
        h3 = conv_output_size(h2, 5, 2)
        w3 = conv_output_size(w2, 5, 2)
        h4 = conv_output_size(h3, 3, 2)
        w4 = conv_output_size(w3, 3, 2)
        h5 = conv_output_size(h4, 3, 2)
        w5 = conv_output_size(w4, 3, 2)

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 8, kernel_size=5, stride=2),
            nn.LayerNorm((8, h1, w1)),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=2),
            nn.LayerNorm((16, h2, w2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.LayerNorm((32, h3, w3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.LayerNorm((64, h4, w4)),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.LayerNorm((128, h5, w5)),
            nn.ReLU(),
        )

        with torch.no_grad():
            sample_input = torch.zeros((1, n_input_channels, self.H, self.W))
            cnn_output = self.cnn(sample_input)
            n_flatten = cnn_output.view(1, -1).shape[1]

        self._init_state_and_output(n_flatten)

    def _init_state_and_output(self, n_flatten: int):
        self.feature_dim = n_flatten
        self.state_hidden_dim = max(64, min(256, self.state_dim * 2))

        self.state_net = nn.Sequential(
            nn.Linear(self.state_dim, self.state_hidden_dim),
            nn.ReLU(),
        )

        self.output = nn.Sequential(
            nn.Linear(n_flatten + self.state_hidden_dim, self.features_dim),
            nn.ReLU(),
        )

    def set_debug(self, debug: bool = True):
        self._debug = debug

    def forward(self, observations) -> torch.Tensor:
        self._forward_count += 1
        image = observations["image"].float()
        state = observations["state"].float()

        if self._debug and self._forward_count <= 3:
            print(f"\n[CustomBEVCNN Forward #{self._forward_count}]")
            print(
                f"  Image batch shape: {image.shape}, min: {image.min():.4f}, "
                f"max: {image.max():.4f}, mean: {image.mean():.4f}"
            )
            print(
                f"  Image non-zero: {(image > 0).sum().item()} / {image.numel()} "
                f"({100 * (image > 0).sum().item() / image.numel():.2f}%)"
            )
            print(
                f"  State batch shape: {state.shape}, min: {state.min():.4f}, "
                f"max: {state.max():.4f}, mean: {state.mean():.4f}"
            )
            print(
                f"  State non-zero: {(state != 0).sum().item()} / {state.numel()} "
                f"({100 * (state != 0).sum().item() / state.numel():.2f}%)"
            )

        x = image.permute(0, 3, 1, 2)

        if self._arch == "branch_attn":
            x1 = x[:, 0:1, :, :]
            x2 = x[:, 1:2, :, :]
            x3 = x[:, 2:3, :, :]

            f1 = self.branch1(x1)
            f2 = self.branch2(x2)
            f3 = self.branch3(x3)

            feats = torch.stack([f1, f2, f3], dim=1)

            weights = self.attn(feats.mean(dim=1))
            weights = weights.unsqueeze(1)
            fused = (feats * weights).sum(dim=1)

            image_feats = fused
        else:
            cnn_features = self.cnn(x)
            image_feats = cnn_features.view(cnn_features.size(0), -1)

        state_feats = self.state_net(state)
        combined = torch.cat([image_feats, state_feats], dim=1)
        output = self.output(combined)

        if self._debug and self._forward_count <= 3:
            print(
                f"  Output shape: {output.shape}, min: {output.min():.4f}, "
                f"max: {output.max():.4f}, mean: {output.mean():.4f}"
            )

        return output
