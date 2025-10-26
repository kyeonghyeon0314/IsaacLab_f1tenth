# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
CNN+MLP hybrid models for F1TENTH racing with LiDAR.

Based on TinyLidarNet architecture:
https://arxiv.org/html/2410.07447v1

Architecture:
    LiDAR (1080) → 1D CNN → compressed features (64)
    Vehicle state (2) → pass through
    Concat (66) → MLP → actions/Q-values
"""

import torch
import torch.nn as nn
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin


class LidarFeatureExtractor(nn.Module):
    """
    1D CNN for LiDAR feature extraction.

    Compresses 1080-dimensional LiDAR scan into 64-dimensional feature vector
    by learning spatial patterns (walls, corners, obstacles).

    Architecture inspired by TinyLidarNet:
        Conv1d(1→32, k=5, s=2) → ReLU    # 1080 → 538
        Conv1d(32→64, k=3, s=2) → ReLU   # 538 → 268
        Conv1d(64→64, k=3, s=2) → ReLU   # 268 → 133
        Flatten                           # 64×133 = 8512
        Linear(8512 → 64)                 # Compress to 64
    """

    def __init__(self, input_size: int = 1080, output_size: int = 64):
        super().__init__()

        self.cnn = nn.Sequential(
            # Conv Block 1: Extract low-level features
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            # Conv Block 2: Extract mid-level features
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # Conv Block 3: Extract high-level features
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Calculate flattened size after convolutions
        # Formula: output_length = floor((input_length + 2*padding - kernel_size) / stride) + 1
        # After Conv1: (1080 + 2*2 - 5)/2 + 1 = 540
        # After Conv2: (540 + 2*1 - 3)/2 + 1 = 270
        # After Conv3: (270 + 2*1 - 3)/2 + 1 = 135
        # Total: 64 * 135 = 8640
        self.flatten_size = 8640

        # Compression layer
        self.fc = nn.Linear(self.flatten_size, output_size)

    def forward(self, lidar: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lidar: (batch_size, 1080) LiDAR scan

        Returns:
            features: (batch_size, 64) compressed features
        """
        # Add channel dimension: (batch, 1080) → (batch, 1, 1080)
        x = lidar.unsqueeze(1)

        # Apply CNN
        x = self.cnn(x)  # → (batch, 64, 135)

        # Flatten
        x = x.flatten(start_dim=1)  # → (batch, 8640)

        # Compress
        x = self.fc(x)  # → (batch, 64)

        return x


class CNNMLPPolicy(GaussianMixin, Model):
    """
    CNN+MLP Policy Network for F1TENTH racing.

    Input:
        - LiDAR scan (1080)
        - Vehicle state (2): forward_speed, steering

    Output:
        - Actions (2): steering_velocity, target_velocity

    Architecture:
        LiDAR (1080) → CNN → features (64)
        Vehicle state (2) → pass through
        Concat (66) → MLP (66→64→64→2) → actions
    """

    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True,
                 min_log_std=-20, max_log_std=2,
                 reduction="sum", initial_log_std=0):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std,
                              min_log_std, max_log_std, reduction)

        # LiDAR feature extractor
        self.lidar_extractor = LidarFeatureExtractor(
            input_size=1080,
            output_size=64
        )

        # MLP: processes concatenated features (64 + 2 = 66)
        # Architecture: 66 → 64 → 64 → 2 (matches YAML config)
        self.mlp = nn.Sequential(
            nn.Linear(66, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions),  # 2 actions
            nn.Tanh()  # squash to [-1, 1]
        )

        # Log std parameter for Gaussian policy
        self.log_std_parameter = nn.Parameter(
            torch.full((self.num_actions,), initial_log_std),
            requires_grad=True
        )

    def compute(self, inputs, role=""):
        """
        Forward pass.

        Args:
            inputs: dict with key "states" containing observations (batch, 1082)
            role: unused (for skrl compatibility)

        Returns:
            mean: action mean (batch, 2)
            log_std: action log std (2,)
            outputs: empty dict (for skrl compatibility)
        """
        states = inputs["states"]

        # Split input into LiDAR and vehicle state
        lidar = states[:, :1080]           # (batch, 1080)
        vehicle_state = states[:, 1080:]   # (batch, 2)

        # Extract LiDAR features
        lidar_features = self.lidar_extractor(lidar)  # (batch, 64)

        # Concatenate features
        features = torch.cat([lidar_features, vehicle_state], dim=1)  # (batch, 66)

        # Pass through MLP
        mean = self.mlp(features)  # (batch, 2)

        return mean, self.log_std_parameter, {}


class CNNMLPCritic(DeterministicMixin, Model):
    """
    CNN+MLP Critic Network (Q-function) for F1TENTH racing.

    Input:
        - LiDAR scan (1080)
        - Vehicle state (2): forward_speed, steering
        - Actions (2): steering_velocity, target_velocity

    Output:
        - Q-value (1): estimated return

    Architecture:
        LiDAR (1080) → CNN → features (64)
        Vehicle state (2) + Actions (2) → concat (4)
        Concat all (68) → MLP (68→256→128→64→1) → Q-value
    """

    def __init__(self, observation_space, action_space, device,
                 clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        # LiDAR feature extractor (shared architecture with policy)
        self.lidar_extractor = LidarFeatureExtractor(
            input_size=1080,
            output_size=64
        )

        # MLP: processes concatenated features (64 + 2 + 2 = 68)
        # Architecture: 68 → 256 → 128 → 64 → 1 (matches YAML config)
        self.mlp = nn.Sequential(
            nn.Linear(68, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Q-value
        )

    def compute(self, inputs, role=""):
        """
        Forward pass.

        Args:
            inputs: dict with keys:
                - "states": observations (batch, 1082)
                - "taken_actions": actions (batch, 2)
            role: unused (for skrl compatibility)

        Returns:
            q_value: estimated Q-value (batch, 1)
            outputs: empty dict (for skrl compatibility)
        """
        states = inputs["states"]
        actions = inputs["taken_actions"]

        # Split states into LiDAR and vehicle state
        lidar = states[:, :1080]           # (batch, 1080)
        vehicle_state = states[:, 1080:]   # (batch, 2)

        # Extract LiDAR features
        lidar_features = self.lidar_extractor(lidar)  # (batch, 64)

        # Concatenate all features
        features = torch.cat([lidar_features, vehicle_state, actions], dim=1)  # (batch, 68)

        # Pass through MLP
        q_value = self.mlp(features)  # (batch, 1)

        return q_value, {}
