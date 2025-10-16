#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Test script for F1TENTH environment.
Usage:
    python test_f1tenth.py --num_envs 16
"""

import argparse

import torch

from isaaclab.app import AppLauncher

# Add command line arguments
parser = argparse.ArgumentParser(description="Test F1TENTH environment")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
# AppLauncher will add --headless automatically
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import Isaac Lab modules (must be after AppLauncher)
import gymnasium as gym

import isaaclab_tasks  # noqa: F401 - Registers F1TENTH environment
from isaaclab_tasks.direct.f1tenth import F1TenthEnvCfg


def main():
    """Test F1TENTH environment with random actions."""

    # Create environment configuration
    env_cfg = F1TenthEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    # Create environment
    env = gym.make("Isaac-F1tenth-Direct-v0", cfg=env_cfg)

    print("\n" + "="*80)
    print("F1TENTH Environment Test")
    print("="*80)
    print(f"Number of environments: {env.unwrapped.num_envs}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Episode length: {env_cfg.episode_length_s}s")
    print("="*80 + "\n")

    # Reset environment
    obs, _ = env.reset()
    print(f"Initial observation shape: {obs['policy'].shape}")
    print(f"  - LiDAR scans: 1080 values")
    print(f"  - Vehicle state: 6 values [vx, vy, yaw_rate, steering, pos_x, pos_y]")

    # Run random actions
    num_episodes = 5
    episode_count = 0
    step_count = 0

    # Track initial position and joint angles for slip detection
    initial_pos_xy = obs["policy"][0, 1084:1086].clone()
    initial_joint_pos = env.unwrapped.robot.data.joint_pos[0].clone()

    print(f"\nRunning {num_episodes} episodes with random actions...\n")

    while episode_count < num_episodes:
        # Random action: [steering_velocity, acceleration]
        # steering_velocity: -1 to 1 (scaled to -3.2 to 3.2 rad/s)
        # acceleration: -1 to 1 (scaled to -9.51 to 9.51 m/s²)
        action = torch.randn((env.unwrapped.num_envs, 2), device=env.unwrapped.device)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

        # Check for episode completion
        if terminated.any() or truncated.any():
            episode_count += 1
            print(f"Episode {episode_count} completed after {step_count} steps!")
            print(f"  - Average reward: {reward.mean().item():.3f}")
            print(f"  - Terminated: {terminated.sum().item()} envs")
            print(f"  - Truncated: {truncated.sum().item()} envs")

            # Analyze termination reasons
            robot_pos = env.unwrapped.robot.data.root_state_w[:, :3]
            robot_vel = env.unwrapped.robot.data.root_state_w[:, 7:10]

            collision = robot_pos[:, 2] < 0.05
            out_of_bounds = torch.norm(robot_pos[:, :2], dim=-1) > 50.0
            reverse_too_much = robot_vel[:, 0] < -1.0

            print(f"\n  Termination reasons:")
            print(f"    - Collision (Z < 0.05m): {collision.sum().item()} envs | Z = {robot_pos[:, 2].mean().item():.3f}m")
            print(f"    - Out of bounds (>50m): {out_of_bounds.sum().item()} envs | Distance = {torch.norm(robot_pos[:, :2], dim=-1).mean().item():.2f}m")
            print(f"    - Reverse too much (<-1 m/s): {reverse_too_much.sum().item()} envs | Vel X = {robot_vel[:, 0].mean().item():.2f} m/s")
            print(f"    - Position: X={robot_pos[:, 0].mean().item():.2f}, Y={robot_pos[:, 1].mean().item():.2f}, Z={robot_pos[:, 2].mean().item():.3f}")

            # Check LiDAR data
            lidar_scans = obs['policy'][:, :1080]
            print(f"\n  - LiDAR min distance: {lidar_scans.min().item():.2f}m")
            print(f"  - LiDAR max distance: {lidar_scans.max().item():.2f}m")

            # Check vehicle state
            vehicle_state = obs['policy'][:, 1080:]
            print(f"  - Avg velocity X: {vehicle_state[:, 0].mean().item():.2f} m/s")
            print(f"  - Avg steering: {vehicle_state[:, 3].mean().item():.3f} rad")

            # Calculate and print wheel slip
            actual_dist = torch.norm(obs["policy"][0, 1084:1086] - initial_pos_xy).item()
            # wheel odometry is sum of absolute wheel rotations * radius
            wheel_radius = 0.0508  # m
            wheel_rotations = env.unwrapped.robot.data.joint_pos[0, [0, 2]] - initial_joint_pos[[0, 2]]
            odom_dist = torch.sum(torch.abs(wheel_rotations)).item() * wheel_radius
            slip = (odom_dist - actual_dist) / (odom_dist + 1e-6)
            print(f"  - Wheel Odometry: {odom_dist:.3f}m | Actual Distance: {actual_dist:.3f}m | Slip: {slip:.2%}")
            print()

            # Reset step counter for next episode
            step_count = 0

            # Reset tracking variables after environment reset
            obs, _ = env.reset()
            initial_pos_xy = obs["policy"][0, 1084:1086].clone()
            initial_joint_pos = env.unwrapped.robot.data.joint_pos[0].clone()

    print("="*80)
    print("Test completed successfully! ✓")
    print("="*80)

    # Close environment
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
