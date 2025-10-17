# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
F1TENTH Racing Environment with LiDAR
Based on f1tenth_gym: https://github.com/f1tenth/f1tenth_gym
"""

from __future__ import annotations

import math
import torch
import os
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCaster, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from isaaclab_assets.robots.f1tenth import F1TENTH_CFG


@configclass
class F1TenthEnvCfg(DirectRLEnvCfg):
    """Configuration for F1TENTH racing environment."""

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=10.0)

    # Simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=2)

    # Robot
    robot: ArticulationCfg = F1TENTH_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Track
    track: UsdFileCfg = UsdFileCfg(
        usd_path=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "/workspace/isaaclab/source/isaaclab_assets/isaaclab_assets/f1tenth/tracks/underground_track_physics.usd"
        ),
    )

    # LiDAR - Attach to laser link (URDF: base_to_laser joint at xyz="0.275 0 0.19")
    lidar = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/laser",  # Use laser link from URDF
        mesh_prim_paths=["/World/ground"],
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-135.0, 135.0),
            horizontal_res=0.25,
        ),
        max_distance=30.0,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),  # No offset needed - laser link already positioned
        ray_alignment="yaw",  # Align rays with yaw only (horizontal plane)
        debug_vis=True,       # Enable visualization
    )

    # Environment settings
    episode_length_s = 10.0  # Reduced from 20.0 to encourage faster driving
    decimation = 2

    # Action space
    action_scale_steering = 3.2
    action_scale_velocity = 9.51
    action_space = 2

    # Observation space (LiDAR + vehicle_state)
    observation_space = 1080 + 2  # LiDAR(1080) + speed(1) + steering(1)
    state_space = 0

    # Vehicle parameters (from f1tenth_gym and actual hardware)
    vehicle_params = {
        # Dynamics model parameters (from f1tenth_gym)
        "mu": 1.0489,        # friction coefficient
        "C_Sf": 4.718,       # front cornering stiffness
        "C_Sr": 5.4562,      # rear cornering stiffness
        "lf": 0.15875,       # distance from CG to front axle [m]
        "lr": 0.17145,       # distance from CG to rear axle [m]
        "h": 0.074,          # height of CG [m]
        "m": 3.74,           # mass [kg] (f1tenth_gym default, actual Traxxas: 2.63 kg)
        "I": 0.04712,        # moment of inertia [kg*m^2]

        # Steering constraints
        "s_min": -0.4189,    # minimum steering angle [rad]
        "s_max": 0.4189,     # maximum steering angle [rad]
        "sv_min": -3.2,      # minimum steering velocity [rad/s]
        "sv_max": 3.2,       # maximum steering velocity [rad/s]

        # Longitudinal constraints
        "v_switch": 7.319,   # switching velocity [m/s]
        "a_max": 9.51,       # maximum acceleration [m/s^2]
        "v_min": -5.0,       # minimum velocity [m/s]
        "v_max": 20.0,       # maximum velocity [m/s]

        # Physical dimensions
        "width": 0.31,       # vehicle width [m]
        "length": 0.58,      # vehicle length [m]
        "wheelbase": 0.324,  # wheelbase [m] (from Traxxas spec)
        "wheel_radius": 0.0508,  # wheel radius [m]
    }

    # Motor control parameters (VESC-inspired velocity control)
    motor_control = {
        "type": "velocity",           # "velocity" or "effort"
        "velocity_damping": 0.0,      # damping for velocity integration (0 = no damping)
        "max_wheel_speed": 393.7,     # max wheel angular velocity [rad/s] (~20 m/s / 0.0508 m)
    }

    # Reward weights
    rew_scale_alive = 0.1
    rew_scale_progress = 1.0
    rew_scale_velocity = 0.5
    rew_scale_steering = -0.01
    rew_scale_collision = -10.0
    rew_scale_boundary = -5.0

    # Termination conditions
    max_lateral_deviation = 3.0
    min_velocity = -1.0

    # Spawn zones for randomization
    vehicle_spawn_zone: dict = {
        "x_range": (0.0, 0.0),           # X axis spawn range [min, max]
        "y_range": (-1.0, -0.5),           # Y axis spawn range [min, max]
        "z_fixed": -0.5,                   # Fixed Z height (to avoid collision)
        "yaw_range": (-math.pi/4, math.pi/4), # Yaw orientation range [min, max]
    }

    # Obstacle spawn zones (for future use)
    # Each zone is a dict with x_range, y_range, z_fixed
    obstacle_spawn_zones: list = []


class F1TenthEnv(DirectRLEnv):
    """F1TENTH racing environment with LiDAR-based navigation."""

    cfg: F1TenthEnvCfg

    def __init__(self, cfg: F1TenthEnvCfg, render_mode: str | None = None, **kwargs):
        # Call parent __init__ first - this will call _setup_scene() internally
        super().__init__(cfg, render_mode, **kwargs)

        # Now we can access self.robot and self.lidar created in _setup_scene()
        self._steering_joint_ids, _ = self.robot.find_joints(".*steering_hinge_joint")
        self._rear_wheel_ids, _ = self.robot.find_joints(".*rear_wheel_joint")

        print(f"[DEBUG] Total joints: {self.robot.num_joints}")
        print(f"[DEBUG] Joint names: {self.robot.joint_names}")
        print(f"[DEBUG] Steering joint IDs: {self._steering_joint_ids}")
        print(f"[DEBUG] Rear wheel joint IDs: {self._rear_wheel_ids}")

        self.action_scale_steering = torch.tensor([cfg.action_scale_steering], device=self.device)
        self.action_scale_velocity = torch.tensor([cfg.action_scale_velocity], device=self.device)
        self.previous_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # Reward calculation variables
        self.previous_steering = torch.zeros(self.num_envs, device=self.device)  # For steering stability
        self.lidar_distances = None  # Store LiDAR distances for reward calculation

        # Motor control: velocity integration (VESC-style)
        self.wheel_radius = cfg.vehicle_params["wheel_radius"]
        self.velocity_damping = cfg.motor_control["velocity_damping"]
        self.max_wheel_speed = cfg.motor_control["max_wheel_speed"]
        # Target linear velocity for each environment
        self.target_velocity = torch.zeros(self.num_envs, device=self.device)

        # Stuck detection: track position for movement check
        # Check if vehicle hasn't moved 0.5m in 0.5 seconds (30 steps at 60Hz)
        self.stuck_check_interval = 30  # steps (reduced from 60)
        self.stuck_threshold = 0.5  # meters (reduced from 1.0)
        self.last_check_pos = torch.zeros(self.num_envs, 2, device=self.device)  # XY position
        self.steps_since_last_check = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # Slip detection: track wheel rotation for odometry calculation
        # Store wheel position at last stuck check (not every step)
        self.wheel_pos_at_last_check = torch.zeros(self.num_envs, len(self._rear_wheel_ids), device=self.device)

    def _setup_scene(self):
        """Setup the simulation scene."""
        # Create Articulation and RayCaster here, after scene is initialized
        self.robot = Articulation(self.cfg.robot)
        self.lidar = RayCaster(self.cfg.lidar)

        # Register them with the scene
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["lidar"] = self.lidar

        # Spawn track
        self.cfg.track.func(prim_path="/World/ground", cfg=self.cfg.track)

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)

        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process actions and prepare control commands.

        Args:
            actions: [steering_velocity, acceleration] in normalized space [-1, 1]
        """
        # Steering velocity command
        steering_vel = actions[:, 0] * self.action_scale_steering
        self.steering_vel = steering_vel.unsqueeze(-1)

        # Acceleration command -> integrate to target velocity (VESC-style)
        acceleration = actions[:, 1] * self.action_scale_velocity

        # Integrate acceleration to get target velocity with damping
        # v_target(t+dt) = v_target(t) * (1 - damping) + a * dt
        dt = self.cfg.sim.dt * self.cfg.decimation
        self.target_velocity = self.target_velocity * (1.0 - self.velocity_damping) + acceleration * dt

        # Clamp target velocity to vehicle limits
        v_min = self.cfg.vehicle_params["v_min"]
        v_max = self.cfg.vehicle_params["v_max"]
        self.target_velocity = torch.clamp(self.target_velocity, v_min, v_max)

    def _apply_action(self) -> None:
        """Apply control commands to the robot.

        Uses velocity control for rear wheels (VESC-inspired).
        """
        # Steering: velocity control
        self.robot.set_joint_velocity_target(self.steering_vel, joint_ids=self._steering_joint_ids)

        # Rear wheels: convert linear velocity to angular velocity
        # omega = v / r
        target_wheel_angular_vel = self.target_velocity / self.wheel_radius
        target_wheel_angular_vel = target_wheel_angular_vel.unsqueeze(-1)

        # Clamp to max wheel speed
        target_wheel_angular_vel = torch.clamp(
            target_wheel_angular_vel,
            -self.max_wheel_speed,
            self.max_wheel_speed
        )

        self.robot.set_joint_velocity_target(target_wheel_angular_vel, joint_ids=self._rear_wheel_ids)

    def _get_observations(self) -> dict:
        lidar_data = self.lidar.data.ray_hits_w[..., :3]
        lidar_distances = torch.norm(lidar_data - self.lidar.data.pos_w.unsqueeze(1), dim=-1)

        # Store LiDAR distances for reward calculation
        self.lidar_distances = lidar_distances

        root_state = self.robot.data.root_state_w
        pos, vel, ang_vel = root_state[:, :3], root_state[:, 7:10], root_state[:, 10:13]
        joint_pos = self.robot.data.joint_pos
        if len(self._steering_joint_ids) > 0 and self._steering_joint_ids[0] < joint_pos.shape[1]:
            steering_angle = joint_pos[:, self._steering_joint_ids[0]]
        else:
            steering_angle = torch.zeros(joint_pos.shape[0], device=self.device)

        # Calculate speed as scalar from velocity vector (user requirement: only speed + steering)
        speed = torch.norm(vel[:, :2], dim=-1, keepdim=True)

        vehicle_state = torch.cat([
            speed,                         # 1 dimension
            steering_angle.unsqueeze(-1)   # 1 dimension
        ], dim=-1)  # Total: 2 dimensions
        obs = torch.cat([lidar_distances, vehicle_state], dim=-1)

        # Debug: Print observation shape on first call
        if not hasattr(self, '_obs_shape_printed'):
            print(f"[DEBUG] Observation shape: {obs.shape}")
            print(f"[DEBUG] LiDAR distances shape: {lidar_distances.shape}")
            print(f"[DEBUG] Vehicle state shape: {vehicle_state.shape}")
            print(f"[DEBUG] Speed shape: {speed.shape}")
            print(f"[DEBUG] Steering shape: {steering_angle.unsqueeze(-1).shape}")
            self._obs_shape_printed = True

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """
        Redesigned reward function:
        - Forward progress (50%): Distance traveled in velocity direction
        - Forward speed (10%): Fast driving encouragement
        - Collision penalty (30%): Based on minimum LiDAR distance
        - Steering stability (10%): Penalize abrupt steering changes
        """
        pos = self.robot.data.root_state_w[:, :3]
        vel = self.robot.data.root_state_w[:, 7:10]

        # Get current steering angle
        joint_pos = self.robot.data.joint_pos
        if len(self._steering_joint_ids) > 0 and self._steering_joint_ids[0] < joint_pos.shape[1]:
            current_steering = joint_pos[:, self._steering_joint_ids[0]]
        else:
            current_steering = torch.zeros(self.num_envs, device=self.device)

        # 1) Forward progress reward (50%)
        # Only reward movement in the current velocity direction
        displacement = pos[:, :2] - self.previous_pos[:, :2]  # XY plane displacement
        vel_xy = vel[:, :2]  # XY plane velocity
        speed = torch.norm(vel_xy, dim=-1, keepdim=True) + 1e-6
        forward_direction = vel_xy / speed  # Normalized velocity direction

        # Project displacement onto forward direction
        forward_progress = torch.sum(displacement * forward_direction, dim=-1)
        forward_progress = torch.clamp(forward_progress, 0.0, 1.0)  # Clamp to [0, 1m]
        reward_forward = 0.5 * forward_progress

        # 2) Forward speed reward (10%)
        # Encourage fast driving, normalized to max 10 m/s
        forward_speed = speed.squeeze(-1)
        reward_speed = 0.1 * torch.clamp(forward_speed / 10.0, 0.0, 1.0)

        # 3) Collision penalty (30%)
        # Penalize when minimum LiDAR distance is below threshold (0.2m)
        if self.lidar_distances is not None:
            min_lidar_dist = torch.min(self.lidar_distances, dim=-1)[0]
            collision_threshold = 0.2  # 20cm
            collision_penalty = torch.where(
                min_lidar_dist < collision_threshold,
                -0.3 * (collision_threshold - min_lidar_dist) / collision_threshold,  # Distance-proportional penalty
                torch.zeros_like(min_lidar_dist)
            )
        else:
            collision_penalty = torch.zeros(self.num_envs, device=self.device)

        # 4) Steering stability reward (10%)
        # Penalize abrupt steering changes to prevent rolling
        steering_change = torch.abs(current_steering - self.previous_steering)
        max_steering_change = 0.05  # Max 0.05 rad change per step
        steering_penalty = -0.1 * torch.clamp(steering_change / max_steering_change, 0.0, 1.0)

        # Total reward
        reward = reward_forward + reward_speed + collision_penalty + steering_penalty

        # Update state for next step
        self.previous_pos[:] = pos
        self.previous_steering[:] = current_steering

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos, vel = self.robot.data.root_state_w[:, :3], self.robot.data.root_state_w[:, 7:10]
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Termination conditions
        out_of_bounds = torch.norm(pos[:, :2], dim=-1) > 50.0

        # Collision detection: terminate if minimum LiDAR distance < 0.2m
        collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.lidar_distances is not None:
            min_lidar_dist = torch.min(self.lidar_distances, dim=-1)[0]
            collision = min_lidar_dist < 0.2  # 20cm collision threshold

        # Stuck detection: check if vehicle hasn't moved enough in XY plane
        self.steps_since_last_check += 1
        stuck = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Check every stuck_check_interval steps
        check_now = self.steps_since_last_check >= self.stuck_check_interval
        if check_now.any():
            current_pos_xy = pos[:, :2]
            movement = torch.norm(current_pos_xy - self.last_check_pos, dim=-1)

            # Mark as stuck if movement < threshold
            stuck[check_now] = movement[check_now] < self.stuck_threshold

            # Debug logging: print slip detection for stuck environments
            if stuck.any():
                wheel_radius = 0.0508  # F1tenth wheel radius in meters
                current_wheel_pos = self.robot.data.joint_pos[:, self._rear_wheel_ids]

                # Calculate wheel rotation change over check interval (not just 1 step!)
                wheel_delta = current_wheel_pos - self.wheel_pos_at_last_check

                # Calculate wheel odometry from rotation change
                wheel_odometry = torch.mean(torch.abs(wheel_delta), dim=-1) * wheel_radius

                # Verification: Calculate actual distance using GT positions
                # This validates both wheel odometry and position tracking
                stuck_env_idx = torch.where(stuck)[0]
                for idx in stuck_env_idx:
                    # Current position
                    current_xy = pos[idx, :2]
                    last_check_xy = self.last_check_pos[idx]

                    # GT-based distance (ground truth)
                    gt_distance = torch.norm(current_xy - last_check_xy).item()

                    # Wheel-based distance (odometry)
                    wheel_dist = wheel_odometry[idx].item()

                    # Compare with movement (should be same as gt_distance)
                    movement_dist = movement[idx].item()

                    # Slip ratio: (wheel_distance - actual_distance) / wheel_distance
                    slip_ratio = (wheel_dist - gt_distance) / (wheel_dist + 1e-6)

                    print(f"\n[STUCK DETECTION] Env {idx.item()} at step {self.episode_length_buf[idx].item()}:")
                    print(f"  Current Position: X={current_xy[0].item():.3f}, Y={current_xy[1].item():.3f}, Z={pos[idx, 2].item():.3f}")
                    print(f"  Last Check Position: X={last_check_xy[0].item():.3f}, Y={last_check_xy[1].item():.3f}")
                    print(f"  GT Distance (current - last): {gt_distance:.3f}m")
                    print(f"  Movement variable: {movement_dist:.3f}m (should match GT)")
                    print(f"  Wheel odometry ({self.stuck_check_interval} steps): {wheel_dist:.3f}m")
                    print(f"  Slip ratio: {slip_ratio:.2%}")

                    # Validation check
                    if abs(gt_distance - movement_dist) > 0.001:
                        print(f"  ⚠️  WARNING: GT distance and movement mismatch! Diff: {abs(gt_distance - movement_dist):.6f}m")

            # Update last check position and wheel position for checked envs only
            self.last_check_pos[check_now] = current_pos_xy[check_now]
            self.steps_since_last_check[check_now] = 0

            # Update wheel position at check time (not every step!)
            current_wheel_pos_all = self.robot.data.joint_pos[:, self._rear_wheel_ids]
            self.wheel_pos_at_last_check[check_now] = current_wheel_pos_all[check_now]

        terminated = out_of_bounds | collision | stuck

        # Debug logging: print termination reasons when any environment terminates
        if terminated.any():
            env_idx = torch.where(terminated)[0]
            for idx in env_idx:
                min_lidar = torch.min(self.lidar_distances[idx]).item() if self.lidar_distances is not None else 0.0
                print(f"\n[TERMINATION DEBUG] Env {idx.item()} terminated at step {self.episode_length_buf[idx].item()}:")
                print(f"  Position: X={pos[idx, 0].item():.3f}, Y={pos[idx, 1].item():.3f}, Z={pos[idx, 2].item():.3f}")
                print(f"  Velocity: vx={vel[idx, 0].item():.3f}, vy={vel[idx, 1].item():.3f}, vz={vel[idx, 2].item():.3f}")
                print(f"  Collision (LiDAR < 0.2m): {collision[idx].item()} | Min LiDAR: {min_lidar:.3f}m")
                print(f"  Out of bounds (dist > 50m): {out_of_bounds[idx].item()} | Distance: {torch.norm(pos[idx, :2]).item():.2f}m")
                print(f"  Stuck (moved < {self.stuck_threshold}m in {self.stuck_check_interval} steps): {stuck[idx].item()}")
                print(f"  Time out: {time_out[idx].item()} | Step: {self.episode_length_buf[idx].item()}/{self.max_episode_length-1}")

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Reset joint positions and velocities
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        # Sample random spawn position within vehicle spawn zone
        num_resets = len(env_ids)
        spawn_zone = self.cfg.vehicle_spawn_zone

        # Sample X position within zone
        x_min, x_max = spawn_zone["x_range"]
        x_pos = sample_uniform(x_min, x_max, (num_resets, 1), device=self.device)

        # Sample Y position within zone
        y_min, y_max = spawn_zone["y_range"]
        y_pos = sample_uniform(y_min, y_max, (num_resets, 1), device=self.device)

        # Fixed Z height
        z_pos = torch.full((num_resets, 1), spawn_zone["z_fixed"], device=self.device)

        # Sample yaw orientation within zone
        yaw_min, yaw_max = spawn_zone["yaw_range"]
        yaw = sample_uniform(yaw_min, yaw_max, (num_resets, 1), device=self.device)

        # Combine position
        spawn_pos = torch.cat([x_pos, y_pos, z_pos], dim=-1)

        # Update root state with spawn zone position
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] = spawn_pos

        # Set orientation (quaternion from yaw)
        default_root_state[:, 3] = torch.cos(yaw / 2).squeeze(-1)  # w
        default_root_state[:, 4] = 0.0  # x
        default_root_state[:, 5] = 0.0  # y
        default_root_state[:, 6] = torch.sin(yaw / 2).squeeze(-1)  # z

        # Write to simulation
        self.robot.write_root_state_to_sim(default_root_state, env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Update previous position tracker
        self.previous_pos[env_ids] = default_root_state[:, :3]

        # Reset previous steering for reward calculation
        self.previous_steering[env_ids] = 0.0

        # Reset motor control state
        self.target_velocity[env_ids] = 0.0

        # Reset stuck detection
        self.last_check_pos[env_ids] = default_root_state[env_ids, :2]  # XY position
        self.steps_since_last_check[env_ids] = 0

        # Reset wheel position tracker for slip detection
        self.wheel_pos_at_last_check[env_ids] = self.robot.data.joint_pos[env_ids][:, self._rear_wheel_ids]
