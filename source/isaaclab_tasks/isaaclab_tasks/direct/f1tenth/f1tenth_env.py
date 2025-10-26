# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
LiDAR를 사용하는 F1TENTH 레이싱 환경
f1tenth_gym 기반: https://github.com/f1tenth/f1tenth_gym
"""

from __future__ import annotations

import gymnasium as gym
import math
import numpy as np
import torch
import torch.nn.functional as F
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
    """F1TENTH 레이싱 환경을 위한 설정입니다."""

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

    # LiDAR - 레이저 링크에 부착 (URDF: base_to_laser 조인트, xyz="0.275 0 0.19")
    lidar = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/laser",  # URDF의 레이저 링크 사용
        mesh_prim_paths=["/World/ground"],
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-135.0, 135.0),
            horizontal_res=0.25,
        ),
        max_distance=30.0,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),  # 오프셋 필요 없음 - 레이저 링크가 이미 위치함
        ray_alignment="yaw",  # 광선을 yaw에만 정렬 (수평면)
        debug_vis=True,       # 시각화 활성화
    )

    # 환경 설정
    episode_length_s = 10.0  # 더 빠른 주행을 장려하기 위해 20.0에서 감소
    decimation = 2

    # 행동 공간
    action_scale_steering = 3.2  # 조향 속도 스케일
    # action[:, 0]: 조향 속도 [-1, 1] → [-3.2, 3.2] rad/s
    # action[:, 1]: 목표 선속도 [-1, 1] → [v_min, v_max] m/s (직접 속도 제어)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)   # SAC 맞춤 설정

    # 관찰 공간 (LiDAR + 차량 상태)
    observation_space = 1080 + 2  # LiDAR(1080) + forward_speed(1, 방향포함) + 조향(1)
    state_space = 0

    # 차량 파라미터
    vehicle_params = {
        # 종방향 제약 조건
        "v_min": 0.0,        # 최소 속도 [m/s] (후진 금지, 전진만 허용)
        "v_max": 5.0,        # 최대 속도 [m/s]
        # 물리적 치수
        "wheel_radius": 0.0508,  # 바퀴 반지름 [m]
    }

    # 모터 제어 파라미터 (VESC 속도 제어 모드)
    motor_control = {
        # VESC 6 MkV: 목표 RPM 설정 시 내장 PID로 빠르게 추종
        "max_wheel_speed": 393.7,     # 최대 바퀴 각속도 [rad/s] (~20 m/s / 0.0508 m)
    }

    # 보상 가중치는 _get_rewards() 함수 내에 하드코딩되어 있습니다.

    # 무작위화를 위한 스폰 영역
    vehicle_spawn_zone: dict = {
        "x_range": (0.0, 0.0),           # X축 스폰 범위 [최소, 최대]
        "y_range": (-0.70, -0.60),           # Y축 스폰 범위 [최소, 최대]
        "z_fixed": -0.5,                   # 고정 Z 높이 (충돌 방지)
        "yaw_range": (-math.pi/4, 0), # Yaw 방향 범위 [최소, 최대]
    }

    # 트랙 중심선 waypoints (X, Y 좌표)
    # 진행 거리 기반 보상 계산에 사용
    centerline_waypoints = [
        [0.0, -0.6],
        [1.86, -0.6],
        [1.86, -1.62],
        [4.87, -1.62],
        [4.87, 2.41],
        [-3.16, 2.41],
        [-3.16, -1.61],
        [-1.14, -1.61],
        [-1.14, -0.6],
        [0.0, -0.6],  # 시작점으로 복귀 (폐곡선)
    ]

    # 디버그 모드 (터미널 출력 제어)
    debug_mode: bool = True  # True로 설정하면 상세 디버그 로그 출력


class F1TenthEnv(DirectRLEnv):
    """LiDAR 기반 내비게이션을 사용하는 F1TENTH 레이싱 환경입니다."""

    cfg: F1TenthEnvCfg

    def __init__(self, cfg: F1TenthEnvCfg, render_mode: str | None = None, **kwargs):
        # 먼저 부모 __init__을 호출합니다. 그러면 내부적으로 _setup_scene()이 호출됩니다.
        super().__init__(cfg, render_mode, **kwargs)

        # 이제 _setup_scene()에서 생성된 self.robot 및 self.lidar에 액세스할 수 있습니다.
        self._steering_joint_ids, _ = self.robot.find_joints(".*steering_hinge_joint")
        self._rear_wheel_ids, _ = self.robot.find_joints(".*rear_wheel_joint")

        self.action_scale_steering = torch.tensor([cfg.action_scale_steering], device=self.device)
        # action_scale_velocity는 직접 속도 제어로 변경되어 더 이상 필요 없음
        self.previous_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # 보상 계산 변수
        self.lidar_distances = None  # 보상 계산을 위해 LiDAR 거리 저장

        # IMU와 유사한 참조 프레임: 스폰 시 초기 방향 저장
        # 차량이 회전할 때 후진 주행이 보상받는 것을 방지합니다.
        self.initial_heading = torch.zeros(self.num_envs, 2, device=self.device)  # (num_envs, 2) XY 방향 벡터

        # _get_dones()와 _get_rewards() 간의 공유된 충돌 감지 결과
        # _get_dones()가 먼저 이것을 계산한 다음 _get_rewards()가 사용합니다.
        self.current_collision_detected = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 모터 제어: 직접 속도 제어 (VESC 속도 제어 모드)
        self.wheel_radius = cfg.vehicle_params["wheel_radius"]
        self.max_wheel_speed = cfg.motor_control["max_wheel_speed"]
        # 각 환경의 목표 선형 속도 (VESC RPM 명령)
        self.target_velocity = torch.zeros(self.num_envs, device=self.device)

        # 막힘 감지: 움직임 확인을 위해 위치 추적
        # 차량이 적절한 시간 내에 움직이지 않았는지 확인
        self.stuck_check_interval = 120  # 스텝 (60Hz에서 2초)
        self.stuck_threshold = 0.01  # 미터 (최소 1cm 움직임)
        self.last_check_pos = torch.zeros(self.num_envs, 2, device=self.device)  # XY 위치
        self.steps_since_last_check = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # 슬립 감지: 주행 거리계 계산을 위해 바퀴 회전 추적
        # 마지막 막힘 확인 시점의 바퀴 위치 저장 (매 스텝 아님)
        self.wheel_pos_at_last_check = torch.zeros(self.num_envs, len(self._rear_wheel_ids), device=self.device)

        # 트랙 중심선 초기화 (진행 거리 기반 보상)
        self._init_centerline()

    def _setup_scene(self):
        """시뮬레이션 장면을 설정합니다."""
        # 장면이 초기화된 후 여기에 Articulation 및 RayCaster를 생성합니다.
        self.robot = Articulation(self.cfg.robot)
        self.lidar = RayCaster(self.cfg.lidar)

        # 장면에 등록합니다.
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["lidar"] = self.lidar

        # 트랙 스폰
        self.cfg.track.func(prim_path="/World/ground", cfg=self.cfg.track)

        # 환경 복제
        self.scene.clone_environments(copy_from_source=False)

        # 조명 추가
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """행동을 처리하고 제어 명령을 준비합니다.

        Args:
            actions: 정규화된 공간 [-1, 1]의 [조향_속도, 목표_속도]
                    - actions[:, 0]: 조향 속도 [-1, 1]
                    - actions[:, 1]: 목표 선속도 [-1, 1] → [v_min, v_max]
        """
        # 조향 속도 명령
        steering_vel = actions[:, 0] * self.action_scale_steering
        self.steering_vel = steering_vel.unsqueeze(-1)

        # 속도 명령 (VESC 속도 제어 모드)
        # VESC 6 MkV는 목표 RPM을 설정하면 내장 PID로 빠르게 추종
        # Action space [-1, 1]을 속도 범위 [v_min, v_max]로 선형 매핑
        v_min = self.cfg.vehicle_params["v_min"]
        v_max = self.cfg.vehicle_params["v_max"]

        # action=-1 → v_min, action=0 → (v_min+v_max)/2, action=+1 → v_max
        self.target_velocity = (actions[:, 1] + 1.0) / 2.0 * (v_max - v_min) + v_min

    def _apply_action(self) -> None:
        """로봇에 제어 명령을 적용합니다.

        후륜에 속도 제어를 사용합니다 (VESC 기반).
        """
        # 조향: 속도 제어
        self.robot.set_joint_velocity_target(self.steering_vel, joint_ids=self._steering_joint_ids)

        # 후륜: 선형 속도를 각속도로 변환
        # omega = v / r
        target_wheel_angular_vel = self.target_velocity / self.wheel_radius
        target_wheel_angular_vel = target_wheel_angular_vel.unsqueeze(-1)

        # 최대 바퀴 속도로 제한
        target_wheel_angular_vel = torch.clamp(
            target_wheel_angular_vel,
            -self.max_wheel_speed,
            self.max_wheel_speed
        )

        self.robot.set_joint_velocity_target(target_wheel_angular_vel, joint_ids=self._rear_wheel_ids)

    def _get_observations(self) -> dict:
        lidar_data = self.lidar.data.ray_hits_w[..., :3]
        lidar_distances = torch.norm(lidar_data - self.lidar.data.pos_w.unsqueeze(1), dim=-1)

        # 보상 계산을 위해 LiDAR 거리 저장
        self.lidar_distances = lidar_distances

        root_state = self.robot.data.root_state_w
        pos, vel, _ = root_state[:, :3], root_state[:, 7:10], root_state[:, 10:13]
        joint_pos = self.robot.data.joint_pos
        if len(self._steering_joint_ids) > 0 and self._steering_joint_ids[0] < joint_pos.shape[1]:
            steering_angle = joint_pos[:, self._steering_joint_ids[0]]
        else:
            steering_angle = torch.zeros(joint_pos.shape[0], device=self.device)

        # 전진 방향 속도 계산 (트랙 중심선 기준, 양수=전진, 음수=후진)
        # 현재 위치에서 가장 가까운 centerline waypoint 찾기
        pos_xy = pos[:, :2]  # (num_envs, 2)
        distances = torch.norm(
            pos_xy.unsqueeze(1) - self.centerline.unsqueeze(0),
            dim=-1
        )  # (num_envs, num_waypoints)
        closest_idx = torch.argmin(distances, dim=-1)  # (num_envs,)
        next_idx = (closest_idx + 1) % len(self.centerline)

        # 트랙 진행 방향 계산 (현재 waypoint → 다음 waypoint)
        current_waypoint = self.centerline[closest_idx]  # (num_envs, 2)
        next_waypoint = self.centerline[next_idx]  # (num_envs, 2)
        forward_direction = next_waypoint - current_waypoint  # (num_envs, 2)
        forward_direction = forward_direction / (torch.norm(forward_direction, dim=-1, keepdim=True) + 1e-6)

        vel_xy = vel[:, :2]  # XY 평면 속도
        forward_speed = torch.sum(vel_xy * forward_direction, dim=-1, keepdim=True)  # 전진 속도 투영

        vehicle_state = torch.cat([
            forward_speed,                 # 1 차원 (방향 포함: +전진, -후진)
            steering_angle.unsqueeze(-1)   # 1 차원
        ], dim=-1)  # 총: 2 차원
        obs = torch.cat([lidar_distances, vehicle_state], dim=-1)
        return {"policy": obs}

    def _detect_collision_consecutive(
        self,
        lidar_distances: torch.Tensor,
        threshold: float = 0.2,
        consecutive_count: int = 5
    ) -> torch.Tensor:
        """
        임계값 이하의 연속적인 LiDAR 포인트를 기반으로 충돌을 감지합니다.

        이 방법은 최소 거리만 사용하는 것보다 더 현실적입니다. 실제
        충돌은 가까운 표면을 감지하는 여러 인접 광선을 포함하기 때문입니다.

        Args:
            lidar_distances: LiDAR 거리 측정값 (num_envs, num_rays)
            threshold: 충돌 거리 임계값 (미터)
            consecutive_count: 충돌에 필요한 연속 포인트 수

        Returns:
            각 환경에 대한 충돌을 나타내는 부울 텐서 (num_envs,)
        """
        if lidar_distances is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 부울 마스크 생성: 거리가 임계값보다 작으면 True
        close_mask = (lidar_distances < threshold).float()  # (num_envs, num_rays)

        # 1D 컨볼루션을 사용하여 연속적인 True 값 계산
        # 커널: 길이가 consecutive_count인 [1, 1, ..., 1]
        kernel = torch.ones(1, 1, consecutive_count, device=self.device)

        # 채널 차원 추가: (num_envs, 1, num_rays)
        close_mask_expanded = close_mask.unsqueeze(1)

        # 컨볼루션: 출력은 연속 값의 합을 보여줍니다.
        # 패딩은 출력이 입력과 동일한 길이를 갖도록 보장합니다.
        conv_result = F.conv1d(close_mask_expanded, kernel, padding=consecutive_count // 2)

        # 어떤 위치의 합이 consecutive_count와 같으면 해당 포인트는 모두 연속적입니다.
        has_consecutive = (conv_result >= consecutive_count).any(dim=-1).squeeze(1)

        return has_consecutive

    def _init_centerline(self):
        """
        트랙 중심선을 초기화하고 waypoints 사이를 보간합니다.
        진행 거리 기반 보상 계산에 사용됩니다.
        """
        # Waypoints를 numpy array로 변환
        import numpy as np
        waypoints_np = np.array(self.cfg.centerline_waypoints, dtype=np.float32)

        # Waypoints 사이를 선형 보간하여 촘촘한 점 생성
        # 각 구간을 10개 점으로 나눔 (해상도 조정 가능)
        interpolated_points = []
        for i in range(len(waypoints_np) - 1):
            start = waypoints_np[i]
            end = waypoints_np[i + 1]
            # 시작점부터 끝점 전까지 (끝점은 다음 구간의 시작점)
            for j in range(10):
                t = j / 10.0
                point = start + t * (end - start)
                interpolated_points.append(point)

        # 마지막 waypoint는 첫 waypoint와 같으므로 제외
        # (폐곡선이므로 중복 방지)

        # Torch tensor로 변환 (num_waypoints, 2)
        self.centerline = torch.tensor(interpolated_points, dtype=torch.float32, device=self.device)

        # 각 waypoint까지의 누적 거리 계산
        distances = torch.norm(self.centerline[1:] - self.centerline[:-1], dim=-1)
        self.centerline_cumulative_dist = torch.cat([
            torch.zeros(1, device=self.device),
            torch.cumsum(distances, dim=0)
        ])

        # 총 트랙 길이
        self.track_length = self.centerline_cumulative_dist[-1].item()

        # 각 환경의 현재 waypoint index 추적 (가장 가까운 waypoint)
        self.current_waypoint_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # 각 환경의 이전 진행 거리 (진행량 계산용)
        self.previous_progress = torch.zeros(self.num_envs, device=self.device)

        # 절대 진행 거리 추적 (Lap counter 방식)
        self.total_distance = torch.zeros(self.num_envs, device=self.device)  # 누적 주행 거리
        self.lap_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)  # 완주 횟수

    def _get_track_progress(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        차량 위치에서 트랙 진행 거리를 계산합니다.

        Args:
            pos: 차량 위치 (num_envs, 3) - XYZ 좌표

        Returns:
            progress: 각 환경의 트랙 진행 거리 (num_envs,) [0, track_length]
            progress_delta: 이전 스텝 대비 진행량 (num_envs,)
        """
        # XY 위치만 사용
        pos_xy = pos[:, :2]  # (num_envs, 2)

        # 각 환경에 대해 가장 가까운 centerline waypoint 찾기
        # Broadcasting: (num_envs, 1, 2) - (1, num_waypoints, 2)
        distances = torch.norm(
            pos_xy.unsqueeze(1) - self.centerline.unsqueeze(0),
            dim=-1
        )  # (num_envs, num_waypoints)

        # 가장 가까운 waypoint index
        closest_idx = torch.argmin(distances, dim=-1)  # (num_envs,)

        # 해당 waypoint까지의 누적 거리 = 진행 거리
        progress = self.centerline_cumulative_dist[closest_idx]  # (num_envs,)

        # 결승선 통과 감지 (90% 지점에서 10% 지점으로 점프)
        finish_line_crossed = (self.previous_progress > self.track_length * 0.9) & \
                              (progress < self.track_length * 0.1)

        # Lap counter 증가 (결승선 통과 시)
        self.lap_count[finish_line_crossed] += 1

        # 절대 진행 거리 계산 (Continuous Progress)
        # = 현재 lap의 진행 거리 + (완주한 lap 수 * track_length)
        absolute_progress = progress + self.lap_count.float() * self.track_length

        # 진행량 계산 (절대 거리 기반이므로 항상 양수)
        progress_delta = absolute_progress - self.total_distance

        # 절대 진행 거리 업데이트
        self.total_distance[:] = absolute_progress

        # 현재 waypoint index 업데이트
        self.current_waypoint_idx[:] = closest_idx

        # 다음 스텝을 위해 현재 진행 거리 저장 (결승선 감지용)
        self.previous_progress[:] = progress

        return progress, progress_delta

    def _get_centerline_direction(self) -> torch.Tensor:
        """
        각 환경의 현재 위치에서 트랙을 따라가는 방향 벡터를 계산합니다.

        Returns:
            direction: (num_envs, 2) - 정규화된 XY 방향 벡터 (트랙 진행 방향)
        """
        # 각 환경의 현재 waypoint와 다음 waypoint 인덱스
        current_idx = self.current_waypoint_idx  # (num_envs,)
        next_idx = (current_idx + 1) % len(self.centerline)  # (num_envs,)

        # 각 환경의 waypoint 좌표 가져오기
        current_waypoint = self.centerline[current_idx]  # (num_envs, 2)
        next_waypoint = self.centerline[next_idx]  # (num_envs, 2)

        # 방향 벡터 계산 및 정규화
        direction = next_waypoint - current_waypoint  # (num_envs, 2)
        direction = direction / (torch.norm(direction, dim=-1, keepdim=True) + 1e-6)

        return direction

    def _get_rewards(self) -> torch.Tensor:
        """
        트랙 진행 거리 기반 보상 함수:
        - 50% 트랙 진행: 중심선을 따라 진행한 거리
        - 0.01% 생존 보상: 오래 살아남을수록 유리
        - 10% 속도 보상: 전방 속도
        - 30% 충돌 페널티: LiDAR 기반 충돌 감지
        """
        pos = self.robot.data.root_state_w[:, :3]
        vel = self.robot.data.root_state_w[:, 7:10]

        # 1) 트랙 진행 거리 보상 (50%)
        # 중심선을 따라 진행한 거리를 보상
        _, progress_delta = self._get_track_progress(pos)

        # 전진한 거리에만 보상 (후진은 0)
        # 0~1m 진행 -> 0~0.5 보상
        reward_forward = torch.clamp(progress_delta, 0.0, 1.0) * 0.5

        # 2) 생존 보상 (0.01%)
        # 스텝당 0.0001 보상 (초기 10 step 제외, 전진 중일 때만, 오래 살아남을수록 유리)
        # 생존 보상은 나중에 forward_speed_raw 계산 후 적용

        # 3) 양의 속도 보상 (10%)
        # 전방 속도 구성 요소에 보상 (트랙 중심선 기준, 양의 속도만)
        forward_direction = self._get_centerline_direction()  # (num_envs, 2) - 트랙 진행 방향
        vel_xy = vel[:, :2]  # XY 평면 속도
        forward_speed_raw = torch.sum(vel_xy * forward_direction, dim=-1)  # 속도를 트랙 방향에 투영

        # 양의 속도(전진)에만 보상
        # 0~10m/s -> 0~0.1 보상
        reward_speed = torch.clamp(forward_speed_raw / 10.0, 0.0, 1.0) * 0.1

        # 생존 보상 계산 (전진 중일 때만)
        is_moving_forward = forward_speed_raw > 0.0  # 트랙 방향으로 움직이는지
        reward_survival = torch.where(
            (self.episode_length_buf >= 10) & is_moving_forward,  # Step 10 이후 & 전진 중
            torch.full((self.num_envs,), 0.001, device=self.device),
            torch.zeros(self.num_envs, device=self.device)
        )

        # 4) 충돌 벌점 (-30%)
        # _get_dones()에서 이미 계산된 충돌 감지 사용 (타이밍 문제 해결)
        # Grace period 적용: 리셋 후 3 스텝 동안 충돌 페널티 비활성화
        collision_penalty = torch.zeros(self.num_envs, device=self.device)

        if self.lidar_distances is not None:
            # _get_dones()의 충돌 감지 재사용
            collision_detected = self.current_collision_detected.clone()

            # Grace period: 리셋 후 3 스텝 동안 충돌 페널티 비활성화
            grace_period = 3
            in_grace_period = self.episode_length_buf < grace_period
            collision_detected = collision_detected & ~in_grace_period

            # 충돌 감지 시 즉시 -0.3 페널티 적용
            collision_penalty[collision_detected] = -0.3

            # 디버그: 보상 계산에서 충돌이 감지되면 출력
            if self.cfg.debug_mode and collision_detected.any():
                detected_envs = torch.where(collision_detected)[0]
                for idx in detected_envs:
                    min_dist = torch.min(self.lidar_distances[idx]).item()
                    close_count = (self.lidar_distances[idx] < 0.1).sum().item()
                    print(f"[REWARD COLLISION] Env {idx.item()} at step {self.episode_length_buf[idx].item()}: Min={min_dist:.3f}m, Close(<10cm)={close_count}, Penalty={collision_penalty[idx].item():.4f}")

        # 총 보상
        reward = reward_forward + reward_survival + reward_speed + collision_penalty
        

        # 다음 스텝을 위해 상태 업데이트
        self.previous_pos[:] = pos

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos, vel = self.robot.data.root_state_w[:, :3], self.robot.data.root_state_w[:, 7:10]
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # 종료 조건
        out_of_bounds = torch.norm(pos[:, :2], dim=-1) > 50.0

        # 충돌 감지 (이중 방법):
        # 1. LiDAR 기반: 전방/측면 충돌 (270° 커버리지)
        # 충돌을 계산하고 _get_rewards()에서 사용하도록 저장 (타이밍 문제 해결)
        self.current_collision_detected = self._detect_collision_consecutive(
            self.lidar_distances,
            threshold=0.1,  # 10cm 충돌 임계값 (0.2m에서 변경)
            consecutive_count=10  # 더 강력한 감지를 위해 5에서 10으로 증가 (2.5° 범위)
        )

        # 유예 기간: 리셋 후 처음 3 스텝 동안 충돌 감지 비활성화
        # 센서 업데이트 전 오래된 LiDAR 데이터로 인한 오탐 방지
        grace_period = 3
        in_grace_period = self.episode_length_buf < grace_period
        collision_lidar = self.current_collision_detected & ~in_grace_period

        # 결합됨: 이제 LiDAR 기반 감지만 사용
        collision = collision_lidar

        # 막힘 감지: 차량이 XY 평면에서 충분히 움직이지 않았는지 확인
        self.steps_since_last_check += 1
        stuck = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # stuck_check_interval 스텝마다 확인
        check_now = self.steps_since_last_check >= self.stuck_check_interval
        if check_now.any():
            current_pos_xy = pos[:, :2]
            movement = torch.norm(current_pos_xy - self.last_check_pos, dim=-1)

            # 움직임이 임계값보다 작으면 막힌 것으로 표시
            stuck[check_now] = movement[check_now] < self.stuck_threshold

            # 디버그 로깅: 막힌 환경에 대한 슬립 감지 출력
            if stuck.any():
                wheel_radius = 0.0508  # F1tenth 바퀴 반지름 (미터)
                current_wheel_pos = self.robot.data.joint_pos[:, self._rear_wheel_ids]

                # 확인 간격 동안의 바퀴 회전 변화 계산 (1 스텝만이 아님!)
                wheel_delta = current_wheel_pos - self.wheel_pos_at_last_check

                # 회전 변화로부터 바퀴 주행 거리계 계산
                wheel_odometry = torch.mean(torch.abs(wheel_delta), dim=-1) * wheel_radius

                # 검증: GT 위치를 사용하여 실제 거리 계산
                # 이는 바퀴 주행 거리계와 위치 추적을 모두 검증합니다.
                stuck_env_idx = torch.where(stuck)[0]
                for idx in stuck_env_idx:
                    # 현재 위치
                    current_xy = pos[idx, :2]
                    last_check_xy = self.last_check_pos[idx]

                    # GT 기반 거리 (실측값)
                    gt_distance = torch.norm(current_xy - last_check_xy).item()

                    # 바퀴 기반 거리 (주행 거리계)
                    wheel_dist = wheel_odometry[idx].item()

                    # 움직임과 비교 (gt_distance와 동일해야 함)
                    movement_dist = movement[idx].item()

                    # 슬립 비율: (바퀴_거리 - 실제_거리) / 바퀴_거리
                    slip_ratio = (wheel_dist - gt_distance) / (wheel_dist + 1e-6)

                    if self.cfg.debug_mode:
                        print(f"\n[STUCK DETECTION] Env {idx.item()} at step {self.episode_length_buf[idx].item()}:")
                        print(f"  Current Position: X={current_xy[0].item():.3f}, Y={current_xy[1].item():.3f}, Z={pos[idx, 2].item():.3f}")
                        print(f"  Last Check Position: X={last_check_xy[0].item():.3f}, Y={last_check_xy[1].item():.3f}")
                        print(f"  GT Distance (current - last): {gt_distance:.3f}m")
                        print(f"  Movement variable: {movement_dist:.3f}m (should match GT)")
                        print(f"  Wheel odometry ({self.stuck_check_interval} steps): {wheel_dist:.3f}m")
                        print(f"  Slip ratio: {slip_ratio:.2%}")

                        # 유효성 검사
                        if abs(gt_distance - movement_dist) > 0.001:
                            print(f"  ⚠️  WARNING: GT distance and movement mismatch! Diff: {abs(gt_distance - movement_dist):.6f}m")

            # 확인된 환경에 대해서만 마지막 확인 위치 및 바퀴 위치 업데이트
            self.last_check_pos[check_now] = current_pos_xy[check_now]
            self.steps_since_last_check[check_now] = 0

            # 확인 시점의 바퀴 위치 업데이트 (매 스텝 아님!)
            current_wheel_pos_all = self.robot.data.joint_pos[:, self._rear_wheel_ids]
            self.wheel_pos_at_last_check[check_now] = current_wheel_pos_all[check_now]

        terminated = out_of_bounds | collision | stuck

        # 디버그 로깅: 환경이 종료될 때 종료 이유 출력
        if self.cfg.debug_mode and terminated.any():
            env_idx = torch.where(terminated)[0]
            for idx in env_idx:
                min_lidar = torch.min(self.lidar_distances[idx]).item() if self.lidar_distances is not None else 0.0
                # 디버그 정보를 위해 연속적인 가까운 포인트 수 계산
                close_count = (self.lidar_distances[idx] < 0.2).sum().item() if self.lidar_distances is not None else 0

                # 속도 디버그 정보
                # Commanded: 목표 속도 (방향 포함, 음수=후진, 양수=전진)
                commanded_speed = self.target_velocity[idx].item()

                # Actual: 전진 방향 속도 (트랙 중심선 기준, 옆 미끄러짐 제외)
                # 현재 waypoint에서 다음 waypoint로의 방향
                current_waypoint_idx = int(self.current_waypoint_idx[idx].item())
                next_waypoint_idx = (current_waypoint_idx + 1) % len(self.centerline)
                forward_direction = self.centerline[next_waypoint_idx] - self.centerline[current_waypoint_idx]
                forward_direction = forward_direction / (torch.norm(forward_direction) + 1e-6)

                vel_xy = vel[idx, :2]
                speed_current = torch.sum(vel_xy * forward_direction).item()

                speed_deficit = commanded_speed - speed_current

                # 디버깅을 위해 차량 방향 계산
                quat = self.robot.data.root_state_w[idx, 3:7]
                w, x, y, z = quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item()
                forward_x = 1 - 2 * (y**2 + z**2)
                forward_y = 2 * (x*y + w*z)
                # 방향에서 yaw 각도 계산
                yaw = math.atan2(forward_y, forward_x) * 180 / math.pi

                # 종료 이유 결정 (우선 순위 순)
                reason = []
                if collision_lidar[idx].item():
                    reason.append("COLLISION (LiDAR)")
                if stuck[idx].item():
                    reason.append("STUCK")
                if out_of_bounds[idx].item():
                    reason.append("OUT OF BOUNDS")
                if time_out[idx].item():
                    reason.append("TIMEOUT")

                reason_str = " + ".join(reason) if reason else "UNKNOWN"

                # 유예 기간인지 확인
                grace_status = f" (Grace period)" if self.episode_length_buf[idx].item() < 3 else ""

                print(f"\n[TERMINATION] Env {idx.item()} at step {self.episode_length_buf[idx].item()}{grace_status} | REASON: {reason_str}")
                print(f"  Position: X={pos[idx, 0].item():.3f}, Y={pos[idx, 1].item():.3f}, Z={pos[idx, 2].item():.3f}")
                print(f"  Velocity: vx={vel[idx, 0].item():.3f}, vy={vel[idx, 1].item():.3f}, vz={vel[idx, 2].item():.3f}")
                print(f"  Heading: forward_x={forward_x:.3f}, forward_y={forward_y:.3f}, yaw={yaw:.1f}°")
                print(f"  LiDAR: Min={min_lidar:.3f}m, Close(<20cm)={close_count}")

                # 바퀴 회전 속도 기반 속도 계산 (휠 주행 거리계, Wheel Odometry)
                # Wheel Odom: 바퀴 각속도 평균 (방향 고려, 음수=후진, 양수=전진)
                wheel_vel = self.robot.data.joint_vel[idx, self._rear_wheel_ids]  # 뒷바퀴 각속도 [rad/s]
                odom_speed = torch.mean(wheel_vel).item() * self.wheel_radius  # 선속도 [m/s], abs() 제거

                print(f"  Speed: Commanded={commanded_speed:+.2f} m/s, Actual={speed_current:+.2f} m/s, Wheel Odom={odom_speed:+.2f} m/s, Deficit={speed_deficit:+.2f} m/s")

                # 디버그: LiDAR 센서 위치 및 가장 가까운 충돌 지점 표시
                if self.lidar_distances is not None:
                    lidar_pos = self.lidar.data.pos_w[idx]  # 월드 프레임에서 LiDAR 센서 위치
                    lidar_hits = self.lidar.data.ray_hits_w[idx]  # 월드 프레임에서 모든 충돌 지점
                    min_idx = torch.argmin(self.lidar_distances[idx])
                    closest_hit = lidar_hits[min_idx]
                    min_distance_stored = self.lidar_distances[idx][min_idx].item()

                    # 거리 계산 확인
                    distance_recalculated = torch.norm(closest_hit - lidar_pos).item()

                    print(f"  [LiDAR DEBUG] Sensor position: X={lidar_pos[0].item():.3f}, Y={lidar_pos[1].item():.3f}, Z={lidar_pos[2].item():.3f}")
                    print(f"  [LiDAR DEBUG] Closest hit point: X={closest_hit[0].item():.3f}, Y={closest_hit[1].item():.3f}, Z={closest_hit[2].item():.3f}")
                    print(f"  [LiDAR DEBUG] Distance (stored): {min_distance_stored:.3f}m | Distance (recalculated): {distance_recalculated:.3f}m")

                    # 거리가 일치하는지 확인
                    if abs(min_distance_stored - distance_recalculated) > 0.01:
                        print(f"  [LiDAR WARNING] Distance mismatch! Diff: {abs(min_distance_stored - distance_recalculated):.3f}m")

                    # 벽까지의 거리 표시
                    wall_y = -0.197  # 관찰에서 알려진 벽 Y 좌표
                    distance_to_wall = abs(lidar_pos[1].item() - wall_y)
                    print(f"  [LiDAR DEBUG] Direct distance to wall (Y={wall_y}): {distance_to_wall:.3f}m")

                # 터미네이션 시 보상 정보 출력
                # _get_dones()가 _get_rewards()보다 먼저 실행되므로 현재 스텝 값으로 근사 계산

                # 1) 이동거리 보상 (50%)
                if hasattr(self, 'previous_pos'):
                    displacement = pos[idx, :2] - self.previous_pos[idx, :2]
                    forward_progress = torch.sum(displacement * forward_direction).item()
                    reward_forward_approx = max(0.0, min(forward_progress, 1.0)) * 0.5
                else:
                    reward_forward_approx = 0.0

                # 2) 생존 보상 (0.01%)
                # Step 10 이후 & 전진 중일 때만 부여
                current_step = self.episode_length_buf[idx].item()
                is_moving_forward = speed_current > 0.0
                reward_survival_approx = 0.001 if (current_step >= 10 and is_moving_forward) else 0.0

                # 3) 속도 보상 (10%)
                forward_speed = speed_current  # 이미 계산됨 (centerline 기준)
                reward_speed_approx = max(0.0, min(forward_speed / 10.0, 1.0)) * 0.1

                # 4) 충돌 벌점 (-30%)
                # Grace period 적용
                in_grace = self.episode_length_buf[idx].item() < 3
                collision_penalty_approx = 0.0 if in_grace else (-0.3 if self.current_collision_detected[idx].item() else 0.0)

                # 총 보상 계산
                total_reward_approx = reward_forward_approx + reward_survival_approx + reward_speed_approx + collision_penalty_approx

                print(f"  [REWARD] forward={reward_forward_approx:.4f}, survival={reward_survival_approx:.4f}, speed={reward_speed_approx:.4f}, collision={collision_penalty_approx:.4f}")
                print(f"  [REWARD] Total (approx): {total_reward_approx:.4f}")

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # 조인트 위치 및 속도 재설정
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        # 차량 스폰 영역 내에서 무작위 스폰 위치 샘플링
        num_resets = len(env_ids)
        spawn_zone = self.cfg.vehicle_spawn_zone

        # 영역 내에서 X 위치 샘플링
        x_min, x_max = spawn_zone["x_range"]
        x_pos = sample_uniform(x_min, x_max, (num_resets, 1), device=self.device)

        # 영역 내에서 Y 위치 샘플링
        y_min, y_max = spawn_zone["y_range"]
        y_pos = sample_uniform(y_min, y_max, (num_resets, 1), device=self.device)

        # 고정 Z 높이
        z_pos = torch.full((num_resets, 1), spawn_zone["z_fixed"], device=self.device)

        # 영역 내에서 yaw 방향 샘플링
        yaw_min, yaw_max = spawn_zone["yaw_range"]
        yaw = sample_uniform(yaw_min, yaw_max, (num_resets, 1), device=self.device)

        # 위치 결합
        spawn_pos = torch.cat([x_pos, y_pos, z_pos], dim=-1)

        # 루트 상태를 스폰 영역 위치로 업데이트
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] = spawn_pos

        # 방향 설정 (yaw에서 쿼터니언)
        default_root_state[:, 3] = torch.cos(yaw / 2).squeeze(-1)  # w
        default_root_state[:, 4] = 0.0  # x
        default_root_state[:, 5] = 0.0  # y
        default_root_state[:, 6] = torch.sin(yaw / 2).squeeze(-1)  # z

        # 시뮬레이션에 쓰기
        self.robot.write_root_state_to_sim(default_root_state, env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # 이전 위치 추적기 업데이트
        self.previous_pos[env_ids] = default_root_state[:, :3]

        # 모터 제어 상태 재설정
        self.target_velocity[env_ids] = 0.0

        # 막힘 감지 재설정
        self.last_check_pos[env_ids] = default_root_state[:, :2]  # XY 위치
        self.steps_since_last_check[env_ids] = 0

        # 슬립 감지를 위해 바퀴 위치 추적기 재설정
        self.wheel_pos_at_last_check[env_ids] = self.robot.data.joint_pos[env_ids][:, self._rear_wheel_ids]

        # 스폰 시 초기 방향 저장 (IMU와 유사한 참조 프레임)
        # 스폰 쿼터니언에서 방향 추출
        quat = default_root_state[:, 3:7]  # [w, x, y, z]
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        # 쿼터니언에 의한 로컬 X축 [1, 0, 0] 회전 -> 월드 전방 방향
        initial_heading_x = 1 - 2 * (y**2 + z**2)
        initial_heading_y = 2 * (x*y + w*z)
        initial_heading = torch.stack([initial_heading_x, initial_heading_y], dim=-1)

        # 정규화 및 저장
        initial_heading = initial_heading / (torch.norm(initial_heading, dim=-1, keepdim=True) + 1e-6)
        self.initial_heading[env_ids] = initial_heading

        # 트랙 진행 거리 초기화
        # 스폰 위치에서 가장 가까운 waypoint의 진행 거리로 초기화
        spawn_pos_xy = spawn_pos[:, :2]
        distances = torch.norm(
            spawn_pos_xy.unsqueeze(1) - self.centerline.unsqueeze(0),
            dim=-1
        )
        closest_idx = torch.argmin(distances, dim=-1)
        self.previous_progress[env_ids] = self.centerline_cumulative_dist[closest_idx]

        # 절대 진행 거리 및 Lap counter 초기화
        self.total_distance[env_ids] = self.centerline_cumulative_dist[closest_idx]
        self.lap_count[env_ids] = 0
        self.current_waypoint_idx[env_ids] = closest_idx

