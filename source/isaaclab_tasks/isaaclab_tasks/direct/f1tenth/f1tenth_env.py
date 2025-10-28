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
from isaaclab.sim import SimulationCfg, RigidBodyMaterialCfg
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
        )
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
    episode_length_s = 20.0  # 지속적인 학습을 위해 긴 에피소드 길이 
    decimation = 2

    # 행동 공간
    max_steering_angle = 0.4  # 최대 조향 각도 (rad, 약 ±23도)
    # action[:, 0]: 목표 조향 각도 [-1, 1] → [-0.4, 0.4] rad (위치 제어)
    # action[:, 1]: 목표 선속도 [-1, 1] → [v_min, v_max] m/s (속도 제어)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)   # SAC 맞춤 설정

    # 관찰 공간 (LiDAR만 사용)
    observation_space = 1080  # LiDAR(1080)
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
        "x_range": (0, 0),           # X축 스폰 범위 [최소, 최대]
        "y_range": (-0.67, -0.65),           # Y축 스폰 범위 [최소, 최대]
        "z_fixed": -0.5,                   # 고정 Z 높이 (충돌 방지)
        "yaw_range": (0, 0), # Yaw 방향 범위 [최소, 최대]
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
        self._steering_joint_ids, self._steering_joint_names = self.robot.find_joints(".*steering_hinge_joint")
        self._rear_wheel_ids, self._rear_wheel_names = self.robot.find_joints(".*rear_wheel_joint")
        self._front_wheel_ids, self._front_wheel_names = self.robot.find_joints(".*front_wheel_joint")
        print("JOINTS -> steering:", getattr(self, "_steering_joint_names", self._steering_joint_ids),
              " rear:", getattr(self, "_rear_wheel_names", self._rear_wheel_ids),
              " front:", getattr(self, "_front_wheel_names", self._front_wheel_ids))
        print("ACTUATORS KEYS:", list(self.robot.actuators.keys()) if hasattr(self.robot, "actuators") else "no actuators")

        # 조향은 위치 제어로 변경되어 action_scale_steering 불필요
        # 속도는 직접 제어로 변경되어 action_scale_velocity 불필요
        self.previous_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.previous_total_distance = torch.zeros(self.num_envs, device=self.device)

        # 보상 계산 변수
        self.lidar_distances = None  # 보상 계산을 위해 LiDAR 거리 저장

        # IMU와 유사한 참조 프레임: 스폰 시 초기 방향 저장
        # 차량이 회전할 때 후진 주행이 보상받는 것을 방지합니다.
        self.initial_heading = torch.zeros(self.num_envs, 2, device=self.device)  # (num_envs, 2) XY 방향 벡터

        # 이 단계에서 계산될 종료 플래그
        self.collision_terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.stuck_terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.out_of_bounds_terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 모터 제어: 직접 속도 제어 (VESC 속도 제어 모드)
        self.wheel_radius = cfg.vehicle_params["wheel_radius"]
        self.max_wheel_speed = cfg.motor_control["max_wheel_speed"]
        # 각 환경의 목표 선형 속도 (VESC RPM 명령)
        self.target_velocity = torch.zeros(self.num_envs, device=self.device)

        # 막힘 감지: 움직임 확인을 위해 위치 추적
        # 차량이 적절한 시간 내에 움직이지 않았는지 확인
        self.stuck_check_interval = 120  # 스텝 (2초)
        self.stuck_threshold = 0.1  # 미터 (2초에 최소 10cm 움직임 = 평균 0.05m/s)
        self.stuck_initial_delay = 300  # 리셋 후 첫 체크까지 지연 시간 (5초)
        self.last_check_pos = torch.zeros(self.num_envs, 2, device=self.device)  # XY 위치
        self.steps_since_last_check = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # 슬립 감지: 주행 거리계 계산을 위해 바퀴 회전 추적
        # 마지막 막힘 확인 시점의 바퀴 위치 저장 (매 스텝 아님!)
        self.wheel_pos_at_last_check = torch.zeros(self.num_envs, len(self._rear_wheel_ids), device=self.device)

        # 에피소드 통계
        self.episode_reward_sum = torch.zeros(self.num_envs, device=self.device)
        self.episode_commanded_speed_sum = torch.zeros(self.num_envs, device=self.device)
        self.episode_actual_speed_sum = torch.zeros(self.num_envs, device=self.device)
        self.episode_commanded_steering_sum = torch.zeros(self.num_envs, device=self.device)
        self.episode_actual_steering_sum = torch.zeros(self.num_envs, device=self.device)
        # 보상 항목별 통계
        self.episode_reward_forward_sum = torch.zeros(self.num_envs, device=self.device)
        self.episode_reward_speed_sum = torch.zeros(self.num_envs, device=self.device)
        self.episode_reward_survival_sum = torch.zeros(self.num_envs, device=self.device)
        self.episode_danger_penalty_sum = torch.zeros(self.num_envs, device=self.device)
        self.episode_termination_penalty_sum = torch.zeros(self.num_envs, device=self.device)

        # -- 추가된 디버깅 통계 --
        self.episode_steering_angle_sum = torch.zeros(self.num_envs, device=self.device)
        self.episode_commanded_wheel_vel_sum = torch.zeros(self.num_envs, device=self.device)
        self.episode_actual_wheel_vel_sum = torch.zeros(self.num_envs, device=self.device)
        self.episode_min_commanded_vel = torch.full((self.num_envs,), float("inf"), device=self.device)



        print(f"Steering joint limits: {self.robot.data.soft_joint_pos_limits[0, self._steering_joint_ids]}")


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
        """
        Args:
            actions: 정규화된 공간 [-1, 1]의 [조향_각도, 목표_속도]
            - actions[:, 0]: 목표 조향 각도 [-1, 1] → [-0.4, 0.4] rad (약 ±23도)
            - actions[:, 1]: 목표 선속도 [-1, 1] → [v_min, v_max]
        """
        # 조향: 위치 제어로 변경 (속도 제어 대신)
        max_steering_angle = 0.4  # 라디안 (약 23도)
        target_steering_angle = actions[:, 0] * max_steering_angle
        self.target_steering = target_steering_angle.unsqueeze(-1)
        
        # 속도 명령
        v_min = self.cfg.vehicle_params["v_min"]
        v_max = self.cfg.vehicle_params["v_max"]
        self.target_velocity = (actions[:, 1] + 1.0) / 2.0 * (v_max - v_min) + v_min
        # 클램핑 전 최소 속도 기록
        self.episode_min_commanded_vel = torch.min(self.episode_min_commanded_vel, self.target_velocity)
        # v_min 이상으로 클램핑하여 후진 방지
        self.target_velocity = torch.clamp(self.target_velocity, min=v_min)

    def _apply_action(self) -> None:
        """로봇에 제어 명령을 적용합니다."""
        # 조향: 위치 제어 (속도 제어보다 안정적)
        self.robot.set_joint_position_target(self.target_steering, joint_ids=self._steering_joint_ids)

        # 후륜만 속도 제어 (RWD - Rear Wheel Drive)
        target_wheel_angular_vel = self.target_velocity / self.wheel_radius
        target_wheel_angular_vel = target_wheel_angular_vel.unsqueeze(-1)
        target_wheel_angular_vel = torch.clamp(
            target_wheel_angular_vel,
            -self.max_wheel_speed,
            self.max_wheel_speed
        )
        # 후륜에만 속도 명령 전송 (RWD, 실제 F1tenth와 동일)
        self.robot.set_joint_velocity_target(target_wheel_angular_vel, joint_ids=self._rear_wheel_ids)
        # 전륜은 프리휠(free-rolling) - 속도 명령 없음

    def _get_observations(self) -> dict:
        lidar_data = self.lidar.data.ray_hits_w[..., :3]
        lidar_distances = torch.norm(lidar_data - self.lidar.data.pos_w.unsqueeze(1), dim=-1)

        # 보상 계산을 위해 LiDAR 거리 저장
        self.lidar_distances = lidar_distances

        # LiDAR 데이터만 반환 (차량 상태 제외)
        obs = lidar_distances
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
        # 각 구간을 100개 점으로 나눔 (1cm 간격으로 진행 감지)
        interpolated_points = []
        for i in range(len(waypoints_np) - 1):
            start = waypoints_np[i]
            end = waypoints_np[i + 1]
            # 시작점부터 끝점 전까지 (끝점은 다음 구간의 시작점)
            for j in range(100):
                t = j / 100.0
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

        개선된 방식: 폐곡선을 직선으로 펼쳐서 생각하고, 차량 위치를
        가장 가까운 centerline 선분에 투영하여 연속적인 진행거리를 계산합니다.
        이렇게 하면 도로 폭 내에서 차량이 어디 있든 정확한 진행을 감지합니다.

        Args:
            pos: 차량 위치 (num_envs, 3) - XYZ 좌표

        Returns:
            progress: 각 환경의 트랙 진행 거리 (num_envs,) [0, track_length]
            progress_delta: 이전 스텝 대비 진행량 (num_envs,)
        """
        # XY 위치만 사용
        pos_xy = pos[:, :2]  # (num_envs, 2)

        # 선분 투영 방식: 차량 위치를 centerline segments에 투영
        num_segments = len(self.centerline) - 1

        # Segment 벡터 계산 (한 번만)
        segment_starts = self.centerline[:-1]  # (num_segments, 2)
        segment_ends = self.centerline[1:]     # (num_segments, 2)
        segment_vectors = segment_ends - segment_starts  # (num_segments, 2)
        segment_len_sq = (segment_vectors ** 2).sum(dim=-1)  # (num_segments,) - 길이의 제곱

        # 각 차량에서 모든 segment 시작점까지의 벡터
        # Broadcasting: (num_envs, num_segments, 2)
        to_vehicle = pos_xy.unsqueeze(1) - segment_starts.unsqueeze(0)  # (num_envs, num_segments, 2)

        # 투영 비율 t 계산
        # t = (to_vehicle · segment_vector) / ||segment_vector||^2
        dot_product = (to_vehicle * segment_vectors.unsqueeze(0)).sum(dim=-1)  # (num_envs, num_segments)
        t = dot_product / (segment_len_sq.unsqueeze(0) + 1e-8)  # (num_envs, num_segments)
        t = torch.clamp(t, 0.0, 1.0)  # Segment 내로 제한

        # 투영점 계산
        projected_points = segment_starts.unsqueeze(0) + t.unsqueeze(-1) * segment_vectors.unsqueeze(0)  # (num_envs, num_segments, 2)

        # 차량에서 투영점까지 거리
        dist_to_proj = torch.norm(pos_xy.unsqueeze(1) - projected_points, dim=-1)  # (num_envs, num_segments)

        # 가장 가까운 segment 선택
        closest_seg = torch.argmin(dist_to_proj, dim=-1)  # (num_envs,)

        # 가장 가까운 segment의 t 값
        batch_indices = torch.arange(pos_xy.shape[0], device=pos_xy.device)
        t_closest = t[batch_indices, closest_seg]  # (num_envs,)

        # 진행거리 계산: segment 시작점 누적거리 + (t × segment 길이)
        seg_start_dist = self.centerline_cumulative_dist[closest_seg]  # (num_envs,)
        seg_length = torch.sqrt(segment_len_sq[closest_seg])  # (num_envs,)
        progress = seg_start_dist + t_closest * seg_length  # (num_envs,)

        # 결승선 통과 감지 (90% 지점에서 10% 지점으로 점프)
        finish_line_crossed = (self.previous_progress > self.track_length * 0.9) & \
                              (progress < self.track_length * 0.1)

        # Lap counter 증가 (결승선 통과 시)
        self.lap_count[finish_line_crossed] += 1

        # 절대 진행 거리 계산 (Continuous Progress)
        # = 현재 lap의 진행 거리 + (완주한 lap 수 * track_length)
        absolute_progress = progress + self.lap_count.float() * self.track_length

        # 진행량 계산 (절대 거리 기반이므로 항상 양수여야 함)
        progress_delta = absolute_progress - self.total_distance

        # 절대 진행 거리 업데이트
        self.total_distance[:] = absolute_progress

        # 현재 segment index 업데이트 (디버그 출력용)
        self.current_waypoint_idx[:] = closest_seg

        # 다음 스텝을 위해 현재 진행 거리 저장 (결승선 감지용)
        self.previous_progress[:] = progress

        return progress, progress_delta

    def _get_centerline_direction(self) -> torch.Tensor:
        """
        각 환경의 현재 위치에서 트랙을 따라가는 방향 벡터를 계산합니다.

        Returns:
            direction: (num_envs, 2) - 정규화된 XY 방향 벡터 (트랙 진행 방향)
        """
        # 각 환경의 현재 segment 인덱스
        # Segment i는 centerline[i]에서 centerline[i+1]로 가는 선분
        segment_idx = self.current_waypoint_idx  # (num_envs,)

        # 각 segment의 시작점과 끝점
        segment_start = self.centerline[segment_idx]  # (num_envs, 2)
        segment_end = self.centerline[segment_idx + 1]  # (num_envs, 2)

        # 방향 벡터 계산 및 정규화 (현재 segment의 방향)
        direction = segment_end - segment_start  # (num_envs, 2)
        direction = direction / (torch.norm(direction, dim=-1, keepdim=True) + 1e-6)

        return direction

    def _get_rewards(self) -> torch.Tensor:
        """
        트랙 진행 거리 기반 보상 함수 (학습 초기 탐험 장려):
        - 트랙 진행: 중심선을 따라 1m당 10점
        - 저속 페널티: 1 m/s 미만일 때 선형 페널티 (0 m/s = -0.01, 1 m/s = 0)
        - 속도 보상: 전방 속도에 비례 (최대 5 m/s)
        - 위험 패널티 (2단계):
          * 벽과 25cm 이내: 매 스텝 -1.5점 (경고)
          * 벽과 10cm 이내: 매 스텝 -20점 (위험)
        - 충돌 패널티: 제거 (연속적 위험 패널티로 대체)
        """
        pos = self.robot.data.root_state_w[:, :3]
        vel = self.robot.data.root_state_w[:, 7:10]

        # -- 종료 조건 계산 (페널티 계산을 위해 먼저 수행) --
        self.out_of_bounds_terminated = torch.norm(pos[:, :2], dim=-1) > 50.0

        # Grace Period 정의
        grace_period = 5  # 3에서 5로 증가하여 더 안전하게
        in_grace_period = self.episode_length_buf < grace_period

  
        collision_detected = self._detect_collision_consecutive(
            self.lidar_distances,
            threshold=0.15,  # 15cm 충돌 임계값
            consecutive_count=10  # 더 강력한 감지를 위해 5에서 10으로 증가 (2.5° 범위)
        )
        # Grace Period 내 환경은 충돌로 판별하지 않음
        
        self.collision_terminated = collision_detected & ~in_grace_period

        # 막힘 감지
        self.steps_since_last_check += 1
        self.stuck_terminated.zero_()
        check_now = self.steps_since_last_check >= self.stuck_check_interval

        if check_now.any():
            current_pos_xy = pos[:, :2]
            movement = torch.norm(current_pos_xy - self.last_check_pos, dim=-1)
            self.stuck_terminated[check_now] = movement[check_now] < self.stuck_threshold
            self.last_check_pos[check_now] = current_pos_xy[check_now]
            self.steps_since_last_check[check_now] = 0

        # -- 보상 계산 --
        # 1) 트랙 진행 거리 보상
        # progress_delta: 트랙 진행 거리 (미터 단위)
        # 가중치를 높여서 진행 거리를 더 중요하게 만듦
        _, progress_delta = self._get_track_progress(pos)
        reward_forward = torch.clamp(progress_delta, min=0.0) * 18  # 10 → 15으로 증가

        # 2) 속도 보상
        forward_direction = self._get_centerline_direction()
        vel_xy = vel[:, :2]
        forward_speed_raw = torch.sum(vel_xy * forward_direction, dim=-1)
        reward_speed = torch.clamp(forward_speed_raw / 5.0, 0.0, 1.0) * 1.2

        # 3) 저속 페널티 (연속형 - Inverse 함수)
        # 속도가 0에 가까워질수록 페널티가 급증
        # 공식: -A / (speed/target) + A  (target 속도에서 0)
        TARGET_SPEED_FOR_PENALTY = 1.0  # 1.0 m/s
        PENALTY_SCALE_SLOW = 1.0
        MIN_SPEED_RATIO = 0.05  # 역수 폭발 방지

        speed_ratio = torch.clamp(forward_speed_raw / TARGET_SPEED_FOR_PENALTY, min=MIN_SPEED_RATIO, max=10.0)
        reward_survival = -PENALTY_SCALE_SLOW / speed_ratio + PENALTY_SCALE_SLOW
        # Clipping to prevent excessive penalty (Phase 1: Reward normalization)
        reward_survival = torch.clamp(reward_survival, min=-10.0, max=1.0)
        # speed = 1.0 m/s → ratio=1.0 → penalty=0
        # speed = 0.5 m/s → ratio=0.5 → penalty=-2.0 (clipped)
        # speed = 0.1 m/s → ratio=0.1 → penalty=-10.0 (clipped from -18)
        # speed = 0.05 m/s → ratio=0.05 → penalty=-10.0 (clipped from -38)

        # 4) 연속적 위험 패널티 (지수 함수)
        # 벽에 가까워질수록 페널티가 지수적으로 증가
        # 공식: -A * (exp(B * proximity) - 1)
        min_distance = torch.min(self.lidar_distances, dim=1).values

        WARNING_DISTANCE = 0.25  # 25cm부터 페널티 시작
        PENALTY_SCALE_DANGER = 4.0
        EXP_STEEPNESS_DANGER = 8.0

        # 근접도 계산 (0보다 작으면 0으로 clamp)
        proximity = torch.clamp(WARNING_DISTANCE - min_distance, min=0.0)

        # 지수적 페널티 (proximity=0일 때 0, 가까워질수록 급증)
        danger_penalty = -PENALTY_SCALE_DANGER * (
            torch.exp(EXP_STEEPNESS_DANGER * proximity) - 1.0
        )
        # Clipping to prevent excessive penalty (Phase 1: Reward normalization)
        danger_penalty = torch.clamp(danger_penalty, min=-10.0, max=0.0)
        # distance = 25cm → proximity=0.0 → penalty=0
        # distance = 20cm → proximity=0.05 → penalty=-1.5 (no clipping)
        # distance = 10cm → proximity=0.15 → penalty=-7.0 (no clipping)
        # distance = 5cm → proximity=0.20 → penalty=-10.0 (clipped from -11.9)

        # 총 보상 (termination_penalty 제거됨 - danger_penalty로 충분)
        reward = reward_forward + reward_survival + reward_speed + danger_penalty

        # 에피소드 통계 누적
        self.episode_reward_sum += reward
        self.episode_commanded_speed_sum += self.target_velocity
        self.episode_actual_speed_sum += forward_speed_raw

        # 조향각 통계 (명령값과 실제값)
        current_steering_angle = self.robot.data.joint_pos[:, self._steering_joint_ids].mean(dim=-1)
        self.episode_commanded_steering_sum += self.target_steering.squeeze(-1)
        self.episode_actual_steering_sum += current_steering_angle

        self.episode_reward_forward_sum += reward_forward
        self.episode_reward_speed_sum += reward_speed
        self.episode_reward_survival_sum += reward_survival
        self.episode_danger_penalty_sum += danger_penalty

        # -- 추가된 디버깅 통계 누적 --
        # 조향각 (평균)
        current_steering_angle = self.robot.data.joint_pos[:, self._steering_joint_ids].mean(dim=-1)
        self.episode_steering_angle_sum += torch.abs(current_steering_angle)
        # 바퀴 각속도 (명령 vs 실제)
        commanded_wheel_vel = self.target_velocity / self.wheel_radius
        self.episode_commanded_wheel_vel_sum += torch.abs(commanded_wheel_vel)
        actual_wheel_vel = self.robot.data.joint_vel[:, self._rear_wheel_ids].mean(dim=-1)
        self.episode_actual_wheel_vel_sum += torch.abs(actual_wheel_vel)

        # 바퀴 속도 로깅 제거됨

        # 다음 스텝을 위해 상태 업데이트
        self.previous_pos[:] = pos
        self.previous_total_distance[:] = self.total_distance

        return reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # _get_rewards()에서 계산된 종료 플래그를 가져옴
        # 참고: 그레이스 기간(grace period)은 _get_rewards의 페널티 계산에만 영향을 미치며,
        # 에피소드 종료 자체는 즉시 발생해야 합니다.
        terminated = self.collision_terminated | self.stuck_terminated | self.out_of_bounds_terminated
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        dones = terminated | time_out

        # 에피소드 종료 시 요약 로그 출력
        if self.cfg.debug_mode and dones.any():
            done_ids = torch.where(dones)[0]
            for idx in done_ids:
                ep_len = self.episode_length_buf[idx].item()
                if ep_len == 0:
                    continue

                # 종료 원인 결정 (로깅용)
                reason = "UNKNOWN"
                if time_out[idx].item():
                    reason = "TIMEOUT"
                elif self.out_of_bounds_terminated[idx].item():
                    reason = "OUT OF BOUNDS"
                elif self.stuck_terminated[idx].item():
                    reason = "STUCK"
                elif self.collision_terminated[idx].item():
                    reason = "COLLISION"

                # 통계 계산
                total_dist = self.total_distance[idx].item()
                lap_num = self.lap_count[idx].item()

                # 평균 속도 계산
                avg_cmd_speed = self.episode_commanded_speed_sum[idx].item() / ep_len
                avg_actual_speed = self.episode_actual_speed_sum[idx].item() / ep_len

                # 보상 정보
                total_reward = self.episode_reward_sum[idx].item()
                avg_reward = total_reward / ep_len
                total_forward_reward = self.episode_reward_forward_sum[idx].item()
                total_speed_reward = self.episode_reward_speed_sum[idx].item()
                total_survival_reward = self.episode_reward_survival_sum[idx].item()
                total_danger_penalty = self.episode_danger_penalty_sum[idx].item()

                # 추가 디버깅 통계 계산
                avg_steering_angle = self.episode_steering_angle_sum[idx].item() / ep_len
                avg_cmd_steering = self.episode_commanded_steering_sum[idx].item() / ep_len
                avg_actual_steering = self.episode_actual_steering_sum[idx].item() / ep_len
                min_cmd_vel = self.episode_min_commanded_vel[idx].item()
                # rad/s를 m/s로 변환
                avg_cmd_wheel_vel_ms = (self.episode_commanded_wheel_vel_sum[idx].item() / ep_len) * self.wheel_radius
                avg_actual_wheel_vel_ms = (self.episode_actual_wheel_vel_sum[idx].item() / ep_len) * self.wheel_radius

                # 요약 로그 출력
                print(f"\n[EPISODE END] Env {idx.item()} | {reason} | Steps: {ep_len}")
                print(f"  > Progress: {total_dist:.2f}m, Laps: {lap_num}")
                print(f"  > Body Speeds (avg m/s): Cmd={avg_cmd_speed:.2f}, Actual={avg_actual_speed:.2f}")
                print(f"  > Steering Angles (avg rad): Cmd={avg_cmd_steering:.3f}, Actual={avg_actual_steering:.3f}, AbsAvg={avg_steering_angle:.3f}")
                print(f"  > Wheel Vel (avg m/s): Cmd={avg_cmd_wheel_vel_ms:.2f}, Actual={avg_actual_wheel_vel_ms:.2f}")
                print(f"  > Min Cmd Vel (m/s): {min_cmd_vel:.3f}")
                print(f"  > Rewards: Total={total_reward:.2f}, Avg={avg_reward:.3f} | (Fwd: {total_forward_reward:.2f}, Speed: {total_speed_reward:.2f}, SlowPen: {total_survival_reward:.2f}, DangerPen: {total_danger_penalty:.2f})")

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
        # 리셋 후 첫 체크를 지연시키기 위해 음수로 초기화
        # 예: stuck_initial_delay=180, stuck_check_interval=60 → -120에서 시작
        # 180스텝 후 60에 도달하여 첫 체크 발생
        self.steps_since_last_check[env_ids] = self.stuck_check_interval - self.stuck_initial_delay

        # 슬립 감지를 위해 바퀴 위치 추적기 재설정 (4WD: 후륜만 추적)
        # 전륜은 조향 때문에 슬립 계산이 복잡하므로 후륜만 사용
        self.wheel_pos_at_last_check[env_ids] = self.robot.data.joint_pos[env_ids][:, self._rear_wheel_ids]

        # 에피소드 통계 재설정
        self.episode_reward_sum[env_ids] = 0.0
        self.episode_commanded_speed_sum[env_ids] = 0.0
        self.episode_actual_speed_sum[env_ids] = 0.0
        self.episode_commanded_steering_sum[env_ids] = 0.0
        self.episode_actual_steering_sum[env_ids] = 0.0
        self.episode_reward_forward_sum[env_ids] = 0.0
        self.episode_reward_speed_sum[env_ids] = 0.0
        self.episode_reward_survival_sum[env_ids] = 0.0
        self.episode_danger_penalty_sum[env_ids] = 0.0
        self.episode_steering_angle_sum[env_ids] = 0.0
        self.episode_commanded_wheel_vel_sum[env_ids] = 0.0
        self.episode_actual_wheel_vel_sum[env_ids] = 0.0
        self.episode_min_commanded_vel[env_ids] = float("inf")

        # LiDAR 센서 데이터 초기화
        if self.lidar_distances is not None:
            self.lidar_distances[env_ids] = self.cfg.lidar.max_distance

        self.collision_terminated[env_ids] = False
        self.stuck_terminated[env_ids] = False
        self.out_of_bounds_terminated[env_ids] = False
        
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

        # 트랙 진행 거리 초기화 (선분 투영 방식)
        spawn_pos_xy = spawn_pos[:, :2]  # (num_resets, 2)

        # Segment 계산
        num_segments = len(self.centerline) - 1
        segment_starts = self.centerline[:-1]
        segment_ends = self.centerline[1:]
        segment_vectors = segment_ends - segment_starts
        segment_len_sq = (segment_vectors ** 2).sum(dim=-1)

        # 스폰 위치에서 segment로의 벡터
        to_spawn = spawn_pos_xy.unsqueeze(1) - segment_starts.unsqueeze(0)

        # 투영 비율 t
        dot_product = (to_spawn * segment_vectors.unsqueeze(0)).sum(dim=-1)
        t = dot_product / (segment_len_sq.unsqueeze(0) + 1e-8)
        t = torch.clamp(t, 0.0, 1.0)

        # 투영점과 거리
        projected = segment_starts.unsqueeze(0) + t.unsqueeze(-1) * segment_vectors.unsqueeze(0)
        dist_to_proj = torch.norm(spawn_pos_xy.unsqueeze(1) - projected, dim=-1)

        # 가장 가까운 segment
        closest_seg = torch.argmin(dist_to_proj, dim=-1)
        batch_idx = torch.arange(len(spawn_pos_xy), device=spawn_pos_xy.device)
        t_closest = t[batch_idx, closest_seg]

        # 초기 진행거리
        seg_start_dist = self.centerline_cumulative_dist[closest_seg]
        seg_len = torch.sqrt(segment_len_sq[closest_seg])
        initial_progress = seg_start_dist + t_closest * seg_len

        # 초기화
        self.previous_progress[env_ids] = initial_progress
        self.total_distance[env_ids] = initial_progress
        self.previous_total_distance[env_ids] = initial_progress
        self.lap_count[env_ids] = 0
        self.current_waypoint_idx[env_ids] = closest_seg


