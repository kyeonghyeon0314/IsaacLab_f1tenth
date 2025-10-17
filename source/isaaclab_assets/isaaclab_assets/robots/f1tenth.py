# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for F1TENTH race car with Hokuyo LiDAR."""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# Configuration
##

# Get the directory of the current file
_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
# Construct the full path to the USD file by navigating from the current file's location
F1TENTH_USD_PATH = os.path.join(os.path.dirname(_CURRENT_DIR), "f1tenth", "f1tenth.usd")

F1TENTH_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=F1TENTH_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=15.0,  # F1TENTH max speed ~10 m/s + margin
            max_angular_velocity=100.0,
            max_depenetration_velocity=5.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,  # Higher for stability
            solver_velocity_iteration_count=4,
            sleep_threshold=0.001,
            stabilization_threshold=0.0005,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.15),  # Start slightly above ground
        rot=(1.0, 0.0, 0.0, 0.0),  # No rotation (w, x, y, z)
        joint_pos={
            # Initialize steering to center
            # Note: Isaac Lab strips path prefix, use only joint names
            "left_steering_hinge_joint": 0.0,
            "right_steering_hinge_joint": 0.0,
            # Wheels at rest
            "left_rear_wheel_joint": 0.0,
            "right_rear_wheel_joint": 0.0,
            "left_front_wheel_joint": 0.0,
            "right_front_wheel_joint": 0.0,
        },
        joint_vel={
            "left_rear_wheel_joint": 0.0,
            "right_rear_wheel_joint": 0.0,
        },
    ),
    actuators={
        # VESC Motor (rear wheels) - Velocity Control
        "rear_wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*rear_wheel_joint"],  # Both rear wheels
            effort_limit_sim=10.0,  # Motor torque limit
            velocity_limit=196.0,  # Max wheel angular velocity (10 m/s / 0.0508 m)
            stiffness=0.0,  # Velocity control mode
            damping=0.01,  # Reduced from 0.5 for better velocity tracking
        ),
        # Servo Steering (Ackermann) - Position Control
        "steering": ImplicitActuatorCfg(
            joint_names_expr=[".*steering_hinge_joint"],  # Both steering joints
            effort_limit_sim=10.0,  # Servo torque
            stiffness=800.0,  # High stiffness for position control
            damping=100.0,
        ),
        # Front wheels (passive rotation)
        "front_wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*front_wheel_joint"],
            effort_limit_sim=0.0,  # No actuation
            stiffness=0.0,
            damping=0.1,  # Small damping for realistic rolling
        ),
    },
)
"""Configuration for F1TENTH race car (Traxxas Fiesta ST Rally 1/10 Scale)."""
