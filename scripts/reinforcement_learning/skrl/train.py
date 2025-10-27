# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--stage", type=int, default=3, choices=[1, 2, 3],
                    help="Training stage: 1=Angle-based (MLP only), 2=CNN (MLP frozen), 3=CNN+MLP (fine-tuning, default: 3)")
parser.add_argument("--mlp_checkpoint", type=str, default=None,
                    help="Path to Stage 1 checkpoint to load MLP weights for Stage 2/3")
parser.add_argument("--cnn_checkpoint", type=str, default=None,
                    help="Path to Stage 2 checkpoint to load CNN weights for Stage 3")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO", "SAC"],
    help="The RL algorithm used for training the skrl agent.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import random
from datetime import datetime

import omni
import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    # Extract algorithm from entry point or use default
    if "sac" in agent_cfg_entry_point.lower():
        algorithm = "sac"
    elif "ppo" in agent_cfg_entry_point.lower():
        algorithm = "ppo"
    elif "amp" in agent_cfg_entry_point.lower():
        algorithm = "amp"
    else:
        algorithm = args_cli.algorithm.lower() if args_cli.algorithm else "unknown"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # ========== F1TENTH: Manual setup with CNN+MLP models ==========
    # skrl Runner doesn't support custom model paths, so we manually create agent/trainer
    use_custom_f1tenth_models = args_cli.task == "Isaac-F1tenth-Direct-v0"

    if use_custom_f1tenth_models:
        try:
            from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
            from skrl.trainers.torch import SequentialTrainer
            from skrl.memories.torch import RandomMemory
            from skrl.resources.preprocessors.torch import RunningStandardScaler
            from isaaclab_tasks.direct.f1tenth.models import CNNMLPPolicy, CNNMLPCritic

            # Stage 선택
            use_angle_based = (args_cli.stage == 1)

            if args_cli.stage == 1:
                stage_name = "Stage 1 (Angle-based + MLP)"
            elif args_cli.stage == 2:
                stage_name = "Stage 2 (CNN only, MLP frozen)"
            else:
                stage_name = "Stage 3 (CNN + MLP fine-tuning)"

            print(f"[INFO] Training {stage_name}")
            print("[INFO] Using CNN+MLP hybrid models for F1TENTH (TinyLidarNet architecture)")
            if use_angle_based:
                print("       - LiDAR: 1080 → Angle Binning (128 bins) → Min Pooling → 128 features")
            else:
                print("       - LiDAR: 1080 → 1D CNN → 128 features")
            print("       - Policy MLP: 130 → 128 → 128 → 64 → 2")
            print("       - Critic MLP: 132 → 256 → 256 → 128 → 1")

            # Create custom models directly
            device = env.device
            models = {}

            # Policy
            models["policy"] = CNNMLPPolicy(
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device,
                clip_actions=agent_cfg["models"]["policy"]["clip_actions"],
                clip_log_std=agent_cfg["models"]["policy"]["clip_log_std"],
                min_log_std=agent_cfg["models"]["policy"]["min_log_std"],
                max_log_std=agent_cfg["models"]["policy"]["max_log_std"],
                initial_log_std=agent_cfg["models"]["policy"]["initial_log_std"],
                use_angle_based=use_angle_based
            )

            # Critics
            for critic_name in ["critic_1", "critic_2", "target_critic_1", "target_critic_2"]:
                models[critic_name] = CNNMLPCritic(
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    device=device,
                    clip_actions=agent_cfg["models"][critic_name]["clip_actions"],
                    use_angle_based=use_angle_based
                )

            # Create memory
            memory = RandomMemory(
                memory_size=agent_cfg["memory"]["memory_size"],
                num_envs=env.num_envs,
                device=device
            )

            # Create SAC agent with custom models
            agent_cfg_sac = SAC_DEFAULT_CONFIG.copy()
            agent_cfg_sac.update(agent_cfg["agent"])
            agent_cfg_sac["state_preprocessor"] = RunningStandardScaler
            agent_cfg_sac["state_preprocessor_kwargs"] = agent_cfg["agent"]["state_preprocessor_kwargs"]

            agent = SAC(
                models=models,
                memory=memory,
                cfg=agent_cfg_sac,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device
            )

            # ===== Stage 2/3: 가중치 로드 =====
            import torch

            if not use_angle_based and args_cli.mlp_checkpoint:
                print(f"[INFO] Loading MLP weights from Stage 1 checkpoint")
                print(f"       Checkpoint: {args_cli.mlp_checkpoint}")

                stage1_checkpoint = torch.load(args_cli.mlp_checkpoint, map_location=device)

                # Policy MLP 로드
                agent.policy.mlp.load_state_dict({k.replace('mlp.', ''): v for k, v in stage1_checkpoint['policy'].items() if 'mlp.' in k}, strict=False)
                agent.policy.log_std_parameter.data = stage1_checkpoint['policy']['log_std_parameter']

                # Critic MLP 로드
                for critic_name in ["critic_1", "critic_2"]:
                    if critic_name in stage1_checkpoint:
                        critic = getattr(agent, critic_name)
                        critic.mlp.load_state_dict({k.replace('mlp.', ''): v for k, v in stage1_checkpoint[critic_name].items()}, strict=False)

                print("[INFO] MLP weights loaded successfully")

            if args_cli.stage == 3 and args_cli.cnn_checkpoint:
                print(f"[INFO] Stage 3: Loading CNN weights from Stage 2 checkpoint")
                print(f"       Checkpoint: {args_cli.cnn_checkpoint}")

                stage2_checkpoint = torch.load(args_cli.cnn_checkpoint, map_location=device)

                # Policy CNN 로드
                agent.policy.lidar_extractor.load_state_dict(
                    {k.replace('lidar_extractor.', ''): v for k, v in stage2_checkpoint['policy'].items() if 'lidar_extractor.' in k},
                    strict=False
                )

                # Critic CNN 로드
                for critic_name in ["critic_1", "critic_2"]:
                    if critic_name in stage2_checkpoint:
                        critic = getattr(agent, critic_name)
                        critic.lidar_extractor.load_state_dict(
                            {k.replace('lidar_extractor.', ''): v for k, v in stage2_checkpoint[critic_name].items() if 'lidar_extractor.' in k},
                            strict=False
                        )

                print("[INFO] CNN weights loaded successfully")

            # ===== Stage별 학습률 설정 =====
            import torch.optim as optim

            if args_cli.stage == 1:
                # ===== Stage 1: MLP만 학습 (Angle-based feature) =====
                print("[INFO] Stage 1: Training MLP only (angle-based features)")

                # Feature extractor는 파라미터 없음 (순수 연산)
                # MLP만 학습
                policy_mlp_params = list(agent.policy.mlp.parameters()) + [agent.policy.log_std_parameter]
                agent.policy_optimizer = optim.Adam(policy_mlp_params, lr=1.0e-4)

                critic_mlp_params = []
                for critic_name in ["critic_1", "critic_2"]:
                    critic = getattr(agent, critic_name)
                    critic_mlp_params.extend(list(critic.mlp.parameters()))

                agent.critic_optimizer = optim.Adam(critic_mlp_params, lr=5.0e-4)

                print(f"  Policy MLP: lr=1.0e-4 (fixed)")
                print(f"  Critic MLP: lr=5.0e-4 (fixed)")

            elif args_cli.stage == 2:
                # ===== Stage 2: CNN 학습 + MLP freeze =====
                print("[INFO] Stage 2: Training CNN (MLP frozen)")

                # CNN 파라미터만 학습 (MLP는 freeze)
                policy_cnn_params = list(agent.policy.lidar_extractor.parameters())

                # MLP freeze (학습률 0)
                for param in agent.policy.mlp.parameters():
                    param.requires_grad = False
                agent.policy.log_std_parameter.requires_grad = False

                # Policy: CNN만 학습
                agent.policy_optimizer = optim.Adam(policy_cnn_params, lr=1.0e-4)

                # Critic: CNN만 학습, MLP freeze
                critic_cnn_params = []
                for critic_name in ["critic_1", "critic_2"]:
                    critic = getattr(agent, critic_name)
                    critic_cnn_params.extend(list(critic.lidar_extractor.parameters()))
                    # MLP freeze
                    for param in critic.mlp.parameters():
                        param.requires_grad = False

                agent.critic_optimizer = optim.Adam(critic_cnn_params, lr=5.0e-4)

                # CNN용 학습률 스케줄러 (Stage 2만)
                def cnn_lr_lambda(epoch):
                    return 0.99995 ** epoch

                agent.policy_cnn_scheduler = optim.lr_scheduler.LambdaLR(
                    agent.policy_optimizer,
                    lr_lambda=cnn_lr_lambda
                )

                agent.critic_cnn_scheduler = optim.lr_scheduler.LambdaLR(
                    agent.critic_optimizer,
                    lr_lambda=cnn_lr_lambda
                )

                # 스케줄러 스텝 함수
                original_update = agent._update
                agent._training_step_counter = 0

                def update_with_scheduler(*args, **kwargs):
                    result = original_update(*args, **kwargs)

                    # CNN 학습률 감소
                    agent.policy_cnn_scheduler.step()
                    agent.critic_cnn_scheduler.step()

                    # 로깅
                    agent._training_step_counter += 1
                    if agent._training_step_counter % 1000 == 0:
                        policy_cnn_lr = agent.policy_optimizer.param_groups[0]['lr']
                        critic_cnn_lr = agent.critic_optimizer.param_groups[0]['lr']
                        print(f"[LR] Step {agent._training_step_counter}: "
                              f"Policy CNN={policy_cnn_lr:.2e}, Critic CNN={critic_cnn_lr:.2e}")

                    return result

                agent._update = update_with_scheduler

                print(f"  Policy CNN: lr=1.0e-4 (scheduled, gamma=0.99995)")
                print(f"  Critic CNN: lr=5.0e-4 (scheduled, gamma=0.99995)")
                print(f"  MLP: frozen (학습 안 함)")

            else:
                # ===== Stage 3: CNN + MLP 동시 학습 (Fine-tuning) =====
                print("[INFO] Stage 3: Fine-tuning CNN + MLP together")

                # CNN과 MLP 모두 학습 (분리된 학습률)
                policy_cnn_params = list(agent.policy.lidar_extractor.parameters())
                policy_mlp_params = list(agent.policy.mlp.parameters()) + [agent.policy.log_std_parameter]

                # CNN은 낮은 학습률, MLP는 더 낮은 학습률 (fine-tuning)
                agent.policy_optimizer = optim.Adam([
                    {'params': policy_cnn_params, 'lr': 5.0e-5},   # CNN: 낮은 학습률
                    {'params': policy_mlp_params, 'lr': 1.0e-5}    # MLP: 더 낮은 학습률 (fine-tune)
                ])

                # Critic도 동일
                critic_cnn_params = []
                critic_mlp_params = []
                for critic_name in ["critic_1", "critic_2"]:
                    critic = getattr(agent, critic_name)
                    critic_cnn_params.extend(list(critic.lidar_extractor.parameters()))
                    critic_mlp_params.extend(list(critic.mlp.parameters()))

                agent.critic_optimizer = optim.Adam([
                    {'params': critic_cnn_params, 'lr': 2.5e-4},   # CNN: 낮은 학습률
                    {'params': critic_mlp_params, 'lr': 5.0e-5}    # MLP: 더 낮은 학습률 (fine-tune)
                ])

                # CNN용 스케줄러 (더 느리게 감소)
                def cnn_lr_lambda(epoch):
                    return 0.99998 ** epoch  # 더 느린 감소

                agent.policy_cnn_scheduler = optim.lr_scheduler.LambdaLR(
                    agent.policy_optimizer,
                    lr_lambda=[cnn_lr_lambda, lambda e: 1.0]  # [CNN, MLP]
                )

                agent.critic_cnn_scheduler = optim.lr_scheduler.LambdaLR(
                    agent.critic_optimizer,
                    lr_lambda=[cnn_lr_lambda, lambda e: 1.0]  # [CNN, MLP]
                )

                # 스케줄러 스텝 함수
                original_update = agent._update
                agent._training_step_counter = 0

                def update_with_scheduler(*args, **kwargs):
                    result = original_update(*args, **kwargs)

                    # CNN 학습률만 감소
                    agent.policy_cnn_scheduler.step()
                    agent.critic_cnn_scheduler.step()

                    # 로깅
                    agent._training_step_counter += 1
                    if agent._training_step_counter % 1000 == 0:
                        policy_cnn_lr = agent.policy_optimizer.param_groups[0]['lr']
                        policy_mlp_lr = agent.policy_optimizer.param_groups[1]['lr']
                        critic_cnn_lr = agent.critic_optimizer.param_groups[0]['lr']
                        critic_mlp_lr = agent.critic_optimizer.param_groups[1]['lr']
                        print(f"[LR] Step {agent._training_step_counter}: "
                              f"Policy(CNN={policy_cnn_lr:.2e}, MLP={policy_mlp_lr:.2e}), "
                              f"Critic(CNN={critic_cnn_lr:.2e}, MLP={critic_mlp_lr:.2e})")

                    return result

                agent._update = update_with_scheduler

                print(f"  Policy: CNN=5.0e-5 (scheduled), MLP=1.0e-5 (fixed)")
                print(f"  Critic: CNN=2.5e-4 (scheduled), MLP=5.0e-5 (fixed)")
                print(f"  Fine-tuning both CNN and MLP with lower learning rates")

            # Create trainer
            trainer_cfg = agent_cfg["trainer"].copy()
            trainer_cfg["close_environment_at_exit"] = False  # We'll close it manually

            trainer = SequentialTrainer(
                env=env,
                agents=agent,
                cfg=trainer_cfg
            )

            # Load checkpoint if specified
            if resume_path:
                print(f"[INFO] Loading model checkpoint from: {resume_path}")
                agent.load(resume_path)

            # Run training
            trainer.train()

            use_custom_f1tenth_models = True  # Mark as handled

        except Exception as e:
            print(f"[WARNING] Failed to setup F1TENTH CNN+MLP models: {e}")
            print("          Falling back to default Runner with MLP models")
            use_custom_f1tenth_models = False

    # ========== Standard Runner (for non-F1TENTH tasks or fallback) ==========
    if not use_custom_f1tenth_models:
        # configure and instantiate the skrl runner
        # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
        runner = Runner(env, agent_cfg)

        # load checkpoint (if specified)
        if resume_path:
            print(f"[INFO] Loading model checkpoint from: {resume_path}")
            runner.agent.load(resume_path)

        # run training
        runner.run()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
