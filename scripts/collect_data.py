"""
SAWM - Milestone 4 Trajectory Dataset Collector
Generates diverse, reproducible multi-episode datasets using SawmEnvironment and BaselinePlanner.

Partitions:
- 80 Train episodes -> data/trajectories/train/
- 20 Validation episodes -> data/trajectories/val/

Subsets:
- Nominal (clear workspace, randomized cube and goal)
- Static Obstacle (pillar hazard placed near transit corridor)
- Moving Proxy (dynamic human proxy encroaching at varied speeds)

Usage:
  python scripts/collect_data.py
  python scripts/collect_data.py --train 80 --val 20 --noise 0.015
"""

import os
import sys
import time
import argparse
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.environment import SawmEnvironment
from simulation.robot import FrankaPandaRobot
from planning.baseline_planner import BaselinePlanner
from world_model.dataset import ObservationBuilder, TrajectoryNormalizer


def parse_args():
    parser = argparse.ArgumentParser(description="SAWM Milestone 4 - Dataset Collector")
    parser.add_argument("--train-episodes", type=int, default=80, help="Number of training episodes")
    parser.add_argument("--val-episodes", type=int, default=20, help="Number of validation episodes")
    parser.add_argument("--output-dir", type=str, default="data/trajectories", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Master PRNG seed")
    parser.add_argument("--noise-std", type=float, default=0.015, help="Action exploration noise std (rad)")
    return parser.parse_args()


def collect_single_episode(
    env: SawmEnvironment,
    obs_builder: ObservationBuilder,
    planner: BaselinePlanner,
    episode_id: int,
    subset: str,
    rng: np.random.RandomState,
    noise_std: float = 0.0,
) -> dict:
    """Run a single randomized episode and record (x_t, a_t, x_{t+1}) sequence."""
    obs_builder.reset()
    env.reset()

    # 1. Domain Randomization: Cube & Goal Positions
    # Cube: nominal [0.55, -0.12, 0.425]
    cube_dx = rng.uniform(-0.03, 0.03)
    cube_dy = rng.uniform(-0.03, 0.03)
    cube_pos = np.array([0.55 + cube_dx, -0.12 + cube_dy, 0.425], dtype=np.float64)

    # Goal: nominal [0.55, 0.20, 0.401]
    goal_dx = rng.uniform(-0.03, 0.03)
    goal_dy = rng.uniform(-0.03, 0.03)
    goal_pos = np.array([0.55 + goal_dx, 0.20 + goal_dy, 0.401], dtype=np.float64)

    # Apply cube position in physics
    cube_qpos_id = env.model.jnt_qposadr[env.model.body_jntadr[env.target_object_body_id]]
    env.data.qpos[cube_qpos_id : cube_qpos_id + 3] = cube_pos
    env.data.qpos[cube_qpos_id + 3 : cube_qpos_id + 7] = [1.0, 0.0, 0.0, 0.0]
    env.data.qvel[env.model.body_dofadr[env.target_object_body_id] : env.model.body_dofadr[env.target_object_body_id] + 6] = 0.0

    # 2. Configure Hazard Subset
    env.obstacle_manager.park_all()
    if subset == "nominal":
        pass  # clear workspace
    elif subset == "static_obstacle":
        obs_x = rng.uniform(0.60, 0.68)
        obs_y = rng.uniform(0.12, 0.22)
        obs_z = 0.52
        env.obstacle_manager.set_static_obstacle_position(np.array([obs_x, obs_y, obs_z]), active=True)
    elif subset == "moving_proxy":
        p_start = np.array([rng.uniform(0.50, 0.60), rng.uniform(0.30, 0.40), rng.uniform(0.55, 0.65)])
        p_target = np.array([rng.uniform(0.48, 0.55), rng.uniform(-0.05, 0.05), rng.uniform(0.55, 0.65)])
        proxy_steps = rng.randint(250, 450)
        env.obstacle_manager.set_proxy_trajectory(p_start, p_target, steps=proxy_steps)

    # Forward kinematics after position adjustments
    import mujoco
    mujoco.mj_forward(env.model, env.data)

    # 3. Plan Baseline Keyframes
    raw_obs = env.get_observation()
    actual_cube = raw_obs["target_object"]["position"]
    actual_goal = raw_obs["goal"]["position"]
    keyframes, ik_ok, ik_err = planner.plan_keyframes(actual_cube, actual_goal)

    if not ik_ok:
        # Fallback to nominal ready pose if IK failed for perturbed target
        keyframes, _, _ = planner.plan_keyframes(np.array([0.55, -0.12, 0.425]), np.array([0.55, 0.20, 0.401]))

    # Define segments
    segments = [
        ("HOME", keyframes["home"], keyframes["home"], 10, planner.open_ctrl),
        ("MOVE_TO_PREGRASP", keyframes["home"], keyframes["pregrasp"], planner.transit_steps, planner.open_ctrl),
        ("DESCEND_TO_OBJECT", keyframes["pregrasp"], keyframes["grasp"], planner.vertical_steps, planner.open_ctrl),
        ("GRASP", keyframes["grasp"], keyframes["grasp"], planner.grasp_dwell, planner.close_ctrl),
        ("LIFT", keyframes["grasp"], keyframes["lift"], planner.vertical_steps, planner.close_ctrl),
        ("MOVE_TO_TARGET", keyframes["lift"], keyframes["transport"], planner.transit_steps + 20, planner.close_ctrl),
        ("RELEASE", keyframes["transport"], keyframes["place"], planner.vertical_steps, planner.close_ctrl),
        ("RELEASE", keyframes["place"], keyframes["place"], planner.release_dwell, planner.open_ctrl),
        ("RETURN_HOME", keyframes["place"], keyframes["retract"], planner.vertical_steps, planner.open_ctrl),
        ("RETURN_HOME", keyframes["retract"], keyframes["home"], planner.transit_steps, planner.open_ctrl),
    ]

    obs_list = []
    act_list = []
    reward_list = []
    term_list = []

    # Initial observation x_0
    x_curr = obs_builder.extract_observation_vector(raw_obs)
    obs_list.append(x_curr)

    total_steps = sum(s[3] for s in segments)
    step_count = 0

    q_min = np.array([env.robot.joint_limits[name][0] for name in env.robot.joint_names], dtype=np.float64)
    q_max = np.array([env.robot.joint_limits[name][1] for name in env.robot.joint_names], dtype=np.float64)

    for stage_name, q_start, q_end, num_steps, gripper_cmd in segments:
        traj = FrankaPandaRobot.interpolate_trajectory(q_start, q_end, steps=num_steps, method="cosine")
        for wp in traj:
            step_count += 1
            # Add action exploration noise if specified
            if noise_std > 0.0:
                noise = rng.normal(0.0, noise_std, size=7)
                wp_cmd = np.clip(wp + noise, q_min, q_max)
            else:
                wp_cmd = wp

            # Action a_t executed at step t
            a_curr = obs_builder.extract_action_vector(wp_cmd, gripper_cmd)
            act_list.append(a_curr)

            # Step simulation -> transition to x_{t+1}
            raw_obs = env.step(arm_targets=wp_cmd, gripper_target=gripper_cmd)
            x_next = obs_builder.extract_observation_vector(raw_obs)
            obs_list.append(x_next)

            # Sparse reward indicator (distance to goal)
            dist_goal = np.linalg.norm(raw_obs["target_object"]["position"][:2] - actual_goal[:2])
            reward_list.append(float(-dist_goal))
            term_list.append(1.0 if step_count == total_steps else 0.0)

    # Verification of trajectory length alignment
    obs_arr = np.array(obs_list, dtype=np.float32)       # (T, 49)
    act_arr = np.array(act_list, dtype=np.float32)       # (T-1, 8)
    rew_arr = np.array(reward_list, dtype=np.float32)    # (T-1,)
    term_arr = np.array(term_list, dtype=np.float32)     # (T-1,)

    assert len(obs_arr) == len(act_arr) + 1, f"Mismatch: {len(obs_arr)} obs vs {len(act_arr)} act"

    final_obj = raw_obs["target_object"]["position"]
    final_err = float(np.linalg.norm(final_obj[:2] - actual_goal[:2]))
    success = final_err < 0.060 and abs(final_obj[2] - 0.400) < 0.035

    metadata = {
        "episode_id": episode_id,
        "subset": subset,
        "success": success,
        "placement_error": final_err,
        "total_steps": len(act_arr),
        "cube_init_pos": cube_pos.tolist(),
        "goal_pos": goal_pos.tolist(),
    }

    return {
        "observations": obs_arr,
        "actions": act_arr,
        "rewards": rew_arr,
        "terminals": term_arr,
        "metadata": metadata,
    }


def main():
    args = parse_args()
    print("=" * 80)
    print("SAWM - Milestone 4 Trajectory Dataset Collector")
    print("=" * 80)
    print(f"Target Train Episodes: {args.train_episodes}")
    print(f"Target Val Episodes:   {args.val_episodes}")
    print(f"Output Directory:      {args.output_dir}")
    print(f"Action Noise std:      {args.noise_std} rad")
    print(f"Master Seed:           {args.seed}")
    print("=" * 80)

    train_dir = os.path.join(args.output_dir, "train")
    val_dir = os.path.join(args.output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    rng = np.random.RandomState(args.seed)
    env = SawmEnvironment()
    obs_builder = ObservationBuilder(env)
    planner = BaselinePlanner(env)

    # Subsets distribution (40% nominal, 30% static, 30% moving proxy)
    def get_subset(idx: int, total: int) -> str:
        ratio = idx / max(1, total)
        if ratio < 0.40:
            return "nominal"
        elif ratio < 0.70:
            return "static_obstacle"
        else:
            return "moving_proxy"

    # 1. Collect Training Episodes
    print("\n[INFO] Collecting Training Episodes (80)...")
    train_obs_all = []
    t0 = time.time()

    for ep_idx in range(args.train_episodes):
        subset = get_subset(ep_idx, args.train_episodes)
        # Apply noise on 50% of training episodes to broaden coverage
        noise = args.noise_std if ep_idx % 2 == 1 else 0.0

        ep_data = collect_single_episode(
            env=env,
            obs_builder=obs_builder,
            planner=planner,
            episode_id=ep_idx,
            subset=subset,
            rng=rng,
            noise_std=noise,
        )

        fpath = os.path.join(train_dir, f"episode_{ep_idx:04d}.npz")
        np.savez_compressed(
            fpath,
            observations=ep_data["observations"],
            actions=ep_data["actions"],
            rewards=ep_data["rewards"],
            terminals=ep_data["terminals"],
            metadata=ep_data["metadata"],
        )

        train_obs_all.append(ep_data["observations"])
        if (ep_idx + 1) % 10 == 0 or ep_idx == args.train_episodes - 1:
            dur = time.time() - t0
            print(f"  [Train] Saved {ep_idx+1:2d}/{args.train_episodes} episodes | subset: {subset:<16s} | Elapsed: {dur:.1f}s")

    # 2. Collect Validation Episodes (Strictly held-out configurations)
    print("\n[INFO] Collecting Validation Episodes (20)...")
    val_rng = np.random.RandomState(args.seed + 9999)  # Independent seed stream

    for val_idx in range(args.val_episodes):
        subset = get_subset(val_idx, args.val_episodes)
        # Validation uses zero noise (true dynamic test)
        ep_data = collect_single_episode(
            env=env,
            obs_builder=obs_builder,
            planner=planner,
            episode_id=args.train_episodes + val_idx,
            subset=subset,
            rng=val_rng,
            noise_std=0.0,
        )

        fpath = os.path.join(val_dir, f"episode_{val_idx:04d}.npz")
        np.savez_compressed(
            fpath,
            observations=ep_data["observations"],
            actions=ep_data["actions"],
            rewards=ep_data["rewards"],
            terminals=ep_data["terminals"],
            metadata=ep_data["metadata"],
        )
        if (val_idx + 1) % 5 == 0 or val_idx == args.val_episodes - 1:
            print(f"  [Val]   Saved {val_idx+1:2d}/{args.val_episodes} episodes | subset: {subset:<16s}")

    # 3. Fit Normalizer Strictly on Training Observations
    print("\n[INFO] Computing Normalizer Statistics on Train Episodes Only...")
    train_concat = np.concatenate(train_obs_all, axis=0)
    normalizer = TrajectoryNormalizer(obs_dim=49)
    normalizer.fit(train_concat)

    norm_path = "checkpoints/normalizer.pt"
    normalizer.save(norm_path)
    print(f"[INFO] Train normalizer successfully saved to: {norm_path}")
    print(f"       Total training transitions: {len(train_concat)}")
    print(f"       Features normalized: {49 - len(normalizer.exempt_indices)} (Exempt: {len(normalizer.exempt_indices)})")

    env.close()
    print("\n" + "=" * 80)
    print("DATASET COLLECTION COMPLETED SUCCESSFULLY")
    print(f"Train directory: {train_dir} ({args.train_episodes} episodes)")
    print(f"Val directory:   {val_dir} ({args.val_episodes} episodes)")
    print("=" * 80)


if __name__ == "__main__":
    main()
