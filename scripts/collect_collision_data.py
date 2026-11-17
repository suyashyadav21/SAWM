"""
Milestone 5: Targeted Collision Data Collector & Manifest Generator.

Generates 40 diverse collision and near-miss episodes across 5 categories:
1. pick_approach_collision (8 episodes)
2. transit_corridor_collision (8 episodes)
3. place_descent_collision (8 episodes)
4. dynamic_proxy_collision (8 episodes)
5. boundary_grazing_hard_negatives (8 episodes)

Constructs the unified dataset manifest across 3 isolated partitions:
- Training:    80 original M4 train + 26 augmented collision = 106 episodes
- Calibration: 10 held-out M4 val  +  7 augmented collision = 17 episodes
- Testing:     10 held-out M4 val  +  7 augmented collision = 17 episodes

Usage:
    python scripts/collect_collision_data.py
"""

import os
import sys
import json
import time
import argparse
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.environment import SawmEnvironment
from world_model.dataset import ObservationBuilder
from planning.baseline_planner import BaselinePlanner
from planning.fsm import TaskState


CATEGORIES = [
    "pick_approach_collision",
    "transit_corridor_collision",
    "place_descent_collision",
    "dynamic_proxy_collision",
    "boundary_grazing_hard_negatives",
]


def setup_scenario_hazards(env: SawmEnvironment, category: str, ep_idx: int, rng: np.random.RandomState):
    """
    Configure hazards according to the specific perturbation category.
    Returns:
        perturbation_params: dict describing hazard placement and trajectory.
    """
    env.obstacle_manager.park_all()
    params = {"category": category, "ep_idx": ep_idx}

    if category == "pick_approach_collision":
        # Static pillar placed in descent corridor to cube
        obs_x = 0.46 + rng.uniform(-0.02, 0.04)
        obs_y = -0.04 + rng.uniform(-0.04, 0.02)
        obs_z = 0.52 + rng.uniform(-0.02, 0.02)
        pos = np.array([obs_x, obs_y, obs_z], dtype=np.float64)
        env.obstacle_manager.set_static_obstacle_position(pos, active=True)
        params["obstacle_pos"] = pos.tolist()

    elif category == "transit_corridor_collision":
        # Static pillar placed in transport transit path
        obs_x = 0.54 + rng.uniform(-0.03, 0.04)
        obs_y = 0.04 + rng.uniform(-0.03, 0.04)
        obs_z = 0.58 + rng.uniform(-0.03, 0.04)
        pos = np.array([obs_x, obs_y, obs_z], dtype=np.float64)
        env.obstacle_manager.set_static_obstacle_position(pos, active=True)
        params["obstacle_pos"] = pos.tolist()

    elif category == "place_descent_collision":
        # Static pillar placed adjacent to goal platform
        obs_x = 0.54 + rng.uniform(-0.03, 0.04)
        obs_y = 0.17 + rng.uniform(-0.03, 0.03)
        obs_z = 0.52 + rng.uniform(-0.02, 0.02)
        pos = np.array([obs_x, obs_y, obs_z], dtype=np.float64)
        env.obstacle_manager.set_static_obstacle_position(pos, active=True)
        params["obstacle_pos"] = pos.tolist()

    elif category == "dynamic_proxy_collision":
        # Dynamic proxy encroaching to intercept robot
        start_x = 0.82 + rng.uniform(-0.05, 0.05)
        start_y = 0.28 + rng.uniform(-0.05, 0.05)
        start_z = 0.60 + rng.uniform(-0.03, 0.03)
        p_start = np.array([start_x, start_y, start_z], dtype=np.float64)

        target_x = 0.52 + rng.uniform(-0.03, 0.03)
        target_y = 0.05 + rng.uniform(-0.05, 0.05)
        target_z = 0.58 + rng.uniform(-0.03, 0.03)
        p_target = np.array([target_x, target_y, target_z], dtype=np.float64)

        speed = 0.18 + rng.uniform(0.0, 0.12)  # m/s
        dist = float(np.linalg.norm(p_target - p_start))
        steps = max(100, int(dist / (speed * 0.01)))

        env.obstacle_manager.set_proxy_trajectory(p_start, p_target, steps=steps)
        params["proxy_start"] = p_start.tolist()
        params["proxy_target"] = p_target.tolist()
        params["proxy_speed"] = float(speed)
        params["proxy_steps"] = steps

    elif category == "boundary_grazing_hard_negatives":
        # Near-miss pillar placed just clear of trajectory (d ~ 2-5 cm)
        obs_x = 0.65 + rng.uniform(0.00, 0.03)
        obs_y = 0.18 + rng.uniform(0.00, 0.03)
        obs_z = 0.52
        pos = np.array([obs_x, obs_y, obs_z], dtype=np.float64)
        env.obstacle_manager.set_static_obstacle_position(pos, active=True)
        params["obstacle_pos"] = pos.tolist()

    return params


def collect_single_collision_episode(
    env: SawmEnvironment,
    obs_builder: ObservationBuilder,
    planner: BaselinePlanner,
    episode_id: int,
    category: str,
    rng: np.random.RandomState,
    max_steps_post_contact: int = 40,
) -> dict:
    """
    Run baseline planner under the hazard scenario and record synchronized trajectory.
    """
    env.reset()
    obs_builder.reset()

    # Slight target cube jitter (nominal: [0.55, -0.12, 0.425])
    cube_dx = rng.uniform(-0.015, 0.015)
    cube_dy = rng.uniform(-0.015, 0.015)
    cube_pos = np.array([0.55 + cube_dx, -0.12 + cube_dy, 0.425], dtype=np.float64)

    cube_qpos_id = env.model.jnt_qposadr[env.model.body_jntadr[env.target_object_body_id]]
    env.data.qpos[cube_qpos_id : cube_qpos_id + 3] = cube_pos
    env.data.qpos[cube_qpos_id + 3 : cube_qpos_id + 7] = [1.0, 0.0, 0.0, 0.0]
    env.data.qvel[env.model.body_dofadr[env.target_object_body_id] : env.model.body_dofadr[env.target_object_body_id] + 6] = 0.0

    # Configure hazards
    perturbation_params = setup_scenario_hazards(env, category, episode_id, rng)

    import mujoco
    mujoco.mj_forward(env.model, env.data)

    # Plan keyframes and discrete steps
    raw_obs = env.get_observation()
    actual_cube = raw_obs["target_object"]["position"]
    actual_goal = raw_obs["goal"]["position"]

    keyframes, ik_ok, ik_err = planner.plan_keyframes(actual_cube, actual_goal)
    if not ik_ok:
        keyframes, _, _ = planner.plan_keyframes(np.array([0.55, -0.12, 0.425]), np.array([0.55, 0.20, 0.401]))

    planner._planned_steps = planner.build_discrete_plan(keyframes)
    planner._current_step_cursor = 0

    obs_list = []
    act_list = []
    stages_list = []
    clearance_list = []
    contact_list = []
    rew_list = []
    term_list = []

    # Initial state x_0
    x_curr = obs_builder.extract_observation_vector(raw_obs)
    obs_list.append(x_curr)
    clearance_list.append(float(raw_obs["distances"].geometric_min_distance))
    c_init = 1.0 if (raw_obs["contacts"]["robot_obstacle_collision"] or raw_obs["contacts"]["robot_proxy_collision"]) else 0.0
    contact_list.append(c_init)

    collision_occurred = False
    first_contact_step = None
    steps_since_contact = 0
    total_steps = len(planner._planned_steps)

    for step_idx, step_info in enumerate(planner._planned_steps):
        wp = step_info["arm_target"]
        g_cmd = step_info["gripper_target"]
        fsm_stage = step_info["fsm_state"]

        act_vec = obs_builder.extract_action_vector(wp, g_cmd)
        act_list.append(act_vec)
        stages_list.append(fsm_stage.value)

        # Step simulation
        planner._current_step_cursor = step_idx
        raw_obs = env.step(arm_targets=wp, gripper_target=g_cmd)

        x_next = obs_builder.extract_observation_vector(raw_obs)
        obs_list.append(x_next)

        d_geom = float(raw_obs["distances"].geometric_min_distance)
        clearance_list.append(d_geom)

        is_hazard_contact = bool(
            raw_obs["contacts"]["robot_obstacle_collision"] or
            raw_obs["contacts"]["robot_proxy_collision"]
        )
        c_flag = 1.0 if is_hazard_contact else 0.0
        contact_list.append(c_flag)

        if is_hazard_contact:
            if not collision_occurred:
                collision_occurred = True
                first_contact_step = step_idx + 1
            steps_since_contact += 1

        dist_goal = float(np.linalg.norm(raw_obs["target_object"]["position"][:2] - actual_goal[:2]))
        rew_list.append(-dist_goal)

        # Early termination after post-contact dwell to avoid unbounded simulation distortion
        if collision_occurred and steps_since_contact >= max_steps_post_contact:
            term_list.append(1.0)
            break
        else:
            term_list.append(1.0 if step_idx == total_steps - 1 else 0.0)

    obs_arr = np.array(obs_list, dtype=np.float32)            # (T, 49)
    act_arr = np.array(act_list, dtype=np.float32)            # (T-1, 8)
    stages_arr = np.array(stages_list, dtype=object)          # (T-1,)
    clearance_arr = np.array(clearance_list, dtype=np.float32)# (T,)
    contact_arr = np.array(contact_list, dtype=np.float32)    # (T,)
    rew_arr = np.array(rew_list, dtype=np.float32)            # (T-1,)
    term_arr = np.array(term_list, dtype=np.float32)          # (T-1,)

    assert len(obs_arr) == len(act_arr) + 1, f"Mismatch: {len(obs_arr)} obs vs {len(act_arr)} act"
    assert len(obs_arr) == len(clearance_arr) == len(contact_arr)
    assert len(act_arr) == len(stages_arr) == len(rew_arr) == len(term_arr)

    min_d = float(np.min(clearance_arr))

    metadata = {
        "episode_id": episode_id,
        "scenario_category": category,
        "seed": int(rng.randint(0, 1000000)),
        "perturbation_params": perturbation_params,
        "collision_occurred": bool(collision_occurred),
        "first_contact_step": first_contact_step,
        "total_steps": len(obs_arr),
        "min_clearance": min_d,
        "cube_init_pos": cube_pos.tolist(),
        "goal_pos": actual_goal.tolist(),
    }

    return {
        "observations": obs_arr,
        "actions": act_arr,
        "fsm_stages": stages_arr,
        "clearances": clearance_arr,
        "hazard_contacts": contact_arr,
        "rewards": rew_arr,
        "terminals": term_arr,
        "metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser(description="Collect 40 augmented collision episodes for SAWM M5")
    parser.add_argument("--output_dir", type=str, default="data/augmented_collisions")
    parser.add_argument("--manifest_path", type=str, default="data/dataset_manifest.json")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--repartition_only", action="store_true", help="Repartition existing episodes into manifest without regenerating data")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    episodes_summary = []
    category_counts = {cat: 0 for cat in CATEGORIES}
    collision_counts = {cat: 0 for cat in CATEGORIES}

    if args.repartition_only:
        print("=" * 80)
        print("SAWM - Milestone 5 Dataset Repartitioning (No Data Generation)")
        print(f"Reading existing episodes from: {args.output_dir}")
        print(f"Target Manifest Path:           {args.manifest_path}")
        print("=" * 80)

        import glob
        aug_files = sorted(glob.glob(os.path.join(args.output_dir, "collision_*.npz")))
        assert len(aug_files) == 40, f"Expected 40 existing augmented files, found {len(aug_files)}"

        for ep_idx, fpath in enumerate(aug_files):
            d = np.load(fpath, allow_pickle=True)
            meta = d["metadata"].item()
            cat = meta["scenario_category"]
            category_counts[cat] += 1
            if meta["collision_occurred"]:
                collision_counts[cat] += 1
            episodes_summary.append({
                "episode_id": ep_idx,
                "filename": os.path.basename(fpath),
                "filepath": fpath.replace("\\", "/"),
                "category": cat,
                "collision_occurred": meta["collision_occurred"],
                "first_contact_step": meta["first_contact_step"],
                "total_steps": meta["total_steps"],
                "min_clearance": meta["min_clearance"],
            })
    else:
        print("=" * 80)
        print("SAWM - Milestone 5 Collision Data Collector")
        print(f"Target Episodes: 40 (8 episodes x 5 categories)")
        print(f"Output Directory: {args.output_dir}")
        print(f"Manifest Path:    {args.manifest_path}")
        print("=" * 80)

        env = SawmEnvironment()
        obs_builder = ObservationBuilder(env)
        planner = BaselinePlanner(env)

        t0 = time.time()

        for ep_idx in range(40):
            cat = CATEGORIES[ep_idx // 8]
            ep_data = collect_single_collision_episode(
                env=env,
                obs_builder=obs_builder,
                planner=planner,
                episode_id=ep_idx,
                category=cat,
                rng=rng,
            )

            filename = f"collision_{ep_idx:04d}.npz"
            filepath = os.path.join(args.output_dir, filename)

            np.savez_compressed(
                filepath,
                observations=ep_data["observations"],
                actions=ep_data["actions"],
                fsm_stages=ep_data["fsm_stages"],
                clearances=ep_data["clearances"],
                hazard_contacts=ep_data["hazard_contacts"],
                rewards=ep_data["rewards"],
                terminals=ep_data["terminals"],
                metadata=ep_data["metadata"],
            )

            meta = ep_data["metadata"]
            category_counts[cat] += 1
            if meta["collision_occurred"]:
                collision_counts[cat] += 1

            contact_info = f"COLLISION at step {meta['first_contact_step']}" if meta["collision_occurred"] else "NO CONTACT"
            print(f"  [{ep_idx+1:02d}/40] {cat:<32s} | Steps={meta['total_steps']:3d} | MinDist={meta['min_clearance']*1000:6.1f}mm | {contact_info}")

            episodes_summary.append({
                "episode_id": ep_idx,
                "filename": filename,
                "filepath": filepath.replace("\\", "/"),
                "category": cat,
                "collision_occurred": meta["collision_occurred"],
                "first_contact_step": meta["first_contact_step"],
                "total_steps": meta["total_steps"],
                "min_clearance": meta["min_clearance"],
            })

        elapsed = time.time() - t0
        print("\n" + "=" * 80)
        print(f"COLLECTION COMPLETE in {elapsed:.1f}s")
        env.close()

    print("Category Breakdown:")
    for cat in CATEGORIES:
        print(f"  {cat:<32s}: {category_counts[cat]} episodes | {collision_counts[cat]} collisions")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Build Stratified Dataset Partition Manifest
    # -------------------------------------------------------------------------
    print("\n[INFO] Assembling stratified 3-way dataset manifest...")
    manifest = {
        "metadata": {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "stratification_method": "deterministic_category_stratified",
            "m4_train_episodes": 80,
            "m4_val_episodes": 20,
            "m5_augmented_episodes": 40,
            "total_episodes": 140,
            "target_partition_sizes": {"train": 106, "calibration": 17, "test": 17},
            "scenario_targets_augmented": {
                "pick_approach_collision": {"train": 6, "calibration": 1, "test": 1},
                "transit_corridor_collision": {"train": 5, "calibration": 2, "test": 1},
                "place_descent_collision": {"train": 5, "calibration": 1, "test": 2},
                "dynamic_proxy_collision": {"train": 5, "calibration": 1, "test": 2},
                "boundary_grazing_hard_negatives": {"train": 5, "calibration": 2, "test": 1},
            },
        },
        "partitions": {
            "train": {
                "description": "80 original M4 training episodes + 26 stratified augmented collision episodes",
                "count": 106,
                "episodes": [],
            },
            "calibration": {
                "description": "10 held-out M4 validation episodes (val/ep_0000..0009, IDs 80..89) + 7 stratified augmented collision episodes",
                "count": 17,
                "episodes": [],
            },
            "test": {
                "description": "10 held-out M4 validation episodes (val/ep_0010..0019, IDs 90..99) + 7 stratified augmented collision episodes",
                "count": 17,
                "episodes": [],
            },
        },
    }

    # 1. Add M4 Train Episodes (0000..0079) to Train partition
    for i in range(80):
        manifest["partitions"]["train"]["episodes"].append({
            "source": "m4_original_train",
            "episode_id": i,
            "filepath": f"data/trajectories/train/episode_{i:04d}.npz",
            "category": "m4_baseline_mixed",
            "augmented_collision": False,
        })

    # 2. Add M4 Val Episodes (metadata IDs 80..89, files episode_0000..0009.npz) to Calibration partition
    for i in range(10):
        manifest["partitions"]["calibration"]["episodes"].append({
            "source": "m4_original_val",
            "global_episode_id": 80 + i,
            "filepath": f"data/trajectories/val/episode_{i:04d}.npz",
            "category": "m4_baseline_mixed",
            "augmented_collision": False,
        })

    # 3. Add M4 Val Episodes (metadata IDs 90..99, files episode_0010..0019.npz) to Test partition
    for i in range(10, 20):
        manifest["partitions"]["test"]["episodes"].append({
            "source": "m4_original_val",
            "global_episode_id": 80 + i,
            "filepath": f"data/trajectories/val/episode_{i:04d}.npz",
            "category": "m4_baseline_mixed",
            "augmented_collision": False,
        })

    # 4. Stratified allocation of augmented episodes across categories:
    # Pick approach (0..7): Train=6, Cal=1, Test=1
    # Transit corridor (8..15): Train=5, Cal=2, Test=1
    # Place descent (16..23): Train=5, Cal=1, Test=2
    # Dynamic proxy (24..31): Train=5, Cal=1, Test=2
    # Boundary grazing (32..39): Train=5 (3 coll, 2 no-contact), Cal=2 (1 no-contact, 1 coll), Test=1 (1 no-contact)
    aug_train_ids = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 16, 17, 18, 19, 20, 24, 25, 26, 27, 28, 32, 33, 34, 36, 37]
    aug_cal_ids   = [6, 13, 14, 21, 29, 35, 39]
    aug_test_ids  = [7, 15, 22, 23, 30, 31, 38]

    assert len(aug_train_ids) == 26
    assert len(aug_cal_ids) == 7
    assert len(aug_test_ids) == 7
    assert len(set(aug_train_ids + aug_cal_ids + aug_test_ids)) == 40

    def add_augmented_to_partition(p_key, ep_ids):
        for ep_id in sorted(ep_ids):
            summary = episodes_summary[ep_id]
            manifest["partitions"][p_key]["episodes"].append({
                "source": "m5_augmented_collision",
                "episode_id": ep_id,
                "filepath": summary["filepath"],
                "category": summary["category"],
                "augmented_collision": True,
                "collision_occurred": summary["collision_occurred"],
                "first_contact_step": summary["first_contact_step"],
                "min_clearance": summary["min_clearance"],
            })

    add_augmented_to_partition("train", aug_train_ids)
    add_augmented_to_partition("calibration", aug_cal_ids)
    add_augmented_to_partition("test", aug_test_ids)

    # Write manifest
    os.makedirs(os.path.dirname(args.manifest_path), exist_ok=True)
    with open(args.manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[SUCCESS] Stratified dataset manifest written to {args.manifest_path}")
    print(f"  Train:       {len(manifest['partitions']['train']['episodes'])} episodes (Target: 106)")
    print(f"  Calibration: {len(manifest['partitions']['calibration']['episodes'])} episodes (Target: 17)")
    print(f"  Test:        {len(manifest['partitions']['test']['episodes'])} episodes (Target: 17)")


if __name__ == "__main__":
    main()
