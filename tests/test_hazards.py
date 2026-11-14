"""
SAWM - Milestone 2 Automated Test Suite: Hazards & Safety Scenarios.

Verifies:
1. Scenario 1 (Normal Operation): SAFE throughout, 0 collisions, 0 near-misses
2. Scenario 2 (Obstacle Near Path): SAFE -> WARNING -> SAFE recovery, 0 collisions
3. Scenario 3 (Restricted Zone): Cartesian world-space ZONE_BREACH -> DANGER, collision=False
4. Scenario 4 (Moving Kinematic Proxy): Proximity DANGER (d < 0.08m), near-miss increment, recovery
5. Scenario 5 (Physical Collision Event): Contact detected, DANGER latched until reset
6. 11-Body Tracking & Geometric vs Link-Center distance queries
"""

import os
import sys
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.environment import SawmEnvironment
from simulation.scenarios import ScenarioManager


def test_11_body_tracking_and_distance_queries():
    """Verify that all 11 robot bodies are tracked and geom vs center distances are distinguished."""
    print("\n--- Test 6: 11-Body Tracking & Distance Engine Verification ---")
    env = SawmEnvironment()
    obs_mgr = env.observation_manager

    # 1. Verify exact 11 bodies
    expected_bodies = [f"link{i}" for i in range(8)] + ["hand", "left_finger", "right_finger"]
    assert len(obs_mgr.robot_body_names) == 11, f"Expected 11 robot bodies, got {len(obs_mgr.robot_body_names)}"
    for b in expected_bodies:
        assert b in obs_mgr.robot_body_names, f"Missing body: {b}"
        assert obs_mgr.robot_body_ids[b] != -1, f"Invalid body ID for: {b}"

    print(f"  Tracked bodies ({len(expected_bodies)}): {', '.join(expected_bodies)}")
    print(f"  Active robot collision geoms: {len(obs_mgr.robot_collision_geoms)}")

    # 2. Place an obstacle and check geom distance vs link center distance
    env.obstacle_manager.set_static_obstacle_position(np.array([0.50, 0.0, 0.55]), active=True)
    dists = obs_mgr.compute_distances(env.obstacle_manager)

    print(f"  Geometric min distance:   {dists.geometric_min_distance:.4f} m (between {dists.nearest_robot_geom} and {dists.nearest_hazard_geom})")
    print(f"  Link-center min distance: {dists.link_center_min_distance:.4f} m (between {dists.nearest_robot_body} and {dists.nearest_hazard_body})")

    assert dists.geometric_min_distance < 900.0, "Geometric distance should be finite"
    assert dists.link_center_min_distance < 900.0, "Link-center distance should be finite"
    assert dists.nearest_points is not None, "Nearest points fromto should be populated"

    print("  [PASS] 11-body tracking and distance engine verified.")
    env.close()


def test_scenario1_normal():
    """Scenario 1: Normal operation with no hazards. Expected: SAFE throughout."""
    print("\n--- Test 1: Scenario 1 - Normal Operation ---")
    env = SawmEnvironment()
    mgr = ScenarioManager(env)
    res = mgr.run_scenario(1)

    print(f"  States seen: {res['states_seen']}")
    print(f"  Min geometric distance: {res['min_geometric_distance']:.3f} m")
    print(f"  Collision count: {res['collision_count']}, Near-miss count: {res['near_miss_count']}")

    assert res["states_seen"] == ["SAFE"], f"Expected only SAFE, got {res['states_seen']}"
    assert res["final_state"] == "SAFE", f"Expected final SAFE, got {res['final_state']}"
    assert res["collision_count"] == 0, "No collisions should occur in normal operation"
    assert res["min_geometric_distance"] > 0.25, f"Min distance {res['min_geometric_distance']} should be > 0.25m"

    print("  [PASS] Scenario 1 (Normal Operation) passed.")
    env.close()


def test_scenario2_obstacle_near_path():
    """Scenario 2: Obstacle near motion path. Expected: SAFE -> WARNING -> SAFE recovery."""
    print("\n--- Test 2: Scenario 2 - Obstacle Near Path ---")
    env = SawmEnvironment()
    mgr = ScenarioManager(env)
    res = mgr.run_scenario(2)

    print(f"  States seen: {res['states_seen']}")
    print(f"  Min geometric distance: {res['min_geometric_distance']:.3f} m")
    print(f"  Collision count: {res['collision_count']}")

    assert "WARNING" in res["states_seen"], "Expected WARNING state to be triggered"
    assert res["final_state"] == "SAFE", f"Expected recovery to SAFE, got {res['final_state']}"
    assert res["collision_count"] == 0, "No collision should occur in proximity warning scenario"
    assert 0.08 <= res["min_geometric_distance"] < 0.20, f"Min distance {res['min_geometric_distance']} should be in warning band [0.08, 0.20)"

    print("  [PASS] Scenario 2 (Proximity Warning & Recovery) passed.")
    env.close()


def test_scenario3_restricted_zone():
    """Scenario 3: Restricted zone breach. Expected: ZONE_BREACH -> DANGER, collision=False."""
    print("\n--- Test 3: Scenario 3 - Restricted Zone Intrusion ---")
    env = SawmEnvironment()
    mgr = ScenarioManager(env)
    res = mgr.run_scenario(3)

    print(f"  States seen: {res['states_seen']}")
    print(f"  Max contextual harm: {res['max_contextual_harm']:.2f}")
    print(f"  Collision count: {res['collision_count']}")

    assert "DANGER" in res["states_seen"], "Expected DANGER state due to zone breach"
    assert res["max_contextual_harm"] == 1.0, f"Expected contextual harm 1.0, got {res['max_contextual_harm']}"
    assert res["collision_count"] == 0, "Physical collision must be False for pure contextual hazard"

    print("  [PASS] Scenario 3 (Contextual Restricted Zone Breach) passed.")
    env.close()


def test_scenario4_danger_proximity():
    """Scenario 4: Moving proxy encroachment. Expected: DANGER (d < 0.08m), near-miss count increment, recovery."""
    print("\n--- Test 4: Scenario 4 - Moving Proxy Encroachment ---")
    env = SawmEnvironment()
    mgr = ScenarioManager(env)
    res = mgr.run_scenario(4)

    print(f"  States seen: {res['states_seen']}")
    print(f"  Min geometric distance: {res['min_geometric_distance']:.3f} m")
    print(f"  Near-miss count: {res['near_miss_count']}, Collision count: {res['collision_count']}")

    assert "DANGER" in res["states_seen"], "Expected DANGER state from critical proximity"
    assert res["min_geometric_distance"] < 0.08, f"Expected d < 0.08m, got {res['min_geometric_distance']:.3f}m"
    assert res["near_miss_count"] >= 1, f"Expected at least 1 near-miss event, got {res['near_miss_count']}"
    assert res["collision_count"] == 0, "Collision should be 0 (near-miss without contact)"
    # Non-collision danger recovers when proxy retreats
    assert res["final_state"] in ["SAFE", "WARNING"], f"Expected recovery from proximity danger, got {res['final_state']}"

    print("  [PASS] Scenario 4 (Danger Proximity & Near-Miss Recovery) passed.")
    env.close()


def test_scenario5_collision_event_and_latching():
    """Scenario 5: Physical collision event. Expected: Contact detected, DANGER latched until reset."""
    print("\n--- Test 5: Scenario 5 - Physical Collision & Latched DANGER ---")
    env = SawmEnvironment()
    mgr = ScenarioManager(env)
    res = mgr.run_scenario(5)

    print(f"  States seen: {res['states_seen']}")
    print(f"  Collision count: {res['collision_count']}")
    print(f"  Final state: {res['final_state']}, Latched: {res['final_is_latched']}")

    assert res["collision_count"] >= 1, "Expected physical collision to be detected"
    assert res["final_state"] == "DANGER", "Collision state must remain DANGER"
    assert res["final_is_latched"] is True, "Collision DANGER must be latched"

    # Verify that unlatching requires explicit reset
    env.safety_monitor.reset()
    assert env.safety_monitor.is_latched is False, "Explicit reset must clear latched collision"

    print("  [PASS] Scenario 5 (Physical Collision & Latched DANGER) passed.")
    env.close()


if __name__ == "__main__":
    print("=" * 70)
    print("SAWM - Milestone 2 Automated Test Suite (Hazards & Safety Scenarios)")
    print("=" * 70)
    try:
        test_11_body_tracking_and_distance_queries()
        test_scenario1_normal()
        test_scenario2_obstacle_near_path()
        test_scenario3_restricted_zone()
        test_scenario4_danger_proximity()
        test_scenario5_collision_event_and_latching()
        print("\n" + "=" * 70)
        print("ALL MILESTONE 2 TESTS PASSED SUCCESSFULLY!")
        print("=" * 70)
    except AssertionError as e:
        print(f"\n[FAIL] Test assertion error: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n[ERROR] Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
