"""
test_foundation.py
==================
Integration smoke test for the 4 foundational modules.
Run from project root: python test_foundation.py
"""
import logging
import os
import sys
import numpy as np

# Ensure imports work when running this script from the project root.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    stream=sys.stdout,
)

# ── Imports ─────────────────────────────────────────────────────────────────
from core_engine.session_box  import SessionBox, DataPriority
from core_engine.dynamic_ecs  import build_default_ecs
from simulation.matrix_solver  import MatrixSolver, ConstraintConfig
from ai_core.adversarial_interrogator import AdversarialInterrogator


def test_session_box():
    print("\n" + "=" * 60)
    print("TEST: SessionBox")
    box = SessionBox.get_instance()
    box.register_channel("test_ndarray", owner="test", dtype="ndarray",
                         shape=(4,), priority=DataPriority.NORMAL)
    box.register_channel("test_dict",    owner="test", dtype="dict")

    arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    assert box.write("test_ndarray", arr)
    read_back = box.read("test_ndarray")
    assert read_back is not None
    assert np.allclose(arr, read_back), f"Read mismatch: {read_back}"

    box.write("test_dict", {"key": "hello"})
    assert box.read("test_dict") == {"key": "hello"}

    box.commit_render_snapshot()
    snap, ver = box.read_render_snapshot()
    assert ver >= 1
    assert "test_ndarray" in snap
    print("SessionBox OK  (channels, read/write, render snapshot)")


def test_ecs():
    print("\n" + "=" * 60)
    print("TEST: ECSRegistry")
    ecs = build_default_ecs()

    # Dynamic NLP-injected component
    schema_dict = {
        "component_name": "Mana",
        "data_type":      "float32",
        "mutability":     "dynamic",
        "scope_binding":  "entity",
        "attributes":     {"current_mana": 0.0, "max_mana": 100.0, "regen_rate": 1.5},
    }
    assert ecs.inject_dynamic_component_from_schema_dict(schema_dict)

    e1 = ecs.create_entity()
    ecs.add_component(e1, "Transform", {"x": 1.0, "y": 5.0, "z": 0.0})
    ecs.add_component(e1, "Velocity",  {"vx": 2.0, "vy": 0.0, "vz": 0.0})
    ecs.add_component(e1, "Mass",      {"mass_kg": 75.0})
    ecs.add_component(e1, "Health")
    ecs.add_component(e1, "Energy")
    ecs.add_component(e1, "Mana",      {"current_mana": 50.0, "max_mana": 100.0})

    assert ecs.has_component(e1, "Mana")

    # Run 3 ticks
    for _ in range(3):
        ecs.tick(dt=0.016)

    snap = ecs.get_component_snapshot(e1, "Transform")
    assert snap is not None
    assert snap["x"] > 1.0, f"Physics integration failed: x={snap['x']}"

    q = ecs.query("Transform", "Velocity", "Mass")
    assert e1 in q

    print(f"ECS OK  (entity={e1}, schemas={len(ecs.list_schemas())}, systems={len(ecs._systems)})")
    print(f"  Transform after 3 ticks: x={snap['x']:.4f}")


def test_matrix_solver():
    print("\n" + "=" * 60)
    print("TEST: MatrixSolver")
    cfg    = ConstraintConfig(E_max=5000.0, M_max=1000.0)
    solver = MatrixSolver(config=cfg)

    N = 10
    masses     = np.random.uniform(50.0, 100.0, N).astype(np.float32)
    velocities = np.random.uniform(-5.0, 5.0, (N, 3)).astype(np.float32)
    heights    = np.random.uniform(0.0, 10.0, N).astype(np.float32)
    bio_energy = np.random.uniform(0.0, 50.0, N).astype(np.float32)

    for tick in range(5):
        result = solver.tick(
            masses, velocities, heights, bio_energy,
            ltspice_bandwidth=2000.0,
            ltspice_snr=35.0,
            dt=0.016
        )
        assert result, "Solver returned empty result"

    state = solver.get_state()
    assert not np.isnan(state.E_total), "NaN in E_total"
    assert not np.isinf(state.E_total), "Inf in E_total"
    assert state.prey    > 0.0, "Prey went negative"
    assert state.predator > 0.0, "Predator went negative"

    print(f"MatrixSolver OK  (5 ticks | E={state.E_total:.2f} | "
          f"prey={state.prey:.1f} pred={state.predator:.1f} | "
          f"v_max={state.v_max_entity:.2f} m/s)")


def test_adversarial_interrogator():
    print("\n" + "=" * 60)
    print("TEST: AdversarialInterrogator")

    prompts_received = []

    def on_prompt(p):
        prompts_received.append(p)
        print(f"  [Socratic] {p.prompt_text[:80]}")

    ai = AdversarialInterrogator(use_torch=False, on_prompt=on_prompt)

    # Normal state — should not violate
    normal_state = {
        "tick": 1, "E_total": 1000.0, "E_max": 5000.0,
        "M_total": 200.0, "M_max": 1000.0,
        "prey": 100.0, "predator": 20.0,
        "v_max": 2.0, "e_burn_rate": 0.1,
        "lorenz": (0.5, 1.2, 0.8),
        "decay_trigger": 0,
        "violations": {k: False for k in
                       ["S1_energy_overflow", "S2_mass_overflow",
                        "S3_prey_extinction", "S3_predator_collapse",
                        "S4_speed_violation", "S5_lorenz_divergence"]},
    }
    result_normal = ai.interrogate(normal_state, ltspice_snr=35.0)
    print(f"  Normal: P_viol={result_normal['P_violation']:.4f} "
          f"violation={result_normal['is_violation']}")

    # Violation state — energy overflow
    viol_state = dict(normal_state)
    viol_state["E_total"]       = 9_000.0
    viol_state["decay_trigger"] = 1
    viol_state["violations"]    = dict(normal_state["violations"])
    viol_state["violations"]["S1_energy_overflow"] = True

    result_viol = ai.interrogate(viol_state, ltspice_snr=35.0)
    print(f"  Violation: P_viol={result_viol['P_violation']:.4f} "
          f"violation={result_viol['is_violation']} "
          f"rca={result_viol['rca']}")

    stats = ai.get_stats()
    print(f"  Stats: queries={stats['total_queries']} "
          f"threshold={stats['current_threshold']:.3f} "
          f"fp_rate={stats['fp_rate_estimate']:.2f}")

    ai.shutdown()
    print("AdversarialInterrogator OK")


if __name__ == "__main__":
    test_session_box()
    test_ecs()
    test_matrix_solver()
    test_adversarial_interrogator()
    print("\n" + "=" * 60)
    print("ALL FOUNDATION TESTS PASSED")
