# PROJECT API REFERENCE

This document summarizes the public API surfaces for the `MA_NGUON_DU_AN_QUOC_GIA` project as observed in the current codebase.

## Table of Contents

- [Project Structure](#project-structure)
- [Core Engine](#core-engine)
  - [SessionBox](#sessionbox)
  - [ECSRegistry](#ecsregistry)
- [Simulation](#simulation)
  - [MatrixSolver](#matrixsolver)
- [AI Core](#ai-core)
  - [AdversarialInterrogator](#adversarialinterrogator)
  - [MLPerceptionEngine](#mlperceptionengine)
  - [FormalVerifier](#formalverifier)
  - [SemanticToMathCompiler](#semantictomathcompiler)
- [Graphics](#graphics)
  - [WorldView](#worldview)
  - [ProceduralEcosystem](#proceduralecosystem)
- [Hardware Bridge](#hardware-bridge)
  - [LTSpiceParser](#ltspiceparser)
  - [LTSpiceDNAMapper](#ltspicednamapper)
- [Evolution](#evolution)
  - [GeneticEvolutionEngine](#geneticevolutionengine)
- [Monitoring](#monitoring)
  - [TelemetryCore](#telemetrycore)
- [Orchestration & Launchers](#orchestration--launchers)
- [Tests](#tests)

---

## Project Structure

The current codebase includes these main packages and modules:

- `main.py`
- `orchestrator.py`
- `runtime_orchestrator.py`
- `core_engine/`
  - `dynamic_ecs.py`
  - `session_box.py`
- `simulation/`
  - `matrix_solver.py`
  - `experiment_manager.py`
- `ai_core/`
  - `adversarial_interrogator.py`
  - `ml_perception_engine.py`
  - `formal_verifier.py`
  - `nlp_idea_parser.py`
- `graphics/`
  - `world_view.py`
  - `procedural_eco.py`
- `hardware_bridge/`
  - `ltspice_parser.py`
  - `ltspice_dna_mapper.py`
- `evolution/`
  - `genetic_evolution.py`
- `monitoring/`
  - `telemetry_core.py`
- `tests/`
  - `test_foundation.py`

---

## Core Engine

### SessionBox

File: `core_engine/session_box.py`

#### Types

- `ChannelState` enum: `ACTIVE`, `PAUSED`, `FLUSHED`, `CLOSED`
- `DataPriority` enum: `CRITICAL`, `HIGH`, `NORMAL`, `LOW`
- `ChannelMetadata`
- `SharedMemoryCell`

#### Class: `SessionBox`

Singleton central shared memory coordinator.

Methods:

- `SessionBox.get_instance() -> SessionBox`
  - Thread-safe singleton accessor.

- `register_channel(name: str, owner: str, dtype: str = "dict", shape: Optional[Tuple[int, ...]] = None, priority: DataPriority = DataPriority.NORMAL) -> None`
  - Register a named channel.
  - Supported dtypes: `ndarray`, `dict`, `scalar`.
  - `shape` required when `dtype == "ndarray"`.

- `deregister_channel(name: str) -> None`
  - Remove a channel safely.

- `list_channels() -> Dict[str, str]`
  - Return `{channel_name: owner}` for active channels.

- `write(channel: str, payload: Any, source: str = "unknown") -> bool`
  - Atomically write payload to a channel.
  - Returns `False` if channel missing or closed.
  - `ndarray` payloads must match registered shape.

- `read(channel: str, source: str = "unknown") -> Optional[Any]`
  - Atomically read a deep copy of channel data.
  - Returns `None` for missing or closed channels.

- `read_version(channel: str) -> int`
  - Return channel version counter.

- `async_write(channel: str, payload: Any, source: str = "unknown") -> bool`
  - Async wrapper around `write()`.

- `async_read(channel: str, source: str = "unknown") -> Optional[Any]`
  - Async wrapper around `read()`.

- `commit_render_snapshot() -> None`
  - Snapshot all active channels into the render buffer.
  - Used by renderer to avoid blocking live data.

- `read_render_snapshot() -> Tuple[Dict[str, Any], int]`
  - Read latest render snapshot and version.

- `subscribe(channel: str, callback: Callable[[str, Any], None]) -> None`
  - Register an observer callback fired on every write.

- `signal_shutdown() -> None`
  - Set global shutdown event.

- `is_shutdown_requested() -> bool`
  - Query shutdown flag.

- `wait_for_shutdown(timeout: Optional[float] = None) -> bool`
  - Block until shutdown signalled.

- `get_channel_stats(channel: str) -> Optional[Dict[str, Any]]`
  - Return metadata for a channel.

- `get_all_stats() -> List[Dict[str, Any]]`
  - Return metadata snapshots for all channels.

- `__repr__()`
  - Provides debug string summarizing channel count and shutdown state.

---

### ECSRegistry

File: `core_engine/dynamic_ecs.py`

#### Types

- `EntityID = int`
- `ComponentType = str`
- `ComponentData = Dict[str, np.float32]`
- `ComponentSchema`
- `SystemDescriptor`

#### Class: `ComponentSchema`

Defines component schema structure.

Fields:

- `type_name: str`
- `attributes: Dict[str, float]`
- `mutability: str = "dynamic"`
- `scope: str = "entity"`
- `is_builtin: bool = True`

#### Class: `ECSRegistry`

Thread-safe ECS registry.

Methods:

- `register_component_schema(schema: ComponentSchema) -> None`
  - Register built-in or dynamic schema.

- `get_schema(comp_type: ComponentType) -> Optional[ComponentSchema]`

- `list_schemas() -> List[str]`

- `create_entity() -> EntityID`
  - Create entity with no components.

- `destroy_entity(entity_id: EntityID) -> bool`

- `entity_exists(entity_id: EntityID) -> bool`

- `get_all_entities() -> List[EntityID]`

- `add_component(entity_id: EntityID, comp_type: ComponentType, overrides: Optional[Dict[str, float]] = None) -> bool`
  - Add component from schema defaults plus overrides.
  - Returns `False` if unknown schema or missing entity.

- `remove_component(entity_id: EntityID, comp_type: ComponentType) -> bool`

- `has_component(entity_id: EntityID, comp_type: ComponentType) -> bool`

- `get_component(entity_id: EntityID, comp_type: ComponentType) -> Optional[ComponentData]`
  - Returns a live reference.

- `get_component_snapshot(entity_id: EntityID, comp_type: ComponentType) -> Optional[ComponentData]`
  - Returns a deep copy safe for read-only access.

- `set_component_attr(entity_id: EntityID, comp_type: ComponentType, attr: str, value: float) -> bool`
  - Atomically set one component attribute.
  - Maintains derived `Mass.inv_mass` when `mass_kg` changes.

- `query(*comp_types: ComponentType) -> Set[EntityID]`
  - Return entities that contain all requested components.

- `query_with_data(*comp_types: ComponentType) -> List[Dict[str, Any]]`
  - Return entity snapshots for requested components.

- `register_system(name: str, callable_fn: Callable[["ECSRegistry", Set[EntityID]], None], required_components: Iterable[ComponentType], priority: int = 100) -> None`
  - Register a system with execution priority.

- `tick(dt: float = 0.016) -> None`
  - Execute all enabled systems in priority order.
  - Each system receives `(registry, entity_set, dt)`.

- `get_system_stats() -> List[Dict[str, Any]]`

- `inject_dynamic_component_from_schema_dict(schema_dict: Dict[str, Any]) -> bool`
  - Inject a component schema at runtime from an NLP-produced JSON.

- `get_entity_profile(entity_id: EntityID) -> Optional[Dict[str, Any]]`

- `entity_count() -> int`

- `component_count(comp_type: ComponentType) -> int`

- `__repr__()`

#### Built-in ECS Systems

- `physics_integration_system(registry, entities, dt)`
  - Requires: `Transform`, `Velocity`, `Mass`
  - Updates position using semi-implicit Euler.

- `energy_consumption_system(registry, entities, dt)`
  - Requires: `Energy`
  - Drains energy, damages `Health` when energy is exhausted.

- `health_regen_system(registry, entities, dt)`
  - Requires: `Health`, `Energy`
  - Regenerates health when entity has energy.

- `build_default_ecs() -> ECSRegistry`
  - Creates `ECSRegistry` and registers the three built-in systems.

---

## Simulation

### MatrixSolver

File: `simulation/matrix_solver.py`

#### Data Classes

- `ConstraintConfig`
  - `E_max`, `M_max`, `alpha`, `beta`, `delta`, `gamma`, `prey_init`, `predator_init`, `k_bw`, `k_snr`, `lorenz_sigma`, `lorenz_rho`, `lorenz_beta`, `lorenz_init`, `max_delta_ratio`, `nan_replacement`

- `ConstraintState`
  - `E_kinetic`, `E_potential`, `E_biological`, `E_total`, `decay_trigger`, `M_total`, `prey`, `predator`, `v_max_entity`, `e_burn_rate`, `lorenz_x`, `lorenz_y`, `lorenz_z`, `violations`, `tick`

#### Class: `MatrixSolver`

Methods:

- `__init__(config: Optional[ConstraintConfig] = None) -> None`

- `tick(entity_masses: np.ndarray, entity_velocities: np.ndarray, entity_heights: np.ndarray, entity_bio_energy: np.ndarray, ltspice_bandwidth: float = 1000.0, ltspice_snr: float = 30.0, dt: float = 0.016) -> Dict[str, Any]`
  - Primary math core tick.
  - Inputs:
    - `entity_masses`: shape `(N,)`
    - `entity_velocities`: shape `(N, 3)`
    - `entity_heights`: shape `(N,)`
    - `entity_bio_energy`: shape `(N,)`
    - `ltspice_bandwidth`, `ltspice_snr`, `dt`
  - Outputs dict includes:
    - `tick`, `E_total`, `E_max`, `M_total`, `M_max`, `decay_trigger`, `prey`, `predator`, `v_max`, `e_burn_rate`, `lorenz`, `violations`

- `get_state() -> ConstraintState`
  - Thread-safe deep copy of the current state.

- `get_v_max() -> float`

- `get_E_max() -> float`

- `check_mass_budget(proposed_mass: float) -> bool`
  - Compare current `M_total + proposed_mass` against `M_max`.

- `get_lorenz_seed() -> Tuple[float, float, float]`

- `get_violations() -> Dict[str, bool]`

- `recalibrate_E_max(new_E_max: float) -> None`

- `__repr__()`

#### Internal Helpers

- `_rk4_lotka_volterra(P, Q, cfg, dt) -> Tuple[float, float]`
- `_rk4_lorenz(x, y, z, cfg, dt) -> Tuple[float, float, float]`
- `_safety_clamp(arr, lo, hi) -> np.ndarray`
- `normalise_01(arr) -> np.ndarray`
- `normalise_sym(arr) -> np.ndarray`

---

## AI Core

### AdversarialInterrogator

File: `ai_core/adversarial_interrogator.py`

#### Supporting Classes

- `NumpyAdversarialNet`
  - `forward(x: np.ndarray) -> np.ndarray`

- `TorchAdversarialNet` (if PyTorch available)
  - `forward(x: torch.Tensor) -> torch.Tensor`

- `SocraticPrompt`
- `SocraticGuide`
  - `guide(violation_type: str, context: Dict[str, Any]) -> None` (async)
  - `reset_hint_level(violation_type: str) -> None`

- `CalibrationState`
  - `record_activation(was_rejected: bool) -> None`
  - `current_fp_rate() -> float`
  - `calibrate() -> str`

- `CausalGraphTracer`
  - `trace(math_state: Dict[str, Any], active_viols: List[str], rca: str) -> str`

#### Class: `AdversarialInterrogator`

Methods:

- `__init__(use_torch: bool = True, on_prompt: Optional[Callable] = None, calibration: Optional[CalibrationState] = None) -> None`
  - `use_torch` prefers PyTorch backend if available.
  - `on_prompt` callback receives `SocraticPrompt` objects.

- `interrogate(math_state: Dict[str, Any], ltspice_snr: Optional[float] = None) -> Dict[str, Any]`
  - Primary gatekeeper API.
  - Returns dict with keys including:
    - `P_valid`, `P_violation`, `is_violation`, `threshold`, `rca`, `active_viols`, `causal_chain`, `calibration_dir`, `tick`, `total_queries`, `total_violations`

- `set_threshold(value: float) -> None`
  - Manual override of calibration threshold.

- `get_stats() -> Dict[str, Any]`
  - Returns diagnostics such as `total_queries`, `total_violations`, `violation_rate`, `current_threshold`, `fp_rate_estimate`, `last_p_violation`, `last_rca`, `network_backend`.

- `shutdown() -> None`
  - Stop internal async event loop.

- `__repr__()`

> Note: The current implementation of `interrogate()` returns a dict, but some callers in the codebase expect a truthy/falsy valid flag.

---

### MLPerceptionEngine

File: `ai_core/ml_perception_engine.py`

Class: `MLPerceptionEngine`

Methods:

- `__init__() -> None`

- `extract_feature_vector(registry, math_state: dict, hardware_state: dict) -> np.ndarray`
  - Produces a 12-dim `float32` feature vector.
  - Computes micro ECS metrics and normalizes them with math/hardware state.

---

### FormalVerifier

File: `ai_core/formal_verifier.py`

Class: `FormalVerifier`

Methods:

- `__init__() -> None`

- `verify_initial_axioms(s1_config: Dict[str, float], s2_config: Dict[str, float], sn_custom_rules: List[Dict[str, Any]]) -> Tuple[bool, str]`
  - Validate input axioms for energy, mass, friction, and custom rules.
  - Returns `(is_valid, reason)`.

---

### SemanticToMathCompiler

File: `ai_core/nlp_idea_parser.py`

Class: `SemanticToMathCompiler`

Methods:

- `__init__(ecs_registry: Any, interrogator_gatekeeper: Callable[[Dict[str, Any]], bool], slm_api_url: str = ..., model_name: str = ..., timeout_seconds: float = 15.0) -> None`

- `_query_local_slm(user_prompt: str) -> Optional[str]` (async)
  - Send prompt to local LLM endpoint.

- `_extract_and_validate_json(raw_text: str) -> Optional[Dict[str, Any]]`
  - Extract JSON object from raw response and validate required keys.

- `compile_and_inject_idea(user_text: str) -> bool` (async)
  - Full pipeline:
    1. query LLM
    2. parse JSON
    3. validate via gatekeeper callback
    4. inject into ECS via `inject_dynamic_component_from_schema_dict`

---

## Graphics

### WorldView

File: `graphics/world_view.py`

Class: `WorldView`

Methods:

- `__init__(width: int = 1200, height: int = 900, grid_size: int = 20) -> None`

- `process_user_events() -> bool`
  - Polls Pygame events.
  - Returns `False` on quit.

- `render_frame(registry: Any, biome_map: np.ndarray, ltspice_state: Dict[str, Any], math_state: Dict[str, Any], gatekeeper_status: str, current_gen: int) -> None`
  - Renders background, agents, HUD.
  - Uses `registry.query(...)` and `registry.get_component_snapshot(...)`.

- `run(app: Any) -> None`
  - Main render loop at 60 FPS.
  - Reads `ltspice`, `math_state`, `adversarial_flags` from `SessionBox`.
  - Generates `biome_map` via random noise.

---

### ProceduralEcosystem

File: `graphics/procedural_eco.py`

Class: `ProceduralEcosystem`

Methods:

- `__init__(ecs_registry: Any, width: int = 1000, height: int = 1000, seed: int = 42) -> None`

- `_generate_pseudo_noise(x: float, y: float, scale: float = 0.1) -> float`

- `generate_biome_map() -> None`

- `spawn_ecosystem(matrix_solver_state: Dict[str, Any]) -> None`
  - Generate biome map and create flora entities constrained by `E_max`.

- `decay_system(dt: float) -> None`
  - Decay `Energy` on flora, destroy entities with zero energy.

---

## Hardware Bridge

### LTSpiceParser

File: `hardware_bridge/ltspice_parser.py`

Class: `LTSpiceParser`

Methods:

- `__init__(ltspice_executable_path: str, timeout_seconds: float = 10.0) -> None`

- `run_simulation_async(asc_file_path: str) -> Optional[str]`
  - Launch LTspice in batch mode asynchronously.
  - Returns `.log` path on success.

- `_read_file_with_fallback_encoding(file_path: str) -> str` (async)
  - Read LTspice log with UTF-16/UTF-8 fallback.

- `parse_hardware_dna(log_file_path: str) -> Dict[str, float]` (async)
  - Extract `bandwidth` and `snr` from log file.

- `Pipeline_Execute(asc_file_path: str) -> Dict[str, float]` (async)
  - Full end-to-end parse pipeline with fallback defaults.

---

### LTSpiceDNAMapper

File: `hardware_bridge/ltspice_dna_mapper.py`

Data class: `HardwareDNA`
- `raw_bandwidth_hz`, `raw_snr_db`, `v_max_theoretical`, `hardware_burn_penalty`, `ai_compute_budget`, `is_valid`

Class: `LTSpiceDNAMapper`

Methods:

- `__init__(ltspice_executable_path: str, timeout_seconds: float = 15.0) -> None`

- `_map_raw_to_dna(raw_metrics: Dict[str, float]) -> HardwareDNA`
  - Map hardware metrics into game parameters.

- `pipeline_execute(asc_file_path: str) -> HardwareDNA` (async)
  - Run LTSpice parser and map results to `HardwareDNA`.

---

## Evolution

### GeneticEvolutionEngine

File: `evolution/genetic_evolution.py`

Classes:

- `BrainTopology`
  - `clone() -> BrainTopology`

- `GeneticEvolutionEngine`

Methods:

- `__init__(mutation_rate: float = 0.05, mutation_strength: float = 0.2) -> None`

- `register_or_get_brain(entity_id: int) -> BrainTopology`

- `physics_reflex_system(registry: Any, entities: Set[int], dt: float) -> None`
  - Inferred required components: `Transform`, `Velocity`, `Energy`, `NeuralBrain`
  - Updates `Velocity` from a small neural network.

- `evaluate_and_evolve(registry: Any) -> None`
  - Selects elites, crossover/mutate brains, resets energy/health.

- `_crossover_and_mutate(parent_a: BrainTopology, parent_b: BrainTopology) -> BrainTopology`

---

## Monitoring

### TelemetryCore

File: `monitoring/telemetry_core.py`

Class: `TelemetryCore`

Methods:

- `__init__(history_size: int = 600, poll_rate_hz: float = 30.0) -> None`

- `register_global_metric(metric_name: str, fetch_callback: Callable[[], Any]) -> None`

- `bind_ecs_interface(ecs_query_func: Callable[[int], Dict[str, Any]]) -> None`

- `set_focus_entity(entity_id: int) -> None`

- `clear_focus() -> None`

- `start() -> None`

- `stop() -> None`

- `get_current_dashboard_data() -> Dict[str, Any]`
  - Returns latest global and focused entity data.

- `get_rca_backtrace(frames: int = 60) -> List[Dict[str, Any]]`

---

## Orchestration & Launchers

### HybridSimulationApplication

File: `orchestrator.py`

Class: `HybridSimulationApplication`

Fields used in code:

- `box: SessionBox`
- `registry: ECSRegistry`
- `solver: MatrixSolver`
- `interrogator: AdversarialInterrogator`
- `ui_prompt_queue`

Methods:

- `__init__() -> None`
  - Instantiates `SessionBox`, `ECSRegistry`, `MatrixSolver`, `AdversarialInterrogator`.
  - Registers channels: `ltspice`, `math_state`, `adversarial_flags`, `telemetry_metrics`.
  - Spawns 100 demo entities.

- `start() -> None`
  - Launches `SimulationThread`, `HardwareThread`, `TelemetryThread`.

- `_simulation_loop() -> None`
  - Calls `registry.tick(dt)`, builds solver inputs, runs `MatrixSolver.tick(...)`, writes `math_state`, calls `interrogator.interrogate(...)`, writes `adversarial_flags`, and `commit_render_snapshot()`.

- `_hardware_bridge_loop() -> None`
  - Async loop writing fake LTspice values every 4 seconds.

- `_telemetry_loop() -> None`
  - Reads `adversarial_flags`, writes `telemetry_metrics` every 1s.

- `stop() -> None`
  - Stops running loops.

### RuntimeOrchestrator

File: `runtime_orchestrator.py`

Class: `RuntimeOrchestrator`

Methods:

- `__init__(ltspice_executable_path: str, use_torch: bool = False) -> None`
  - Creates `LTSpiceParser`, `MatrixSolver`, `FormalVerifier`, `TelemetryCore`, and `HybridSimulationApplication`.

- `start() -> None`
  - Starts the contained application.

- `stop() -> None`

- `run_async() -> None` (async)
  - Start and keep running until stopped.

---

## Tests

### `tests/test_foundation.py`

Covers integration smoke tests for:

- `SessionBox`
- `ECSRegistry` and dynamic component injection
- `MatrixSolver`
- `AdversarialInterrogator`

Key patterns:

- `SessionBox.register_channel()` and `write()/read()` for dict/ndarray
- `build_default_ecs()` and `ecs.tick(dt)`
- `MatrixSolver.tick(...)` and state assertions
- `AdversarialInterrogator.interrogate(...)`, `get_stats()`, and `shutdown()`

---

## Notes & Observations

- `WorldView.render_frame()` uses `registry.query("NeuralBrain", "Transform", "Velocity")` and `registry.get_component_snapshot(...)`.
- `WorldView.run()` currently generates `biome_map` from `np.random.uniform(...)` rather than a dedicated procedural subsystem.
- `SessionBox.commit_render_snapshot()` is the intended render buffer synchronization point.
- `AdversarialInterrogator.interrogate()` currently returns a dict, while `orchestrator.py` treats it as a value used directly in `box.write("adversarial_flags", {"valid": gate_result, ...})`.
- `LTSpiceDNAMapper.pipeline_execute()` calls `LTSpiceParser.parse_hardware_metrics(...)` in comments, but the parser file actually exposes `parse_hardware_dna(...)`.

---

## Component & ECS Notes

### Built-in Component Schemas

Defined in `core_engine/dynamic_ecs.py`:

- `Transform`
- `Velocity`
- `Mass`
- `Health`
- `Energy`
- `NeuralBrain`

Each schema is a `ComponentSchema` with default float values.

### Dynamic Component Injection

- `ECSRegistry.inject_dynamic_component_from_schema_dict(schema_dict)` accepts the NLP-produced schema format:
  - `component_name`, `data_type`, `mutability`, `scope_binding`, `attributes`

This is the integration point for `SemanticToMathCompiler` to extend ECS at runtime.

This is new API
async def process_prompt(self, prompt: str):
