"""
core_engine/dynamic_ecs.py
==========================
Lightweight Entity Component System (ECS) for the Hybrid AI Simulation Platform.

Architecture
------------
    - ENTITIES are pure unsigned integers (entity IDs).  No class overhead.
    - COMPONENTS are raw Python dicts with typed float32 numpy scalar attributes.
      Schema is validated against a registered component definition.
    - SYSTEMS are callable objects (classes with __call__) that receive a filtered
      view of entities sharing a specific component set and mutate them in-place.
    - The ECS Registry maintains three thread-safe indexes for O(1) lookup:
        entity_index  : entity_id  -> {component_type -> component_dict}
        component_index: comp_type -> set of entity_ids having that component
        system_registry: ordered list of System objects with execution priority
    - NLP-injected components (from nlp_idea_parser.py) are dynamically registered
      via register_component_schema() at runtime without restarting the ECS.

Standard Built-in Component Schemas
-------------------------------------
    Transform  : position(x,y,z), rotation(rx,ry,rz), scale(sx,sy,sz)
    Velocity   : vx, vy, vz, angular_vx, angular_vy, angular_vz
    Mass       : mass_kg, inv_mass (computed), is_static(bool as 0/1)
    Health     : current_hp, max_hp, regen_rate, is_alive(0/1)
    Energy     : current_energy, max_energy, consumption_rate
    NeuralBrain: weight_hash(float32 checksum), is_possessed(0/1)

Dynamic Example (NLP-injected):
    Mana       : current_mana, max_mana, regen_rate
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
EntityID       = int
ComponentType  = str          # e.g. "Transform", "Velocity", "Mana"
ComponentData  = Dict[str, np.float32]


# ---------------------------------------------------------------------------
# Component Schema Definition
# ---------------------------------------------------------------------------

@dataclass
class ComponentSchema:
    """
    Defines the structure and defaults for a component type.
    All attribute values are stored as np.float32 scalars inside ComponentData dicts.

    Parameters
    ----------
    type_name   : Unique string identifier (e.g. "Mana").
    attributes  : Dict mapping attribute name -> default float32 value.
    mutability  : "dynamic" (value can change every tick) or "constant".
    scope       : "entity" (per-entity) or "global" (shared singleton).
    is_builtin  : True if registered at startup, False if injected at runtime via NLP.
    """
    type_name:  ComponentType
    attributes: Dict[str, float]           # name -> default value
    mutability: str  = "dynamic"           # "dynamic" | "constant"
    scope:      str  = "entity"            # "entity"  | "global"
    is_builtin: bool = True


# ---------------------------------------------------------------------------
# Built-in Schema Definitions
# ---------------------------------------------------------------------------

BUILTIN_SCHEMAS: List[ComponentSchema] = [
    ComponentSchema(
        type_name  = "Transform",
        attributes = {"x": 0.0, "y": 0.0, "z": 0.0,
                      "rx": 0.0, "ry": 0.0, "rz": 0.0,
                      "sx": 1.0, "sy": 1.0, "sz": 1.0},
    ),
    ComponentSchema(
        type_name  = "Velocity",
        attributes = {"vx": 0.0, "vy": 0.0, "vz": 0.0,
                      "angular_vx": 0.0, "angular_vy": 0.0, "angular_vz": 0.0},
    ),
    ComponentSchema(
        type_name  = "Mass",
        attributes = {"mass_kg": 1.0, "inv_mass": 1.0, "is_static": 0.0},
    ),
    ComponentSchema(
        type_name  = "Health",
        attributes = {"current_hp": 100.0, "max_hp": 100.0,
                      "regen_rate": 0.1,   "is_alive": 1.0},
    ),
    ComponentSchema(
        type_name  = "Energy",
        attributes = {"current_energy": 100.0, "max_energy": 100.0,
                      "consumption_rate": 0.5},
    ),
    ComponentSchema(
        type_name  = "NeuralBrain",
        attributes = {"weight_hash": 0.0, "is_possessed": 0.0,
                      "fitness_score": 0.0, "generation": 0.0},
        mutability = "dynamic",
    ),
]


# ---------------------------------------------------------------------------
# System Base Class
# ---------------------------------------------------------------------------

@dataclass(order=True)
class SystemDescriptor:
    """Metadata wrapper around a System callable, used for ordered execution."""
    priority:    int                          # Lower == executed first
    name:        str                          = field(compare=False)
    required_components: FrozenSet[str]       = field(compare=False, default_factory=frozenset)
    callable_fn: Callable[["ECSRegistry", Set[EntityID], float], None] = field(compare=False, default=None)  # type: ignore
    enabled:     bool                         = field(compare=False, default=True)
    tick_count:  int                          = field(compare=False, default=0)
    last_exec_ms: float                       = field(compare=False, default=0.0)


# ---------------------------------------------------------------------------
# ECS Registry
# ---------------------------------------------------------------------------

class ECSRegistry:
    """
    Thread-safe Entity Component System registry.

    Thread Safety
    -------------
    All structural mutations (create/destroy entity, add/remove component,
    register schema/system) acquire the global `_structural_lock`.
    Per-tick system execution acquires per-entity component locks
    for write operations to maximise parallelism while preventing torn reads.

    The coarse structural lock is separate from fine-grained component data locks
    so that the render thread can snapshot component data without blocking
    system execution on unrelated entities.
    """

    def __init__(self) -> None:
        # Schema registry: component_type -> ComponentSchema
        self._schemas: Dict[ComponentType, ComponentSchema] = {}

        # Primary data store: entity_id -> {comp_type -> component_data}
        self._entity_components: Dict[EntityID, Dict[ComponentType, ComponentData]] = {}

        # Inverted index: comp_type -> set of entity_ids
        self._component_index: Dict[ComponentType, Set[EntityID]] = {}

        # System registry (sorted by priority ascending)
        self._systems: List[SystemDescriptor] = []

        # Monotonic entity ID counter
        self._next_entity_id: int = 1

        # Threading primitives
        self._structural_lock: threading.Lock         = threading.Lock()
        # Per-entity component-level RW locks
        self._entity_locks: Dict[EntityID, threading.Lock] = {}

        # Register all built-in schemas at startup
        for schema in BUILTIN_SCHEMAS:
            self.register_component_schema(schema)

        logger.info("[ECSRegistry] Initialised with %d built-in component schemas.", len(BUILTIN_SCHEMAS))

    # ------------------------------------------------------------------
    # Schema Management
    # ------------------------------------------------------------------

    def register_component_schema(self, schema: ComponentSchema) -> None:
        """
        Register a new component schema.
        Safe to call at runtime for NLP-injected dynamic components.
        Idempotent: re-registering an existing schema logs a warning and returns.
        """
        with self._structural_lock:
            if schema.type_name in self._schemas:
                logger.warning("[ECSRegistry] Schema '%s' already registered. Skipping.", schema.type_name)
                return
            self._schemas[schema.type_name] = schema
            self._component_index[schema.type_name] = set()
            logger.debug("[ECSRegistry] Schema '%s' registered (mutability=%s, scope=%s, builtin=%s).",
                         schema.type_name, schema.mutability, schema.scope, schema.is_builtin)

    def get_schema(self, comp_type: ComponentType) -> Optional[ComponentSchema]:
        return self._schemas.get(comp_type)

    def list_schemas(self) -> List[str]:
        return list(self._schemas.keys())

    # ------------------------------------------------------------------
    # Entity Lifecycle
    # ------------------------------------------------------------------

    def create_entity(self) -> EntityID:
        """
        Allocate and return a new unique Entity ID.
        The entity starts with zero components; add components via add_component().
        """
        with self._structural_lock:
            eid = self._next_entity_id
            self._next_entity_id += 1
            self._entity_components[eid] = {}
            self._entity_locks[eid] = threading.Lock()
        logger.debug("[ECSRegistry] Entity %d created.", eid)
        return eid

    def destroy_entity(self, entity_id: EntityID) -> bool:
        """
        Remove an entity and all its components from all indexes.
        Returns True on success, False if entity does not exist.
        """
        with self._structural_lock:
            if entity_id not in self._entity_components:
                logger.warning("[ECSRegistry] Destroy called on non-existent entity %d.", entity_id)
                return False
            comp_types = list(self._entity_components[entity_id].keys())
            for ct in comp_types:
                self._component_index[ct].discard(entity_id)
            del self._entity_components[entity_id]
            del self._entity_locks[entity_id]
        logger.debug("[ECSRegistry] Entity %d destroyed.", entity_id)
        return True

    def entity_exists(self, entity_id: EntityID) -> bool:
        return entity_id in self._entity_components

    def get_all_entities(self) -> List[EntityID]:
        with self._structural_lock:
            return list(self._entity_components.keys())

    # ------------------------------------------------------------------
    # Component Management
    # ------------------------------------------------------------------

    def add_component(
        self,
        entity_id:  EntityID,
        comp_type:  ComponentType,
        overrides:  Optional[Dict[str, float]] = None,
    ) -> bool:
        """
        Attach a component of `comp_type` to entity `entity_id`.

        All attribute values are initialised from the schema defaults,
        then overridden by the `overrides` dict if provided.
        Values are cast to np.float32 for memory efficiency.

        Returns True on success, False if entity or schema is not found.
        """
        schema = self._schemas.get(comp_type)
        if schema is None:
            logger.error("[ECSRegistry] add_component: Unknown schema '%s'. Register it first.", comp_type)
            return False

        with self._structural_lock:
            if entity_id not in self._entity_components:
                logger.error("[ECSRegistry] add_component: Entity %d does not exist.", entity_id)
                return False
            if comp_type in self._entity_components[entity_id]:
                logger.debug("[ECSRegistry] Entity %d already has component '%s'. Skipping.", entity_id, comp_type)
                return True

        # Build component data from schema defaults + overrides
        comp_data: ComponentData = {
            attr: np.float32(overrides.get(attr, default) if overrides else default)
            for attr, default in schema.attributes.items()
        }

        # Special computation: inv_mass for Mass component (avoid division by zero)
        if comp_type == "Mass" and "mass_kg" in comp_data:
            mass = float(comp_data["mass_kg"])
            comp_data["inv_mass"] = np.float32(1.0 / mass if mass != 0.0 else 0.0)

        with self._structural_lock:
            self._entity_components[entity_id][comp_type] = comp_data
            self._component_index[comp_type].add(entity_id)

        logger.debug("[ECSRegistry] Component '%s' added to entity %d.", comp_type, entity_id)
        return True

    def remove_component(self, entity_id: EntityID, comp_type: ComponentType) -> bool:
        """Detach a component from an entity. Returns True on success."""
        with self._structural_lock:
            if entity_id not in self._entity_components:
                return False
            if comp_type not in self._entity_components[entity_id]:
                return False
            del self._entity_components[entity_id][comp_type]
            self._component_index[comp_type].discard(entity_id)
        logger.debug("[ECSRegistry] Component '%s' removed from entity %d.", comp_type, entity_id)
        return True

    def has_component(self, entity_id: EntityID, comp_type: ComponentType) -> bool:
        try:
            return comp_type in self._entity_components[entity_id]
        except KeyError:
            return False

    def get_component(self, entity_id: EntityID, comp_type: ComponentType) -> Optional[ComponentData]:
        """
        Return a REFERENCE to the component data dict.
        Callers must hold the entity lock when mutating attributes during system execution.
        For read-only access outside system ticks, use get_component_snapshot().
        """
        try:
            return self._entity_components[entity_id].get(comp_type)
        except KeyError:
            return None

    def get_component_snapshot(self, entity_id: EntityID, comp_type: ComponentType) -> Optional[ComponentData]:
        """Return a deep-copied snapshot of a component (safe for reads outside system ticks)."""
        lock = self._entity_locks.get(entity_id)
        if lock is None:
            return None
        with lock:
            comp = self._entity_components.get(entity_id, {}).get(comp_type)
            if comp is None:
                return None
            return {k: np.float32(v) for k, v in comp.items()}

    def set_component_attr(
        self,
        entity_id: EntityID,
        comp_type: ComponentType,
        attr:      str,
        value:     float,
    ) -> bool:
        """
        Atomically set a single attribute of a component under the entity lock.
        Validates that the attribute exists in the component schema before writing.
        """
        schema = self._schemas.get(comp_type)
        if schema is None or attr not in schema.attributes:
            logger.error("[ECSRegistry] set_component_attr: Unknown attr '%s' on '%s'.", attr, comp_type)
            return False

        lock = self._entity_locks.get(entity_id)
        if lock is None:
            return False

        with lock:
            comp = self._entity_components.get(entity_id, {}).get(comp_type)
            if comp is None:
                return False
            comp[attr] = np.float32(value)

            # Maintain derived fields
            if comp_type == "Mass" and attr == "mass_kg":
                comp["inv_mass"] = np.float32(1.0 / value if value != 0.0 else 0.0)

        return True

    # ------------------------------------------------------------------
    # Entity Queries
    # ------------------------------------------------------------------

    def query(self, *comp_types: ComponentType) -> Set[EntityID]:
        """
        Return the set of entity IDs that possess ALL of the specified component types.
        Uses set intersection starting from the smallest set for efficiency.
        """
        if not comp_types:
            return set()
        sets = [self._component_index.get(ct, set()) for ct in comp_types]
        sets.sort(key=len)   # Start intersection from smallest set
        result = sets[0].copy()
        for s in sets[1:]:
            result &= s
        return result

    def query_with_data(
        self, *comp_types: ComponentType
    ) -> List[Dict[str, Any]]:
        """
        Query entities and return their component data in a structured list.
        Each entry: {"entity_id": int, comp_type: component_data_dict, ...}
        Returns deep-copied snapshots — safe for consumption outside system tick.
        """
        results = []
        entity_ids = self.query(*comp_types)
        for eid in entity_ids:
            lock = self._entity_locks.get(eid)
            if lock is None:
                continue
            with lock:
                entry: Dict[str, Any] = {"entity_id": eid}
                for ct in comp_types:
                    comp = self._entity_components.get(eid, {}).get(ct)
                    if comp is not None:
                        entry[ct] = {k: np.float32(v) for k, v in comp.items()}
                results.append(entry)
        return results

    # ------------------------------------------------------------------
    # System Registration & Execution
    # ------------------------------------------------------------------

    def register_system(
        self,
        name:                str,
        callable_fn:         Callable[["ECSRegistry", Set[EntityID], float], None],
        required_components: Iterable[ComponentType],
        priority:            int = 100,
    ) -> None:
        """
        Register a System.

        Parameters
        ----------
        name                : Human-readable identifier.
        callable_fn         : fn(registry, entity_set, dt) -> None — mutates components.
        required_components : Entity must possess ALL of these to be included.
        priority            : Execution order; lower values execute first.
        """
        desc = SystemDescriptor(
            priority            = priority,
            name                = name,
            required_components = frozenset(required_components),
            callable_fn         = callable_fn,
        )
        with self._structural_lock:
            self._systems.append(desc)
            self._systems.sort()   # Sort by priority ascending
        logger.info("[ECSRegistry] System '%s' registered (priority=%d, requires=%s).",
                    name, priority, desc.required_components)

    def tick(self, dt: float = 0.016) -> None:
        """
        Execute one ECS tick: iterate all enabled systems in priority order.

        Parameters
        ----------
        dt : Delta time in seconds since last tick. Injected into entity component data
             as a temporary '__dt' key before each system call, then removed.
        """
        # Snapshot system list to avoid mutation during iteration
        with self._structural_lock:
            systems_snapshot = list(self._systems)

        for desc in systems_snapshot:
            if not desc.enabled:
                continue

            # Determine the entity set satisfying the system's component requirements
            if desc.required_components:
                entity_set = self.query(*desc.required_components)
            else:
                entity_set = set(self._entity_components.keys())

            if not entity_set:
                continue

            t0 = time.perf_counter()
            try:
                desc.callable_fn(self, entity_set, dt)
            except Exception as exc:
                logger.exception("[ECSRegistry] System '%s' raised an exception: %s", desc.name, exc)
            finally:
                desc.tick_count  += 1
                desc.last_exec_ms = (time.perf_counter() - t0) * 1000.0

    def get_system_stats(self) -> List[Dict[str, Any]]:
        """Return execution statistics for all registered systems."""
        return [
            {
                "name":         s.name,
                "priority":     s.priority,
                "enabled":      s.enabled,
                "tick_count":   s.tick_count,
                "last_exec_ms": s.last_exec_ms,
                "requires":     list(s.required_components),
            }
            for s in self._systems
        ]

    # ------------------------------------------------------------------
    # Dynamic Component Injection (NLP Pipeline Entry Point)
    # ------------------------------------------------------------------

    def inject_dynamic_component_from_schema_dict(
        self,
        schema_dict: Dict[str, Any],
    ) -> bool:
        """
        Entry point called by nlp_idea_parser.py to inject a runtime-defined component.

        Expected schema_dict format (produced by the NLP parser):
        {
            "component_name": "Mana",
            "data_type":      "float32",
            "mutability":     "dynamic",
            "scope_binding":  "entity",
            "attributes": {
                "current_mana": 0.0,
                "max_mana":     100.0,
                "regen_rate":   1.0
            }
        }
        Returns True if the schema was successfully registered.
        """
        required_keys = {"component_name", "data_type", "mutability", "scope_binding", "attributes"}
        if not required_keys.issubset(schema_dict.keys()):
            missing = required_keys - set(schema_dict.keys())
            logger.error("[ECSRegistry] inject_dynamic_component: Missing keys: %s", missing)
            return False

        schema = ComponentSchema(
            type_name   = schema_dict["component_name"],
            attributes  = {
                k: float(v) for k, v in schema_dict["attributes"].items()
            },
            mutability  = schema_dict["mutability"],
            scope       = schema_dict["scope_binding"],
            is_builtin  = False,
        )
        self.register_component_schema(schema)
        logger.info("[ECSRegistry] Dynamic component '%s' injected via NLP pipeline.", schema.type_name)
        return True

    # ------------------------------------------------------------------
    # Diagnostics / Introspection
    # ------------------------------------------------------------------

    def get_entity_profile(self, entity_id: EntityID) -> Optional[Dict[str, Any]]:
        """Return a full snapshot of all components attached to an entity."""
        lock = self._entity_locks.get(entity_id)
        if lock is None:
            return None
        with lock:
            components = self._entity_components.get(entity_id, {})
            return {
                "entity_id":  entity_id,
                "components": {
                    ct: {k: float(v) for k, v in data.items()}
                    for ct, data in components.items()
                }
            }

    def entity_count(self) -> int:
        return len(self._entity_components)

    def component_count(self, comp_type: ComponentType) -> int:
        return len(self._component_index.get(comp_type, set()))

    def __repr__(self) -> str:
        return (f"<ECSRegistry entities={self.entity_count()} "
                f"schemas={len(self._schemas)} "
                f"systems={len(self._systems)}>")


# ---------------------------------------------------------------------------
# Built-in System Implementations
# ---------------------------------------------------------------------------

def physics_integration_system(
    registry:   ECSRegistry,
    entities:   Set[EntityID],
    dt:         float,
) -> None:
    """
    System: PhysicsIntegration (priority=10)
    Integrates velocity into position using semi-implicit Euler integration.
    Requires: Transform, Velocity, Mass
    """
    for eid in entities:
        lock = registry._entity_locks.get(eid)
        if lock is None:
            continue
        with lock:
            transform = registry._entity_components[eid].get("Transform")
            velocity  = registry._entity_components[eid].get("Velocity")
            mass      = registry._entity_components[eid].get("Mass")
            if transform is None or velocity is None or mass is None:
                continue
            if float(mass["is_static"]) > 0.5:
                continue   # Static bodies are not integrated

            # Semi-implicit Euler: x(t+dt) = x(t) + v(t+dt)*dt
            transform["x"] = np.float32(float(transform["x"]) + float(velocity["vx"]) * dt)
            transform["y"] = np.float32(float(transform["y"]) + float(velocity["vy"]) * dt)
            transform["z"] = np.float32(float(transform["z"]) + float(velocity["vz"]) * dt)


def energy_consumption_system(
    registry:   ECSRegistry,
    entities:   Set[EntityID],
    dt:         float,
) -> None:
    """
    System: EnergyConsumption (priority=20)
    Drains entity energy each tick according to consumption_rate.
    If energy reaches 0 and Health component is present, starts draining HP.
    Requires: Energy
    """
    for eid in entities:
        lock = registry._entity_locks.get(eid)
        if lock is None:
            continue
        with lock:
            energy = registry._entity_components[eid].get("Energy")
            if energy is None:
                continue
            new_energy = float(energy["current_energy"]) - float(energy["consumption_rate"]) * dt
            energy["current_energy"] = np.float32(max(0.0, new_energy))

            if new_energy < 0.0:
                health = registry._entity_components[eid].get("Health")
                if health is not None:
                    starvation_damage = abs(new_energy) * 0.1
                    new_hp = float(health["current_hp"]) - starvation_damage
                    health["current_hp"] = np.float32(max(0.0, new_hp))
                    if new_hp <= 0.0:
                        health["is_alive"] = np.float32(0.0)


def health_regen_system(
    registry:   ECSRegistry,
    entities:   Set[EntityID],
    dt:         float,
) -> None:
    """
    System: HealthRegen (priority=30)
    Regenerates HP each tick for living entities that have sufficient energy.
    Requires: Health, Energy
    """
    for eid in entities:
        lock = registry._entity_locks.get(eid)
        if lock is None:
            continue
        with lock:
            health = registry._entity_components[eid].get("Health")
            energy = registry._entity_components[eid].get("Energy")
            if health is None or energy is None:
                continue
            if float(health["is_alive"]) < 0.5:
                continue
            if float(energy["current_energy"]) <= 0.0:
                continue
            new_hp = float(health["current_hp"]) + float(health["regen_rate"]) * dt
            health["current_hp"] = np.float32(min(float(health["max_hp"]), new_hp))


def build_default_ecs() -> ECSRegistry:
    """
    Factory function: creates a pre-wired ECSRegistry with all built-in systems registered.
    Call this from the main entry point to get a fully operational ECS.
    """
    registry = ECSRegistry()
    registry.register_system("PhysicsIntegration",  physics_integration_system,
                              ["Transform", "Velocity", "Mass"], priority=10)
    registry.register_system("EnergyConsumption",   energy_consumption_system,
                              ["Energy"],                          priority=20)
    registry.register_system("HealthRegen",         health_regen_system,
                              ["Health", "Energy"],               priority=30)
    logger.info("[ECSRegistry] Default ECS built with 3 built-in systems.")
    return registry
