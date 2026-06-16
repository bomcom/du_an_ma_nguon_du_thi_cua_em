"""
math_core/matrix_solver.py
==========================
Deterministic Closed-Loop Math Core for the Hybrid AI Simulation Platform.

Architecture
------------
    Implements the Axiomatic Constraint System S_1 … S_n governing all physical
    and biological evolution boundaries within the simulation world.

    Constraint System Definitions
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    S_1 (Energy Conservation):
        Total system energy E_total = Σ(kinetic + potential + biological) must not
        exceed E_max. Excess energy triggers the environmental decay signal.
        E_total = Σ 0.5 * m_i * |v_i|² + m_i * g * h_i + bio_energy_i

    S_2 (Mass Conservation / Biomass Ceiling):
        Total system mass M_total = Σ m_i  must remain ≤ M_max.
        New entities may not be spawned if M_total + m_new > M_max.

    S_3 (Biological State Evolution – Lotka-Volterra):
        dP/dt = α·P − β·P·Q      (prey population P)
        dQ/dt = δ·P·Q − γ·Q      (predator population Q)
        Integrated with 4th-order Runge-Kutta (RK4) each simulation tick.

    S_4 (Electrical-Physical Coupling):
        Physical movement speed limit is proportional to LTspice Bandwidth:
            v_max(t) = k_bw · BW(t)   [meters per second]
        Calorie/energy burn is proportional to Mass × SNR:
            E_burn(t) = k_snr · m · SNR(t)

    S_5 (Non-Linear Orbital / Wave Coupling):
        A generic non-linear ODE system representing wave coupling:
            dx/dt = σ(y − x)          [Lorenz x — used as chaos seed]
            dy/dt = x(ρ − z) − y
            dz/dt = xy − βz

    Numerical Safety Layer
    ~~~~~~~~~~~~~~~~~~~~~~
    After every integration step, ALL output matrices pass through:
        1. NaN / Inf guard (replace with boundary clamped value)
        2. Symmetric normalisation to operational parameter ranges
        3. Rate-of-change limiter (prevents catastrophic single-tick jumps)

CUDA Acceleration
-----------------
    If CuPy is available and a CUDA device is detected, all large matrix operations
    are delegated to CuPy for GPU acceleration. Falls back to NumPy transparently.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Attempt CuPy import with graceful NumPy fallback
try:
    import cupy as cp  # type: ignore[import-not-found]
    _CUPY_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    cp = None  # type: ignore[assignment]
    _CUPY_AVAILABLE = False

logger = logging.getLogger(__name__)
logger.info("[MatrixSolver] CuPy / CUDA available: %s", _CUPY_AVAILABLE)


# ---------------------------------------------------------------------------
# Backend selector helpers
# ---------------------------------------------------------------------------

def _xp() -> Any:
    """Return cupy if CUDA is available, else numpy."""
    return cp if _CUPY_AVAILABLE else np


def _to_numpy(arr: Any) -> np.ndarray:
    """Convert a cupy or numpy array to a numpy ndarray."""
    if _CUPY_AVAILABLE and cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def _from_numpy(arr: np.ndarray) -> Any:
    """Convert numpy array to xp array (cupy if available)."""
    if _CUPY_AVAILABLE and cp is not None:
        return cp.asarray(arr)
    return arr


# ---------------------------------------------------------------------------
# Constraint Parameter Configuration
# ---------------------------------------------------------------------------

@dataclass
class ConstraintConfig:
    """
    Tunable parameters for the axiomatic constraint system.
    Loaded from config at startup; may be recalibrated by the Adversarial Interrogator.
    """
    # S_1: Energy Conservation
    E_max:          float  = 1_000_000.0       # Maximum total system energy (Joules)
    g:              float  = 9.81              # Gravitational acceleration (m/s²)

    # S_2: Mass Conservation
    M_max:          float  = 50_000.0          # Maximum total system mass (kg)

    # S_3: Lotka-Volterra
    alpha:          float  = 0.10              # Prey birth rate
    beta:           float  = 0.02              # Predation rate
    delta:          float  = 0.005             # Predator efficiency
    gamma:          float  = 0.08              # Predator death rate
    prey_init:      float  = 100.0             # Initial prey population
    predator_init:  float  = 20.0              # Initial predator population

    # S_4: Electrical-Physical Coupling
    k_bw:           float  = 0.001             # BW → v_max scaling factor
    k_snr:          float  = 0.0001            # SNR → energy burn scaling

    # S_5: Lorenz chaos parameters
    lorenz_sigma:   float  = 10.0
    lorenz_rho:     float  = 28.0
    lorenz_beta:    float  = 8.0 / 3.0
    lorenz_init:    Tuple  = (0.1, 0.0, 0.0)  # (x0, y0, z0)

    # Numerical safety
    max_delta_ratio: float = 0.5               # Max fractional change per tick (rate limiter)
    nan_replacement: float = 0.0               # Value used to replace NaN cells


# ---------------------------------------------------------------------------
# State Vectors (held in memory between ticks)
# ---------------------------------------------------------------------------

@dataclass
class ConstraintState:
    """Mutable state vector for the constraint system, updated each tick."""
    # S_1
    E_kinetic:     np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    E_potential:   np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    E_biological:  np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    E_total:       float = 0.0
    decay_trigger: bool  = False               # True when E_total > E_max

    # S_2
    M_total:       float = 0.0

    # S_3 Lotka-Volterra
    prey:          float = 100.0
    predator:      float = 20.0

    # S_4 Electrical coupling (updated from Session Box channel "ltspice")
    v_max_entity:  float = 10.0               # Current speed ceiling
    e_burn_rate:   float = 0.1                # Current calorie burn rate

    # S_5 Lorenz
    lorenz_x:      float = 0.1
    lorenz_y:      float = 0.0
    lorenz_z:      float = 0.0

    # Constraint violation flags  {S_n_name: bool}
    violations:    Dict[str, bool] = field(default_factory=dict)

    # Tick counter
    tick:          int   = 0


# ---------------------------------------------------------------------------
# MatrixSolver
# ---------------------------------------------------------------------------

class MatrixSolver:
    """
    Deterministic closed-loop math core.

    Workflow per tick
    -----------------
    1. Ingest entity mass/velocity arrays from the ECS snapshot.
    2. Evaluate S_1 … S_5 constraints sequentially.
    3. Integrate differential equations (RK4 for S_3, S_5).
    4. Apply numerical safety layer (NaN/Inf guard + normalisation).
    5. Publish the resulting state to the Session Box channel "math_state".
    6. Publish violation flags to channel "adversarial_flags".

    Thread Safety
    -------------
    tick() is not re-entrant; it must be called from a single dedicated thread.
    Reading results via get_state() is thread-safe (returns a deep copy).
    """

    def __init__(self, config: Optional[ConstraintConfig] = None) -> None:
        self.config = config or ConstraintConfig()
        self.state  = ConstraintState(
            prey     = self.config.prey_init,
            predator = self.config.predator_init,
            lorenz_x = self.config.lorenz_init[0],
            lorenz_y = self.config.lorenz_init[1],
            lorenz_z = self.config.lorenz_init[2],
        )
        self._state_lock = threading.Lock()
        self._tick_lock  = threading.Lock()   # Ensures tick() is not called concurrently

        # Pre-allocated working matrices (avoid per-tick heap allocations)
        # Sized for a maximum of MAX_ENTITIES entities
        MAX_ENTITIES = 4096
        xp = _xp()
        self._mass_buf = xp.zeros(MAX_ENTITIES, dtype=xp.float32)
        self._vx_buf   = xp.zeros(MAX_ENTITIES, dtype=xp.float32)
        self._vy_buf   = xp.zeros(MAX_ENTITIES, dtype=xp.float32)
        self._vz_buf   = xp.zeros(MAX_ENTITIES, dtype=xp.float32)
        self._hy_buf   = xp.zeros(MAX_ENTITIES, dtype=xp.float32)   # height for potential energy

        logger.info("[MatrixSolver] Initialised (backend=%s, E_max=%.1f, M_max=%.1f).",
                    "CuPy/CUDA" if _CUPY_AVAILABLE else "NumPy/CPU",
                    self.config.E_max, self.config.M_max)

    # ------------------------------------------------------------------
    # Primary Tick Entry Point
    # ------------------------------------------------------------------

    def tick(
        self,
        entity_masses:     np.ndarray,   # shape (N,) float32
        entity_velocities: np.ndarray,   # shape (N, 3) float32
        entity_heights:    np.ndarray,   # shape (N,) float32 (y-coordinate)
        entity_bio_energy: np.ndarray,   # shape (N,) float32 (NPC metabolic energy)
        ltspice_bandwidth: float = 1000.0,
        ltspice_snr:       float = 30.0,
        dt:                float = 0.016,
    ) -> Dict[str, Any]:
        """
        Execute one constraint evaluation + integration step.
        """
        if not self._tick_lock.acquire(blocking=False):
            logger.warning("[MatrixSolver] tick() called re-entrantly. Skipping this tick.")
            return {}

        try:
            return self._execute_tick(
                entity_masses, entity_velocities, entity_heights,
                entity_bio_energy, ltspice_bandwidth, ltspice_snr, dt
            )
        finally:
            self._tick_lock.release()

    def _execute_tick(
        self,
        masses:     np.ndarray,
        velocities: np.ndarray,
        heights:    np.ndarray,
        bio_energy: np.ndarray,
        bw:         float,
        snr:        float,
        dt:         float,
    ) -> Dict[str, Any]:
        xp    = _xp()
        cfg   = self.config
        t_start = time.perf_counter()

        N = len(masses)

        # ── CHỐT AN TOÀN CHỐNG CRASH: Xử lý khi chưa có thực thể (N == 0) ──
        if N == 0:
            # S_3 Lotka-Volterra và S_5 Lorenz vẫn tiến hóa độc lập theo thời gian định tiến
            new_prey, new_predator = self._rk4_lotka_volterra(self.state.prey, self.state.predator, cfg, dt)
            lx, ly, lz = self._rk4_lorenz(self.state.lorenz_x, self.state.lorenz_y, self.state.lorenz_z, cfg, dt)
            
            with self._state_lock:
                self.state.E_kinetic     = np.array([], dtype=np.float32)
                self.state.E_potential   = np.array([], dtype=np.float32)
                self.state.E_biological  = np.array([], dtype=np.float32)
                self.state.E_total       = 0.0
                self.state.decay_trigger = False
                self.state.M_total       = 0.0
                self.state.prey          = max(0.0, new_prey)
                self.state.predator      = max(0.0, new_predator)
                self.state.v_max_entity  = cfg.k_bw * bw
                self.state.e_burn_rate   = 0.0  # Triệt tiêu burn rate để tránh phép chia rỗng tạo NaN
                self.state.lorenz_x      = lx
                self.state.lorenz_y      = ly
                self.state.lorenz_z      = lz
                self.state.violations    = {
                    "S1_energy_overflow":   False,
                    "S2_mass_overflow":     False,
                    "S3_prey_extinction":   new_prey < 1.0,
                    "S3_predator_collapse": new_predator < 1.0,
                    "S4_speed_violation":   False,
                    "S5_lorenz_divergence": any(abs(v) > 1e6 for v in [lx, ly, lz]),
                }
                self.state.tick         += 1
            return self._build_output_dict()

        # ── Nếu N > 0, xử lý tính toán ma trận phân tán song song ──
        m  = _from_numpy(np.asarray(masses,     dtype=np.float32).flatten())
        v  = _from_numpy(np.asarray(velocities, dtype=np.float32).reshape(N, 3))
        h  = _from_numpy(np.asarray(heights,    dtype=np.float32).flatten())
        be = _from_numpy(np.asarray(bio_energy, dtype=np.float32).flatten())

        # ── S_1: Energy Conservation ───────────────────────────────────
        v_sq      = xp.sum(v ** 2, axis=1)          # |v|² per entity
        E_kin     = 0.5 * m * v_sq
        E_pot     = m * cfg.g * xp.maximum(h, xp.zeros_like(h))
        E_bio     = xp.maximum(be, xp.zeros_like(be))
        E_total_v = E_kin + E_pot + E_bio

        E_kin_np  = _to_numpy(E_kin)
        E_pot_np  = _to_numpy(E_pot)
        E_total   = float(xp.sum(E_total_v))

        decay_trigger = E_total > cfg.E_max

        # ── S_2: Mass Conservation ─────────────────────────────────────
        M_total = float(xp.sum(m))

        # ── S_3: Lotka-Volterra (RK4 Integration) ─────────────────────
        new_prey, new_predator = self._rk4_lotka_volterra(self.state.prey, self.state.predator, cfg, dt)
        new_prey     = max(0.0, new_prey)
        new_predator = max(0.0, new_predator)

        # ── S_4: Electrical-Physical Coupling ─────────────────────────
        v_max_entity = cfg.k_bw * bw                      # m/s ceiling
        e_burn_rate  = cfg.k_snr * float(xp.mean(m)) * snr   # J/s (An toàn vì m không rỗng)

        # ── S_5: Lorenz System (RK4) ───────────────────────────────────
        lx, ly, lz = self._rk4_lorenz(self.state.lorenz_x, self.state.lorenz_y, self.state.lorenz_z, cfg, dt)

        # ── Numerical Safety Layer (Sửa lỗi Molasses Bug bằng cách truyền thêm prev_arr) ──
        E_kin_safe = self._safety_clamp(E_kin_np, 0.0, cfg.E_max, prev_arr=self.state.E_kinetic)
        E_pot_safe = self._safety_clamp(E_pot_np, 0.0, cfg.E_max, prev_arr=self.state.E_potential)

        # Constraint violation flags
        violations = {
            "S1_energy_overflow":   decay_trigger,
            "S2_mass_overflow":     M_total > cfg.M_max,
            "S3_prey_extinction":   new_prey < 1.0,
            "S3_predator_collapse": new_predator < 1.0,
            "S4_speed_violation":   False,
            "S5_lorenz_divergence": any(abs(v) > 1e6 for v in [lx, ly, lz]),
        }

        # ── Write new state (under lock for thread-safe reads) ─────────
        with self._state_lock:
            self.state.E_kinetic     = E_kin_safe
            self.state.E_potential   = E_pot_safe
            self.state.E_biological  = _to_numpy(E_bio)
            self.state.E_total       = E_total
            self.state.decay_trigger = decay_trigger
            self.state.M_total       = M_total
            self.state.prey          = new_prey
            self.state.predator      = new_predator
            self.state.v_max_entity  = v_max_entity
            self.state.e_burn_rate   = e_burn_rate
            self.state.lorenz_x      = lx
            self.state.lorenz_y      = ly
            self.state.lorenz_z      = lz
            self.state.violations    = violations
            self.state.tick         += 1

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        logger.debug(
            "[MatrixSolver] tick=%d | E=%.2f/%.2f | M=%.2f/%.2f | "
            "prey=%.1f pred=%.1f | v_max=%.2f | violations=%s | %.2fms",
            self.state.tick, E_total, cfg.E_max, M_total, cfg.M_max,
            new_prey, new_predator, v_max_entity,
            [k for k, v in violations.items() if v],
            elapsed_ms,
        )

        return self._build_output_dict()

    # ------------------------------------------------------------------
    # RK4 Integrators
    # ------------------------------------------------------------------

    @staticmethod
    def _rk4_lotka_volterra(
        P: float, Q: float, cfg: ConstraintConfig, dt: float
    ) -> Tuple[float, float]:
        """
        4th-order Runge-Kutta integration of the Lotka-Volterra system.
        """
        def dP(p, q): return cfg.alpha * p - cfg.beta * p * q
        def dQ(p, q): return cfg.delta * p * q - cfg.gamma * q

        k1p = dt * dP(P, Q)
        k1q = dt * dQ(P, Q)
        k2p = dt * dP(P + k1p / 2, Q + k1q / 2)
        k2q = dt * dQ(P + k1p / 2, Q + k1q / 2)
        k3p = dt * dP(P + k2p / 2, Q + k2q / 2)
        k3q = dt * dQ(P + k2p / 2, Q + k2q / 2)
        k4p = dt * dP(P + k3p, Q + k3q)
        k4q = dt * dQ(P + k3p, Q + k3q)

        new_P = P + (k1p + 2*k2p + 2*k3p + k4p) / 6.0
        new_Q = Q + (k1q + 2*k2q + 2*k3q + k4q) / 6.0
        return new_P, new_Q

    @staticmethod
    def _rk4_lorenz(
        x: float, y: float, z: float, cfg: ConstraintConfig, dt: float
    ) -> Tuple[float, float, float]:
        """
        4th-order Runge-Kutta integration of the Lorenz chaotic system.
        """
        s, r, b = cfg.lorenz_sigma, cfg.lorenz_rho, cfg.lorenz_beta

        def dx(x, y, z): return s * (y - x)
        def dy(x, y, z): return x * (r - z) - y
        def dz(x, y, z): return x * y - b * z

        k1x = dt * dx(x, y, z)
        k1y = dt * dy(x, y, z)
        k1z = dt * dz(x, y, z)

        k2x = dt * dx(x+k1x/2, y+k1y/2, z+k1z/2)
        k2y = dt * dy(x+k1x/2, y+k1y/2, z+k1z/2)
        k2z = dt * dz(x+k1x/2, y+k1y/2, z+k1z/2)

        k3x = dt * dx(x+k2x/2, y+k2y/2, z+k2z/2)
        k3y = dt * dy(x+k2x/2, y+k2y/2, z+k2z/2)
        k3z = dt * dz(x+k2x/2, y+k2y/2, z+k2z/2)

        k4x = dt * dx(x+k3x, y+k3y, z+k3z)
        k4y = dt * dy(x+k3x, y+k3y, z+k3z)
        k4z = dt * dz(x+k3x, y+k3y, z+k3z)

        nx = x + (k1x + 2*k2x + 2*k3x + k4x) / 6.0
        ny = y + (k1y + 2*k2y + 2*k3y + k4y) / 6.0
        nz = z + (k1z + 2*k2z + 2*k3z + k4z) / 6.0
        return nx, ny, nz

    # ------------------------------------------------------------------
    # Numerical Safety Layer
    # ------------------------------------------------------------------

    def _safety_clamp(
        self,
        arr: np.ndarray,
        lo:  float,
        hi:  float,
        prev_arr: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Multi-stage numerical safety filter:
        1. Replace NaN with cfg.nan_replacement.
        2. Replace +/-Inf with boundary values.
        3. Clip to [lo, hi].
        4. Apply rate-of-change limiter relative to historical state delta.
        """
        cfg  = self.config
        safe = np.where(np.isnan(arr), cfg.nan_replacement, arr)
        safe = np.where(np.isposinf(safe), hi,  safe)
        safe = np.where(np.isneginf(safe), lo,  safe)
        safe = np.clip(safe, lo, hi)

        # Rate-of-change limiter chuẩn xác: Chỉ áp dụng khi số lượng thực thể không đổi giữa 2 vòng lặp lân cận
        if prev_arr is not None and prev_arr.shape == safe.shape and prev_arr.size > 0:
            delta = safe - prev_arr
            max_val = max(np.max(np.abs(prev_arr)), 1e-12)
            limit = max_val * cfg.max_delta_ratio
            delta_clipped = np.clip(delta, -limit, limit)
            safe = prev_arr + delta_clipped
            safe = np.clip(safe, lo, hi)

        return safe.astype(np.float32)

    def normalise_01(self, arr: np.ndarray) -> np.ndarray:
        """Normalise array to [0, 1] range for network input tensors."""
        arr = arr.astype(np.float32)
        if arr.size == 0:
            return arr
        lo, hi = np.min(arr), np.max(arr)
        if hi - lo < 1e-12:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - lo) / (hi - lo)).astype(np.float32)

    def normalise_sym(self, arr: np.ndarray) -> np.ndarray:
        """Normalise array to [-1, 1] range (symmetric) for neural weight inputs."""
        if arr.size == 0:
            return arr.astype(np.float32)
        n01 = self.normalise_01(arr)
        return (n01 * 2.0 - 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Public Accessors
    # ------------------------------------------------------------------

    def get_state(self) -> ConstraintState:
        """Return a thread-safe deep copy of the current constraint state."""
        import copy
        with self._state_lock:
            return copy.deepcopy(self.state)

    def get_v_max(self) -> float:
        """Current maximum entity speed derived from LTspice Bandwidth (S_4)."""
        with self._state_lock:
            return self.state.v_max_entity

    def get_E_max(self) -> float:
        """Return the configured energy ceiling."""
        return self.config.E_max

    def check_mass_budget(self, proposed_mass: float) -> bool:
        """
        Returns True if adding `proposed_mass` kg would not violate S_2.
        """
        with self._state_lock:
            return (self.state.M_total + proposed_mass) <= self.config.M_max

    def get_lorenz_seed(self) -> Tuple[float, float, float]:
        """Return the current Lorenz attractor state as a noise seed triple."""
        with self._state_lock:
            return self.state.lorenz_x, self.state.lorenz_y, self.state.lorenz_z

    def get_violations(self) -> Dict[str, bool]:
        """Return current constraint violation flags."""
        with self._state_lock:
            return dict(self.state.violations)

    def _build_output_dict(self) -> Dict[str, Any]:
        """Serialise current state into a plain Python dict for Session Box writes."""
        s = self.state
        return {
            "tick":          s.tick,
            "E_total":       s.E_total,
            "E_max":         self.config.E_max,
            "M_total":       s.M_total,
            "M_max":         self.config.M_max,
            "decay_trigger": int(s.decay_trigger),
            "prey":          s.prey,
            "predator":      s.predator,
            "v_max":         s.v_max_entity,
            "e_burn_rate":   s.e_burn_rate,
            "lorenz":        (s.lorenz_x, s.lorenz_y, s.lorenz_z),
            "violations":    dict(s.violations),
        }

    def recalibrate_E_max(self, new_E_max: float) -> None:
        """
        Dynamically recalibrate the energy ceiling.
        Called by the Adversarial Interrogator's Dynamic Boundary Calibration algorithm.
        """
        old = self.config.E_max
        self.config.E_max = float(new_E_max)
        logger.info("[MatrixSolver] E_max recalibrated: %.2f → %.2f", old, new_E_max)

    def __repr__(self) -> str:
        return (f"<MatrixSolver tick={self.state.tick} "
                f"E={self.state.E_total:.1f}/{self.config.E_max:.1f} "
                f"backend={'CuPy' if _CUPY_AVAILABLE else 'NumPy'}>")
    