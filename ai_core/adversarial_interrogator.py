"""
ai_core/adversarial_interrogator.py
====================================
Adversarial Gatekeeper Network — ADVERSARIAL_INTERROGATOR_NET

Architecture
------------
    The Adversarial Interrogator is a lightweight feed-forward neural network
    implemented in raw NumPy (with optional PyTorch acceleration) that acts as
    the strict logical gatekeeper for the simulation's axiomatic constraint system.

    Network Function
    ~~~~~~~~~~~~~~~~
    Input (12-dim float32 feature vector):
        [E_ratio, M_ratio, prey_norm, predator_norm,
         v_max_norm, e_burn_norm, lorenz_x_n, lorenz_y_n, lorenz_z_n,
         decay_flag, any_population_collapse, raw_violation_count]

    Output (2-dim softmax):
        [P_valid, P_violation]
        If P_violation > threshold → trigger validation alert + Socratic guidance.

    Network Architecture:
        Input(12) → Dense(32, ReLU) → Dropout(0.1) → Dense(16, ReLU) → Dense(2, Softmax)

    Over-Sensitivity Problem & Dynamic Boundary Calibration
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    A gatekeeper operating on fixed thresholds becomes pathologically over-sensitive:
    legitimate high-energy events (e.g., a combat burst) are rejected as violations.
    This system implements a two-stage mitigation strategy:

    Stage 1 — Dynamic Threshold Calibration:
        The rejection threshold starts at `base_threshold` (default 0.65).
        A rolling window of the last N activations is maintained.
        If the false-positive rate (valid actions rejected) exceeds
        `fp_rate_ceiling` (default 0.20), the threshold is raised by
        `calibration_step` (default 0.03), up to `max_threshold` (0.90).
        If the false-positive rate falls below `fp_rate_floor` (default 0.05),
        the threshold is lowered by `calibration_step`, down to `min_threshold` (0.50).

    Stage 2 — Socratic Guidance Cascade:
        When a violation is detected, instead of hard-rejecting the action,
        the interrogator invokes `SocraticGuide.guide()` asynchronously to
        present the user with targeted self-correction prompts, leading
        them to discover the constraint they violated independently.

    Weight Initialisation
    ~~~~~~~~~~~~~~~~~~~~~
    Weights use He (Kaiming) initialisation for ReLU layers.
    All matrices are float32 to match the ECS component dtype.
"""
"""

Tên tệp: adversarial_interrogator.py
Mô tả: Bộ kiểm duyệt đối kháng cho hệ thống ECS.

Bản quyền © 2026 Phạm Hồng Hải Đăng.

Mọi quyền được bảo lưu.

Tài liệu này thuộc sở hữu trí tuệ của Phạm Hồng Hải  Hải Đăng.

"""

import asyncio
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

logger = logging.getLogger(__name__)

# Attempt PyTorch import (optional acceleration)
_TORCH_AVAILABLE = False
try:
    import torch  # type: ignore[import-not-found]
    import torch.nn as nn  # type: ignore[import-not-found]
    _TORCH_AVAILABLE = True
except ImportError:
    pass

logger.info("[AdversarialInterrogator] PyTorch available: %s", _TORCH_AVAILABLE)


# ---------------------------------------------------------------------------
# Network Dimensions (constants)
# ---------------------------------------------------------------------------
INPUT_DIM  = 12
HIDDEN1    = 32
HIDDEN2    = 16
OUTPUT_DIM = 2    # [P_valid, P_violation]


# ---------------------------------------------------------------------------
# NumPy-native Micro Neural Network
# ---------------------------------------------------------------------------

class NumpyAdversarialNet:
    """
    Pure NumPy feed-forward network.
    Architecture: Input(12) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(2, Softmax)
    Used when PyTorch is unavailable or when minimal dependencies are preferred.
    """

    def __init__(self, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)

        # He (Kaiming) initialisation for ReLU networks
        # W ~ N(0, sqrt(2 / fan_in))
        def he_init(fan_in: int, fan_out: int) -> np.ndarray:
            std = np.sqrt(2.0 / fan_in)
            return rng.normal(0.0, std, size=(fan_out, fan_in)).astype(np.float32)

        # Layer 1: (HIDDEN1, INPUT_DIM)
        self.W1 = he_init(INPUT_DIM, HIDDEN1)
        self.b1 = np.zeros(HIDDEN1, dtype=np.float32)

        # Layer 2: (HIDDEN2, HIDDEN1)
        self.W2 = he_init(HIDDEN1, HIDDEN2)
        self.b2 = np.zeros(HIDDEN2, dtype=np.float32)

        # Layer 3 (output): (OUTPUT_DIM, HIDDEN2)
        self.W3 = he_init(HIDDEN2, OUTPUT_DIM)
        self.b3 = np.zeros(OUTPUT_DIM, dtype=np.float32)

        logger.debug("[NumpyAdversarialNet] Weights initialised (He). Params=%d",
                     self._param_count())

    # ── Activation functions ─────────────────────────────────────────

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))     # Numerically stable softmax
        return e / np.sum(e)

    # ── Forward pass ────────────────────────────────────────────────

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        Parameters
        ----------
        x : float32 array of shape (INPUT_DIM,)
        Returns
        -------
        float32 array of shape (OUTPUT_DIM,) — [P_valid, P_violation]
        """
        assert x.shape == (INPUT_DIM,), f"Expected ({INPUT_DIM},), got {x.shape}"
        h1 = self._relu(self.W1 @ x + self.b1)
        h2 = self._relu(self.W2 @ h1 + self.b2)
        logits = self.W3 @ h2 + self.b3
        return self._softmax(logits)

    def _param_count(self) -> int:
        return sum(w.size for w in [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3])


# ---------------------------------------------------------------------------
# PyTorch Adversarial Net (optional)
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:
    class TorchAdversarialNet(nn.Module):
        """
        Optional PyTorch implementation of the same architecture.
        Includes Dropout(0.1) for inference-time calibration uncertainty.
        """
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(INPUT_DIM, HIDDEN1),
                nn.ReLU(),
                nn.Dropout(p=0.10),
                nn.Linear(HIDDEN1, HIDDEN2),
                nn.ReLU(),
                nn.Linear(HIDDEN2, OUTPUT_DIM),
                nn.Softmax(dim=-1),
            )
            # He initialisation
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                    nn.init.zeros_(m.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
            return self.net(x)


# ---------------------------------------------------------------------------
# Socratic Guide (async sub-system)
# ---------------------------------------------------------------------------

@dataclass
class SocraticPrompt:
    """A structured suggestion delivered to the user when a violation is detected."""
    violation_type: str
    prompt_text:    str
    hint_level:     int   = 1   # 1=gentle, 2=specific, 3=direct


class SocraticGuide:
    """
    Asynchronous Socratic Guidance Sub-system.
    Queues targeted prompt suggestions to lead users toward self-correction
    rather than receiving opaque hard rejections.
    """

    # Prompt templates indexed by violation type
    PROMPT_LIBRARY: Dict[str, List[str]] = {
        "S1_energy_overflow": [
            "Hệ thống đang cảm nhận một lượng năng lượng rất lớn. "
            "Bạn có thể giải thích tại sao thực thể này cần năng lượng vượt mức E_max không?",
            "Thử nghĩ: nếu tổng năng lượng vượt quá E_max, "
            "điều gì sẽ xảy ra với sự bảo toàn năng lượng trong hệ kín?",
            "Gợi ý trực tiếp: Giảm giá trị current_energy hoặc max_energy xuống "
            "sao cho Σ E_i ≤ E_max = {E_max:.1f} J.",
        ],
        "S2_mass_overflow": [
            "Tổng khối lượng hệ thống đang tiến gần đến giới hạn M_max. "
            "Tại sao thực thể mới này cần được tạo ra lúc này?",
            "Xem xét: Có thực thể nào có thể bị loại bỏ để nhường chỗ không?",
            "Gợi ý trực tiếp: M_total hiện tại = {M_total:.1f} kg, M_max = {M_max:.1f} kg. "
            "Khối lượng đề xuất của bạn vượt quá ngân sách còn lại.",
        ],
        "S3_prey_extinction": [
            "Quần thể con mồi đang tiến gần đến tuyệt chủng. "
            "Điều này sẽ ảnh hưởng như thế nào đến chuỗi thức ăn?",
            "Trong mô hình Lotka-Volterra, khi P→0, điều gì xảy ra với Q theo thời gian?",
            "Gợi ý: Giảm tỷ lệ săn mồi β hoặc tăng α (tỷ lệ sinh sản con mồi).",
        ],
        "S4_speed_violation": [
            "Tốc độ của thực thể này vượt quá v_max được xác định bởi Băng thông LTspice. "
            "Điều này vi phạm ràng buộc vật lý S_4.",
            "Gợi ý: v_max hiện tại = {v_max:.2f} m/s dựa trên BW = {bw:.1f} Hz.",
        ],
        "default": [
            "Hành động của bạn có vẻ vi phạm một ràng buộc hệ thống. "
            "Bạn có muốn xem xét lại các thông số không?",
        ],
    }

    def __init__(
        self,
        on_prompt: Optional[Callable[[SocraticPrompt], None]] = None
    ) -> None:
        """
        Parameters
        ----------
        on_prompt : Optional callback invoked when a prompt is ready for display.
                    Signature: callback(SocraticPrompt) -> None
        """
        self._on_prompt = on_prompt
        self._prompt_queue: asyncio.Queue = asyncio.Queue()
        self._hint_history: Dict[str, int] = {}   # violation_type -> last hint level used

    async def guide(
        self,
        violation_type:  str,
        context:         Dict[str, Any],
    ) -> None:
        """
        Asynchronously select and deliver the appropriate Socratic prompt.
        Escalates hint_level on repeated violations of the same type.
        """
        templates = self.PROMPT_LIBRARY.get(violation_type, self.PROMPT_LIBRARY["default"])
        level = self._hint_history.get(violation_type, 0)
        level = min(level, len(templates) - 1)

        # Format template with context values
        try:
            text = templates[level].format(**context)
        except KeyError:
            text = templates[0]

        prompt = SocraticPrompt(
            violation_type = violation_type,
            prompt_text    = text,
            hint_level     = level + 1,
        )

        self._hint_history[violation_type] = level + 1

        logger.info("[SocraticGuide] Delivering level-%d hint for '%s': %s",
                    prompt.hint_level, violation_type, prompt.prompt_text[:60])

        if self._on_prompt:
            self._on_prompt(prompt)

        await self._prompt_queue.put(prompt)

    def reset_hint_level(self, violation_type: str) -> None:
        """Reset escalation level after a successful correction."""
        self._hint_history.pop(violation_type, None)


# ---------------------------------------------------------------------------
# Dynamic Threshold Calibration State
# ---------------------------------------------------------------------------

@dataclass
class CalibrationState:
    """
    Tracks the rolling statistics used for Dynamic Boundary Calibration.
    """
    threshold:          float = 0.65          # Current active rejection threshold
    base_threshold:     float = 0.65
    min_threshold:      float = 0.50
    max_threshold:      float = 0.90
    calibration_step:   float = 0.03
    fp_rate_ceiling:    float = 0.20          # Too many rejections → raise threshold
    fp_rate_floor:      float = 0.05          # Too few rejections → lower threshold
    window_size:        int   = 50            # Rolling window for FP rate estimation

    # Rolling window: True = was a rejection, False = was accepted
    _activation_window: deque = field(default_factory=lambda: deque(maxlen=50))

    def record_activation(self, was_rejected: bool) -> None:
        self._activation_window.append(was_rejected)

    def current_fp_rate(self) -> float:
        if not self._activation_window:
            return 0.0
        return sum(self._activation_window) / len(self._activation_window)

    def calibrate(self) -> str:
        """
        Adjust threshold based on observed false-positive rate.
        Returns a string label: "raised", "lowered", or "unchanged".
        """
        fp_rate = self.current_fp_rate()
        if fp_rate > self.fp_rate_ceiling and self.threshold < self.max_threshold:
            self.threshold = min(self.max_threshold, self.threshold + self.calibration_step)
            logger.info("[Calibration] FP rate=%.2f > ceiling=%.2f → threshold raised to %.3f",
                        fp_rate, self.fp_rate_ceiling, self.threshold)
            return "raised"
        elif fp_rate < self.fp_rate_floor and self.threshold > self.min_threshold:
            self.threshold = max(self.min_threshold, self.threshold - self.calibration_step)
            logger.info("[Calibration] FP rate=%.2f < floor=%.2f → threshold lowered to %.3f",
                        fp_rate, self.fp_rate_floor, self.threshold)
            return "lowered"
        return "unchanged"


# ---------------------------------------------------------------------------
# Causal Graph Tracer (Thêm mới theo yêu cầu)
# ---------------------------------------------------------------------------

class CausalGraphTracer:
    """
    Theo vết nguyên nhân hệ quả (Causal Tracing).
    Thay vì chỉ báo lỗi, nó xây dựng một chuỗi logic dẫn đến vi phạm
    (VD: Biến động môi trường -> Thay đổi chỉ số -> Chạm ngưỡng giới hạn -> Vi phạm).
    """
    
    def trace(self, math_state: Dict[str, Any], active_viols: List[str], rca: str) -> str:
        """
        Dựng chuỗi nguyên nhân - kết quả (A -> B -> C -> Violation).
        """
        if not active_viols:
            return "Stable: No violations detected."

        primary_viol = active_viols[0]
        chain = []

        # Giải mã chuỗi logic dựa trên loại vi phạm chính
        if primary_viol == "S1_energy_overflow":
            e_tot = math_state.get('E_total', 0)
            e_max = math_state.get('E_max', 1)
            chain.extend([
                "Excessive Action/Spawn Triggered",
                f"Energy Pool Spirals: E_total ({e_tot:.1f}) exceeds E_max ({e_max:.1f})",
                "Axiom S1 Bounding Failure",
                "Violation: S1_energy_overflow"
            ])
            
        elif primary_viol == "S2_mass_overflow":
            m_tot = math_state.get('M_total', 0)
            m_max = math_state.get('M_max', 1)
            chain.extend([
                "Uncontrolled Entity Reproduction",
                f"Biomass Capacity Breached: M_total ({m_tot:.1f}) > M_max ({m_max:.1f})",
                "Axiom S2 Spatial Squeeze",
                "Violation: S2_mass_overflow"
            ])
            
        elif "S3" in primary_viol:  # Prey or Predator collapse
            prey = math_state.get('prey', 0)
            pred = math_state.get('predator', 0)
            chain.extend([
                "Lotka-Volterra Equilibrium Disrupted",
                f"Population Imbalance: Prey={prey:.0f}, Predators={pred:.0f}",
                "Ecosystem Collapse Imminent",
                f"Violation: {primary_viol}"
            ])
            
        elif primary_viol == "S4_speed_violation":
            v_max = math_state.get('v_max', 0)
            chain.extend([
                f"Agent Requested Acceleration to v={v_max:.1f} m/s",
                f"Underlying Hardware Bottleneck (RCA: {rca})",
                "Axiom S4 Velocity Mismatch",
                "Violation: S4_speed_violation"
            ])
            
        else:
            chain.extend([
                "Unknown Trigger Event",
                f"System Detected Anomaly (RCA: {rca})",
                "Interrogator Threshold Exceeded",
                f"Violation: {primary_viol}"
            ])

        # Kết nối chuỗi bằng mũi tên logic
        return " ➔ ".join(chain)


# ---------------------------------------------------------------------------
# AdversarialInterrogator — Main Class
# ---------------------------------------------------------------------------

class AdversarialInterrogator:
    """
    The central Adversarial Gatekeeper Network controller.

    Responsibilities
    ----------------
    1. Feature extraction from constraint state dict (math_state channel).
    2. Forward pass through ADVERSARIAL_INTERROGATOR_NET.
    3. Threshold comparison with dynamic calibration.
    4. Violation flag publication to Session Box channel "adversarial_flags".
    5. Asynchronous Socratic guidance dispatch on violation detection.
    6. Root Cause Analysis (RCA) routing: electrical vs. mathematical origin.
    7. Causal Graph Tracing for transparent debugging (A -> B -> C -> Violation).
    """

    def __init__(
        self,
        use_torch:       bool                    = True,
        on_prompt:       Optional[Callable]      = None,
        calibration:     Optional[CalibrationState] = None,
    ) -> None:
        """
        Parameters
        ----------
        use_torch    : Prefer PyTorch net if available; fallback to NumPy.
        on_prompt    : Callback for Socratic prompts (forwarded to SocraticGuide).
        calibration  : Optional pre-configured CalibrationState; defaults to new instance.
        """
        # Initialise network backend
        if use_torch and _TORCH_AVAILABLE:
            self._net_type = "torch"
            self._torch_net = TorchAdversarialNet()
            self._torch_net.eval()
            logger.info("[AdversarialInterrogator] Using PyTorch backend.")
        else:
            self._net_type = "numpy"
            self._numpy_net = NumpyAdversarialNet()
            logger.info("[AdversarialInterrogator] Using NumPy backend.")

        self._calibration = calibration or CalibrationState()
        self._socratic    = SocraticGuide(on_prompt=on_prompt)
        
        # Thêm thuộc tính causal_graph_tracer
        self.causal_graph_tracer = CausalGraphTracer()
        
        self._async_loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock        = threading.Lock()

        # Telemetry
        self._total_queries:     int   = 0
        self._total_violations:  int   = 0
        self._last_p_violation:  float = 0.0
        self._last_rca:          str   = "none"

        # Async event loop for dispatching Socratic guidance coroutines
        self._loop_thread: Optional[threading.Thread] = None
        self._start_async_loop()

        logger.info("[AdversarialInterrogator] Initialised. Threshold=%.2f.",
                    self._calibration.threshold)

    # ------------------------------------------------------------------
    # Async Event Loop Management
    # ------------------------------------------------------------------

    def _start_async_loop(self) -> None:
        """Start a dedicated daemon thread running an asyncio event loop for async guidance."""
        self._async_loop = asyncio.new_event_loop()

        def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        self._loop_thread = threading.Thread(
            target=_run_loop,
            args=(self._async_loop,),
            daemon=True,
            name="AdversarialInterrogator-AsyncLoop",
        )
        self._loop_thread.start()
        logger.debug("[AdversarialInterrogator] Async event loop started on daemon thread.")

    def _dispatch_socratic(self, violation_type: str, context: Dict[str, Any]) -> None:
        """Schedule a Socratic guidance coroutine on the dedicated async loop."""
        if self._async_loop is None or not self._async_loop.is_running():
            logger.warning("[AdversarialInterrogator] Async loop not available; skipping Socratic dispatch.")
            return
        future = asyncio.run_coroutine_threadsafe(
            self._socratic.guide(violation_type, context),
            self._async_loop,
        )
        # Non-blocking: we don't await the result here
        future.add_done_callback(lambda f: None)

    # ------------------------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------------------------

    def _build_feature_vector(self, math_state: Dict[str, Any]) -> np.ndarray:
        """
        Extract and normalise the 12-dimensional input feature vector
        from the raw math_state dict published by MatrixSolver.
        """
        E_total = float(math_state.get("E_total", 0.0))
        E_max   = float(math_state.get("E_max",   1e6))
        M_total = float(math_state.get("M_total", 0.0))
        M_max   = float(math_state.get("M_max",   5e4))
        prey    = float(math_state.get("prey",    100.0))
        pred    = float(math_state.get("predator", 20.0))
        v_max   = float(math_state.get("v_max",    10.0))
        e_burn  = float(math_state.get("e_burn_rate", 0.1))
        lorenz  = math_state.get("lorenz", (0.1, 0.0, 0.0))
        decay   = float(math_state.get("decay_trigger", 0))
        viols   = math_state.get("violations", {})

        pop_sum = prey + pred + 1e-9
        vc      = sum(1 for v in viols.values() if v)

        feat = np.array([
            np.clip(E_total / max(E_max, 1e-9), 0.0, 2.0),
            np.clip(M_total / max(M_max, 1e-9), 0.0, 2.0),
            prey    / pop_sum,
            pred    / pop_sum,
            np.clip(v_max / 1000.0, 0.0, 1.0),
            np.clip(e_burn / 100.0, 0.0, 1.0),
            float(np.tanh(lorenz[0] / 30.0)),
            float(np.tanh(lorenz[1] / 30.0)),
            float(np.tanh(lorenz[2] / 50.0)),
            decay,
            float(prey < 1.0 or pred < 1.0),
            np.clip(vc / 6.0, 0.0, 1.0),
        ], dtype=np.float32)

        # Final NaN / Inf guard on feature vector itself
        feat = np.where(np.isnan(feat) | np.isinf(feat), 0.0, feat)
        return feat

    # ------------------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------------------

    def _run_network(self, feature_vec: np.ndarray) -> Tuple[float, float]:
        """
        Execute the network forward pass.
        Returns (P_valid, P_violation) as Python floats.
        """
        if self._net_type == "torch" and _TORCH_AVAILABLE:
            with torch.no_grad():
                t = torch.from_numpy(feature_vec).float()
                out = self._torch_net(t).numpy()
            return float(out[0]), float(out[1])
        else:
            out = self._numpy_net.forward(feature_vec)
            return float(out[0]), float(out[1])

    # ------------------------------------------------------------------
    # Root Cause Analysis
    # ------------------------------------------------------------------

    def _root_cause_analysis(
        self,
        violations:  Dict[str, bool],
        ltspice_snr: Optional[float] = None,
    ) -> str:
        """
        Perform Root Cause Analysis to classify the origin of a violation.
        Returns a string label.
        """
        active = [k for k, v in violations.items() if v]
        if not active:
            return "none"

        causes = []
        if "S4_speed_violation" in active and ltspice_snr is not None and ltspice_snr < 10.0:
            causes.append("electrical_noise")
        if "S5_lorenz_divergence" in active:
            causes.append("mathematical_singularity")
        if "S3_prey_extinction" in active or "S3_predator_collapse" in active:
            causes.append("population_collapse")
        if "S1_energy_overflow" in active or "S2_mass_overflow" in active:
            causes.append("resource_ceiling_breach")

        if not causes:
            causes = ["unknown_violation"]

        return "compound" if len(causes) > 1 else causes[0]

    # ------------------------------------------------------------------
    # Primary Interrogate Method
    # ------------------------------------------------------------------

    def interrogate(
        self,
        math_state:  Dict[str, Any],
        ltspice_snr: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Run a full interrogation cycle on the current constraint state.
        Now includes Causal Graph Tracing.
        """
        with self._lock:
            self._total_queries += 1

            # 1. Feature extraction
            feat = self._build_feature_vector(math_state)

            # 2. Network forward pass
            p_valid, p_viol = self._run_network(feat)
            self._last_p_violation = p_viol

            # 3. Threshold comparison
            threshold    = self._calibration.threshold
            is_violation = p_viol > threshold

            # 4. Record activation for calibration
            self._calibration.record_activation(is_violation)

            # 5. Dynamic Boundary Calibration
            cal_direction = self._calibration.calibrate()

            violations  = math_state.get("violations", {})
            active_viols = [k for k, v in violations.items() if v]

            causal_chain = "None"

            if is_violation:
                self._total_violations += 1

                # 6. Root Cause Analysis
                rca = self._root_cause_analysis(violations, ltspice_snr)
                self._last_rca = rca
                
                # BỔ SUNG: Dựng chuỗi hệ quả Causal Graph
                causal_chain = self.causal_graph_tracer.trace(math_state, active_viols, rca)

                logger.warning(
                    "[AdversarialInterrogator] VIOLATION DETECTED | "
                    "P_viol=%.4f > thr=%.3f | RCA=%s | active=%s",
                    p_viol, threshold, rca, active_viols
                )
                logger.warning("[Causal Chain] %s", causal_chain)

                # 7. Socratic guidance dispatch (async, non-blocking)
                primary_viol = active_viols[0] if active_viols else "default"
                context = {
                    "E_max":   math_state.get("E_max",    1e6),
                    "M_total": math_state.get("M_total",  0.0),
                    "M_max":   math_state.get("M_max",    5e4),
                    "v_max":   math_state.get("v_max",    10.0),
                    "bw":      ltspice_snr or 0.0,
                }
                self._dispatch_socratic(primary_viol, context)

                # 8. Reset hint escalation for resolved violations (graceful re-entry)
                for vt in list(self._socratic._hint_history.keys()):
                    if vt not in active_viols:
                        self._socratic.reset_hint_level(vt)
            else:
                rca = "none"
                self._last_rca = "none"

            result = {
                "P_valid":         p_valid,
                "P_violation":     p_viol,
                "is_violation":    is_violation,
                "threshold":       threshold,
                "rca":             rca,
                "active_viols":    active_viols,
                "causal_chain":    causal_chain,    # Đã thêm vào Output Dict
                "calibration_dir": cal_direction,
                "tick":            math_state.get("tick", 0),
                "total_queries":   self._total_queries,
                "total_violations":self._total_violations,
            }

            logger.debug(
                "[AdversarialInterrogator] P_valid=%.4f P_viol=%.4f | "
                "violation=%s | cal=%s | threshold=%.3f",
                p_valid, p_viol, is_violation, cal_direction, threshold
            )

            return result

    # ------------------------------------------------------------------
    # Manual Threshold Override (for diagnostics / Telemetry RCA)
    # ------------------------------------------------------------------

    def set_threshold(self, value: float) -> None:
        """Manually override the current rejection threshold (0.0–1.0)."""
        clamped = float(np.clip(value, 0.01, 0.99))
        with self._lock:
            self._calibration.threshold = clamped
        logger.info("[AdversarialInterrogator] Threshold manually set to %.3f.", clamped)

    def get_stats(self) -> Dict[str, Any]:
        """Return diagnostic statistics."""
        with self._lock:
            return {
                "total_queries":       self._total_queries,
                "total_violations":    self._total_violations,
                "violation_rate":      (self._total_violations / max(self._total_queries, 1)),
                "current_threshold":   self._calibration.threshold,
                "fp_rate_estimate":    self._calibration.current_fp_rate(),
                "last_p_violation":    self._last_p_violation,
                "last_rca":            self._last_rca,
                "network_backend":     self._net_type,
            }

    def shutdown(self) -> None:
        """Stop the async event loop gracefully."""
        if self._async_loop and self._async_loop.is_running():
            self._async_loop.call_soon_threadsafe(self._async_loop.stop)
        logger.info("[AdversarialInterrogator] Shut down.")

    def __repr__(self) -> str:
        return (f"<AdversarialInterrogator "
                f"backend={self._net_type} "
                f"threshold={self._calibration.threshold:.3f} "
                f"queries={self._total_queries} "
                f"violations={self._total_violations}>")
    