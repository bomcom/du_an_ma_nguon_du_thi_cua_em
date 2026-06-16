# runtime_orchestrator.py
"""

Tên tệp: runtime_orchestrator.py
Giai đoạn: 5E - Điều phối Thời gian Thực

Bản quyền © 2026  Phạm Hồng Hải Đăng.

Mọi quyền được bảo lưu.

Tài liệu này thuộc sở hữu trí tuệ của Phạm Hồng Hải Đăng.

"""

import asyncio
import logging

from hardware_bridge.ltspice_parser import LTSpiceParser
from simulation.matrix_solver import MatrixSolver
from orchestrator import HybridSimulationApplication
from ai_core.formal_verifier import FormalVerifier
from monitoring.telemetry_core import TelemetryCore

logger = logging.getLogger("RuntimeOrchestrator")
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

class RuntimeOrchestrator:
    def __init__(
        self,
        ltspice_executable_path: str,
        use_torch: bool = False,
    ) -> None:
        self.parser = LTSpiceParser(ltspice_executable_path=ltspice_executable_path)
        self.solver = MatrixSolver()
        self.formal_verifier = FormalVerifier()
        self.telemetry = TelemetryCore()
        self.app = HybridSimulationApplication()
        self.world_state = {
            "tick_rate": 30.0,
            "v_max": 5.0,
            "energy_burn_rate": 1.5,
            "is_simulation_running": True,
        }
        self.current_circuit_path = "circuits/active_core_hardware.asc"
        self._running = False

    def start(self) -> None:
        logger.info("[RuntimeOrchestrator] Starting runtime orchestrator.")
        self._running = True
        self.app.start()

    def stop(self) -> None:
        if self._running:
            self._running = False
            self.app.stop()
            logger.info("[RuntimeOrchestrator] Runtime orchestration stopped.")

    async def run_async(self) -> None:
        self.start()
        try:
            while self._running:
                await asyncio.sleep(0.5)
        finally:
            self.stop()

if __name__ == "__main__":
    LTSPICE_EXE = "C:\\Users\\HaiDang\\AppData\\Local\\LTspice\\LTspice.exe"
    runtime = RuntimeOrchestrator(ltspice_executable_path=LTSPICE_EXE)
    runtime.start()
