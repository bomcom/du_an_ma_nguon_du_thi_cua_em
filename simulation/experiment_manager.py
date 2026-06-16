"""
Filename: simulation/experiment_manager.py
Description: EXPERIMENT_MANAGER — Phân hệ quản lý giả thuyết khoa học (Hypothesis Manager).
             Thiết lập, cô lập và điều phối các luồng tính toán song song (A/B Testing) 
             cho nhiều mô hình thực tại, thu thập viễn trắc luồng chéo an toàn (Thread-Safe).
Author: Chuyên gia phần mềm AI/Simulation
"""

import time
import logging
import threading
import copy
from queue import Queue
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

logger = logging.getLogger("ExperimentManager")

@dataclass
class ExperimentConfig:
    """Cấu hình bộ gen môi trường và ràng buộc tiên đề của một thí nghiệm."""
    experiment_id: str
    description: str
    gravity: float = 9.81
    energy_burn_rate: float = 5.5
    max_energy_ceiling: float = 1200.0
    initial_population: int = 10
    mutation_rate: float = 0.05
    hardware_bandwidth: float = 25000.0  # Kết nối từ LTspice

@dataclass
class TelemetrySnapshot:
    """Bản ghi dữ liệu viễn trắc an toàn luồng phục vụ vẽ đồ họa HUD và XAI."""
    experiment_id: str
    timestamp: float
    current_fps: float
    entity_count: int
    total_energy: float
    total_mass: float
    violation_count: int
    causal_history: List[str] = field(default_factory=list)
    entity_positions: List[Dict[str, Any]] = field(default_factory=list)


class ExperimentWorkerThread(threading.Thread):
    """
    Luồng công nhân cô lập (Isolated Worker Thread).
    Vận hành một vòng lặp logic toán/vật lý khép kín cho duy nhất một thế giới giả thuyết.
    """
    def __init__(self, 
                 config: ExperimentConfig, 
                 registry_template: Any, 
                 ecosystem_class: Any, 
                 perception_engine: Any, 
                 interrogator: Any,
                 global_ui_queue: Queue):
        super().__init__()
        self.config = config
        self.name = f"ExpWorker-{config.experiment_id}"
        self.daemon = True
        self._running = False
        
        # Tạo bản sao sâu (Deep Copy) hệ thống ECS Registry để cô lập bộ nhớ hoàn toàn giữa các thế giới
        self.registry = copy.deepcopy(registry_template)
        
        # Khởi tạo cục bộ phân hệ sinh thái thủ tục dựa trên thực tại cô lập này
        self.ecosystem = ecosystem_class(self.registry, width=1000, height=1000, seed=42)
        self.perception_engine = perception_engine
        self.interrogator = interrogator
        self.global_ui_queue = global_ui_queue
        
        # Hệ thống khóa cục bộ và bộ đệm hoán đổi (Double-Buffering) tránh Race Condition khi UI đọc dữ liệu
        self.telemetry_lock = threading.Lock()
        self.latest_snapshot: Optional[TelemetrySnapshot] = None
        self.causal_logs: List[str] = []

    def stop(self):
        self._running = False

    def log_causal_chain(self, event_msg: str):
        """Ghi vết chuỗi nhân quả phục vụ tính giải thích được (Explainability Layer)."""
        timestamp = time.time()
        formatted_msg = f"[{time.strftime('%H:%M:%S', time.localtime(timestamp))}] {event_msg}"
        self.causal_logs.append(formatted_msg)
        if len(self.causal_logs) > 5:
            self.causal_logs.pop(0)

    def run(self):
        logger.info(f"Kích hoạt thành công luồng tính toán độc lập cho thế giới: {self.config.experiment_id}")
        self._running = True
        last_time = time.time()
        
        # Cấu hình lại các thực thể ban đầu dựa theo thông số thí nghiệm riêng biệt
        with self.registry.lock:
            # Thiết lập trọng lượng vật lý ban đầu phụ thuộc vào biến số môi trường (Ví dụ: Trọng lực ảnh hưởng khối cơ)
            for eid in list(self.registry.entities):
                mass_factor = self.config.gravity / 9.81
                self.registry.set_component_attr(eid, "Mass", "mass_kg", 70.0 * mass_factor)

        self.log_causal_chain(f"Thế giới được khởi tạo với g={self.config.gravity}, BW={self.config.hardware_bandwidth}Hz")

        while self._running:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            if dt <= 0:
                dt = 0.016
            
            # 1. Đồng bộ hóa cấu hình phần cứng giả lập vào trạng thái cục bộ
            hardware_state = {"bandwidth": self.config.hardware_bandwidth}
            
            # 2. Xây dựng bản đồ trạng thái vĩ mô (Math State Map) cục bộ cho riêng thế giới này
            math_state = {
                "E_total": 0.0,
                "E_max": self.config.max_energy_ceiling,
                "M_total": 0.0,
                "M_max": 5000.0 * (self.config.gravity / 9.81),
                "prey": 50.0,
                "predator": 10.0,
                "v_max": self.config.hardware_bandwidth / 1000.0,
                "e_burn_rate": self.config.energy_burn_rate,
                "lorenz": (12.5, 18.2, 35.1),
                "decay_trigger": 0,
                "violations": {}
            }
            
            # Run Ecosystem logic
            self.ecosystem.Ecosystem_Tick(dt, math_state)
            
            # 3. KẾT NỐI TRUNG GIAN ĐA LUỒNG: Đẩy dữ liệu qua ML_PERCEPTION_ENGINE
            math_state["ml_features"] = self.perception_engine.extract_feature_vector(
                registry=self.registry,
                math_state=math_state,
                hardware_state=hardware_state
            )
            
            # 4. THẨM ĐỊNH ADVERSARIAL: Kiểm tra tính nhất quán logic của thế giới này
            # Ghi đè hàm callback của Interrogator để định hướng Prompt về hàng đợi UI tổng
            is_valid = self.interrogator.interrogate(math_state)
            if not is_valid:
                self.log_causal_chain("CẢNH BÁO: Phát hiện vi phạm hệ tiên đề nhân quả!")
            
            # 5. TRÍCH XUẤT VIỄN TRẮC AN TOÀN (DOUBLE BUFFERING)
            entities_snapshot_data = []
            with self.registry.lock:
                for eid in list(self.registry.entities):
                    tf = self.registry.get_component_snapshot(eid, "Transform")
                    entities_snapshot_data.append({
                        "id": eid, "x": tf.get("x", 0.0), "y": tf.get("y", 0.0)
                    })
                    
            # Đóng gói Snapshot mới và dùng Lock bảo vệ để luồng chính Pygame rút dữ liệu bất cứ lúc nào
            with self.telemetry_lock:
                self.latest_snapshot = TelemetrySnapshot(
                    experiment_id=self.config.experiment_id,
                    timestamp=current_time,
                    current_fps=1.0 / dt if dt > 0 else 60.0,
                    entity_count=len(entities_snapshot_data),
                    total_energy=float(math_state["E_total"]),
                    total_mass=float(math_state["M_total"]),
                    violation_count=len(math_state["violations"]),
                    causal_history=list(self.causal_logs),
                    entity_positions=entities_snapshot_data
                )
                
            # Khống chế chu kỳ tính toán logic tần số cao (~60 Hz)
            time.sleep(0.016)


class ExperimentManager:
    """
    Trục quản lý luồng trung gian (Orchestrator Hub).
    Khởi tạo, lưu trữ và đồng bộ hóa thông tin giữa các luồng thí nghiệm song song.
    """
    def __init__(self, registry_template: Any, ecosystem_class: Any, perception_engine: Any, interrogator: Any, global_ui_queue: Queue):
        self.registry_template = registry_template
        self.ecosystem_class = ecosystem_class
        self.perception_engine = perception_engine
        self.interrogator = interrogator
        self.global_ui_queue = global_ui_queue
        
        # Từ điển lưu trữ các thực thể luồng công nhân đang chạy ngầm
        self.active_workers: Dict[str, ExperimentWorkerThread] = {}
        logger.info("Khởi tạo EXPERIMENT_MANAGER hoàn tất. Hệ thống sẵn sàng phân tách đa luồng.")

    def register_and_start_experiment(self, config: ExperimentConfig):
        """Khai sinh một thế giới mô phỏng mới và đẩy nó vào một luồng phần cứng riêng biệt."""
        if config.experiment_id in self.active_workers:
            logger.warning(f"Thí nghiệm {config.experiment_id} đang chạy. Bỏ qua yêu cầu.")
            return

        worker = ExperimentWorkerThread(
            config=config,
            registry_template=self.registry_template,
            ecosystem_class=self.ecosystem_class,
            perception_engine=self.perception_engine,
            interrogator=self.interrogator,
            global_ui_queue=self.global_ui_queue
        )
        self.active_workers[config.experiment_id] = worker
        worker.start()
        logger.info(f"Đã kích hoạt thành công tiến trình song song cho mã cấu hình: {config.experiment_id}")

    def get_all_telemetry_snapshots(self) -> Dict[str, TelemetrySnapshot]:
        """Luồng chính gọi hàm này để lấy ảnh chụp dữ liệu viễn trắc an toàn từ tất cả các thế giới."""
        snapshots = {}
        for exp_id, worker in self.active_workers.items():
            with worker.telemetry_lock:
                if worker.latest_snapshot is not None:
                    snapshots[exp_id] = copy.copy(worker.latest_snapshot)
        return snapshots

    def shutdown_all(self):
        """Tắt an toàn (Graceful Shutdown) toàn bộ luồng ngầm khi dừng chương trình."""
        logger.info("Đang phát lệnh dừng toàn bộ các luồng thí nghiệm song song...")
        for worker in self.active_workers.values():
            worker.stop()
        for worker in self.active_workers.values():
            worker.join(timeout=1.0)
        logger.info("Toàn bộ các luồng công nhân thí nghiệm đã giải phóng tài nguyên.")

        