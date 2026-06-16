import threading
import time
import logging
from collections import deque
from typing import Any, Callable, Dict, Optional, List

# Cấu hình logging
logger = logging.getLogger("SystemDiagnosticsCore")

class TelemetryCore:
    """
    Phân hệ Vi viễn trắc và Chẩn đoán độc lập (Independent Diagnostics & Telemetry).
    Vận hành trên một luồng riêng biệt (Isolated Thread), thu thập dữ liệu thời gian thực
    bằng cơ chế Dynamic Registry (không hard-code). Hỗ trợ song song Chế độ Vĩ mô và Chế độ Cá thể.
    """

    def __init__(self, history_size: int = 600, poll_rate_hz: float = 30.0):
        # Bộ đệm lưu trữ dữ liệu theo thời gian (Hỗ trợ RCA Backtrace khi sập hệ thống)
        # 600 frames @ 30Hz = Lưu trữ 20 giây lịch sử gần nhất
        self.history_size = history_size
        self.poll_rate = 1.0 / poll_rate_hz
        
        self.global_history = deque(maxlen=history_size)
        self.entity_history = deque(maxlen=history_size)

        # CƠ CHẾ ĐĂNG KÝ ĐỘNG (Dynamic Registry)
        # Lưu trữ các hàm (callbacks) để trích xuất dữ liệu thay vì gán cứng biến số
        self._global_metrics_registry: Dict[str, Callable[[], Any]] = {}
        
        # Hàm callback gọi đến ECS để trích xuất toàn bộ dữ liệu của một Entity ID
        self._ecs_query_callback: Optional[Callable[[int], Dict[str, Any]]] = None

        # TRẠNG THÁI TIÊU ĐIỂM (Focus State)
        self._focused_entity_id: Optional[int] = None
        
        # Threading & Thread Safety
        self._lock = threading.Lock()
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None

    # =====================================================================
    # 1. API ĐĂNG KÝ DỮ LIỆU ĐỘNG (Dành cho các module khác gọi tới)
    # =====================================================================

    def register_global_metric(self, metric_name: str, fetch_callback: Callable[[], Any]) -> None:
        """
        Đăng ký một thông số vĩ mô mới vào hệ thống viễn trắc.
        Ví dụ: telemetry.register_global_metric("LTSpice_SNR", parser.get_snr)
        """
        with self._lock:
            self._global_metrics_registry[metric_name] = fetch_callback
            logger.info(f"[Telemetry] Đã đăng ký thông số vĩ mô mới: {metric_name}")

    def bind_ecs_interface(self, ecs_query_func: Callable[[int], Dict[str, Any]]) -> None:
        """
        Kết nối Telemetry với Hệ thống ECS. 
        Truyền vào một hàm có khả năng nhận Entity ID và trả về toàn bộ Component (Dict).
        Đảm bảo các thuộc tính như "Mana" do NLP tạo ra sẽ tự động được lấy.
        """
        with self._lock:
            self._ecs_query_callback = ecs_query_func
            logger.info("[Telemetry] Đã liên kết giao diện truy xuất ECS Động.")

    # =====================================================================
    # 2. CƠ CHẾ CHUYỂN ĐỔI NGỮ CẢNH TƯƠNG TÁC (Dual Data Harvesting Modes)
    # =====================================================================

    def set_focus_entity(self, entity_id: int) -> None:
        """
        Người dùng click vào vật thể/NPC -> Chuyển sang Chế độ Tiêu điểm (Entity Profiling Mode).
        """
        with self._lock:
            self._focused_entity_id = entity_id
            self.entity_history.clear() # Reset lịch sử cá thể cũ
            logger.info(f"[Telemetry] Chuyển tiêu điểm giám sát sang Entity ID: {entity_id}")

    def clear_focus(self) -> None:
        """
        Người dùng click ra ngoài -> Trở về Chế độ Vĩ mô (Global Macro-State Mode).
        """
        with self._lock:
            self._focused_entity_id = None
            self.entity_history.clear()
            logger.info("[Telemetry] Đã hủy tiêu điểm. Trở về giám sát hệ thống Vĩ mô.")

    # =====================================================================
    # 3. LUỒNG THỰC THI NỀN (Isolated Background Thread)
    # =====================================================================

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._monitor_thread = threading.Thread(target=self._telemetry_loop, daemon=True, name="TelemetryThread")
        self._monitor_thread.start()
        logger.info("[Telemetry] Tiến trình Vi viễn trắc đã khởi chạy độc lập.")

    def stop(self) -> None:
        self._running = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join()
        logger.info("[Telemetry] Tiến trình Vi viễn trắc đã đóng.")

    def _telemetry_loop(self) -> None:
        """
        Vòng lặp thu hoạch dữ liệu (Data Harvesting Loop).
        Chạy ngầm liên tục mà không làm ảnh hưởng đến Tầng Lõi Toán hay Đồ họa.
        """
        while self._running:
            start_time = time.perf_counter()

            current_global_state: Dict[str, Any] = {"timestamp": time.time()}
            current_entity_state = None

            with self._lock:
                # 3.1. Thu hoạch dữ liệu Vĩ Mô (Global State)
                for name, fetch_func in self._global_metrics_registry.items():
                    try:
                        current_global_state[name] = fetch_func()
                    except Exception as e:
                        current_global_state[name] = f"Error: {str(e)}"
                
                self.global_history.append(current_global_state)

                # 3.2. Thu hoạch dữ liệu Vi Mô (Nếu có Entity đang được Focus)
                if self._focused_entity_id is not None and self._ecs_query_callback is not None:
                    try:
                        # Gọi callback kéo TOÀN BỘ dữ liệu của Entity (bao gồm cả Mana, Health, XYZ...)
                        current_entity_state = self._ecs_query_callback(self._focused_entity_id)
                        current_entity_state["timestamp"] = time.time()
                        self.entity_history.append(current_entity_state)
                    except Exception as e:
                        logger.warning(f"[Telemetry] Lỗi truy xuất Entity {self._focused_entity_id}: {e}")
                        self._focused_entity_id = None # Mất focus nếu Entity chết/biến mất

            # Giữ nhịp độ viễn trắc độc lập
            elapsed = time.perf_counter() - start_time
            sleep_time = max(0.0, self.poll_rate - elapsed)
            time.sleep(sleep_time)

    # =====================================================================
    # 4. API TRUY XUẤT DỮ LIỆU ĐỂ HIỂN THỊ (Dành cho Tầng Giao Diện / World View)
    # =====================================================================

    def get_current_dashboard_data(self) -> Dict[str, Any]:
        """
        Trả về Snapshot dữ liệu hiện tại để vẽ lên màn hình UI.
        Tự động nhận biết ngữ cảnh (Có trả về Entity Data hay không).
        """
        with self._lock:
            latest_global = self.global_history[-1] if self.global_history else {}
            latest_entity = self.entity_history[-1] if self.entity_history and self._focused_entity_id is not None else None
            
            return {
                "mode": "ENTITY_FOCUS" if latest_entity else "GLOBAL_MACRO",
                "focused_id": self._focused_entity_id,
                "global_metrics": latest_global,
                "entity_metrics": latest_entity
            }

    def get_rca_backtrace(self, frames: int = 60) -> List[Dict[str, Any]]:
        """
        Truy xuất dữ liệu lịch sử để Root Cause Analysis (Chẩn đoán nguyên nhân gốc rễ).
        Sử dụng khi Hệ thống báo lỗi (Adversarial Net kích hoạt).
        """
        with self._lock:
            history_list = list(self.global_history)
            return history_list[-frames:] if len(history_list) >= frames else history_list
        