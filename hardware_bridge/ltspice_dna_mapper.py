"""
Filename: hardware_bridge/ltspice_dna_mapper.py
Description: LTSPICE_DNA_MAPPER — Phân hệ Ánh xạ DNA Phần cứng.
             Điều phối tiến trình chạy LTspice ẩn, bóc tách tín hiệu vi mạch (BW, SNR)
             và chuyển hóa (Map) thành các tham số cấu hình hệ sinh thái (Hardware DNA).
Author: Chuyên gia phần mềm AI/Simulation
"""

import os
import logging
from typing import Dict
from dataclasses import dataclass

# Giả định module phân tích thô đã tồn tại ở tầng dưới
from hardware_bridge.ltspice_parser import LTSpiceParser

logger = logging.getLogger("LTSpiceDNAMapper")

@dataclass
class HardwareDNA:
    """Cấu trúc dữ liệu bất biến chứa DNA phần cứng sau khi ánh xạ."""
    raw_bandwidth_hz: float = 20000.0
    raw_snr_db: float = 40.0
    
    # --- CÁC THAM SỐ ÁNH XẠ (MAPPED PARAMETERS) ---
    v_max_theoretical: float = 20.0       # Tốc độ di chuyển tối đa của Tác nhân (m/s)
    hardware_burn_penalty: float = 1.0    # Hệ số phạt năng lượng do nhiễu mạch
    ai_compute_budget: int = 100          # Độ sâu/Số nơ-ron tối đa được phép kích hoạt
    is_valid: bool = True                 # Trạng thái mạch (Mạch lỗi -> False)

class LTSpiceDNAMapper:
    """
    Trục điều phối phần cứng trung gian.
    Tách biệt hoàn toàn luồng I/O của hệ điều hành khỏi vòng lặp tính toán Lõi toán học.
    """
    
    def __init__(self, ltspice_executable_path: str, timeout_seconds: float = 15.0):
        # Khởi tạo bộ phân tích thô ở tầng dưới
        self.parser = LTSpiceParser(
            ltspice_executable_path=ltspice_executable_path, 
            timeout_seconds=timeout_seconds
        )
        logger.info("Khởi tạo LTSPICE_DNA_MAPPER thành công. Cầu nối phần cứng đã sẵn sàng.")

    def _map_raw_to_dna(self, raw_metrics: Dict[str, float]) -> HardwareDNA:
        """
        Lõi toán học ánh xạ: Biến đổi các con số điện tử thành quy luật vật lý.
        Đây là nơi thể hiện định lý "Phần cứng định đoạt Trí tuệ".
        """
        bw = raw_metrics.get("bandwidth", 20000.0)
        snr = raw_metrics.get("snr", 40.0)
        
        # 1. Ánh xạ Vận tốc: Băng thông càng rộng, xung nhịp truyền tải cao, NPC chạy càng nhanh
        # Công thức: v_max = BW * 0.001 (Điều chỉnh hệ số tùy môi trường)
        mapped_v_max = max(1.0, bw * 0.001)
        
        # 2. Ánh xạ Hệ số Tiêu hao: Mạch nhiễu (SNR thấp) sinh ra nhiệt và hao phí điện năng
        # Công thức: Penalty = 100 / (SNR + 1). SNR càng cao, Penalty càng tiến về 0.
        mapped_penalty = 100.0 / (snr + 1.0) if snr > 0 else 10.0
        
        # 3. Ánh xạ Ngân sách Trí tuệ (AI Compute Budget): Mạch sạch cho phép chạy mạng nơ-ron sâu hơn
        # Ví dụ: Mạch SNR > 40dB cho phép 128 nơ-ron, dưới 20dB chỉ cho phép 32 nơ-ron.
        if snr > 40.0:
            ai_budget = 128
        elif snr > 20.0:
            ai_budget = 64
        else:
            ai_budget = 32

        dna = HardwareDNA(
            raw_bandwidth_hz=bw,
            raw_snr_db=snr,
            v_max_theoretical=mapped_v_max,
            hardware_burn_penalty=mapped_penalty,
            ai_compute_budget=ai_budget,
            is_valid=True
        )
        
        logger.debug(f"[DNA Mapped] V_max: {dna.v_max_theoretical:.1f} m/s | "
                     f"Penalty: {dna.hardware_burn_penalty:.2f} | AI Budget: {dna.ai_compute_budget}")
        return dna

    async def pipeline_execute(self, asc_file_path: str) -> HardwareDNA:
        """
        Đường ống tích hợp (Pipeline) toàn phần được di dời từ lớp Parser thô.
        Chạy mô phỏng ẩn -> Đọc file -> Bóc tách -> Ánh xạ thành DNA.
        """
        logger.info(f"[DNA Pipeline] Bắt đầu giải mã sơ đồ mạch: {os.path.basename(asc_file_path)}")
        
        # 1. Gọi tiến trình con bất đồng bộ (Tránh nghẽn luồng)
        log_path = await self.parser.run_simulation_async(asc_file_path)
        
        if log_path:
            # 2. Đọc file bất đồng bộ và trích xuất dữ liệu thô
            raw_metrics = await self.parser.parse_hardware_dna(log_path)
            
            # 3. Ánh xạ thành cấu trúc DNA Sinh tồn
            mapped_dna = self._map_raw_to_dna(raw_metrics)
            
            logger.info("[DNA Pipeline] Ánh xạ Phần cứng thành công. Trạng thái: HỢP LỆ.")
            return mapped_dna
            
        logger.warning("[DNA Pipeline] Mô phỏng thất bại. Kích hoạt DNA Cấu hình An toàn Mặc định.")
        # Fallback về cấu hình an toàn nếu LTspice báo lỗi mạch
        return HardwareDNA(is_valid=False)

# =====================================================================
# HƯỚNG DẪN TÍCH HỢP VÀO LUỒNG ĐỒNG BỘ TRONG MAIN.PY
# =====================================================================
"""
# Tích hợp vào hàm hardware_bridge_loop() của file main.py

async def run_periodic_simulation():
    mapper = LTSpiceDNAMapper(ltspice_executable_path="C:\\LTspice\\LTspice.exe")
    
    while self.running:
        # Gọi hàm Pipeline_Execute đã được tái cấu trúc
        dna_result = await mapper.pipeline_execute(asc_circuit_path)
        
        # Khóa luồng an toàn để đẩy dữ liệu DNA vào bộ nhớ chung
        with self.hardware_lock:
            self.hardware_dna["bandwidth"] = dna_result.raw_bandwidth_hz
            self.hardware_dna["snr_db"] = dna_result.raw_snr_db
            self.hardware_dna["v_max"] = dna_result.v_max_theoretical
            self.hardware_dna["burn_penalty"] = dna_result.hardware_burn_penalty
            self.hardware_dna["ai_budget"] = dna_result.ai_compute_budget
            
        await asyncio.sleep(4.0)
"""
