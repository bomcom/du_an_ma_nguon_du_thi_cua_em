import os
import re
import asyncio
import logging
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LTSpiceDNAMapper")

class LTSpiceParser:
    def __init__(self, ltspice_executable_path: str, timeout_seconds: float = 10.0):
        self.ltspice_path = ltspice_executable_path
        self.timeout = timeout_seconds
        # Tối ưu regex để bắt linh hoạt hơn các khoảng trắng và ký tự đơn vị nếu có
        self.bw_regex = re.compile(r"bandwidth:\s+from\s+[\d\.]+\s+to\s+([\d\.e\+\-]+)", re.IGNORECASE)
        self.snr_regex = re.compile(r"snr:\s+.*=\s+([\d\.e\+\-]+)", re.IGNORECASE)

    async def run_simulation_async(self, asc_file_path: str) -> Optional[str]:
        if not os.path.exists(asc_file_path):
            logger.error(f"Không tìm thấy tệp sơ đồ mạch nguyên lý: {asc_file_path}")
            return None

        args = [self.ltspice_path, "-b", "-Run", asc_file_path]
        logger.info(f"Đang khởi chạy tiến trình nhánh LTSpice cho: {asc_file_path}")
        
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout)

            if process.returncode == 0:
                log_file_path = os.path.splitext(asc_file_path)[0] + ".log"
                return log_file_path
            else:
                logger.error(f"LTSpice lỗi (Mã: {process.returncode}): {stderr.decode(errors='ignore').strip()}")
                return None

        except asyncio.TimeoutError:
            logger.error(f"Mô phỏng vượt quá thời gian ({self.timeout}s). Đang cưỡng bức hủy và dọn dẹp...")
            if process:
                try:
                    process.kill()
                    await process.wait() # Đảm bảo dọn dẹp triệt để tiến trình con
                except Exception:
                    pass
            return None
        except Exception as e:
            logger.error(f"Lỗi hệ thống khi gọi tiến trình CLI LTSpice: {str(e)}")
            return None

    async def _read_file_with_fallback_encoding(self, file_path: str) -> str:
        """Hàm bổ trợ đọc file xử lý bẫy encoding của LTSpice log"""
        loop = asyncio.get_running_loop()
        
        def _read():
            # Thử đọc với UTF-16 LE trước (mặc định của LTSpice trên nhiều hệ thống)
            try:
                with open(file_path, "r", encoding="utf-16-le") as f:
                    return f.read()
            except UnicodeDecodeError:
                # Fallback về UTF-8 nếu UTF-16 thất bại
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
                    
        return await loop.run_in_executor(None, _read)

    async def parse_hardware_dna(self, log_file_path: str) -> Dict[str, float]:
        dna_metrics = {"bandwidth": 20000.0, "snr": 40.0}

        if not log_file_path or not os.path.exists(log_file_path):
            return dna_metrics

        logger.info(f"Đang phân rã dữ liệu động trong tệp: {log_file_path}")
        try:
            content = await self._read_file_with_fallback_encoding(log_file_path)

            bw_match = self.bw_regex.search(content)
            if bw_match:
                dna_metrics["bandwidth"] = float(bw_match.group(1))

            snr_match = self.snr_regex.search(content)
            if snr_match:
                dna_metrics["snr"] = float(snr_match.group(1))

            logger.info(f"Cập nhật DNA Phần cứng -> Bandwidth: {dna_metrics['bandwidth']} Hz, SNR: {dna_metrics['snr']} dB")
            return dna_metrics

        except Exception as e:
            logger.error(f"Thất bại trong việc bóc tách thông số tệp .log: {str(e)}")
            return dna_metrics

    async def Pipeline_Execute(self, asc_file_path: str) -> Dict[str, float]:
        log_path = await self.run_simulation_async(asc_file_path)
        if log_path:
            return await self.parse_hardware_dna(log_path)
        logger.warning("Quy trình mô phỏng thất bại. Sử dụng cấu hình DNA phần cứng an toàn mặc định.")
        return {"bandwidth": 20000.0, "snr": 40.0}
    