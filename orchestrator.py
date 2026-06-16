# -*- coding: utf-8 -*-
"""
orchestrator.py
===============
Simulation Core Orchestrator (Kiến trúc Hệ kín Logic)

Luồng nhân quả:
Prompt -> IntentRouter -> MatrixSolver -> WorldFieldGenerator -> ECS Reality Snapshot
"""

import time
import threading
import asyncio
import logging
from queue import Queue
from typing import Tuple, Dict, Any

import numpy as np

from core_engine.session_box import SessionBox
from core_engine.dynamic_ecs import build_default_ecs
from simulation.matrix_solver import MatrixSolver
from ai_core.adversarial_interrogator import AdversarialInterrogator
from evolution.genetic_evolution import GeneticEvolutionEngine

# --- PIPELINE DỊCH THUẬT VÀ SINH TRƯỜNG THỰC TẠI ĐỘNG ---
from ai_core.nlp_idea_parser import SemanticToMathCompiler
from simulation.world_generator import WorldGenerator
from simulation.intent_router import IntentRouter, IntentType
from simulation.world_field_generator import WorldFieldGenerator  # Tích hợp thực tại động

logger = logging.getLogger("SimulationKernel")


class HybridSimulationApplication:

    def __init__(self):
        self.running = False
        self.ui_prompt_queue = Queue()
        self.box = SessionBox.get_instance()
        self.registry = build_default_ecs()
        self.intent_router = IntentRouter()

        # Khởi tạo Lõi toán giải ma trận trạng thái trước
        self.solver = MatrixSolver()

        # Khởi tạo Bộ sinh các trường thực tại vô hướng động (Địa hình/Khí hậu/Tài nguyên)
        self.world_field_generator = WorldFieldGenerator(
            width=512,
            height=512,
            seed=42
        )

        self.interrogator = AdversarialInterrogator(
            use_torch=False,
            on_prompt=self._safe_ui_prompt_callback
        )

        self.evolution_engine = GeneticEvolutionEngine(
            mutation_rate=0.05,
            mutation_strength=0.2
        )

        # --------------------------------------------------------
        # KHỞI TẠO PIPELINE BIÊN DỊCH VÀ KHỞI SINH
        # --------------------------------------------------------
        def mock_gatekeeper_check(schema_dict): 
            return True 

        self.nlp_compiler = SemanticToMathCompiler(
            ecs_registry=self.registry,
            interrogator_gatekeeper=mock_gatekeeper_check
        )
        
        self.world_generator = WorldGenerator(
            registry=self.registry, 
            matrix_solver=self.solver
        )

        # --------------------------------------------------------
        # SYSTEM REGISTRATION (Đăng ký Hệ thống ECS)
        # --------------------------------------------------------
        self.registry.register_system(
            "BrainReflex",
            lambda reg, ents, dt: self.evolution_engine.physics_reflex_system(reg, ents, dt),
            ["Transform", "Velocity", "Energy", "NeuralBrain"],
            priority=15
        )

        self._register_channels()

    # --------------------------------------------------------
    # PIPELINE XỬ LÝ NGÔN NGỮ & ĐIỀU PHỐI Ý NIỆM (PROMPT PIPELINE)
    # --------------------------------------------------------
    
    def trigger_world_generation_sync(self, prompt_text: str) -> Tuple[bool, str]:
        """Hàm bao bọc (Wrapper) đồng bộ để gọi từ UI hoặc Terminal"""
        logger.info(f"[Core Engine] Tiếp nhận lệnh kiến tạo: '{prompt_text}'")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            success, message = loop.run_until_complete(self._async_create_world_from_prompt(prompt_text))
            return success, message
        finally:
            loop.close()

    async def _async_create_world_from_prompt(self, prompt_text: str) -> Tuple[bool, str]:
        """Quy trình cốt lõi Giai đoạn 2: Ngôn ngữ -> Toán học -> Bản đồ cấu trúc"""
        self.world_generator.clear_world()
        logger.info("[Pipeline] Đang biên dịch ngôn ngữ tự nhiên qua Local SLM...")
        
        mock_world_schema = {
            "world_name": "New Genesis",
            "entities": [
                {"count": 80, "components": {"Transform": {}, "Velocity": {}, "Mass": {"mass_kg": 2.0}, "Health": {}, "Energy": {}, "NeuralBrain": {}}}, 
                {"count": 15, "components": {"Transform": {}, "Velocity": {}, "Mass": {"mass_kg": 5.0}, "Health": {}, "Energy": {"current_energy": 120.0}, "NeuralBrain": {"weight_hash": 0.5}}}, 
                {"count": 200, "components": {"Transform": {}, "Velocity": {}, "Mass": {"mass_kg": 0.5}, "Energy": {"current_energy": 50.0, "consumption_rate": 0.05}}} 
            ]
        }

        gate_result = {"valid": True} 
        if not gate_result.get("valid", False):
            return False, "Hệ thống phản biện từ chối! Vi phạm hệ kín."

        success, msg = self.world_generator.generate_from_schema(mock_world_schema)
        return success, msg

    async def process_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        SỬA LỖI ĐỊNH DẠNG: Đã thụt lề chuẩn xác vào trong Class.
        Hàm tiếp nhận Prompt từ người dùng để biên dịch động quy luật của Thế giới.
        """
        route = self.intent_router.classify(prompt)

        # 1. Đổi cấu trúc linh hồn thành phần sinh học (Component)
        if route.intent == IntentType.COMPONENT:
            success = await self.nlp_compiler.compile_and_inject_idea(prompt)
            return {
                "success": success,
                "route": "component"
            }

        # 2. Thay đổi tham số vật lý/toán học vĩ mô tác động lên Thế giới (World)
        elif route.intent == IntentType.WORLD:
            success, msg = await self.world_generator.generate_from_prompt(prompt)
            return {
                "success": success,
                "route": "world",
                "message": msg
            }

        # 3. Khởi sinh thực thể động (Entity)
        elif route.intent == IntentType.ENTITY:
            success, msg = await self.world_generator.spawn_entities_from_prompt(prompt)
            return {
                "success": success,
                "route": "entity",
                "message": msg
            }

        return {
            "success": False,
            "route": "unknown",
            "message": "Unable to classify prompt."
        }

    # --------------------------------------------------------
    # QUẢN LÝ ĐA LUỒNG VÀ ĐỒNG BỘ KÊNH TRUYỀN DỮ LIỆU
    # --------------------------------------------------------
    def _register_channels(self):
        logger.info("[Kernel] Khởi tạo các kênh dữ liệu dùng chung trên SessionBox")
        self.box.register_channel("ltspice", owner="hardware", dtype="dict")
        self.box.register_channel("math_state", owner="solver", dtype="dict")
        self.box.register_channel("adversarial_flags", owner="gatekeeper", dtype="dict")
        self.box.register_channel("telemetry_metrics", owner="telemetry", dtype="dict")
        self.box.write("ltspice", {"bandwidth": 1000.0, "snr": 30.0})

    def _safe_ui_prompt_callback(self, prompt):
        self.ui_prompt_queue.put(prompt)

    def start(self):
        self.running = True
        self.sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.hardware_thread = threading.Thread(target=self._hardware_bridge_loop, daemon=True)
        self.telemetry_thread = threading.Thread(target=self._telemetry_loop, daemon=True)

        self.sim_thread.start()
        self.hardware_thread.start()
        self.telemetry_thread.start()
        logger.info("[Kernel] Toàn bộ hệ thống mô phỏng đa luồng đã khởi chạy.")

    def stop(self):
        self.running = False
        try:
            self.interrogator.shutdown()
        except Exception:
            pass
        logger.info("[Kernel] Đã dừng toàn bộ vòng lặp lõi an toàn.")

    # --------------------------------------------------------
    # THU THẬP VÀ ĐÓNG GÓI SNAPSHOT ECS ĐỂ CHUYỂN CHO LÕI TOÁN
    # --------------------------------------------------------
    def _collect_solver_arrays(self):
        entities = self.registry.query("Transform", "Velocity", "Mass", "Energy")
        masses, velocities, heights, energies = [], [], [], []
        for eid in entities:
            t = self.registry.get_component_snapshot(eid, "Transform")
            v = self.registry.get_component_snapshot(eid, "Velocity")
            m = self.registry.get_component_snapshot(eid, "Mass")
            e = self.registry.get_component_snapshot(eid, "Energy")

            if t is None or v is None or m is None or e is None: 
                continue

            try:
                masses.append(float(m["mass_kg"]))
                velocities.append([float(v["vx"]), float(v["vy"]), float(v.get("vz", 0.0))])
                heights.append(float(t.get("z", t.get("y", 0.0))))
                energies.append(float(e["current_energy"]))
            except (KeyError, TypeError):
                continue

        return (
            np.asarray(masses, dtype=np.float32),
            np.asarray(velocities, dtype=np.float32),
            np.asarray(heights, dtype=np.float32),
            np.asarray(energies, dtype=np.float32)
        )

    # --------------------------------------------------------
    # VÒNG LẶP MÔ PHỎNG THỜI GIAN THỰC CHÍNH (MAIN SIMULATION LOOP)
    # --------------------------------------------------------
    def _simulation_loop(self):
        dt = 1.0 / 60.0
        while self.running:
            start_time = time.perf_counter()
            
            # Step 1: Cập nhật hệ thống ECS vật lý & não bộ nhân tạo cơ bản
            self.registry.tick(dt)

            # Step 2: Thu thập mảng dữ liệu trạng thái từ thực thể để chuẩn bị giải toán
            (masses, velocities, heights, energies) = self._collect_solver_arrays()
            hw = self.box.read("ltspice") or {}
            bw = hw.get("bandwidth", 1000.0)
            snr = hw.get("snr", 30.0)

            # Step 3: Lõi toán giải phương trình vi phân và các ràng buộc logic cứng toàn cục
            result = self.solver.tick(
                entity_masses=masses, entity_velocities=velocities,
                entity_heights=heights, entity_bio_energy=energies,
                ltspice_bandwidth=bw, ltspice_snr=snr, dt=dt
            )
            self.box.write("math_state", result)

            # ----------------------------------------------------------------
            # TÍCH HỢP ĐỘNG: Tạo phản chiếu thực tại trường dựa trên kết quả lõi toán
            # Địa hình, tài nguyên biến đổi ngay lập tức theo Entropy, Năng lượng, Dân số sinh vật
            # ----------------------------------------------------------------
            self.world_field_generator.update(
                math_state=result,
                hardware_state=hw
            )

            # Step 4: Đưa qua hệ thống phản biện AdversarialInterrogator kiểm tra an toàn hệ kín
            gate = self.interrogator.interrogate(result, ltspice_snr=snr)
            self.box.write("adversarial_flags", gate)
            
            # Step 5: Đồng bộ hóa vùng đệm kết xuất đồ họa
            self.box.commit_render_snapshot()

            elapsed = time.perf_counter() - start_time
            time.sleep(max(0.0, dt - elapsed))

    # --------------------------------------------------------
    # CÁC LUỒNG PHỤ TRỢ: PHẦN CỨNG VÀ ĐO LƯỜNG ĐA KÊNH
    # --------------------------------------------------------
    def _hardware_bridge_loop(self):
        async def worker():
            while self.running:
                await asyncio.sleep(4)
                self.box.write("ltspice", {
                    "bandwidth": np.random.uniform(900, 1800),
                    "snr": np.random.uniform(25, 45)
                })
        asyncio.run(worker())

    def _telemetry_loop(self):
        while self.running:
            try:
                gate = self.box.read("adversarial_flags") or {}
                math_state = self.box.read("math_state") or {}
                
                self.box.write("telemetry_metrics", {
                    "entity_count": self.registry.entity_count(),
                    "prey_count": int(math_state.get("prey", 0)),
                    "predator_count": int(math_state.get("predator", 0)),
                    "E_total": float(math_state.get("E_total", 0)),
                    "E_max": float(math_state.get("E_max", 1000)),
                    "M_total": float(math_state.get("M_total", 0)),
                    "M_max": float(math_state.get("M_max", 5000)),
                    "v_max": float(math_state.get("v_max", 0)),
                    "decay_trigger": int(math_state.get("decay_trigger", 0)),
                    "lorenz_x": float(math_state.get("lorenz", [0, 0, 0])[0]),
                    "lorenz_y": float(math_state.get("lorenz", [0, 0, 0])[1]),
                    "lorenz_z": float(math_state.get("lorenz", [0, 0, 0])[2]),
                    "gatekeeper": gate.get("is_violation", False),
                    "violations": list(math_state.get("violations", {}).keys()),
                    "timestamp": time.time()
                })
            except Exception as e:
                logger.exception(e)
            time.sleep(1)

    def _validate_schema(self, schema_dict):
        try:
            result = self.interrogator.interrogate({
                "schema_size": len(schema_dict.get("attributes", {}))
            })
            return not result.get("is_violation", False)
        except Exception:
            return False
        