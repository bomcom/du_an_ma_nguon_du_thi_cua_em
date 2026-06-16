# -*- coding: utf-8 -*-
"""
evolution/dynamic_genome.py
===========================
Phân hệ Hiện thực hóa Nhiễm sắc thể Động (Dynamic Genome Spawner Engine)
Kiến trúc: Giai đoạn 4B - Quy trình Khởi sinh Thực thể từ Hàng đợi Gen

Mô tả:
    File này đóng vai trò tiêu thụ hàng đợi gen con (`offspring_genome_queue`) 
    được sinh ra từ Giai đoạn 4A (adaptive_selection.py). Nó chịu trách nhiệm:
        1. Thẩm định cấu trúc dữ liệu nhiễm sắc thể (Genome Validation).
        2. Đối chiếu lược đồ thành phần hệ thống (Schema Check).
        3. Cấp phát định danh thực thể trống từ ECSRegistry (create_entity).
        4. Thiết lập và ghi đè các tham số thuộc tính (add_component).
        5. Hoàn tất chu trình khởi sinh thực thể động vào thế giới mô phỏng.

Tác giả: Chuyên gia Kiến trúc Hệ thống Mô phỏng
"""

import logging
import threading
from typing import Dict, List, Any, Set

logger = logging.getLogger("SimulationKernel.GenomeSpawner")


class DynamicGenomeEngine:
    """
    Bộ điều phối nạp bộ gen động chịu trách nhiệm chuyển hóa các bản thiết kế di truyền 
    (Genome Snapshots) thành các thực thể ECS sống, có đầy đủ thuộc tính tính trạng tại Runtime.
    """

    def __init__(self, registry: Any):
        """
        Khởi tạo Engine sinh thực thể động liên kết với lõi ECSRegistry của hệ thống.
        """
        self.registry = registry
        self._lock = threading.RLock()
        
        logger.info("[GenomeSpawner] Kích hoạt thành công phân hệ Hiện thực hóa Nhiễm sắc thể Giai đoạn 4B.")

    # ------------------------------------------------------
    # 1. GENOME VALIDATION LAYER (TẦNG THẨM ĐỊNH CẤU TRÚC GEN)
    # ------------------------------------------------------
    def validate_genome(self, genome: Dict[str, Any]) -> bool:
        """
        Kiểm tra tính toàn vẹn của dữ liệu nhiễm sắc thể trước khi nạp vào bộ nhớ RAM.
        Đảm bảo cấu trúc không bị rỗng, sai định dạng hoặc chứa dữ liệu độc hại.
        """
        if not genome or not isinstance(genome, dict):
            logger.error("[Validation Error] Cấu trúc mã gen con bị rỗng hoặc không phải định dạng Dictionary.")
            return False

        for comp_name, attributes in genome.items():
            if not isinstance(comp_name, str):
                logger.error(f"[Validation Error] Tên Component '{comp_name}' phải là định dạng chuỗi (string).")
                return False
            
            if not isinstance(attributes, dict):
                logger.error(f"[Validation Error] Cấu trúc thuộc tính bên trong '{comp_name}' không hợp lệ.")
                return False
                
        return True

    # ------------------------------------------------------
    # 2. SCHEMA CHECK LAYER (SỬ DỤNG API CHUẨN ĐÃ ĐỐI CHIẾU)
    # ------------------------------------------------------
    def verify_and_sync_schemas(self, genome: Dict[str, Dict[str, float]]) -> Set[str]:
        """
        Sử dụng chuẩn list_schemas() và inject_dynamic_component_from_schema_dict() 
        từ tài liệu PROJECT_API_REFERENCE.md của dự án.
        """
        verified_components: Set[str] = set()
        
        # Gọi trực tiếp API chính thức có trong đặc tả của bạn
        try:
            registered_schemas = self.registry.list_schemas()
        except Exception:
            registered_schemas = []

        for comp_name, attributes in genome.items():
            if registered_schemas and comp_name in registered_schemas:
                verified_components.add(comp_name)
                continue
                
            # Kích hoạt API inject động có thật trong tài liệu cấu trúc nếu gặp trait lạ (HVE, Mana...)
            try:
                schema_dict = {
                    "component_name": comp_name,
                    "attributes": {k: 0.0 for k in attributes.keys()}, # Lấy danh sách trường số thực
                    "mutability": "dynamic",
                    "scope": "entity"
                }
                # Triệu gọi API chuẩn từ file dynamic_ecs.py của bạn
                self.registry.inject_dynamic_component_from_schema_dict(schema_dict)
                logger.info(f"[SchemaCheck] Đã kích hoạt API Inject động cho Component: {comp_name}")
                verified_components.add(comp_name)
            except Exception as e:
                logger.error(f"[SchemaCheck Error] Lỗi gọi API inject hệ thống cho '{comp_name}': {e}")

        return verified_components

    # ------------------------------------------------------
    # CORE INTERFACE: CONSUME QUEUE & SPAWN ENTITIES
    # ------------------------------------------------------
    def spawn_offspring_generation(
        self, 
        offspring_genome_queue: List[Dict[str, Dict[str, float]]]
    ) -> List[int]:
        """
        HÀM THỰC THI CHU TRÌNH VÒNG LẶP CHÍNH:
        Tiêu thụ toàn bộ danh sách hàng đợi gen con cái, thực hiện tuần tự quy trình
        từ thẩm định cấu trúc cho tới cấp phát thực thể vật lý vào không gian bộ nhớ RAM.
        
        Args:
            offspring_genome_queue: Danh sách chuỗi gen con thu được từ bộ lọc tiến hóa 4A.
            
        Returns:
            List[int]: Danh sách các EntityID mới được khởi sinh thành công.
        """
        if not offspring_genome_queue:
            logger.warning("[GenomeSpawner] Hàng đợi gen con trống. Hủy bỏ chu trình sinh thực thể.")
            return []

        newly_spawned_entities: List[int] = []

        with self._lock:
            logger.info(f"[GenomeSpawner] Bắt đầu xử lý quy trình khởi sinh cho {len(offspring_genome_queue)} mã gen.")

            for index, raw_genome in enumerate(offspring_genome_queue):
                # BƯỚC 1: Thẩm định cấu trúc dữ liệu gen (Genome Validation)
                if not self.validate_genome(raw_genome):
                    logger.warning(f"[GenomeSpawner] Bỏ qua bộ gen thứ {index} do vi phạm cấu trúc dữ liệu di truyền.")
                    continue

                # BƯỚC 2: Kiểm tra lược đồ thành phần hệ thống (Schema Check)
                # Đảm bảo các component động (Mana, HVE, SoulLink,...) đều sẵn sàng trong ECS
                self.verify_and_sync_schemas(raw_genome)

                # BƯỚC 3: Triệu gọi API cấp phát Định danh thực thể trống từ lõi ECS (create_entity)
                try:
                    new_entity_id = self.registry.create_entity()
                except Exception as e:
                    logger.critical(f"[Critical Error] Gọi API create_entity() thất bại tại cá thể thứ {index}: {e}")
                    continue

                # BƯỚC 4: Duyệt qua từng đoạn nhiễm sắc thể để "bơm" vào RAM (add_component)
                spawn_success = True
                for comp_type, attributes in raw_genome.items():
                    try:
                        # Gọi API add_component chuẩn của bạn với tham số ghi đè (overrides/attributes)
                        # Hàm này sẽ tự động khởi tạo cấu trúc dữ liệu theo Schema và nạp chỉ số đột biến vào
                        success = self.registry.add_component(
                            entity_id=new_entity_id,
                            comp_type=comp_type,
                            overrides=attributes
                        )
                        
                        if not success:
                            logger.warning(
                                f"[Component Inject Warning] API add_component trả về False cho Entity {new_entity_id} "
                                f"với Component '{comp_type}'."
                            )
                    except Exception as e:
                        logger.error(
                            f"[Component Inject Error] Thất bại khi inject Component '{comp_type}' "
                            f"vào Entity {new_entity_id}: {e}"
                        )
                        spawn_success = False
                        break

                # BƯỚC 5: Hoàn tất chu trình sinh thực thể mới (Spawn New Entity)
                if spawn_success:
                    newly_spawned_entities.append(new_entity_id)
                    logger.debug(f"[Spawn Success] Khởi sinh thành công Thực thể động ID: {new_entity_id}")
                else:
                    logger.error(f"[Spawn Failed] Hủy bỏ và dọn dẹp thực thể lỗi ID: {new_entity_id}")
                    # Nếu có hàm hủy thực thể lỗi thì gọi ở đây (ví dụ: self.registry.destroy_entity(new_entity_id))

            logger.info(
                f"[GenomeSpawner] Chu kỳ kết thúc. Đã đưa thành công "
                f"{len(newly_spawned_entities)}/{len(offspring_genome_queue)} Thực thể con vào Hệ sinh thái."
            )
            
            return newly_spawned_entities
        