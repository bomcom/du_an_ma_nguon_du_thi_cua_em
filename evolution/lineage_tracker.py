# -*- coding: utf-8 -*-
"""
evolution/lineage_tracker.py
============================
Phân hệ Theo vết Phả hệ và Lịch sử Tiến hóa (Evolutionary Lineage Tracker Engine)
Kiến trúc: Giai đoạn 4C - Tối ưu hóa cho Mô phỏng quy mô lớn (Large-Scale Production)

Cải tiến Kiến trúc:
    - Sử dụng Lineage UUID độc lập với ECS Entity ID (tránh lỗi Recycle ID).
    - Quản lý bộ nhớ nghiêm ngặt: Archive thực thể chết, giới hạn buffer bằng deque.
    - Cấu trúc O(1) và O(N) chuẩn hóa cho các thao tác truy vấn dạng Cây (Tree/Graph).
"""

import time
import uuid
import logging
import threading
from collections import deque, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional

logger = logging.getLogger("SimulationKernel.LineageTracker")


# ==========================================================
# CẤU TRÚC DỮ LIỆU TỐI ƯU (OPTIMIZED DATA STRUCTURES)
# ==========================================================

@dataclass
class GlobalEvent:
    """Định dạng chuẩn cho sự kiện vĩ mô giúp dễ dàng Audit và Replay"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = "UNKNOWN"
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LineageRecord:
    """
    Bản ghi định danh di truyền độc lập.
    Quản trị vòng đời bằng UUID thay vì EntityID của ECS.
    """
    lineage_id: str                          # Danh tính sinh học bất biến
    last_known_entity_id: int                # Runtime handle hiện tại trong ECS
    generation: int
    parent_lineage_ids: List[str] = field(default_factory=list) # Trỏ tới UUID của bố mẹ
    species_id: str = "unclassified"         # Bộ đệm phân loại loài (Chờ 4D quyết định)
    
    birth_time: float = field(default_factory=time.time)
    death_time: Optional[float] = None
    
    # Tối ưu RAM 1: Dùng Counter thay vì lưu từng dictionary sự kiện
    injected_traits_counter: Counter = field(default_factory=Counter)
    
    # Tối ưu RAM 2: Giới hạn lịch sử đột biến tối đa 50 sự kiện gần nhất
    major_mutations: deque = field(default_factory=lambda: deque(maxlen=50))


class LineageTracker:
    def __init__(self):
        self._lock = threading.RLock()
        
        # Ánh xạ Runtime ECS ID -> Lineage UUID (Giải quyết bài toán Recycle ID)
        self.active_handles: Dict[int, str] = {}
        
        # Kho lưu trữ Live (Thực thể đang sống)
        self.live_records: Dict[str, LineageRecord] = {}
        
        # Kho lưu trữ Cold/Archive (Thực thể đã chết, dọn rác RAM đồ họa)
        self.archived_records: Dict[str, LineageRecord] = {}
        
        # Chỉ mục phụ O(1) vẽ cây phả hệ: parent_uuid -> Set[child_uuid]
        self.children_map: Dict[str, Set[str]] = {}
        
        # Biên niên sử sự kiện hệ thống
        self.global_events: deque = deque(maxlen=10000)

        logger.info("[LineageTracker] Kích hoạt phân hệ Lịch sử. Bảo mật ID tái sử dụng: BẬT.")

    # ------------------------------------------------------
    # API GHI NHẬN (MUTATORS)
    # ------------------------------------------------------

    def register_birth(self, entity_id: int, generation: int, parent_entity_ids: List[int], species_id: str = "unclassified") -> str:
        """Khai sinh bản ghi mới. Chuyển đổi Entity ID của cha mẹ thành Lineage UUID"""
        with self._lock:
            new_lineage_id = str(uuid.uuid4())
            
            # Phân giải UUID của bố mẹ từ Entity ID (nếu bố mẹ còn sống)
            parent_uuids = []
            for p_id in parent_entity_ids:
                if p_id in self.active_handles:
                    parent_uuids.append(self.active_handles[p_id])

            record = LineageRecord(
                lineage_id=new_lineage_id,
                last_known_entity_id=entity_id,
                generation=generation,
                parent_lineage_ids=parent_uuids,
                species_id=species_id
            )
            
            self.live_records[new_lineage_id] = record
            self.active_handles[entity_id] = new_lineage_id

            # Cập nhật bản đồ con cái (Children Map)
            for p_uuid in parent_uuids:
                if p_uuid not in self.children_map:
                    self.children_map[p_uuid] = set()
                self.children_map[p_uuid].add(new_lineage_id)

            return new_lineage_id

    def register_death(self, entity_id: int) -> None:
        """Ghi nhận cái chết và giải phóng handle của ECS để an toàn cho đợt Recycle ID"""
        with self._lock:
            lineage_id = self.active_handles.pop(entity_id, None)
            if not lineage_id or lineage_id not in self.live_records:
                return

            record = self.live_records[lineage_id]
            record.death_time = time.time()

    def compact_history(self) -> int:
        """
        GARBAGE COLLECTION: Di chuyển các record đã chết sang kho Archive.
        Hàm này nên được gọi định kỳ (ví dụ: sau mỗi 1000 tick) để dọn RAM.
        """
        with self._lock:
            dead_uuids = [uid for uid, rec in self.live_records.items() if rec.death_time is not None]
            for uid in dead_uuids:
                self.archived_records[uid] = self.live_records.pop(uid)
            return len(dead_uuids)

    def register_trait_injection(self, entity_id: int, trait_name: str, source: str = "prompt") -> None:
        """Lưu vết Prompt nạp Trait bằng Counter siêu nhẹ RAM"""
        with self._lock:
            lineage_id = self.active_handles.get(entity_id)
            if not lineage_id:
                return
            
            record = self.live_records.get(lineage_id)
            if record:
                record.injected_traits_counter[trait_name] += 1
                
            self.global_events.append(GlobalEvent(
                event_type="TRAIT_INJECTION",
                details={"lineage_id": lineage_id, "trait": trait_name, "source": source}
            ))

    def register_mutation_event(self, entity_id: int, component_name: str, delta: float) -> None:
        """Lưu vết đột biến an toàn RAM nhờ deque maxlen=50"""
        with self._lock:
            lineage_id = self.active_handles.get(entity_id)
            if not lineage_id:
                return
                
            record = self.live_records.get(lineage_id)
            if record:
                record.major_mutations.append({
                    "c": component_name, "d": round(delta, 4), "t": time.time()
                })

    # ------------------------------------------------------
    # API TRUY VẤN VÀ ĐỒ THỊ (QUERIES & GRAPH)
    # ------------------------------------------------------

    def _get_record(self, lineage_id: str) -> Optional[LineageRecord]:
        return self.live_records.get(lineage_id) or self.archived_records.get(lineage_id)

    def get_descendants(self, lineage_id: str) -> List[str]:
        """
        SỬA LỖI O(N^2): Thuật toán BFS quét con cháu tốc độ cao sử dụng collections.deque.
        """
        with self._lock:
            descendants: List[str] = []
            queue = deque([lineage_id])  # Khởi tạo deque chuẩn
            visited: Set[str] = set()

            while queue:
                current_id = queue.popleft()  # O(1) pop thay vì pop(0)
                if current_id in visited:
                    continue
                visited.add(current_id)

                children = self.children_map.get(current_id, set())
                for child in children:
                    if child not in visited:
                        descendants.append(child)
                        queue.append(child)
                        
            return descendants

    def export_evolution_tree(self, root_lineage_id: str, max_depth: int = 5) -> Dict[str, Any]:
        """
        SỬA LỖI UI CRASH: Xuất đồ thị có giới hạn độ sâu (Pagination/Depth Limit).
        Tránh đẩy toàn bộ 500k entity làm nổ giao diện.
        """
        with self._lock:
            nodes = []
            edges = []
            queue = deque([(root_lineage_id, 0)])
            visited = set()

            while queue:
                current_id, depth = queue.popleft()
                
                if depth > max_depth or current_id in visited:
                    continue
                visited.add(current_id)

                rec = self._get_record(current_id)
                if not rec:
                    continue

                nodes.append({
                    "id": current_id,
                    "ecs_id": rec.last_known_entity_id,
                    "species": rec.species_id,
                    "alive": rec.death_time is None
                })

                if depth < max_depth:
                    children = self.children_map.get(current_id, set())
                    for child_id in children:
                        edges.append({"from": current_id, "to": child_id})
                        queue.append((child_id, depth + 1))

            return {"nodes": nodes, "edges": edges}
        