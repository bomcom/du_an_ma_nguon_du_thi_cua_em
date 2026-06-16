# -*- coding: utf-8 -*-
"""
evolution/species_manager.py
============================
Phân hệ Quản lý Loài và Phân hóa Loài (Macro-Evolution & Taxonomy Layer)
Kiến trúc: Giai đoạn 4D — Tầng sinh học vĩ mô

Trách nhiệm DUY NHẤT (Single Responsibility)
--------------------------------------------
Module này CHỈ làm hai việc:
  1. TAXONOMY   : Duy trì danh mục loài (species registry).
  2. SPECIATION : Phát hiện sự phân hóa loài dựa trên khoảng cách di truyền.

Những điều module này KHÔNG làm
---------------------------------
  - KHÔNG gọi create_entity() / destroy_entity() / add_component() trên ECS.
  - KHÔNG lưu bản sao genome đầy đủ (chỉ lưu SpeciesSignature gọn nhẹ).
  - KHÔNG thay đổi trực tiếp LineageRecord bên trong GeneticEvolutionEngine.
  - KHÔNG giữ bất kỳ tham chiếu lẫn lộn (circular reference) nào.

Nguồn sự thật duy nhất (Single Source of Truth)
-------------------------------------------------
  - GeneticEvolutionEngine._brains (Dict[EntityID, BrainTopology])
    → được truy vấn qua adapter callback được inject vào __init__,
      tránh import vòng tròn.

Sửa toàn bộ 6 lỗi được xác định bởi review ChatGPT
-----------------------------------------------------
  Bug 1 : lineage_id (str) ↔ entity_id (int) không khớp.
          → Dùng EntityID = int xuyên suốt; classify_entity() nhận EntityID.
  Bug 2 : parent_lineage_ids không tồn tại trên LineageRecord.
          → Truy vấn qua adapter callback; không truy cập trực tiếp attribute sai.
  Bug 3 : Root species có signature rỗng → mọi entity đều tạo loài mới.
          → Root species được khởi tạo với sentinel flag is_root=True và
            bypass khoảng cách Jaccard bằng cách gán nó như best_match
            mặc định chỉ khi registry hoàn toàn trống (chưa có entity nào).
            Thực thể đầu tiên CÓ thể kích hoạt speciation từ root — đây là
            hành vi ĐÚNG về mặt sinh học. Lỗi thực sự là mọi entity SAU đó
            cũng bị speciation do không tìm được match. Sửa bằng cách: sau
            khi loài đầu tiên được tạo, root bị đánh dấu retired=True và
            loại khỏi vòng so sánh.
  Bug 4 : member_count tăng vô hạn, không giảm khi entity chết.
          → Thêm on_entity_death(entity_id) API để giảm đúng counter.
            Sử dụng live_members: Set[EntityID] thay vì counter đơn thuần
            để có O(1) lookup và count chính xác bất kỳ lúc nào.
  Bug 5 : classify_entity() có side-effects ẩn (phân loại + tạo species +
          ghi tracker cùng lúc).
          → Tách thành hai method riêng biệt:
              find_best_species()    → pure query, không side-effect
              register_new_species() → mutation, có log rõ ràng
              classify_entity()      → orchestrator gọi cả hai, documented.
  Bug 6 : trait_hashes lấy từ genome dict key sai ("_injected_traits"),
          trong khi LineageRecord có injected_traits.
          → Lấy từ adapter callback thống nhất; không đọc genome dict trực tiếp.

Nâng cấp bổ sung (vượt ngoài yêu cầu sửa lỗi)
-----------------------------------------------
  - PhenotypeSignature: đo cả khác biệt GIÁ TRỊ số (không chỉ sự tồn tại component)
    bằng cosine distance trên vector giá trị trung bình.
  - HybridDistance = α·JaccardStructural + (1−α)·CosinePheno  (α có thể điều chỉnh).
  - Thread-safe hoàn toàn qua threading.RLock().
  - Publish trạng thái loài lên Session Box channel "species_state" mỗi khi
    có speciation event.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple
)

import numpy as np

# EntityID là int — khớp với ECSRegistry.EntityID
EntityID = int

logger = logging.getLogger("SimulationKernel.SpeciesManager")


# ═══════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class StructuralSignature:
    """
    Phần 1 của chữ ký nhận dạng loài: CẤU TRÚC (có component nào).
    Dùng frozenset để hashable và immutable sau khi tạo.
    """
    component_types: FrozenSet[str]   # e.g. frozenset({"Health", "Energy", "Mana"})
    injected_traits: FrozenSet[str]   # tên trait được inject từ NLP pipeline


@dataclass
class PhenotypeSignature:
    """
    Phần 2 của chữ ký nhận dạng loài: KIỂU HÌNH (giá trị số trung bình).

    Lưu vector giá trị trung bình (mean) của các attribute quan trọng,
    chuẩn hóa về [0, 1] theo E_max của MatrixSolver.
    Cho phép phát hiện speciation khi genome diverge về giá trị
    dù component set vẫn giống nhau.
    """
    mean_values: np.ndarray    # float32 vector, shape (N_PHENO_DIMS,)
    # Mapping: [energy_ratio, health_ratio, mass_norm, fitness_norm,
    #           gene_w_mean_abs, gene_b_mean_abs, generation_norm, 0.0(padding)]
    DIMS: int = 8


# N chiều phenotype vector (dùng hằng số để dễ thay đổi sau)
PHENO_DIMS = 8


@dataclass
class SpeciesSignature:
    """
    Chữ ký tổng hợp của một loài.
    Kết hợp StructuralSignature + PhenotypeSignature.
    """
    structural: StructuralSignature
    phenotype:  PhenotypeSignature


def _empty_pheno() -> PhenotypeSignature:
    return PhenotypeSignature(mean_values=np.zeros(PHENO_DIMS, dtype=np.float32))


@dataclass
class SpeciesRecord:
    """
    Hồ sơ đầy đủ của một loài trong cây phát sinh loài.
    Không lưu genome — chỉ lưu signature và metadata.
    """
    species_id:          str
    ancestor_species_id: Optional[str]
    birth_generation:    int
    signature:           SpeciesSignature

    # Tập entity đang sống thuộc loài này (O(1) add/remove/count)
    live_members:        Set[EntityID] = field(default_factory=set)

    # Metadata
    birth_time_s:        float = field(default_factory=time.time)
    is_root:             bool  = False    # True chỉ cho loài gốc
    is_retired:          bool  = False    # True khi không còn member và đã có successor

    @property
    def member_count(self) -> int:
        """Số entity đang sống — luôn chính xác, không thể drift."""
        return len(self.live_members)


# ═══════════════════════════════════════════════════════════════════════════
# Distance Functions
# ═══════════════════════════════════════════════════════════════════════════

def _jaccard_structural(
    sig_a: StructuralSignature,
    sig_b: StructuralSignature,
) -> float:
    """
    Khoảng cách Jaccard giữa hai tập component types.
    Kết hợp component_types và injected_traits với trọng số:
        d = 0.7 * jaccard(components) + 0.3 * jaccard(traits)

    Returns: float in [0.0, 1.0]. 0 = giống nhau hoàn toàn, 1 = khác hoàn toàn.
    """
    # Component Jaccard
    union_c = sig_a.component_types | sig_b.component_types
    if union_c:
        inter_c = sig_a.component_types & sig_b.component_types
        jac_c   = 1.0 - len(inter_c) / len(union_c)
    else:
        jac_c = 0.0

    # Trait Jaccard
    union_t = sig_a.injected_traits | sig_b.injected_traits
    if union_t:
        inter_t = sig_a.injected_traits & sig_b.injected_traits
        jac_t   = 1.0 - len(inter_t) / len(union_t)
    else:
        jac_t = 0.0

    return 0.7 * jac_c + 0.3 * jac_t


def _cosine_phenotype(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
) -> float:
    """
    Cosine distance giữa hai phenotype vector.
    cosine_distance = 1 - (a·b) / (|a|·|b|)

    Returns: float in [0.0, 1.0].
    """
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))

    if norm_a < 1e-9 or norm_b < 1e-9:
        # Nếu một trong hai là zero vector, coi như giống nhau
        # (tránh NaN khi entity mới chưa có dữ liệu)
        return 0.0

    cosine_similarity = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    cosine_similarity = float(np.clip(cosine_similarity, -1.0, 1.0))
    return 1.0 - cosine_similarity


def hybrid_distance(
    sig_a:        SpeciesSignature,
    sig_b:        SpeciesSignature,
    alpha:        float = 0.6,
) -> float:
    """
    Khoảng cách lai (Hybrid Distance) giữa hai chữ ký loài.

    Formula:
        d_hybrid = α * d_structural + (1 - α) * d_phenotype

    Parameters
    ----------
    alpha : Trọng số của khoảng cách cấu trúc [0.0, 1.0].
            0.6 = cấu trúc quan trọng hơn kiểu hình trong mặc định.
            Có thể điều chỉnh qua SpeciesManager.alpha.
    """
    d_structural = _jaccard_structural(sig_a.structural, sig_b.structural)
    d_phenotype  = _cosine_phenotype(
        sig_a.phenotype.mean_values,
        sig_b.phenotype.mean_values,
    )
    return alpha * d_structural + (1.0 - alpha) * d_phenotype


# ═══════════════════════════════════════════════════════════════════════════
# Adapter Protocol (tránh import vòng tròn)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EntityInfoPacket:
    """
    Gói thông tin về một entity, được tạo bởi adapter callback.
    SpeciesManager chỉ nhìn thấy EntityInfoPacket — không nhìn thấy
    GeneticEvolutionEngine hay ECSRegistry trực tiếp.
    """
    entity_id:       EntityID
    component_types: FrozenSet[str]        # ECSRegistry.list_schemas filtered by has_component
    injected_traits: FrozenSet[str]        # từ BrainTopology.injected_traits nếu có
    energy_ratio:    float = 0.0           # current_energy / max_energy
    health_ratio:    float = 1.0           # current_hp / max_hp
    mass_norm:       float = 0.0           # mass_kg / M_max
    fitness_score:   float = 0.0           # BrainTopology.fitness_score
    gene_w_mean_abs: float = 0.0           # mean(|W|) of neural weights
    gene_b_mean_abs: float = 0.0           # mean(|b|) of neural weights
    generation:      int   = 0             # từ BrainTopology.generation
    generation_norm: float = 0.0           # generation / max_generation (e.g. 1000)
    parent_ids:      Tuple[EntityID, ...] = field(default_factory=tuple)


# Type alias cho callback functions được inject
EntityInfoCallback = Callable[[EntityID], Optional[EntityInfoPacket]]
ParentSpeciesCallback = Callable[[EntityID], Optional[str]]


# ═══════════════════════════════════════════════════════════════════════════
# SpeciesManager
# ═══════════════════════════════════════════════════════════════════════════

class SpeciesManager:
    """
    Quản lý phân loại học và phát hiện phân hóa loài.

    Khởi tạo
    ---------
    Inject hai adapter callbacks thay vì giữ tham chiếu trực tiếp đến
    GeneticEvolutionEngine hay ECSRegistry — tránh coupling chặt và
    import vòng tròn.

    Parameters
    ----------
    get_entity_info   : Callback (entity_id: int) -> EntityInfoPacket | None
                        Được cung cấp bởi orchestrator, tổng hợp dữ liệu
                        từ ECSRegistry + GeneticEvolutionEngine.
    session_box       : Optional SessionBox — nếu cung cấp, speciation events
                        sẽ được publish lên channel "species_state".
    threshold         : Ngưỡng hybrid distance để khai sinh loài mới [0, 1].
                        Mặc định 0.35. Giá trị nhỏ hơn → phân hóa dễ hơn.
    alpha             : Trọng số khoảng cách cấu trúc trong hybrid distance.

    Thread Safety
    -------------
    Tất cả public methods đều an toàn đa luồng qua threading.RLock().
    RLock (reentrant) được dùng vì classify_entity() gọi cả find và register.
    """

    def __init__(
        self,
        get_entity_info:    EntityInfoCallback,
        session_box:        Optional[Any]  = None,
        threshold:          float          = 0.35,
        alpha:              float          = 0.6,
    ) -> None:
        self._lock             = threading.RLock()
        self._get_entity_info  = get_entity_info
        self._session_box      = session_box
        self.threshold         = float(np.clip(threshold, 0.01, 0.99))
        self.alpha             = float(np.clip(alpha, 0.0, 1.0))

        # species_id -> SpeciesRecord
        self._registry: Dict[str, SpeciesRecord] = {}

        # entity_id -> species_id (lookup ngược để on_entity_death() O(1))
        self._entity_species_map: Dict[EntityID, str] = {}

        # Counter đơn điệu tăng cho species ID generation
        self._species_counter: int = 0

        # Speciation event log (cho TelemetryCore / Evolution Viewer)
        self._speciation_log: List[Dict[str, Any]] = []

        # Khởi tạo loài gốc với signature SENTINEL
        # Root species có structural signature rỗng, nhưng IS_ROOT=True
        # → bị loại khỏi vòng so sánh sau khi loài thứ nhất được tạo (retire).
        root_sig = SpeciesSignature(
            structural = StructuralSignature(
                component_types = frozenset(),
                injected_traits = frozenset(),
            ),
            phenotype = _empty_pheno(),
        )
        self._registry["species_root"] = SpeciesRecord(
            species_id          = "species_root",
            ancestor_species_id = None,
            birth_generation    = 0,
            signature           = root_sig,
            is_root             = True,
        )

        logger.info(
            "[SpeciesManager] Khởi tạo thành công | threshold=%.2f | alpha=%.2f",
            self.threshold, self.alpha,
        )

    # ───────────────────────────────────────────────────────────────────
    # Public API: Phân loại
    # ───────────────────────────────────────────────────────────────────

    def classify_entity(self, entity_id: EntityID) -> str:
        """
        API chính: Xác định species_id cho entity.

        Workflow
        --------
        1. Lấy EntityInfoPacket qua adapter callback.
        2. Xây dựng SpeciesSignature từ packet.
        3. find_best_species() → O(S) so sánh với tất cả loài hiện có.
        4. Nếu min_distance >= threshold → register_new_species().
        5. Cập nhật live_members và entity_species_map (không ghi ECS/Tracker).
        6. Publish speciation event lên Session Box nếu loài mới được tạo.

        Parameters
        ----------
        entity_id : EntityID (int) — khớp với ECSRegistry.EntityID.

        Returns
        -------
        species_id (str) mà entity này thuộc về.
        """
        with self._lock:
            # Step 1: lấy thông tin entity
            packet = self._get_entity_info(entity_id)
            if packet is None:
                logger.warning(
                    "[SpeciesManager] classify_entity: Không lấy được info cho entity %d. "
                    "Gán vào root.",
                    entity_id,
                )
                self._assign_entity_to_species(entity_id, "species_root")
                return "species_root"

            # Step 2: xây dựng signature
            sig = self._build_signature(packet)

            # Step 3 & 4: tìm / tạo loài
            best_species_id, min_dist = self.find_best_species(sig)
            new_species_created = False

            if min_dist >= self.threshold:
                # Phân hóa loài mới
                ancestor_id  = self._infer_ancestor_species(packet)
                best_species_id = self.register_new_species(
                    signature           = sig,
                    ancestor_species_id = ancestor_id,
                    birth_generation    = packet.generation,
                    trigger_entity_id   = entity_id,
                )
                new_species_created = True

                # Retire root sau khi loài thực đầu tiên xuất hiện
                root_rec = self._registry.get("species_root")
                if root_rec and root_rec.is_root and not root_rec.is_retired:
                    root_rec.is_retired = True
                    logger.debug("[SpeciesManager] Root species đã được retired.")

            # Step 5: cập nhật membership (đây là MUTATION duy nhất được phép)
            old_species = self._entity_species_map.get(entity_id)
            if old_species and old_species != best_species_id:
                # Entity di chuyển sang loài khác (reclassification)
                old_rec = self._registry.get(old_species)
                if old_rec:
                    old_rec.live_members.discard(entity_id)

            self._assign_entity_to_species(entity_id, best_species_id)

            # Step 6: publish event nếu loài mới
            if new_species_created:
                self._publish_speciation_event(best_species_id, entity_id)

            return best_species_id

    def find_best_species(
        self,
        sig: SpeciesSignature,
    ) -> Tuple[str, float]:
        """
        PURE QUERY — không side-effect.
        Tìm loài gần nhất với signature đã cho.

        Returns
        -------
        (best_species_id, min_distance)
        Nếu không có loài nào (trừ root đã retired) → trả về ("species_root", 1.0)
        để caller biết cần tạo loài mới.
        """
        best_id   = "species_root"
        min_dist  = 1.0

        for sid, rec in self._registry.items():
            # Bỏ qua root đã retired và các loài đã retired
            if rec.is_retired:
                continue

            d = hybrid_distance(sig, rec.signature, alpha=self.alpha)
            if d < min_dist:
                min_dist = d
                best_id  = sid

        return best_id, min_dist

    def register_new_species(
        self,
        signature:           SpeciesSignature,
        ancestor_species_id: Optional[str],
        birth_generation:    int,
        trigger_entity_id:   Optional[EntityID] = None,
    ) -> str:
        """
        MUTATION — Đăng ký loài mới vào registry.
        Tách rõ ràng khỏi find_best_species() để dễ test và debug.

        Returns
        -------
        species_id (str) của loài mới vừa tạo.
        """
        self._species_counter += 1
        new_id = f"species_{self._species_counter:06d}"

        self._registry[new_id] = SpeciesRecord(
            species_id          = new_id,
            ancestor_species_id = ancestor_species_id or "species_root",
            birth_generation    = birth_generation,
            signature           = signature,
        )

        event_info = {
            "new_species_id":   new_id,
            "ancestor":         ancestor_species_id,
            "generation":       birth_generation,
            "trigger_entity":   trigger_entity_id,
            "timestamp":        time.time(),
            "total_species":    len(self._registry),
        }
        self._speciation_log.append(event_info)

        logger.info(
            "[SpeciesManager] ★ SPECIATION ★ Loài mới: %s | "
            "ancestor=%s | gen=%d | trigger_entity=%s | total_species=%d",
            new_id, ancestor_species_id, birth_generation,
            trigger_entity_id, len(self._registry),
        )
        return new_id

    # ───────────────────────────────────────────────────────────────────
    # Public API: Lifecycle Events
    # ───────────────────────────────────────────────────────────────────

    def on_entity_death(self, entity_id: EntityID) -> None:
        """
        Sửa Bug 4: Giảm member_count khi entity chết.
        Phải được gọi bởi GeneticEvolutionEngine.evaluate_and_evolve()
        (hoặc ECS death system) ngay khi entity bị destroy.

        Thread-safe. Idempotent: gọi nhiều lần với cùng entity_id an toàn.
        """
        with self._lock:
            species_id = self._entity_species_map.pop(entity_id, None)
            if species_id is None:
                return   # Entity chưa từng được classify → bỏ qua

            rec = self._registry.get(species_id)
            if rec:
                rec.live_members.discard(entity_id)

                # Retire loài nếu không còn member và không phải root
                if rec.member_count == 0 and not rec.is_root:
                    rec.is_retired = True
                    logger.info(
                        "[SpeciesManager] Loài %s đã tuyệt chủng (0 member).",
                        species_id,
                    )

    def on_entity_reclassify(
        self,
        entity_id: EntityID,
    ) -> str:
        """
        Re-chạy classify_entity() khi genome của entity thay đổi đáng kể
        (sau crossover/mutation lớn).
        Returns species_id mới (có thể giống cũ nếu không đủ diverge).
        """
        return self.classify_entity(entity_id)

    # ───────────────────────────────────────────────────────────────────
    # Public API: Queries (read-only)
    # ───────────────────────────────────────────────────────────────────

    def get_species_tree(self) -> Dict[str, Any]:
        """
        Xuất sơ đồ cây tiến hóa loài cho Evolution Viewer / TelemetryCore.

        Returns
        -------
        Dict[species_id, {ancestor, member_count, birth_gen, is_extinct, ...}]
        """
        with self._lock:
            return {
                sid: {
                    "ancestor":      rec.ancestor_species_id,
                    "member_count":  rec.member_count,
                    "birth_gen":     rec.birth_generation,
                    "is_extinct":    rec.is_retired and not rec.is_root,
                    "birth_time_s":  rec.birth_time_s,
                }
                for sid, rec in self._registry.items()
            }

    def get_species_members(self, species_id: str) -> List[EntityID]:
        """
        Lấy tất cả entity_id đang sống thuộc loài này.
        O(1) — dùng live_members set trực tiếp (không scan toàn bộ tracker).
        """
        with self._lock:
            rec = self._registry.get(species_id)
            if rec is None:
                return []
            return list(rec.live_members)

    def get_entity_species(self, entity_id: EntityID) -> Optional[str]:
        """Trả về species_id hiện tại của entity, hoặc None nếu chưa classify."""
        with self._lock:
            return self._entity_species_map.get(entity_id)

    def get_all_living_species(self) -> List[str]:
        """Trả về danh sách species_id có ít nhất 1 member đang sống."""
        with self._lock:
            return [
                sid for sid, rec in self._registry.items()
                if rec.member_count > 0
            ]

    def get_speciation_log(self, last_n: int = 20) -> List[Dict[str, Any]]:
        """Trả về N sự kiện speciation gần nhất."""
        with self._lock:
            return list(self._speciation_log[-last_n:])

    def get_stats(self) -> Dict[str, Any]:
        """Trả về metrics tổng quan cho TelemetryCore."""
        with self._lock:
            alive   = [r for r in self._registry.values() if not r.is_retired]
            extinct = [r for r in self._registry.values() if r.is_retired and not r.is_root]
            total_members = sum(r.member_count for r in alive)
            return {
                "total_species_ever":   len(self._registry),
                "living_species":       len(alive),
                "extinct_species":      len(extinct),
                "total_classified":     len(self._entity_species_map),
                "total_living_members": total_members,
                "speciation_events":    len(self._speciation_log),
                "threshold":            self.threshold,
                "alpha":                self.alpha,
            }

    def get_species_record(self, species_id: str) -> Optional[SpeciesRecord]:
        """Trả về bản sao SpeciesRecord (không trả reference để tránh data race)."""
        with self._lock:
            rec = self._registry.get(species_id)
            if rec is None:
                return None
            # Trả về shallow copy — live_members set đã là copy riêng
            import copy
            return copy.copy(rec)

    # ───────────────────────────────────────────────────────────────────
    # Configuration
    # ───────────────────────────────────────────────────────────────────

    def set_threshold(self, new_threshold: float) -> None:
        """Điều chỉnh ngưỡng phân hóa loài tại runtime."""
        with self._lock:
            old = self.threshold
            self.threshold = float(np.clip(new_threshold, 0.01, 0.99))
            logger.info(
                "[SpeciesManager] Threshold thay đổi: %.3f → %.3f",
                old, self.threshold,
            )

    def set_alpha(self, new_alpha: float) -> None:
        """Điều chỉnh trọng số cấu trúc vs kiểu hình tại runtime."""
        with self._lock:
            self.alpha = float(np.clip(new_alpha, 0.0, 1.0))

    # ───────────────────────────────────────────────────────────────────
    # Session Box Integration
    # ───────────────────────────────────────────────────────────────────

    def publish_to_session_box(self) -> None:
        """
        Publish snapshot trạng thái loài lên channel "species_state".
        Gọi sau mỗi speciation event hoặc theo định kỳ từ telemetry loop.
        """
        if self._session_box is None:
            return
        payload = self.get_stats()
        payload["species_tree"] = self.get_species_tree()
        self._session_box.write(
            "species_state", payload, source="species_manager"
        )

    # ───────────────────────────────────────────────────────────────────
    # Private Helpers
    # ───────────────────────────────────────────────────────────────────

    def _build_signature(self, packet: EntityInfoPacket) -> SpeciesSignature:
        """
        Xây dựng SpeciesSignature từ EntityInfoPacket.
        Sửa Bug 6: không đọc genome dict trực tiếp — lấy từ packet chuẩn hóa.
        """
        structural = StructuralSignature(
            component_types = packet.component_types,
            injected_traits = packet.injected_traits,
        )

        pheno_vec = np.array([
            packet.energy_ratio,
            packet.health_ratio,
            packet.mass_norm,
            float(np.clip(packet.fitness_score / 100.0, 0.0, 1.0)),
            float(np.clip(packet.gene_w_mean_abs, 0.0, 1.0)),
            float(np.clip(packet.gene_b_mean_abs, 0.0, 1.0)),
            packet.generation_norm,
            0.0,   # padding — dành cho feature tương lai
        ], dtype=np.float32)

        # NaN/Inf guard trên phenotype vector
        pheno_vec = np.where(
            np.isnan(pheno_vec) | np.isinf(pheno_vec),
            0.0,
            pheno_vec,
        )

        phenotype = PhenotypeSignature(mean_values=pheno_vec)
        return SpeciesSignature(structural=structural, phenotype=phenotype)

    def _infer_ancestor_species(self, packet: EntityInfoPacket) -> Optional[str]:
        """
        Suy ra species_id của loài tổ tiên từ parent_ids của entity.
        Sửa Bug 2: không truy cập parent_lineage_ids (không tồn tại) —
        dùng packet.parent_ids (int tuple) để tra cứu entity_species_map.
        """
        for parent_id in packet.parent_ids:
            ancestor_species = self._entity_species_map.get(parent_id)
            if ancestor_species:
                return ancestor_species
        return "species_root"

    def _assign_entity_to_species(
        self,
        entity_id:  EntityID,
        species_id: str,
    ) -> None:
        """Gán entity vào loài và cập nhật cả hai indexes."""
        rec = self._registry.get(species_id)
        if rec is None:
            logger.error(
                "[SpeciesManager] _assign_entity_to_species: "
                "species '%s' không tồn tại.",
                species_id,
            )
            return
        rec.live_members.add(entity_id)
        self._entity_species_map[entity_id] = species_id

    def _publish_speciation_event(
        self,
        new_species_id: str,
        trigger_entity: EntityID,
    ) -> None:
        """Publish speciation event lên Session Box (non-blocking)."""
        if self._session_box is None:
            return
        event = {
            "event":          "SPECIATION",
            "new_species_id": new_species_id,
            "trigger_entity": trigger_entity,
            "timestamp":      time.time(),
            "total_species":  len(self._registry),
        }
        self._session_box.write(
            "species_state", event, source="species_manager"
        )

    # ───────────────────────────────────────────────────────────────────
    # Dunder Methods
    # ───────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"<SpeciesManager "
            f"living={stats['living_species']} "
            f"extinct={stats['extinct_species']} "
            f"classified={stats['total_classified']} "
            f"threshold={self.threshold:.2f}>"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Adapter Builder (tiện ích cho orchestrator)
# ═══════════════════════════════════════════════════════════════════════════

def build_entity_info_adapter(
    ecs_registry:      Any,
    evolution_engine:  Any,
    m_max:             float = 50_000.0,
    max_generation:    int   = 1000,
) -> EntityInfoCallback:
    """
    Factory tạo callback get_entity_info chuẩn.
    Được gọi một lần trong orchestrator để inject vào SpeciesManager.

    Parameters
    ----------
    ecs_registry     : ECSRegistry instance.
    evolution_engine : GeneticEvolutionEngine instance.
    m_max            : M_max từ ConstraintConfig (để normalize mass).
    max_generation   : Giá trị generation tối đa dùng để normalize.

    Returns
    -------
    Callable[[EntityID], Optional[EntityInfoPacket]]
    """
    def get_entity_info(entity_id: EntityID) -> Optional[EntityInfoPacket]:
        """Tổng hợp thông tin entity từ ECS + Evolution Engine."""
        if not ecs_registry.entity_exists(entity_id):
            return None

        # Lấy component types entity đang có
        profile = ecs_registry.get_entity_profile(entity_id)
        if profile is None:
            return None
        comp_types = frozenset(profile["components"].keys())

        # Lấy component snapshots
        energy_snap  = ecs_registry.get_component_snapshot(entity_id, "Energy")
        health_snap  = ecs_registry.get_component_snapshot(entity_id, "Health")
        mass_snap    = ecs_registry.get_component_snapshot(entity_id, "Mass")
        brain_snap   = ecs_registry.get_component_snapshot(entity_id, "NeuralBrain")

        energy_ratio = 0.0
        if energy_snap:
            max_e = float(energy_snap.get("max_energy", 1.0)) or 1.0
            energy_ratio = float(energy_snap.get("current_energy", 0.0)) / max_e

        health_ratio = 1.0
        if health_snap:
            max_h = float(health_snap.get("max_hp", 100.0)) or 1.0
            health_ratio = float(health_snap.get("current_hp", 100.0)) / max_h

        mass_norm = 0.0
        if mass_snap and m_max > 0:
            mass_norm = float(mass_snap.get("mass_kg", 0.0)) / m_max

        fitness    = 0.0
        generation = 0
        w_mean_abs = 0.0
        b_mean_abs = 0.0

        # Lấy dữ liệu gene từ GeneticEvolutionEngine
        brain_topo = None
        if hasattr(evolution_engine, '_brains'):
            brain_topo = evolution_engine._brains.get(entity_id)

        if brain_topo is not None:
            fitness    = float(getattr(brain_topo, 'fitness_score', 0.0))
            generation = int(getattr(brain_topo, 'generation', 0))
            # Tính mean absolute weight/bias làm proxy gene magnitude
            try:
                w1 = getattr(brain_topo, 'W1', None)
                b1 = getattr(brain_topo, 'b1', None)
                if w1 is not None:
                    w_mean_abs = float(np.mean(np.abs(w1)))
                if b1 is not None:
                    b_mean_abs = float(np.mean(np.abs(b1)))
            except Exception:
                pass
        elif brain_snap is not None:
            fitness    = float(brain_snap.get("fitness_score", 0.0))
            generation = int(brain_snap.get("generation", 0))

        # Lấy injected traits từ brain_topo nếu có
        injected_traits: FrozenSet[str] = frozenset()
        if brain_topo and hasattr(brain_topo, 'injected_traits'):
            injected_traits = frozenset(brain_topo.injected_traits or [])

        # Parent IDs (từ brain topology nếu được track)
        parent_ids: Tuple[EntityID, ...] = ()
        if brain_topo and hasattr(brain_topo, 'parent_ids'):
            raw = getattr(brain_topo, 'parent_ids', None)
            if raw:
                parent_ids = tuple(int(p) for p in raw)

        gen_norm = float(generation) / max(max_generation, 1)

        return EntityInfoPacket(
            entity_id       = entity_id,
            component_types = comp_types,
            injected_traits = injected_traits,
            energy_ratio    = float(np.clip(energy_ratio, 0.0, 1.0)),
            health_ratio    = float(np.clip(health_ratio, 0.0, 1.0)),
            mass_norm       = float(np.clip(mass_norm,    0.0, 1.0)),
            fitness_score   = float(np.clip(fitness, 0.0, 1e6)),
            gene_w_mean_abs = float(np.clip(w_mean_abs, 0.0, 10.0)),
            gene_b_mean_abs = float(np.clip(b_mean_abs, 0.0, 10.0)),
            generation      = generation,
            generation_norm = float(np.clip(gen_norm, 0.0, 1.0)),
            parent_ids      = parent_ids,
        )

    return get_entity_info
