"""
core_engine/session_box.py
==========================
Central Session Box: The global data pipeline hub for the Hybrid AI Simulation Platform.

Architecture:
    - Implements a thread-safe Shared Memory Matrix using threading.Lock primitives.
    - Coordinates asynchronous data flow between the Math Core (matrix solver tick loop)
      and the 3D Graphics Rendering loop via a double-buffered state snapshot mechanism.
    - All subsystems register named "channels" into the Session Box and read/write
      via atomic lock-guarded operations, preventing race conditions entirely.
    - Supports asyncio-compatible async_get / async_set wrappers for coroutine contexts.

Data Flow Topology:
    [LTspice Parser] ──► [Session Box: channel "ltspice"]
    [NLP Parser]     ──► [Session Box: channel "ecs_mutations"]
    [Math Core]      ──► [Session Box: channel "math_state"]
    [AI Core]        ──► [Session Box: channel "adversarial_flags"]
    [Evolution Eng.] ──► [Session Box: channel "npc_states"]
    [Renderer]       ◄── [Session Box: snapshot buffer] (read-only, non-blocking)
    [Telemetry]      ◄── [Session Box: channel "telemetry_metrics"]
"""

import asyncio
import copy
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ChannelState(Enum):
    """Lifecycle state of a registered data channel."""
    ACTIVE   = auto()
    PAUSED   = auto()
    FLUSHED  = auto()
    CLOSED   = auto()


class DataPriority(Enum):
    """Priority level for write arbitration when multiple producers compete."""
    CRITICAL  = 0   # e.g., adversarial kill-switch signals
    HIGH      = 1   # e.g., math core conservation law violations
    NORMAL    = 2   # e.g., NPC state updates
    LOW       = 3   # e.g., telemetry metrics


# ---------------------------------------------------------------------------
# Internal Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ChannelMetadata:
    """Per-channel bookkeeping stored inside the Session Box registry."""
    name:           str
    owner:          str                        # Module name that registered this channel
    dtype:          str          = "dict"      # "ndarray" | "dict" | "scalar"
    state:          ChannelState = ChannelState.ACTIVE
    write_count:    int          = 0
    read_count:     int          = 0
    last_write_ts:  float        = 0.0         # Unix timestamp of last write
    last_read_ts:   float        = 0.0
    priority:       DataPriority = DataPriority.NORMAL
    subscribers:    List[Callable] = field(default_factory=list)


@dataclass
class SharedMemoryCell:
    """
    An atomic cell wrapping a numpy array or plain Python dict/scalar.
    The lock is per-cell to maximise concurrency (channel-level granularity).
    """
    data:       Any                          # numpy array OR dict payload
    lock:       threading.Lock = field(default_factory=threading.Lock)
    version:    int            = 0           # Monotonic version counter
    dtype:      str            = "dict"      # "ndarray" | "dict" | "scalar"


# ---------------------------------------------------------------------------
# SessionBox Singleton
# ---------------------------------------------------------------------------

class SessionBox:
    """
    The central Shared Memory Matrix and data pipeline coordinator.

    Usage
    -----
    >>> box = SessionBox.get_instance()
    >>> box.register_channel("math_state", owner="matrix_solver", dtype="ndarray",
    ...                      shape=(4,), priority=DataPriority.HIGH)
    >>> box.write("math_state", np.array([1.0, 0.5, 0.3, 0.9], dtype=np.float32))
    >>> snapshot = box.read("math_state")

    Async Usage (from asyncio coroutines)
    --------------------------------------
    >>> await box.async_write("math_state", payload)
    >>> value = await box.async_read("math_state")
    """

    _instance:  Optional["SessionBox"] = None
    _init_lock: threading.Lock         = threading.Lock()

    # ------------------------------------------------------------------
    # Singleton construction
    # ------------------------------------------------------------------

    def __new__(cls) -> "SessionBox":
        with cls._init_lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
        return cls._instance

    @classmethod
    def get_instance(cls) -> "SessionBox":
        """Thread-safe singleton accessor."""
        return cls()

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        # Primary shared memory registry: channel_name -> SharedMemoryCell
        self._cells:    Dict[str, SharedMemoryCell]  = {}
        self._meta:     Dict[str, ChannelMetadata]   = {}
        self._registry_lock = threading.Lock()         # Protects structural changes to _cells/_meta

        # Double-buffered snapshot for the renderer (non-blocking reads)
        self._render_snapshot:      Dict[str, Any]  = {}
        self._render_snapshot_lock: threading.Lock  = threading.Lock()
        self._snapshot_version:     int             = 0

        # Global kill-switch: set True to gracefully stop all subsystems
        self._shutdown_event: threading.Event = threading.Event()

        # Async event loop reference (set externally by the main coroutine runner)
        self._async_loop: Optional[asyncio.AbstractEventLoop] = None

        logger.info("[SessionBox] Initialised. Shared Memory Matrix online.")

    # ------------------------------------------------------------------
    # Channel Management
    # ------------------------------------------------------------------

    def register_channel(
        self,
        name:     str,
        owner:    str,
        dtype:    str            = "dict",
        shape:    Optional[Tuple[int, ...]] = None,
        priority: DataPriority   = DataPriority.NORMAL,
    ) -> None:
        """
        Register a new named channel in the Shared Memory Matrix.

        Parameters
        ----------
        name     : Unique channel identifier string.
        owner    : Name of the registering module (for diagnostics).
        dtype    : "ndarray" initialises a numpy float32 zero array of `shape`;
                   "dict"    initialises an empty Python dict;
                   "scalar"  initialises 0.0.
        shape    : Required when dtype == "ndarray".
        priority : Write arbitration priority (DataPriority enum).
        """
        with self._registry_lock:
            if name in self._cells:
                logger.warning("[SessionBox] Channel '%s' already registered by '%s'. Skipping.",
                               name, self._meta[name].owner)
                return

            # Initialise the cell payload based on dtype
            if dtype == "ndarray":
                if shape is None:
                    raise ValueError(f"Channel '{name}': shape must be provided for ndarray dtype.")
                initial_data = np.zeros(shape, dtype=np.float32)
            elif dtype == "dict":
                initial_data = {}
            elif dtype == "scalar":
                initial_data = 0.0
            else:
                raise ValueError(f"Unsupported dtype '{dtype}' for channel '{name}'.")

            self._cells[name] = SharedMemoryCell(data=initial_data, dtype=dtype)
            self._meta[name]  = ChannelMetadata(
                name=name, owner=owner, dtype=dtype, priority=priority
            )
            logger.debug("[SessionBox] Channel '%s' registered by '%s' (dtype=%s).", name, owner, dtype)

    def deregister_channel(self, name: str) -> None:
        """Safely remove a channel from the registry."""
        with self._registry_lock:
            if name not in self._cells:
                logger.warning("[SessionBox] Attempt to deregister unknown channel '%s'.", name)
                return
            self._meta[name].state = ChannelState.CLOSED
            del self._cells[name]
            del self._meta[name]
            logger.info("[SessionBox] Channel '%s' deregistered.", name)

    def list_channels(self) -> Dict[str, str]:
        """Return a dict of {channel_name: owner} for all active channels."""
        with self._registry_lock:
            return {n: m.owner for n, m in self._meta.items()}

    # ------------------------------------------------------------------
    # Synchronous Read / Write
    # ------------------------------------------------------------------

    def write(self, channel: str, payload: Any, source: str = "unknown") -> bool:
        """
        Atomically write payload into the named channel cell.

        Parameters
        ----------
        channel : Target channel name.
        payload : Data to write. Must match channel dtype.
        source  : Caller identifier (used for logging).

        Returns
        -------
        True on success, False if channel is not found or closed.
        """
        try:
            cell = self._cells[channel]
            meta = self._meta[channel]
        except KeyError:
            logger.error("[SessionBox] Write failed: channel '%s' not registered.", channel)
            return False

        if meta.state == ChannelState.CLOSED:
            logger.warning("[SessionBox] Write to closed channel '%s' rejected.", channel)
            return False

        with cell.lock:
            # For ndarray channels, validate shape compatibility before overwriting
            if cell.dtype == "ndarray":
                if not isinstance(payload, np.ndarray):
                    logger.error("[SessionBox] Channel '%s' expects ndarray, got %s.", channel, type(payload))
                    return False
                if payload.shape != cell.data.shape:
                    logger.error("[SessionBox] Channel '%s' shape mismatch: expected %s, got %s.",
                                 channel, cell.data.shape, payload.shape)
                    return False
                np.copyto(cell.data, payload.astype(np.float32))
            else:
                cell.data = copy.deepcopy(payload)

            cell.version      += 1
            meta.write_count  += 1
            meta.last_write_ts = time.time()

        # Notify subscribers asynchronously (fire-and-forget)
        self._notify_subscribers(channel, payload)
        return True

    def read(self, channel: str, source: str = "unknown") -> Optional[Any]:
        """
        Atomically read a deep-copied snapshot from the named channel cell.

        Returns None if channel is not found or is closed.
        """
        try:
            cell = self._cells[channel]
            meta = self._meta[channel]
        except KeyError:
            logger.error("[SessionBox] Read failed: channel '%s' not registered.", channel)
            return None

        if meta.state == ChannelState.CLOSED:
            return None

        with cell.lock:
            if cell.dtype == "ndarray":
                result = cell.data.copy()
            else:
                result = copy.deepcopy(cell.data)

            meta.read_count  += 1
            meta.last_read_ts = time.time()

        return result

    def read_version(self, channel: str) -> int:
        """Return the current monotonic version of a channel without reading data."""
        try:
            with self._cells[channel].lock:
                return self._cells[channel].version
        except KeyError:
            return -1

    # ------------------------------------------------------------------
    # Async wrappers (for asyncio coroutine callers)
    # ------------------------------------------------------------------

    async def async_write(self, channel: str, payload: Any, source: str = "unknown") -> bool:
        """Non-blocking async write: offloads synchronous lock acquisition to thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.write, channel, payload, source)

    async def async_read(self, channel: str, source: str = "unknown") -> Optional[Any]:
        """Non-blocking async read: offloads synchronous lock acquisition to thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.read, channel, source)

    # ------------------------------------------------------------------
    # Double-Buffered Render Snapshot
    # ------------------------------------------------------------------

    def commit_render_snapshot(self) -> None:
        """
        Atomically snapshot all ACTIVE channels into the render buffer.
        Called once per math tick to prepare a consistent read-only view for the renderer.
        The renderer always reads from this snapshot — never directly from live cells —
        ensuring the render thread never blocks the math computation thread.
        """
        snapshot: Dict[str, Any] = {}
        with self._registry_lock:
            channel_names = list(self._cells.keys())

        for name in channel_names:
            try:
                cell = self._cells[name]
                with cell.lock:
                    if cell.dtype == "ndarray":
                        snapshot[name] = cell.data.copy()
                    else:
                        snapshot[name] = copy.deepcopy(cell.data)
            except KeyError:
                pass  # Channel was deregistered mid-snapshot; skip gracefully

        with self._render_snapshot_lock:
            self._render_snapshot   = snapshot
            self._snapshot_version += 1

        logger.debug("[SessionBox] Render snapshot committed (version=%d, channels=%d).",
                     self._snapshot_version, len(snapshot))

    def read_render_snapshot(self) -> Tuple[Dict[str, Any], int]:
        """
        Non-blocking read of the latest render snapshot.
        Returns (snapshot_dict, version_int).
        The snapshot_dict is a shallow copy — DO NOT mutate its values.
        """
        with self._render_snapshot_lock:
            return dict(self._render_snapshot), self._snapshot_version

    # ------------------------------------------------------------------
    # Subscriber / Observer Pattern
    # ------------------------------------------------------------------

    def subscribe(self, channel: str, callback: Callable[[str, Any], None]) -> None:
        """
        Register a callback to be fired on every write to `channel`.
        Callbacks are invoked in a daemon thread to avoid blocking the writer.
        Signature: callback(channel_name: str, payload: Any) -> None
        """
        with self._registry_lock:
            if channel not in self._meta:
                raise KeyError(f"Cannot subscribe to unregistered channel '{channel}'.")
            self._meta[channel].subscribers.append(callback)
        logger.debug("[SessionBox] Subscriber registered on channel '%s'.", channel)

    def _notify_subscribers(self, channel: str, payload: Any) -> None:
        """Fire all registered callbacks for a channel in separate daemon threads."""
        try:
            subs = self._meta[channel].subscribers
        except KeyError:
            return
        for cb in subs:
            t = threading.Thread(target=cb, args=(channel, payload), daemon=True)
            t.start()

    # ------------------------------------------------------------------
    # Shutdown Coordination
    # ------------------------------------------------------------------

    def signal_shutdown(self) -> None:
        """Signal all subsystems to terminate their processing loops."""
        self._shutdown_event.set()
        logger.info("[SessionBox] Global shutdown signal sent.")

    def is_shutdown_requested(self) -> bool:
        """Poll whether a shutdown has been requested."""
        return self._shutdown_event.is_set()

    def wait_for_shutdown(self, timeout: Optional[float] = None) -> bool:
        """Block the calling thread until shutdown is signalled."""
        return self._shutdown_event.wait(timeout=timeout)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_channel_stats(self, channel: str) -> Optional[Dict[str, Any]]:
        """Return a metadata snapshot dict for a given channel."""
        try:
            meta = self._meta[channel]
            return {
                "name":          meta.name,
                "owner":         meta.owner,
                "state":         meta.state.name,
                "priority":      meta.priority.name,
                "write_count":   meta.write_count,
                "read_count":    meta.read_count,
                "last_write_ts": meta.last_write_ts,
                "last_read_ts":  meta.last_read_ts,
                "version":       self._cells[channel].version if channel in self._cells else -1,
            }
        except KeyError:
            return None

    def get_all_stats(self) -> List[Dict[str, Any]]:
        """Return metadata snapshots for every registered channel."""
        with self._registry_lock:
            names = list(self._meta.keys())
        return [s for name in names if (s := self.get_channel_stats(name)) is not None]

    def __repr__(self) -> str:
        with self._registry_lock:
            n = len(self._cells)
        return (f"<SessionBox channels={n} "
                f"snapshot_v={self._snapshot_version} "
                f"shutdown={'YES' if self.is_shutdown_requested() else 'NO'}>")
