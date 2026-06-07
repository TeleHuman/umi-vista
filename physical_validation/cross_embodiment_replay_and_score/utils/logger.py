# Log recording utility, running in a dedicated thread.

import os
import time
import threading
import queue
from typing import Optional, Sequence, Tuple

class TrajLogger:
    """
        Simple thread-safe logger for recording actuator positions (act_pos)
        and target positions (tgt_pos) per control cycle.

        File naming options:
        - Provide `filename_prefix` (default) and optionally add a timestamp.
        - Provide a full `filename` to use directly; optionally append a timestamp before the extension.

        Default log format for `log()` writes rows:
            ts cycle act_0 .. act_N tgt_0 .. tgt_N
        Where vectors are expected length `vector_len`.
        `log2()` writes only the `tgt_pos` values.
    """
    def __init__(self,
                 out_dir: str = "log",
                 filename_prefix: Optional[str] = "replay",
                 vector_len: int = 20,
                 async_mode: bool = True,
                 flush_interval: float = 0.1,
                 buffer_limit: int = 10000,
                 filename: Optional[str] = None):
        self.out_dir = os.path.abspath(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)
        ts_str = time.strftime("%Y%m%d_%H%M%S")
        # Build final filename (two modes only):
        # 1) If `filename` provided -> use exactly as-is.
        # 2) Else use `filename_prefix` + _YYYYmmdd_HHMMSS + .txt
        if filename and isinstance(filename, str) and filename.strip():
            final_name = filename
        else:
            prefix = filename_prefix if (isinstance(filename_prefix, str) and filename_prefix) else "replay"
            final_name = f"{prefix}_{ts_str}.txt"
        self.file_path = os.path.join(self.out_dir, final_name)
        self.vector_len = int(vector_len)
        self._lock = threading.Lock()
        self._cycle = 0
        self._fh = open(self.file_path, "w")
        # Async writer setup
        self._async = bool(async_mode)
        self._flush_interval = float(flush_interval)
        self._buffer_limit = int(buffer_limit)
        self._q: Optional[queue.Queue[Tuple[str]]] = None
        self._stop = threading.Event()
        self._writer_thread: Optional[threading.Thread] = None
        if self._async:
            self._q = queue.Queue(maxsize=self._buffer_limit)
            self._writer_thread = threading.Thread(target=self._writer_loop, name="TrajectoryLoggerWriter", daemon=True)
            self._writer_thread.start()

    def log(self, act_pos: Sequence[float], tgt_pos: Sequence[float], ts: Optional[float] = None):
        """Append one row. act_pos and tgt_pos should be sequences of length == vector_len.
        If longer, they will be truncated; if shorter, they will be padded with zeros.
        """
        if ts is None:
            ts = time.time()
        def _fit(vec):
            v = list(vec)
            if len(v) < self.vector_len:
                v = v + [0.0] * (self.vector_len - len(v))
            elif len(v) > self.vector_len:
                v = v[:self.vector_len]
            return [float(x) for x in v]
        row = [f"{float(ts):.6f}", str(int(self._cycle))] + [f"{x:.9f}" for x in _fit(act_pos)] + [f"{x:.9f}" for x in _fit(tgt_pos)]
        line = " ".join(row) + "\n"
        if self._async and self._q is not None:
            try:
                self._q.put(line, block=False)
            except queue.Full:
                # fallback to direct write if queue is full
                with self._lock:
                    self._fh.write(line)
            with self._lock:
                self._cycle += 1
        else:
            with self._lock:
                self._fh.write(line)
                self._fh.flush()
                self._cycle += 1
    
    def log2(self, tgt_pos: Sequence[float]):
        """Append one row containing only tgt_pos values.
        The sequence is padded/truncated to match vector_len.
        """
        def _fit(vec):
            v = list(vec)
            if len(v) < self.vector_len:
                v = v + [0.0] * (self.vector_len - len(v))
            elif len(v) > self.vector_len:
                v = v[:self.vector_len]
            return [float(x) for x in v]
        row_vals = [f"{x:.9f}" for x in _fit(tgt_pos)]
        line = " ".join(row_vals) + "\n"
        if self._async and self._q is not None:
            try:
                self._q.put(line, block=False)
            except queue.Full:
                # fallback to direct write if queue is full
                with self._lock:
                    self._fh.write(line)
            with self._lock:
                self._cycle += 1
        else:
            with self._lock:
                self._fh.write(line)
                self._fh.flush()
                self._cycle += 1

    def _writer_loop(self):
        last_flush = time.time()
        while not self._stop.is_set():
            try:
                line = self._q.get(timeout=self._flush_interval) if self._q is not None else None
            except queue.Empty:
                line = None
            if line is not None:
                with self._lock:
                    self._fh.write(line)
            # periodic flush
            now = time.time()
            if now - last_flush >= self._flush_interval:
                with self._lock:
                    try:
                        self._fh.flush()
                    except Exception:
                        pass
                last_flush = now
        # drain remaining lines on stop
        if self._q is not None:
            while True:
                try:
                    line = self._q.get_nowait()
                except queue.Empty:
                    break
                with self._lock:
                    self._fh.write(line)
        with self._lock:
            try:
                self._fh.flush()
            except Exception:
                pass

    def close(self):
        # stop writer thread if running
        if self._async and self._writer_thread is not None:
            self._stop.set()
            try:
                self._writer_thread.join(timeout=2.0)
            except Exception:
                pass
        with self._lock:
            try:
                self._fh.flush()
            except Exception:
                pass
            try:
                self._fh.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
