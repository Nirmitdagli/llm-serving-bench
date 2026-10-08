"""
Samples GPU memory in a background thread with nvidia-smi while a load level runs.

Why sample and not read once: memory grows as the engine fills its KV cache with
more concurrent sequences, so we keep the peak. Note that vLLM and SGLang reserve
most of the GPU for KV cache at startup (--gpu-memory-utilization / --mem-fraction-static),
so "used" mostly reflects that setting. bench.py records it; run_suite.sh sets the
same fraction on every engine so the numbers are comparable.
"""
import shutil
import subprocess
import threading


def _query(gpu_index):
    out = subprocess.run(
        ["nvidia-smi", f"--id={gpu_index}", "--query-gpu=memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5,
    ).stdout.strip()
    used, total, util = (float(x) for x in out.split(","))
    return used, total, util


class GpuMemoryMonitor:
    def __init__(self, gpu_index=0, interval_s=0.25):
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self.available = shutil.which("nvidia-smi") is not None
        self._stop = threading.Event()
        self._thread = None
        self._peak = self._total = 0.0
        self._utils = []

    def _loop(self):
        while not self._stop.is_set():
            try:
                used, total, util = _query(self.gpu_index)
                self._peak, self._total = max(self._peak, used), total
                self._utils.append(util)
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def start(self):
        if not self.available:
            return
        self._stop.clear()
        self._peak, self._utils = 0.0, []
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Returns peak memory and mean utilization for the window since start()."""
        if not self.available or self._thread is None:
            return {"gpu_mem_peak_mib": None, "gpu_mem_total_mib": None, "gpu_util_mean_pct": None}
        self._stop.set()
        self._thread.join()
        mean_util = sum(self._utils) / len(self._utils) if self._utils else None
        return {"gpu_mem_peak_mib": round(self._peak), "gpu_mem_total_mib": round(self._total),
                "gpu_util_mean_pct": None if mean_util is None else round(mean_util, 1)}
