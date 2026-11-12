"""Polite use of a shared machine: pick only free GPUs and an idle share of the CPUs.

``plan_resources`` runs once, before any CUDA work, and decides:

* **GPUs**: survey every visible GPU with ``nvidia-smi`` (sampled for ``sample_seconds``,
  worst case kept), and treat one as free only if its utilisation is <=
  ``max_utilization`` % and it has >= ``min_free_memory_gb`` free. Take up to
  ``max_gpus`` free GPUs (most free memory first). If none is free: wait and re-poll
  (``on_busy="wait"``, up to ``wait_timeout_min``), fall back to the CPU (``"cpu"``),
  or stop (``"fail"``).
* **Isolation**: ``CUDA_VISIBLE_DEVICES`` is narrowed to the chosen GPUs (and
  ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` so indices mean the same as in ``nvidia-smi``), so this
  process and its DDP ranks can never create a CUDA context on anyone else's GPU. A
  ``CUDA_VISIBLE_DEVICES`` set by you or a scheduler is respected (we only narrow it).
* **Memory cap**: each rank caps its PyTorch allocator at ``memory_share`` of the memory
  that was free on its GPU (``GPUMemoryCap``), so growth on our side cannot push the
  other users' jobs out of memory; we would hit our own OOM first.
* **CPU**: the budget is ``cpu_share`` of the currently idle cores (affinity minus load
  average); it is split into DataLoader workers and torch threads per rank, and the
  process niceness is raised to ``nice`` so other users' CPU work takes precedence.
* **Batch**: ``batch_size`` in the config is the *global* batch; with n GPUs each gets
  ``ceil(batch_size / n)``, so the number of optimiser steps (and the step-wise lr
  decay) stays the same as on one device.

* **NCCL check** (``nccl_check``, multi-GPU only): a 2-way broadcast + all-reduce + barrier
  between the chosen GPUs, in a throw-away subprocess with a time limit. Some machines
  advertise GPU peer-to-peer copies that do not work; NCCL then hangs on the first
  collective and DDP only fails after its 30 min timeout. If the check fails, it is
  retried with ``NCCL_P2P_DISABLE=1`` (kept in ``plan.env`` for every rank), and if that
  fails too the run falls back to one GPU.

Multi-GPU runs use DDP. Lightning starts rank k > 0 by re-running the command, so the plan
(and the run directory) is made once by rank 0 and handed to the other ranks through the
``G4GNN_RUN`` environment variable (``handoff`` / ``inherited``); ranks never re-plan
(rank 0's own GPUs would look busy to them). A rank > 0 exits as soon as rank 0 is gone
(``exit_with_parent``) instead of waiting in a collective until the DDP timeout.

``nvidia-smi`` is called directly (no extra dependency) and CUDA is never initialised
while planning.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import pytorch_lightning as pl
import torch

__all__ = ["ResourceConfig", "GPUInfo", "ResourcePlan", "query_gpus", "cpu_budget",
           "plan_resources", "check_nccl", "apply_process_settings", "ensure_open_file_capacity",
           "prepare_worker_fork", "GPUMemoryCap", "handoff",
           "inherited", "exit_with_parent", "HANDOFF_ENV", "gpus_available", "GPUWatcher"]

HANDOFF_ENV = "G4GNN_RUN"
_GB = 1024  # nvidia-smi reports MiB


@dataclass
class ResourceConfig:
    max_gpus: Optional[int] = None        # None: every free GPU
    min_free_memory_gb: float = 8.0       # a GPU needs this much free memory to be used
    max_utilization: float = 10.0         # ... and at most this compute utilisation (%)
    memory_share: float = 0.9             # cap our allocator at this share of its free memory
    sample_seconds: float = 3.0           # utilisation sampled over this window (worst case)
    on_busy: str = "wait"                 # no free GPU: "wait" | "cpu" | "fail"
    wait_timeout_min: float = 120.0
    poll_seconds: float = 60.0
    cpu_share: float = 0.5                # use at most this share of the idle CPU cores
    max_workers_per_device: int = 8
    nice: int = 10                        # process niceness (0 = normal priority)
    nccl_check: bool = True               # multi-GPU: test NCCL between the GPUs before training
    rescale_check_minutes: float = 15.0   # --auto-resume: look for more free GPUs this often (0: off)

    def __post_init__(self):
        if self.on_busy not in ("wait", "cpu", "fail"):
            raise ValueError("resources.on_busy must be 'wait', 'cpu' or 'fail'")
        if not 0 < self.memory_share <= 1 or not 0 < self.cpu_share <= 1:
            raise ValueError("resources.memory_share and cpu_share must be in (0, 1]")


@dataclass
class GPUInfo:
    index: int          # nvidia-smi (PCI bus) index
    uuid: str
    name: str
    total_mb: float
    used_mb: float
    utilization: float  # %

    @property
    def free_mb(self) -> float:
        return self.total_mb - self.used_mb


@dataclass
class ResourcePlan:
    accelerator: str                      # "gpu" | "cpu"
    devices: int                          # devices per node (GPUs are 0..n-1 after narrowing)
    strategy: str                         # "auto" | "ddp" | user's choice
    visible_devices: Optional[str]        # value given to CUDA_VISIBLE_DEVICES (None: untouched)
    gpus: list = field(default_factory=list)              # chosen GPUInfo dicts, in rank order
    memory_fraction: list = field(default_factory=list)   # allocator cap per rank (GPU only)
    num_workers: int = 0                  # DataLoader workers per rank
    torch_threads: int = 1                # intra-op threads per rank
    nice: int = 0
    env: dict = field(default_factory=dict)  # extra environment for every rank (e.g. NCCL_P2P_DISABLE)
    reason: str = ""
    allowed: Optional[list] = None        # CUDA_VISIBLE_DEVICES tokens we may use (None: all GPUs)

    def per_device_batch(self, global_batch: int) -> int:
        return max(1, math.ceil(global_batch / max(1, self.devices)))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ResourcePlan":
        return cls(**d)


# ---- surveys ------------------------------------------------------------------------------
def _run_smi(args: list[str]) -> str:
    return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, check=True,
                          timeout=30).stdout


def query_gpus(smi: Optional[Callable[[list[str]], str]] = None) -> list[GPUInfo]:
    """One ``nvidia-smi`` snapshot of every GPU (empty if there is no usable nvidia-smi)."""
    if smi is None:
        if shutil.which("nvidia-smi") is None:
            return []
        smi = _run_smi
    try:
        out = smi(["--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
                   "--format=csv,noheader,nounits"])
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        num = [float(p) if p.replace(".", "", 1).isdigit() else math.nan for p in parts[3:6]]
        gpus.append(GPUInfo(int(parts[0]), parts[1], parts[2], *num))
    return gpus


def _survey(smi, samples: int, interval: float, sleep) -> list[GPUInfo]:
    """Worst case (max used memory, max utilisation) over ``samples`` snapshots; a GPU whose
    numbers nvidia-smi cannot read (``[N/A]``) counts as fully busy."""
    worst: dict[int, GPUInfo] = {}
    for i in range(max(1, samples)):
        if i:
            sleep(interval)
        for g in query_gpus(smi):
            if math.isnan(g.used_mb) or math.isnan(g.total_mb):
                g.used_mb = g.total_mb if not math.isnan(g.total_mb) else 0.0
            if math.isnan(g.utilization):
                g.utilization = 100.0
            w = worst.get(g.index)
            worst[g.index] = g if w is None else GPUInfo(
                g.index, g.uuid, g.name, g.total_mb, max(w.used_mb, g.used_mb),
                max(w.utilization, g.utilization))
    return [worst[k] for k in sorted(worst)]


def cpu_budget(share: float, cpu_info: Optional[Callable[[], tuple[int, float]]] = None) -> int:
    """Cores we may use: ``share`` of (cores we are allowed on - 1-min load average), >= 1."""
    if cpu_info is None:
        cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
        cpu_info = lambda: (cores, os.getloadavg()[0])  # noqa: E731
    cores, load = cpu_info()
    idle = max(1.0, cores - load)
    return max(1, int(idle * share))


def _visible_entries() -> Optional[list[str]]:
    v = os.environ.get("CUDA_VISIBLE_DEVICES")
    if v is None:
        return None
    return [e.strip() for e in v.split(",") if e.strip() and e.strip() != "-1"]


def _match_visible(gpus: list[GPUInfo], entries: Optional[list[str]]) -> list[tuple[GPUInfo, str]]:
    """GPUs we are allowed to see, each with the CUDA_VISIBLE_DEVICES token that names it."""
    if entries is None:
        return [(g, str(g.index)) for g in gpus]
    out = []
    for e in entries:
        for g in gpus:
            if e == str(g.index) or (e.startswith("GPU-") and g.uuid.startswith(e)):
                out.append((g, e))
                break
    return out


# ---- NCCL check ---------------------------------------------------------------------------
NCCL_CHECK_SECONDS = 120.0

# Every GPU sends to every other one: broadcast from the LAST rank (a broadcast from rank 0
# alone can pass while the reverse direction hangs), then all-reduce and barrier.
_NCCL_CHECK_SCRIPT = """
import datetime, sys, torch, torch.distributed as dist, torch.multiprocessing as mp

def work(rank, world, init):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=init, rank=rank, world_size=world,
                            timeout=datetime.timedelta(seconds=60))
    t = torch.full((1,), float(rank + 1), device="cuda")
    dist.broadcast(t, world - 1)
    dist.all_reduce(t)
    dist.barrier()
    torch.cuda.synchronize()
    ok = t.item() == world * world
    dist.destroy_process_group()
    sys.exit(0 if ok else 3)

if __name__ == "__main__":
    world = torch.cuda.device_count()
    if world < 2:
        sys.exit(4)
    mp.spawn(work, args=(world, sys.argv[1]), nprocs=world)
"""


def check_nccl(visible_devices: str, extra_env: Optional[dict] = None,
               timeout: float = NCCL_CHECK_SECONDS) -> bool:
    """True if NCCL collectives work between the GPUs ``visible_devices`` (a
    CUDA_VISIBLE_DEVICES value). Runs in a separate process group (CUDA is never initialised
    here) that is killed whole if it does not finish within ``timeout`` seconds."""
    env = {k: v for k, v in os.environ.items() if k not in (HANDOFF_ENV, "LOCAL_RANK")}
    env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=visible_devices,
               **(extra_env or {}))
    with tempfile.TemporaryDirectory(prefix="g4gnn_nccl_") as tmp:
        script = Path(tmp) / "nccl_check.py"
        script.write_text(_NCCL_CHECK_SCRIPT)
        init = (Path(tmp) / "rendezvous").as_uri()
        proc = subprocess.Popen([sys.executable, str(script), init], env=env, cwd=tmp,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                start_new_session=True)
        try:
            return proc.wait(timeout=timeout) == 0
        except subprocess.TimeoutExpired:
            return False
        finally:
            try:  # also the spawned ranks, which may be stuck in NCCL
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            proc.wait()


def _nccl_preflight(chosen: list, check: Callable[[str, dict], bool], log) -> tuple[list, dict]:
    """(GPUs to use, extra env): all of them if NCCL works (if need be without P2P), else one."""
    tokens = ",".join(t for _, t in chosen)
    if check(tokens, {}):
        return chosen, {}
    log(f"[resources] NCCL check between GPUs {tokens} failed or hung; "
        "retrying with NCCL_P2P_DISABLE=1")
    p2p_off = {"NCCL_P2P_DISABLE": "1"}
    if check(tokens, p2p_off):
        log("[resources] NCCL works without GPU peer-to-peer: using NCCL_P2P_DISABLE=1")
        return chosen, p2p_off
    log(f"[resources] NCCL fails between GPUs {tokens} even without peer-to-peer; "
        "falling back to ONE GPU (check the machine: nvidia-smi topo -m, NCCL_DEBUG=INFO)")
    return chosen[:1], {}


# ---- planning -----------------------------------------------------------------------------
def plan_resources(
    rc: ResourceConfig,
    accelerator: str = "auto",
    devices: Any = "auto",
    strategy: str = "auto",
    num_workers: Any = "auto",
    *,
    smi: Optional[Callable[[list[str]], str]] = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    cpu_info: Optional[Callable[[], tuple[int, float]]] = None,
    log: Callable[[str], None] = print,
    set_env: bool = True,
    nccl_check: Optional[Callable[[str, dict], bool]] = None,
) -> ResourcePlan:
    """Decide GPUs / CPU use (see module docstring). ``accelerator``/``devices``/``strategy``
    come from the trainer config: ``"auto"`` lets the planner choose; ``"cpu"`` forces the
    CPU; an explicit GPU count or list limits the choice to that many / those GPUs.
    ``nccl_check(visible_devices, extra_env) -> bool`` replaces ``check_nccl`` (tests)."""
    accelerator = str(accelerator).lower()
    gpu_ok = _gpu_requested(accelerator) and (smi is not None or torch.backends.cuda.is_built())
    chosen: list[tuple[GPUInfo, str]] = []
    reason = "accelerator=cpu requested" if accelerator == "cpu" else ""
    allowed_tokens = _allowed_tokens(devices)

    if gpu_ok:
        if torch.cuda.is_initialized():
            raise RuntimeError("plan_resources must run before CUDA is initialised")
        max_n = _max_gpus(rc, devices)
        deadline = clock() + 60 * rc.wait_timeout_min
        samples = max(1, int(rc.sample_seconds))
        while True:
            visible = _match_visible(_survey(smi, samples, rc.sample_seconds / samples, sleep),
                                     allowed_tokens)
            if not visible:
                reason = "no visible GPU (nvidia-smi found none or CUDA_VISIBLE_DEVICES hides all)"
                break
            free = [(g, t) for g, t in visible if _is_free(g, rc)]
            busy = [g for g, _ in visible if all(g is not f for f, _ in free)]
            for g in busy:
                log(f"[resources] GPU {g.index} ({g.name}) busy: {g.utilization:.0f}% util, "
                    f"{g.free_mb / _GB:.1f}/{g.total_mb / _GB:.1f} GB free")
            if free:
                free.sort(key=lambda x: (-x[0].free_mb, x[0].utilization, x[0].index))
                chosen = free[:max_n] if max_n else free
                break
            if rc.on_busy == "fail":
                raise RuntimeError("no free GPU (resources.on_busy='fail')")
            if rc.on_busy == "cpu":
                reason = "no free GPU; on_busy='cpu'"
                break
            if clock() >= deadline:
                raise RuntimeError(f"no GPU became free within {rc.wait_timeout_min} min")
            log(f"[resources] no free GPU, re-checking in {rc.poll_seconds:.0f}s")
            sleep(rc.poll_seconds)

    extra_env: dict = {}
    if len(chosen) > 1 and rc.nccl_check:
        chosen, extra_env = _nccl_preflight(chosen, nccl_check or check_nccl, log)

    if chosen:
        n = len(chosen)
        tokens = ",".join(t for _, t in chosen)
        if set_env:
            os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
            os.environ["CUDA_VISIBLE_DEVICES"] = tokens
        fractions = [round(rc.memory_share * g.free_mb / g.total_mb, 4) for g, _ in chosen]
        plan = ResourcePlan(
            accelerator="gpu", devices=n,
            strategy=strategy if strategy != "auto" else ("ddp" if n > 1 else "auto"),
            visible_devices=tokens, gpus=[asdict(g) for g, _ in chosen],
            memory_fraction=fractions, env=extra_env,
            reason=f"{n} free GPU(s): " + ", ".join(
                f"{g.index} ({g.free_mb / _GB:.0f} GB free, {g.utilization:.0f}%)" for g, _ in chosen)
            + "".join(f"; {k}={v}" for k, v in extra_env.items()))
    else:
        n = devices if isinstance(devices, int) and devices > 0 else 1
        plan = ResourcePlan(accelerator="cpu", devices=n,
                            strategy=strategy if strategy != "auto" else ("ddp" if n > 1 else "auto"),
                            visible_devices=None,
                            reason=reason or "no CUDA build of torch / no nvidia-smi")

    budget = max(1, cpu_budget(rc.cpu_share, cpu_info) // plan.devices)
    if num_workers == "auto" or num_workers is None:
        # GPU: the device computes, so the cores feed it (1 thread kept for the main loop);
        # CPU: the cores also compute the model, so split them between loading and compute
        loaders = budget - 1 if plan.accelerator == "gpu" else budget // 2
        plan.num_workers = min(rc.max_workers_per_device, max(0, loaders))
    else:
        plan.num_workers = int(num_workers)
    plan.torch_threads = max(1, budget - plan.num_workers)
    plan.nice = rc.nice
    plan.allowed = allowed_tokens
    log(f"[resources] {plan.accelerator} x{plan.devices} ({plan.strategy}); {plan.reason}; "
        f"{plan.num_workers} workers + {plan.torch_threads} threads per rank; nice {plan.nice}")
    return plan


def _gpu_requested(accelerator: str) -> bool:
    return str(accelerator).lower() in ("auto", "gpu", "cuda")


def _allowed_tokens(devices: Any) -> Optional[list[str]]:
    """CUDA_VISIBLE_DEVICES tokens we may use (None: every GPU), before any narrowing."""
    allowed = _visible_entries()
    if isinstance(devices, (list, tuple)):  # explicit physical indices
        want = [str(d) for d in devices]
        allowed = [t for t in (allowed or want) if t in want]
    return allowed


def _max_gpus(rc: ResourceConfig, devices: Any) -> Optional[int]:
    max_n = rc.max_gpus
    if isinstance(devices, int) and devices > 0:
        max_n = devices if max_n is None else min(max_n, devices)
    return max_n


def _is_free(g: GPUInfo, rc: ResourceConfig) -> bool:
    return g.utilization <= rc.max_utilization and g.free_mb >= rc.min_free_memory_gb * _GB


def gpus_available(rc: ResourceConfig, plan: ResourcePlan, accelerator: str = "auto",
                   devices: Any = "auto", *, smi: Optional[Callable[[list[str]], str]] = None,
                   sample_seconds: Optional[float] = None,
                   sleep: Callable[[float], None] = time.sleep) -> int:
    """How many GPUs a new plan could take now: the ones ``plan`` holds (busy because we use
    them) plus the other free ones we may use, capped like ``plan_resources``."""
    if not _gpu_requested(accelerator):
        return 0
    secs = rc.sample_seconds if sample_seconds is None else sample_seconds
    samples = max(1, int(secs))
    ours = {g["uuid"] for g in plan.gpus}
    visible = _match_visible(_survey(smi, samples, secs / samples, sleep), plan.allowed)
    n = len(ours) + sum(1 for g, _ in visible if g.uuid not in ours and _is_free(g, rc))
    max_n = _max_gpus(rc, devices)
    return min(n, max_n) if max_n else n


class GPUWatcher:
    """Rank 0 of a supervised run (``--auto-resume``): a background thread that re-surveys
    the GPUs every ``rc.rescale_check_minutes`` (utilisation sampled for >= 30 s) and sets
    ``triggered`` once two checks in a row find more GPUs than the run holds; the run then
    pauses and the supervisor resumes it with a new plan. Never touches CUDA."""

    def __init__(self, rc: ResourceConfig, plan: ResourcePlan, accelerator: str = "auto",
                 devices: Any = "auto", *, available: Optional[Callable[[], int]] = None,
                 log: Callable[[str], None] = print):
        self.interval = 60.0 * rc.rescale_check_minutes
        self.held = len(plan.gpus)
        self.available = available or (lambda: gpus_available(
            rc, plan, accelerator, devices, sample_seconds=max(30.0, rc.sample_seconds)))
        self.log = log
        self.triggered = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def check(self, streak: int) -> int:
        """One survey; returns the new streak of checks that found more GPUs."""
        try:
            n = self.available()
        except Exception as e:  # a failing nvidia-smi must not stop training
            self.log(f"[resources] GPU re-check failed: {e}")
            return 0
        if n <= self.held:
            return 0
        streak += 1
        self.log(f"[resources] {n} GPU(s) usable now, run holds {self.held} (check {streak}/2)")
        if streak >= 2:
            self.triggered = True
        return streak

    def start(self) -> "GPUWatcher":
        if self.interval <= 0:
            return self

        def loop():
            streak = 0
            while not self._stop.wait(self.interval) and not self.triggered:
                streak = self.check(streak)

        self._thread = threading.Thread(target=loop, daemon=True, name="g4gnn-gpu-watcher")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()


def ensure_open_file_capacity(min_files: int = 65536, rlimit=None) -> tuple[int, str]:
    """DataLoader workers hand every tensor of a batch to the main process as an open file
    descriptor (torch's default ``file_descriptor`` sharing). A PyG batch carries ~30 tensors,
    so ~14 workers x prefetch per rank exceed the common soft limit of 1024 and the pin-memory
    thread dies with "received 0 items of ancdata". Raise the soft ``RLIMIT_NOFILE`` to the hard
    limit; if that is still below ``min_files``, switch torch to the ``file_system`` strategy
    (no descriptor per tensor). Returns (soft limit, sharing strategy)."""
    if rlimit is None:
        try:
            import resource as rlimit
        except ImportError:  # not POSIX
            return -1, torch.multiprocessing.get_sharing_strategy()
    soft, hard = rlimit.getrlimit(rlimit.RLIMIT_NOFILE)
    target = hard if hard != rlimit.RLIM_INFINITY else max(soft, min_files)
    if soft != rlimit.RLIM_INFINITY and soft < target:
        try:
            rlimit.setrlimit(rlimit.RLIMIT_NOFILE, (target, hard))
            soft = target
        except (ValueError, OSError):
            pass
    if soft != rlimit.RLIM_INFINITY and soft < min_files:
        torch.multiprocessing.set_sharing_strategy("file_system")
    return soft, torch.multiprocessing.get_sharing_strategy()


def prepare_worker_fork() -> int:
    """Call right before DataLoader workers are forked from a process that holds a CUDA context
    (e.g. the test pass after a DDP fit). Cyclic garbage that still owns GPU memory (the DDP
    reducer's buckets, old optimizer state) would otherwise be freed by a forked worker's own
    garbage collector: a CUDA call in the child -> "CUDA error: initialization error" thrown in a
    destructor -> ``terminate``, and every worker dies. Collect it here, where CUDA is valid, then
    freeze what is left so the children's collectors never touch inherited objects. Returns the
    number of objects collected."""
    import gc

    n = gc.collect()
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    gc.freeze()
    return n


def apply_process_settings(plan: ResourcePlan) -> None:
    """Per process (every rank): absolute niceness, torch thread count, CUDA visibility,
    ``plan.env``, open-file capacity for many DataLoader workers (``ensure_open_file_capacity``)."""
    ensure_open_file_capacity()
    if plan.nice and hasattr(os, "setpriority"):
        try:
            current = os.getpriority(os.PRIO_PROCESS, 0)
            os.setpriority(os.PRIO_PROCESS, 0, max(current, plan.nice))  # never raise priority
        except (OSError, PermissionError):
            pass
    torch.set_num_threads(plan.torch_threads)
    if plan.visible_devices is not None:
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ["CUDA_VISIBLE_DEVICES"] = plan.visible_devices
    os.environ.update(plan.env)


# ---- DDP handoff --------------------------------------------------------------------------
def handoff(run_dir: str | Path, plan: ResourcePlan) -> None:
    """Rank 0: publish the run directory + plan for the ranks Lightning will start."""
    os.environ[HANDOFF_ENV] = json.dumps({"run_dir": str(run_dir), "plan": plan.to_dict()})


def inherited() -> Optional[tuple[Path, ResourcePlan]]:
    """Rank > 0 (started by Lightning): rank 0's run directory and plan, else None."""
    raw = os.environ.get(HANDOFF_ENV)
    if not raw or "LOCAL_RANK" not in os.environ or os.environ.get("LOCAL_RANK") == "0":
        return None
    d = json.loads(raw)
    return Path(d["run_dir"]), ResourcePlan.from_dict(d["plan"])


def exit_with_parent(poll_seconds: float = 5.0, log: Callable[[str], None] = print) -> None:
    """Rank > 0: exit as soon as the process that launched us (rank 0) is gone. Without
    this, a rank whose rank 0 was killed waits in its next collective until the DDP timeout
    (30 min) and keeps its GPU meanwhile."""
    parent = os.getppid()

    def watch():
        while True:
            time.sleep(poll_seconds)
            if os.getppid() != parent:  # re-parented: rank 0 died
                log(f"[resources] rank 0 (pid {parent}) is gone; exiting this rank")
                os._exit(1)

    threading.Thread(target=watch, name="exit-with-parent", daemon=True).start()


# ---- per-rank GPU memory cap --------------------------------------------------------------
class GPUMemoryCap(pl.Callback):
    """Caps each rank's CUDA allocator at its planned share and records the peak use
    (``<run_dir>/resources_rank<k>.json``) so you can check we stayed within it."""

    def __init__(self, fractions: list[float], run_dir: Optional[str | Path] = None):
        self.fractions = list(fractions)
        self.run_dir = None if run_dir is None else Path(run_dir)

    def setup(self, trainer, pl_module, stage):
        dev = trainer.strategy.root_device
        if dev.type == "cuda" and self.fractions:
            frac = self.fractions[dev.index if dev.index is not None else 0]
            torch.cuda.set_per_process_memory_fraction(frac, dev)
            torch.cuda.reset_peak_memory_stats(dev)

    def teardown(self, trainer, pl_module, stage):
        dev = trainer.strategy.root_device
        if dev.type != "cuda" or self.run_dir is None:
            return
        i = dev.index or 0
        total = torch.cuda.get_device_properties(dev).total_memory
        rec = {"stage": stage, "rank": trainer.global_rank, "device": str(dev),
               "cap_fraction": self.fractions[i] if self.fractions else None,
               "cap_bytes": int(self.fractions[i] * total) if self.fractions else None,
               "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(dev)),
               "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(dev)),
               "total_bytes": int(total)}
        path = self.run_dir / f"resources_rank{trainer.global_rank}_{stage}.json"
        path.write_text(json.dumps(rec, indent=1))
