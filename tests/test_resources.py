"""Shared-machine resource planning (src/resources.py), with a fake nvidia-smi, plus a real
2-process DDP run on CPU through the CLI (same code path as multi-GPU)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from g4shape.resources import (HANDOFF_ENV, ResourceConfig, ResourcePlan, cpu_budget, handoff,
                           inherited, plan_resources)

ROOT = Path(__file__).resolve().parents[1]
GB = 1024


def fake_smi(*snapshots):
    """nvidia-smi stand-in: each call returns the next snapshot (the last one repeats).
    A snapshot is a list of (index, total_GB, used_GB, util%)."""
    calls = {"n": 0}

    def smi(args):
        snap = snapshots[min(calls["n"], len(snapshots) - 1)]
        calls["n"] += 1
        return "\n".join(f"{i}, GPU-{i:04d}-uuid, Tesla X, {t * GB}, {u * GB}, {util}"
                         for i, t, u, util in snap)
    smi.calls = calls
    return smi


@pytest.fixture
def env(monkeypatch):
    for k in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", HANDOFF_ENV, "LOCAL_RANK"):
        monkeypatch.delenv(k, raising=False)
    # On a GPU machine, earlier tests that run trainer.fit on CPU leave CUDA initialised
    # (Lightning's isolate_rng collects every visible GPU's RNG state). These tests use a fake
    # nvidia-smi and no real GPU, so plan_resources' guard must not see that leftover state;
    # test_refuses_after_cuda_is_initialised checks the guard itself.
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    return monkeypatch


def _plan(rc=None, smi=None, **kw):
    kw.setdefault("cpu_info", lambda: (16, 2.0))
    kw.setdefault("sleep", lambda s: None)
    kw.setdefault("nccl_check", lambda visible, env: True)
    return plan_resources(rc or ResourceConfig(sample_seconds=1), smi=smi, log=lambda m: None, **kw)


SHARED = [(0, 40, 30, 95),   # someone training: busy
          (1, 40, 1, 0),     # free, 39 GB
          (2, 24, 20, 0),    # idle but only 4 GB free: not enough
          (3, 40, 5, 3)]     # free, 35 GB, light util


def test_uses_only_free_gpus_most_free_memory_first(env):
    p = _plan(smi=fake_smi(SHARED))
    assert p.accelerator == "gpu" and p.devices == 2 and p.strategy == "ddp"
    assert [g["index"] for g in p.gpus] == [1, 3]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1,3"  # others' GPUs are invisible to us
    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    # allocator cap = memory_share x free/total, per rank
    assert p.memory_fraction == [round(0.9 * 39 / 40, 4), round(0.9 * 35 / 40, 4)]


def test_worst_case_over_samples(env):
    """A GPU that spikes during the sampling window is not taken."""
    spiky = [[(0, 40, 1, 0), (1, 40, 1, 0)], [(0, 40, 1, 0), (1, 40, 20, 80)]]
    p = _plan(ResourceConfig(sample_seconds=2), smi=fake_smi(*spiky))
    assert [g["index"] for g in p.gpus] == [0]


def test_max_gpus_and_explicit_devices(env):
    assert _plan(ResourceConfig(sample_seconds=1, max_gpus=1), smi=fake_smi(SHARED)).devices == 1
    env.delenv("CUDA_VISIBLE_DEVICES")
    p = _plan(smi=fake_smi(SHARED), devices=[3])
    assert [g["index"] for g in p.gpus] == [3] and p.strategy == "auto"


def test_respects_preset_cuda_visible_devices(env):
    env.setenv("CUDA_VISIBLE_DEVICES", "0,3")  # e.g. set by a scheduler / the admin
    p = _plan(smi=fake_smi(SHARED))
    assert [g["index"] for g in p.gpus] == [3] and os.environ["CUDA_VISIBLE_DEVICES"] == "3"
    env.setenv("CUDA_VISIBLE_DEVICES", "GPU-0001")  # UUID form
    p = _plan(smi=fake_smi(SHARED))
    assert [g["index"] for g in p.gpus] == [1] and os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-0001"


def test_all_busy_policies(env):
    busy = [(0, 40, 39, 99), (1, 40, 30, 90)]
    assert _plan(ResourceConfig(sample_seconds=1, on_busy="cpu"), smi=fake_smi(busy)).accelerator == "cpu"
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    with pytest.raises(RuntimeError, match="no free GPU"):
        _plan(ResourceConfig(sample_seconds=1, on_busy="fail"), smi=fake_smi(busy))


def test_waits_until_a_gpu_frees_up(env):
    busy, later = [(0, 40, 39, 99)], [(0, 40, 2, 0)]
    slept = []
    smi = fake_smi(busy, busy, later)
    p = _plan(ResourceConfig(sample_seconds=1, poll_seconds=30), smi=smi, sleep=slept.append)
    assert p.accelerator == "gpu" and slept == [30, 30]


def test_wait_times_out(env):
    t = {"now": 0.0}

    def sleep(s):
        t["now"] += s
    with pytest.raises(RuntimeError, match="within"):
        _plan(ResourceConfig(sample_seconds=1, poll_seconds=60, wait_timeout_min=2),
              smi=fake_smi([(0, 40, 39, 99)]), sleep=sleep, clock=lambda: t["now"])


def test_no_nvidia_smi_or_cpu_requested(env):
    assert _plan(smi=lambda a: "").accelerator == "cpu"
    p = _plan(smi=fake_smi(SHARED), accelerator="cpu", devices=2)
    assert p.accelerator == "cpu" and p.devices == 2 and p.strategy == "ddp"


def test_unreadable_gpu_counts_as_busy(env):
    smi = lambda a: "0, GPU-a, X, 40960, [N/A], [N/A]\n1, GPU-b, X, 40960, 1024, 0"  # noqa: E731
    p = _plan(smi=smi)
    assert [g["index"] for g in p.gpus] == [1]


def test_cpu_budget_leaves_room_for_others(env):
    assert cpu_budget(0.5, lambda: (16, 2.0)) == 7       # half of the 14 idle cores
    assert cpu_budget(0.5, lambda: (16, 15.9)) == 1      # machine busy: one core
    p = _plan(smi=fake_smi(SHARED), cpu_info=lambda: (32, 8.0))  # 12 cores for 2 ranks
    assert (p.num_workers, p.torch_threads) == (5, 1)
    assert _plan(smi=fake_smi(SHARED), num_workers=2).num_workers == 2  # explicit wins
    p = _plan(smi=lambda a: "", cpu_info=lambda: (16, 0.0))  # CPU run: 8 cores, half compute
    assert p.accelerator == "cpu" and (p.num_workers, p.torch_threads) == (4, 4)


def test_global_batch_is_split_over_devices():
    p = ResourcePlan(accelerator="gpu", devices=3, strategy="ddp", visible_devices="0,1,2")
    assert p.per_device_batch(256) == 86 and ResourcePlan("cpu", 1, "auto", None).per_device_batch(256) == 256


def test_handoff_only_reaches_other_ranks(env, tmp_path):
    plan = ResourcePlan("gpu", 2, "ddp", "1,3", memory_fraction=[0.8, 0.7])
    handoff(tmp_path, plan)
    assert inherited() is None              # rank 0 (no LOCAL_RANK) plans itself
    env.setenv("LOCAL_RANK", "0")
    assert inherited() is None
    env.setenv("LOCAL_RANK", "1")           # a rank Lightning re-launched
    run_dir, p = inherited()
    assert run_dir == tmp_path and p == plan


def test_nccl_check_falls_back_to_no_p2p_then_one_gpu(env):
    """GPUs whose NCCL hangs: retry without peer-to-peer (kept for every rank), else 1 GPU."""
    calls = []

    def no_p2p_only(visible, extra):
        calls.append((visible, dict(extra)))
        return extra.get("NCCL_P2P_DISABLE") == "1"
    p = _plan(smi=fake_smi(SHARED), nccl_check=no_p2p_only)
    assert calls == [("1,3", {}), ("1,3", {"NCCL_P2P_DISABLE": "1"})]
    assert p.devices == 2 and p.strategy == "ddp" and p.env == {"NCCL_P2P_DISABLE": "1"}
    assert ResourcePlan.from_dict(p.to_dict()).env == p.env  # reaches the other ranks

    p = _plan(smi=fake_smi(SHARED), nccl_check=lambda visible, extra: False)
    assert p.devices == 1 and p.strategy == "auto" and p.env == {}
    assert [g["index"] for g in p.gpus] == [1] and os.environ["CUDA_VISIBLE_DEVICES"] == "1"

    calls.clear()  # one GPU or check disabled: nothing to check
    _plan(ResourceConfig(sample_seconds=1, max_gpus=1), smi=fake_smi(SHARED), nccl_check=no_p2p_only)
    _plan(ResourceConfig(sample_seconds=1, nccl_check=False), smi=fake_smi(SHARED),
          nccl_check=no_p2p_only)
    assert calls == []


def test_check_nccl_kills_a_hung_check(env, monkeypatch, tmp_path):
    """A check that never finishes is killed at the time limit (with its children) -> False."""
    import g4shape.resources as r
    pidfile = tmp_path / "child.pid"
    monkeypatch.setattr(r, "_NCCL_CHECK_SCRIPT",
                        "import subprocess, sys, time\n"
                        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                        f"open({str(pidfile)!r}, 'w').write(str(c.pid))\n"
                        "time.sleep(60)\n")
    assert r.check_nccl("0", timeout=3) is False
    child = int(pidfile.read_text())
    import time
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)  # the spawned "rank" is gone too


def test_rank_exits_when_rank0_dies(tmp_path):
    """A rank > 0 whose launcher (rank 0) is killed exits by itself instead of hanging."""
    import signal
    import time
    pidfile = tmp_path / "rank1.pid"
    rank1 = ("import os, time; from g4shape.resources import exit_with_parent; "
             "open(%r, 'w').write(str(os.getpid())); exit_with_parent(0.1, log=lambda m: None); "
             "time.sleep(60)") % str(pidfile)
    rank0 = subprocess.Popen([sys.executable, "-c",
                              f"import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', {rank1!r}]); "
                              "time.sleep(60)"], cwd=ROOT)
    for _ in range(100):
        if pidfile.exists() and pidfile.read_text():
            break
        time.sleep(0.1)
    child = int(pidfile.read_text())
    rank0.send_signal(signal.SIGKILL)
    rank0.wait()
    for _ in range(50):  # well under a second of polling at 0.1 s
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        os.kill(child, signal.SIGKILL)
        pytest.fail("rank 1 kept running after rank 0 died")


def test_two_process_ddp_run_through_cli(data, tmp_path):
    """Real DDP (2 CPU processes, gloo) via `python -m g4shape.train`: rank 1 is re-launched by
    Lightning and must reuse rank 0's run dir; metrics are gathered; test runs on 1 device."""
    d, _, _ = data
    cfg = tmp_path / "ddp.yaml"
    cfg.write_text(yaml.safe_dump({
        "data": {"genome_dir": str(d / "genome"), "index_dir": str(d / "index"),
                 "batch_size": 4, "eval_batch_size": 4, "num_workers": 0},
        "model": {"hidden_dim": 8, "graph_encoder_kwargs": {"num_layers": 1, "num_heads": 2}},
        "trainer": {"accelerator": "cpu", "devices": 2, "out_dir": str(tmp_path / "runs"),
                    "run_name": "ddp"},
        "resources": {"nice": 0},
    }))
    env = {k: v for k, v in os.environ.items() if k not in (HANDOFF_ENV, "LOCAL_RANK")}
    res = subprocess.run([sys.executable, "-m", "g4shape.train", "--config", str(cfg)], cwd=ROOT,
                         env=env, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stderr[-3000:]
    runs = list((tmp_path / "runs").iterdir())
    assert [r.name for r in runs] == ["ddp"]  # rank 1 did not create its own run dir
    plan = json.loads((runs[0] / "resources.json").read_text())
    assert plan["devices"] == 2 and plan["strategy"] == "ddp"
    test = json.loads((runs[0] / "test_metrics.json").read_text())
    assert {"test_pearson", "test_jaccard", "test_auprc"} <= set(test)
    assert test["test_n_samples"] == 10  # whole test split, on one device
    import csv
    val_n = {float(r["val_n_samples"]) for r in csv.DictReader(open(runs[0] / "metrics.csv"))
             if r.get("val_n_samples")}
    assert val_n == {10.0}  # gathered from both ranks (5 + 5), not one rank's half


class _FakeRlimit:
    """``resource`` module stand-in for ensure_open_file_capacity."""
    RLIMIT_NOFILE, RLIM_INFINITY = 7, -1

    def __init__(self, soft, hard):
        self.limits = (soft, hard)

    def getrlimit(self, which):
        return self.limits

    def setrlimit(self, which, limits):
        self.limits = limits


@pytest.mark.parametrize("soft, hard, want_soft, want_strategy", [
    (1024, 524288, 524288, "file_descriptor"),   # raised to the hard limit, fds are enough
    (1024, 4096, 4096, "file_system"),           # hard limit too low: no fd per tensor
    (1024, -1, 65536, "file_descriptor"),        # unlimited hard limit
])
def test_open_file_capacity(soft, hard, want_soft, want_strategy):
    import torch.multiprocessing as mp

    from g4shape.resources import ensure_open_file_capacity

    before = mp.get_sharing_strategy()
    try:
        mp.set_sharing_strategy("file_descriptor")
        rl = _FakeRlimit(soft, hard)
        assert ensure_open_file_capacity(rlimit=rl) == (want_soft, want_strategy)
        assert rl.limits[0] == want_soft
    finally:
        mp.set_sharing_strategy(before)


def test_prepare_worker_fork_collects_and_freezes():
    import gc

    from g4shape.resources import prepare_worker_fork

    class Cycle:
        pass

    a, b = Cycle(), Cycle()
    a.other, b.other = b, a
    del a, b                       # unreachable cycle: only a collection frees it
    try:
        assert prepare_worker_fork() >= 2
        assert gc.get_freeze_count() > 0  # inherited objects are invisible to a forked child's gc
    finally:
        gc.unfreeze()



def test_refuses_after_cuda_is_initialised(env):
    env.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="before CUDA is initialised"):
        _plan(smi=fake_smi(SHARED))
