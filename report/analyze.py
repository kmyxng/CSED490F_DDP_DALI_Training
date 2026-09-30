"""Summarize the Lab 3 Nsight logs (exported to .sqlite) into report/results.json.

Usage (at the repository root):
    python report/analyze.py

Every metric is measured inside the NVTX "Epoch 0" range, so setup time
(data loader / model / NCCL init) is reported separately and does not dilute it.
Only the Python standard library is used.
"""
import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

LOG_ROOT = Path("nsight_logs")
OUT = Path("report/results.json")
NVIDIA_SMI = Path("logs/nvidia_smi.csv")  # sampled every 200 ms during our Vast.ai runs

RUNS = {
    # (source, mode, #GPU): sqlite path
    ("ta", "dp", 1): "ta/dp/gpu_1.sqlite",
    ("ta", "dp", 2): "ta/dp/gpu_2.sqlite",
    ("ta", "dp", 4): "ta/dp/gpu_4.sqlite",
    ("ta", "ddp", 1): "ta/ddp/gpu_1.sqlite",
    ("ta", "ddp", 2): "ta/ddp/gpu_2.sqlite",
    ("ta", "ddp", 4): "ta/ddp/gpu_4.sqlite",
    ("ta", "ddp_dali", 1): "ta/ddp_dali/gpu_1.sqlite",
    ("ta", "ddp_dali", 2): "ta/ddp_dali/gpu_2.sqlite",
    ("ta", "ddp_dali", 4): "ta/ddp_dali/gpu_4.sqlite",
    ("ours", "dp", 1): "dp_20260929_141639/gpu_1.sqlite",
    ("ours", "dp", 2): "dp_20260929_141639/gpu_2.sqlite",
    ("ours", "ddp", 1): "ddp_20260929_141819/gpu_1.sqlite",
    ("ours", "ddp", 2): "ddp_20260929_141819/gpu_2.sqlite",
    ("ours", "ddp_dali", 1): "ddp_dali_20260929_141948/gpu_1.sqlite",
    ("ours", "ddp_dali", 2): "ddp_dali_20260929_141948/gpu_2.sqlite",
}

# NVTX ranges annotated in train_cifar.py and handler/{DP,DDP}/train.py
SETUP_RANGES = ["Set data loader", "Set Data Parallelism", "Sync model parameters"]
STEP_RANGES = ["upload data to GPU", "forward", "loss", "backward"]

GPU_METRICS = [
    "GPU Active [Throughput %]",
    "SMs Active [Throughput %]",
    "SM Issue [Throughput %]",
    "Tensor Active [Throughput %]",
    "DRAM Read Bandwidth [Throughput %]",
    "DRAM Write Bandwidth [Throughput %]",
    "PCIe RX Throughput [Throughput %]",
    "PCIe TX Throughput [Throughput %]",
]

MEMCPY_KINDS = {1: "HtoD", 2: "DtoH", 8: "DtoD", 10: "PtoP"}


def nvtx_rows(db):
    """All closed NVTX ranges as (name, start, end, globalTid)."""
    return db.execute("""
        SELECT coalesce(n.text, s.value), n.start, n.end, n.globalTid
        FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId = s.id
        WHERE n.end IS NOT NULL
    """).fetchall()


def union_length(intervals, lo, hi):
    """Total length of the union of [start, end) intervals clipped to [lo, hi)."""
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted((max(s, lo), min(e, hi)) for s, e in intervals if e > lo and s < hi):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def has_table(db, name):
    return db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def analyze(path):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = nvtx_rows(db)

    # One "Epoch 0" range per training process (1 for DP, #GPU for DDP)
    epochs = {tid: (s, e) for name, s, e, tid in rows if name == "Epoch 0"}
    lo = min(s for s, _ in epochs.values())
    hi = max(e for _, e in epochs.values())

    # Per-rank breakdown of the training loop
    ranks = []
    for tid, (es, ee) in epochs.items():
        mine = [(n, s, e) for n, s, e, t in rows if t == tid and es <= s and e <= ee]
        batches = [(s, e) for n, s, e in mine if n.startswith("Batch ") or n.startswith("Train batch")]
        n_iter = len(batches)
        step = {k: sum(e - s for n, s, e in mine if n == k) / n_iter / 1e6 for k in STEP_RANGES}
        batch_ms = sum(e - s for s, e in batches) / n_iter / 1e6
        epoch_ms = (ee - es) / 1e6
        ranks.append({
            "epoch_s": epoch_ms / 1e3,
            "iters": n_iter,
            "iter_ms": epoch_ms / n_iter,
            "batch_range_ms": batch_ms,
            # time between two batch ranges: waiting for the data loader (+ metric all-reduce/logging)
            "outside_batch_ms": epoch_ms / n_iter - batch_ms,
            "step_ms": step,
        })

    # Setup ranges (before the epoch)
    setup = {k: max([(e - s) / 1e9 for n, s, e, _ in rows if n == k], default=None) for k in SETUP_RANGES}

    # Library NVTX ranges inside the epoch window: NCCL calls, DALI reader
    lib = {}
    for n, s, e, _ in rows:
        if lo <= s and e <= hi and (n.startswith("nccl") or n.startswith("[DALI][Loader]")):
            d = lib.setdefault(n, [0, 0.0])
            d[0] += 1
            d[1] += (e - s) / 1e6
    lib = {k: {"count": c, "total_ms": round(t, 2)} for k, (c, t) in sorted(lib.items(), key=lambda x: -x[1][1])}

    # Kernel timeline per device: busy ratio and NCCL share
    kernels = db.execute("""
        SELECT k.deviceId, k.start, k.end, s.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName = s.id
        WHERE k.end > ? AND k.start < ?
    """, (lo, hi)).fetchall()
    # NCCL kernels also run while waiting for the other ranks, so compute kernels are counted separately
    devices = {}
    for dev in sorted({k[0] for k in kernels}):
        ks = [(s, e) for d, s, e, _ in kernels if d == dev]
        nccl = [(s, e) for d, s, e, n in kernels if d == dev and "nccl" in n.lower()]
        compute = [(s, e) for d, s, e, n in kernels if d == dev and "nccl" not in n.lower()]
        devices[dev] = {
            "kernel_busy_pct": 100 * union_length(ks, lo, hi) / (hi - lo),
            "compute_busy_pct": 100 * union_length(compute, lo, hi) / (hi - lo),
            "compute_ms_per_iter": union_length(compute, lo, hi) / 1e6 / ranks[0]["iters"],
            "nccl_kernel_ms": union_length(nccl, lo, hi) / 1e6,
        }

    # Memcpy inside the epoch, per kind
    memcpy = {}
    for kind, n, b, t in db.execute("""
        SELECT copyKind, count(*), sum(bytes), sum(end - start)
        FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start >= ? AND end <= ? GROUP BY copyKind
    """, (lo, hi)):
        memcpy[MEMCPY_KINDS.get(kind, str(kind))] = {"count": n, "MB": b / 2**20, "ms": t / 1e6}

    # Peak allocated device memory (only when captured with --cuda-memory-usage)
    mem_peak = None
    if has_table(db, "CUDA_GPU_MEMORY_USAGE_EVENTS"):
        mem_peak = {}
        cur = {}
        for dev, b, op in db.execute(
                "SELECT deviceId, bytes, memoryOperationType FROM CUDA_GPU_MEMORY_USAGE_EVENTS ORDER BY start"):
            cur[dev] = cur.get(dev, 0) + (b if op == 0 else -b)
            mem_peak[dev] = max(mem_peak.get(dev, 0), cur[dev])
        mem_peak = {d: v / 2**20 for d, v in sorted(mem_peak.items())}

    # Hardware GPU metrics (TA logs only; needs privileges we did not have on Vast.ai)
    gpu_metrics = None
    if has_table(db, "GPU_METRICS"):
        ids = {name: mid for mid, name in db.execute(
            "SELECT DISTINCT metricId, metricName FROM TARGET_INFO_GPU_METRICS")}
        wanted = {ids[m]: m for m in GPU_METRICS if m in ids}
        gpu_metrics = {}
        q = f"""
            SELECT typeId, metricId, avg(value) FROM GPU_METRICS
            WHERE timestamp BETWEEN ? AND ? AND metricId IN ({",".join(map(str, wanted))})
            GROUP BY typeId, metricId
        """
        types = sorted({t for t, _, _ in db.execute(q, (lo, hi))})
        for t, mid, v in db.execute(q, (lo, hi)):
            gpu_metrics.setdefault(types.index(t), {})[wanted[mid]] = v

    # nvidia-smi samples inside the epoch window (our runs only; TA runs have no overlapping samples)
    smi = None
    if NVIDIA_SMI.exists():
        t0 = db.execute("SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME").fetchone()[0]
        a, b = (t0 + lo) / 1e9, (t0 + hi) / 1e9
        samples = {}
        with NVIDIA_SMI.open() as f:
            reader = csv.reader(f, skipinitialspace=True)
            next(reader)
            for ts, idx, util, mem_util, mem_used in reader:
                t = datetime.strptime(ts, "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
                if a <= t <= b and int(idx) in devices:
                    samples.setdefault(int(idx), []).append(
                        (float(util.split()[0]), float(mem_util.split()[0]), float(mem_used.split()[0])))
        if samples:
            smi = {i: {"util_gpu_pct": sum(x[0] for x in v) / len(v),
                       "util_mem_pct": sum(x[1] for x in v) / len(v),
                       "mem_used_max_MB": max(x[2] for x in v),
                       "samples": len(v)}
                   for i, v in sorted(samples.items())}

    db.close()
    return {
        "epoch_window_s": (hi - lo) / 1e9,
        "ranks": ranks,
        "setup_s": setup,
        "library_nvtx": lib,
        "devices": devices,
        "memcpy": memcpy,
        "mem_peak_MB": mem_peak,
        "gpu_metrics": gpu_metrics,
        "nvidia_smi": smi,
    }


def main():
    results = {}
    for (src, mode, n), rel in RUNS.items():
        path = LOG_ROOT / rel
        print(f"[analyze] {src:4s} {mode:8s} {n} GPU  {path}", flush=True)
        results[f"{src}/{mode}/{n}"] = analyze(path)
    OUT.write_text(json.dumps(results, indent=1))
    print(f"[analyze] wrote {OUT}")


if __name__ == "__main__":
    main()
