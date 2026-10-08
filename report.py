"""
Turns results/*.jsonl into Markdown tables (one per model + tag) and, if matplotlib
is installed, a throughput-vs-latency plot per model.

  python report.py results            # prints tables, writes results/REPORT.md and results/*.png
"""
import glob
import json
import os
import sys
from collections import defaultdict

COLUMNS = [
    ("concurrency", "Concurrency"),
    ("engine", "Engine"),
    ("request_throughput", "Req/s"),
    ("output_tok_per_s", "Output tok/s"),
    ("ttft_ms_p50", "TTFT p50 ms"),
    ("ttft_ms_p99", "TTFT p99 ms"),
    ("tpot_ms_p50", "TPOT p50 ms"),
    ("e2e_ms_p50", "E2E p50 ms"),
    ("gpu_mem_peak_mib", "GPU peak MiB"),
    ("succeeded", "OK"),
]


def load(result_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(result_dir, "*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:,.1f}" if v >= 100 else f"{v:.2f}"
    return str(v)


def tables(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r.get("tag", ""), r["input_len"], r["output_len"])].append(r)
    out = []
    for (model, tag, il, ol), rs in groups.items():
        rs.sort(key=lambda r: (r["concurrency"], r["engine"]))
        out.append(f"### {model} ({tag or 'default'}), {il} input / {ol} output tokens\n")
        out.append("| " + " | ".join(h for _, h in COLUMNS) + " |")
        out.append("| " + " | ".join("---" for _ in COLUMNS) + " |")
        for r in rs:
            cells = [fmt(r.get(k)) for k, _ in COLUMNS]
            if r["requests"] != r["succeeded"]:
                cells[-1] += f"/{r['requests']}"
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def plot(rows, result_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    paths = []
    by_model = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_model[(r["model"], r.get("tag", ""))][r["engine"]].append(r)
    for (model, tag), engines in by_model.items():
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for engine, rs in sorted(engines.items()):
            rs.sort(key=lambda r: r["concurrency"])
            x = [r["output_tok_per_s"] for r in rs]
            y = [r["tpot_ms_p50"] for r in rs]
            ax.plot(x, y, marker="o", label=engine)
            for r, xi, yi in zip(rs, x, y):
                ax.annotate(f"c={r['concurrency']}", (xi, yi), fontsize=7, xytext=(4, 4),
                            textcoords="offset points")
        ax.set_xlabel("Output tokens/s (all users)")
        ax.set_ylabel("TPOT p50 (ms per token, per user)")
        ax.set_title(f"{model.split('/')[-1]} {tag}: throughput vs per-user speed")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = os.path.join(result_dir, f"{model.split('/')[-1]}-{tag or 'default'}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


if __name__ == "__main__":
    result_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    rows = load(result_dir)
    if not rows:
        raise SystemExit(f"no *.jsonl results in {result_dir}")
    md = tables(rows)
    gpu = os.path.join(result_dir, "gpu.txt")
    header = f"GPU: {open(gpu).read().strip().splitlines()[-1]}\n\n" if os.path.exists(gpu) else ""
    with open(os.path.join(result_dir, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write(header + md)
    print(header + md)
    for p in plot(rows, result_dir):
        print("wrote", p)
