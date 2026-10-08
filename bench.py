"""
Load generator for OpenAI-compatible LLM servers (vLLM, SGLang, TensorRT-LLM).

All three engines expose POST /v1/completions with streaming, so one client
measures all of them the same way. For every request we record:

  TTFT  time to first token      = first streamed chunk - request sent      (prefill + queueing)
  TPOT  time per output token    = (end - first chunk) / (output tokens - 1) (decode speed per user)
  E2E   end-to-end latency       = end - request sent

and for the whole run: request throughput, output tokens/s, and peak GPU memory.

Load model: closed loop. C workers each send a request, wait for it to finish,
then send the next, until N requests are done. Sweeping C (1, 4, 16, 64) shows
how each engine trades per-user latency for total throughput as batches grow.

Usage:
  python bench.py --url http://localhost:8000 --model Qwen/Qwen2.5-1.5B-Instruct \
      --engine vllm --concurrency 1 4 16 64 --num-prompts 128 \
      --input-len 512 --output-len 128 --out results/vllm.jsonl
"""
import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass, asdict, field

import aiohttp

from gpu_monitor import GpuMemoryMonitor


@dataclass
class RequestResult:
    ok: bool
    ttft: float = 0.0            # seconds
    e2e: float = 0.0             # seconds
    output_tokens: int = 0
    prompt_tokens: int = 0
    chunk_times: list = field(default_factory=list)   # arrival time of each streamed chunk
    error: str = ""

    @property
    def tpot(self):
        """Decode time per token, excluding the first token (which TTFT already covers)."""
        if self.output_tokens < 2:
            return None
        return (self.e2e - self.ttft) / (self.output_tokens - 1)


# -----------------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------------
_WORDS = ("the model reads every token of this prompt during prefill and then writes "
          "new tokens one at a time during decode which is memory bound on the gpu").split()


def make_prompts(n, input_len, tokenizer=None, seed=0):
    """n prompts of about input_len tokens each. Random words, so no two prompts
    share a prefix and prefix caching cannot make one engine look faster."""
    rng = random.Random(seed)
    prompts = []
    for i in range(n):
        words = [f"[{i}]"] + [rng.choice(_WORDS) for _ in range(input_len)]
        text = " ".join(words)
        if tokenizer is not None:                     # trim to the exact token count
            ids = tokenizer(text, add_special_tokens=False)["input_ids"][:input_len]
            text = tokenizer.decode(ids)
        prompts.append(text)
    return prompts


# -----------------------------------------------------------------------------
# One streamed request
# -----------------------------------------------------------------------------
async def send_request(session, url, model, prompt, output_len):
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_len,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,                            # always generate output_len tokens: fair comparison
        "stream_options": {"include_usage": True},     # server reports exact token counts in the last chunk
    }
    res = RequestResult(ok=False)
    start = time.perf_counter()
    try:
        async with session.post(url + "/v1/completions", json=body) as resp:
            if resp.status != 200:
                res.error = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                return res
            text_chunks = 0
            async for raw in resp.content:             # server-sent events, one "data: {...}" per line
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                msg = json.loads(data)
                now = time.perf_counter()
                if msg.get("usage"):
                    res.output_tokens = msg["usage"].get("completion_tokens", 0)
                    res.prompt_tokens = msg["usage"].get("prompt_tokens", 0)
                choices = msg.get("choices") or []
                if choices and choices[0].get("text"):
                    if text_chunks == 0:
                        res.ttft = now - start
                    res.chunk_times.append(now - start)
                    text_chunks += 1
            res.e2e = time.perf_counter() - start
            if res.output_tokens == 0:                 # server did not send usage: one chunk ~ one token
                res.output_tokens = text_chunks
            res.ok = text_chunks > 0
            if not res.ok:
                res.error = "no tokens streamed"
    except Exception as exc:                           # connection reset, timeout, bad JSON
        res.error = f"{type(exc).__name__}: {exc}"
    return res


# -----------------------------------------------------------------------------
# One load level (closed loop with C workers)
# -----------------------------------------------------------------------------
async def run_level(url, model, prompts, output_len, concurrency, timeout_s=600):
    queue = asyncio.Queue()
    for p in prompts:
        queue.put_nowait(p)
    results = []

    async def worker(session):
        while True:
            try:
                p = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            results.append(await send_request(session, url, model, p, output_len))

    conn = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=timeout_s)) as s:
        t0 = time.perf_counter()
        await asyncio.gather(*(worker(s) for _ in range(concurrency)))
        duration = time.perf_counter() - t0
    return results, duration


def pct(values, q):
    """q-th percentile, linear interpolation (same as numpy's default)."""
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def summarize(results, duration, concurrency):
    ok = [r for r in results if r.ok]
    ttft = [r.ttft for r in ok]
    tpot = [r.tpot for r in ok if r.tpot is not None]
    e2e = [r.e2e for r in ok]
    out_tok = sum(r.output_tokens for r in ok)
    in_tok = sum(r.prompt_tokens for r in ok)
    ms = lambda x: None if x is None else round(x * 1000, 2)
    return {
        "concurrency": concurrency,
        "requests": len(results),
        "succeeded": len(ok),
        "errors": sorted({r.error for r in results if not r.ok})[:3],
        "duration_s": round(duration, 3),
        "request_throughput": round(len(ok) / duration, 3),
        "output_tok_per_s": round(out_tok / duration, 1),
        "total_tok_per_s": round((in_tok + out_tok) / duration, 1),
        "mean_prompt_tokens": round(in_tok / len(ok), 1) if ok else None,
        "mean_output_tokens": round(out_tok / len(ok), 1) if ok else None,
        "ttft_ms_mean": ms(statistics.mean(ttft)) if ttft else None,
        "ttft_ms_p50": ms(pct(ttft, 50)),
        "ttft_ms_p99": ms(pct(ttft, 99)),
        "tpot_ms_mean": ms(statistics.mean(tpot)) if tpot else None,
        "tpot_ms_p50": ms(pct(tpot, 50)),
        "tpot_ms_p99": ms(pct(tpot, 99)),
        "e2e_ms_mean": ms(statistics.mean(e2e)) if e2e else None,
        "e2e_ms_p50": ms(pct(e2e, 50)),
        "e2e_ms_p99": ms(pct(e2e, 99)),
    }


async def wait_ready(url, timeout_s=900):
    """Poll /v1/models until the server answers (model loading can take minutes)."""
    deadline = time.time() + timeout_s
    async with aiohttp.ClientSession() as s:
        while time.time() < deadline:
            try:
                async with s.get(url + "/v1/models", timeout=aiohttp.ClientTimeout(total=5)) as r:
                    if r.status == 200:
                        return True
            except Exception:
                pass
            await asyncio.sleep(2)
    return False


async def main_async(args):
    if not await wait_ready(args.url, args.ready_timeout):
        raise SystemExit(f"server at {args.url} not ready after {args.ready_timeout}s")

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Warm-up: first requests pay for CUDA graph capture, kernel autotuning and cache allocation.
    warm = make_prompts(args.warmup, args.input_len, tokenizer, seed=1)
    await run_level(args.url, args.model, warm, args.output_len, concurrency=min(4, args.warmup))

    monitor = GpuMemoryMonitor(args.gpu_index)
    rows = []
    for c in args.concurrency:
        prompts = make_prompts(args.num_prompts, args.input_len, tokenizer, seed=100 + c)
        monitor.start()
        results, duration = await run_level(args.url, args.model, prompts, args.output_len, c)
        mem = monitor.stop()
        row = {"engine": args.engine, "model": args.model, "tag": args.tag,
               "input_len": args.input_len, "output_len": args.output_len,
               **summarize(results, duration, c), **mem}
        rows.append(row)
        print(f"[{args.engine}] c={c:>3}  {row['request_throughput']:.2f} req/s  "
              f"{row['output_tok_per_s']:.0f} out tok/s  TTFT p50 {row['ttft_ms_p50']} ms  "
              f"TPOT p50 {row['tpot_ms_p50']} ms  ok {row['succeeded']}/{row['requests']}  "
              f"GPU peak {row.get('gpu_mem_peak_mib')} MiB", flush=True)
        if args.out:
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
    return rows


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--model", required=True, help="model name the server was started with")
    p.add_argument("--engine", default="unknown", help="label for the results (vllm, sglang, trtllm)")
    p.add_argument("--tag", default="", help="extra label, e.g. fp16 or awq-int4")
    p.add_argument("--tokenizer", default=None, help="HF tokenizer for exact input lengths (optional)")
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16, 64])
    p.add_argument("--num-prompts", type=int, default=128)
    p.add_argument("--input-len", type=int, default=512)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--ready-timeout", type=int, default=900)
    p.add_argument("--out", default=None, help="append one JSON line per concurrency level")
    return p.parse_args(argv)


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
