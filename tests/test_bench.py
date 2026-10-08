"""
Tests the harness against mock_server.py, whose timings are known.
Runs on any machine: python -m pytest tests -q
"""
import asyncio
import os
import sys

import pytest
from aiohttp import web

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench                                    # noqa: E402
from mock_server import make_app                # noqa: E402

PREFILL_MS, DECODE_MS, OUT = 60.0, 8.0, 32


async def _with_server(app, fn, port):
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    try:
        return await fn(f"http://127.0.0.1:{port}")
    finally:
        await runner.cleanup()


def run(app, fn, port):
    return asyncio.run(_with_server(app, fn, port))


def test_ttft_tpot_match_known_timings():
    async def go(url):
        prompts = bench.make_prompts(8, 64)
        results, duration = await bench.run_level(url, "mock", prompts, OUT, concurrency=4)
        return bench.summarize(results, duration, 4)

    s = run(make_app(PREFILL_MS, DECODE_MS), go, 8761)
    assert s["succeeded"] == 8
    assert s["mean_output_tokens"] == OUT
    assert s["mean_prompt_tokens"] == 65                       # "[i]" + 64 words
    assert PREFILL_MS <= s["ttft_ms_p50"] < PREFILL_MS + 40    # sleep never returns early
    assert DECODE_MS * 0.9 <= s["tpot_ms_p50"] < DECODE_MS * 1.5
    expected_e2e = PREFILL_MS + DECODE_MS * (OUT - 1)
    assert expected_e2e <= s["e2e_ms_p50"] < expected_e2e * 1.6


def test_concurrency_raises_throughput():
    """Mock requests do not compete for a GPU, so 8 workers should be ~8x faster than 1."""
    async def go(url):
        prompts = bench.make_prompts(16, 16)
        r1, d1 = await bench.run_level(url, "mock", prompts, 8, concurrency=1)
        r8, d8 = await bench.run_level(url, "mock", prompts, 8, concurrency=8)
        return bench.summarize(r1, d1, 1), bench.summarize(r8, d8, 8)

    s1, s8 = run(make_app(20, 2), go, 8762)
    assert s8["request_throughput"] > 4 * s1["request_throughput"]


def test_multi_token_chunks_and_missing_usage():
    """Engines may stream several tokens per chunk. With usage, counts stay exact;
    without usage, the harness falls back to counting chunks."""
    async def go(url):
        results, d = await bench.run_level(url, "mock", bench.make_prompts(2, 8), 16, concurrency=2)
        return bench.summarize(results, d, 2)

    with_usage = run(make_app(10, 1, tokens_per_chunk=4, send_usage=True), go, 8763)
    no_usage = run(make_app(10, 1, tokens_per_chunk=4, send_usage=False), go, 8764)
    assert with_usage["mean_output_tokens"] == 16
    assert no_usage["mean_output_tokens"] == 4


async def _overloaded(_):
    return web.Response(status=503, text="overloaded")


def test_server_errors_are_counted_not_raised():
    async def go(url):
        results, d = await bench.run_level(url, "mock", bench.make_prompts(3, 8), 8, concurrency=3)
        return bench.summarize(results, d, 3)

    app = web.Application()
    app.router.add_post("/v1/completions", _overloaded)
    s = run(app, go, 8765)
    assert s["succeeded"] == 0 and s["requests"] == 3
    assert "HTTP 503" in s["errors"][0]


def test_percentile_matches_numpy_linear():
    assert bench.pct([1, 2, 3, 4], 50) == 2.5
    assert bench.pct([10], 99) == 10
    assert bench.pct(list(range(101)), 99) == 99


def test_prompts_are_unique_and_deterministic():
    a, b = bench.make_prompts(5, 20, seed=3), bench.make_prompts(5, 20, seed=3)
    assert a == b and len(set(a)) == 5
