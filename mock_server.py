"""
A fake OpenAI-compatible streaming server with KNOWN timings, used to test bench.py
without a GPU. It waits prefill_ms before the first token, then decode_ms per token,
so the harness must report TTFT ~ prefill_ms and TPOT ~ decode_ms.

  python mock_server.py --port 8099 --prefill-ms 50 --decode-ms 10
"""
import argparse
import asyncio
import json

from aiohttp import web


def make_app(prefill_ms=50.0, decode_ms=10.0, tokens_per_chunk=1, send_usage=True):
    async def models(_):
        return web.json_response({"object": "list", "data": [{"id": "mock", "object": "model"}]})

    async def completions(request):
        body = await request.json()
        n = int(body.get("max_tokens", 16))
        prompt_tokens = len(str(body.get("prompt", "")).split())
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)

        async def send(obj):
            await resp.write(f"data: {json.dumps(obj)}\n\n".encode())

        # Sleep until absolute deadlines, not for fixed intervals: timer granularity
        # (about 15 ms on Windows) then delays single tokens but does not accumulate.
        loop = asyncio.get_running_loop()
        t0 = loop.time()

        async def wait_until(t):
            delay = t - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)

        await wait_until(t0 + prefill_ms / 1000)
        sent = 0
        while sent < n:
            k = min(tokens_per_chunk, n - sent)
            if sent > 0:
                await wait_until(t0 + (prefill_ms + decode_ms * (sent + k - 1)) / 1000)
            await send({"choices": [{"index": 0, "text": " tok" * k, "finish_reason": None}]})
            sent += k
        if send_usage:
            await send({"choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": n,
                                                 "total_tokens": prompt_tokens + n}})
        await resp.write(b"data: [DONE]\n\n")
        return resp

    app = web.Application()
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/completions", completions)
    return app


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8099)
    p.add_argument("--prefill-ms", type=float, default=50)
    p.add_argument("--decode-ms", type=float, default=10)
    a = p.parse_args()
    web.run_app(make_app(a.prefill_ms, a.decode_ms), port=a.port)
