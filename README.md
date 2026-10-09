# LLM Serving Bench: vLLM vs SGLang, with TensorRT-LLM support

A repeatable harness that measures LLM inference engines on identical hardware, with the
same model, the same prompts and the same GPU memory budget. It reports what users feel
(time to first token, time per output token) and what the operator pays for
(tokens per second and peak GPU memory), for full-precision and 4-bit quantized models.

| File | What it does |
| --- | --- |
| `bench.py` | Async load generator for any OpenAI-compatible `/v1/completions` server. Streams every response and times each token. |
| `gpu_monitor.py` | Samples `nvidia-smi` in a background thread during each load level; keeps peak memory and mean utilization. |
| `run_suite.sh` | Starts one engine at a time with matched settings, runs the sweep, shuts it down, then builds the report. |
| `report.py` | Turns `results/*.jsonl` into Markdown tables and a throughput-vs-latency plot. |
| `mock_server.py` | Fake streaming server with known timings, so the harness can be tested without a GPU. |
| `tests/` | Checks TTFT, TPOT, token counting, multi-token chunks and error handling against the mock. |
| `llm_serving_bench_colab.ipynb` | Runs vLLM and SGLang (fp16 and AWQ int4) on a free Colab T4. |

## What is measured

| Metric | Definition | Why it matters |
| --- | --- | --- |
| **TTFT** | first streamed token minus request sent | Prefill cost plus queueing. What a chat user waits for before text appears. |
| **TPOT** | (end minus first token) / (output tokens minus 1) | Decode speed per user. Decode is memory-bandwidth bound: each step reads all weights and the KV cache. |
| **E2E** | end minus request sent | Total latency for one request. |
| **Output tok/s** | output tokens of all requests / wall time | Total throughput: how many users one GPU can serve. |
| **GPU peak MiB** | max `nvidia-smi` memory.used during the run | Weights plus reserved KV cache. Quantization shrinks the weights and leaves more room for KV cache. |

Each metric is reported as mean, p50 and p99. Token counts come from the server's `usage`
field (`stream_options.include_usage`), so engines that stream several tokens per chunk are
still counted exactly.

## Method (what keeps the comparison fair)

- **Closed-loop concurrency sweep** (1, 4, 16, 64 users). Each user sends a request, waits
  for it, then sends the next. Low concurrency shows best-case latency; high concurrency
  shows how well each engine's continuous batching and KV-cache management scale.
- **Fixed output length.** `ignore_eos: true` and `max_tokens = 128`, so every engine
  generates exactly the same number of tokens.
- **Exact input length.** Prompts are trimmed to 512 tokens with the model's own tokenizer.
- **No prefix-cache shortcuts.** Every prompt is random and unique, so automatic prefix
  caching (SGLang's RadixAttention, vLLM's APC) cannot make one engine look faster.
- **Same memory budget.** All engines get the same memory fraction. vLLM and SGLang reserve
  it at startup for KV cache, so "peak memory" mostly reflects that setting. TensorRT-LLM's
  fraction applies to memory left after the weights load, which is noted next to its results.
- **Warm-up first.** 8 requests before timing, so CUDA graph capture and kernel autotuning
  are not counted.
- **Greedy decoding** (`temperature: 0`) on all engines.

## Results (Tesla T4, 16 GB)

Colab T4, driver 580.82, vLLM 0.31.0, SGLang 0.5.21. Qwen2.5-1.5B-Instruct in fp16 and its
AWQ 4-bit version. 512 input / 128 output tokens, 128 requests per level, memory fraction 0.85,
greedy decoding. Every run completed 128/128 requests. Both engines ran Triton attention
kernels on the T4 (vLLM picked `TRITON_ATTN` itself; SGLang was started with `--attention-backend triton`).

**Summary**

- **vLLM led SGLang on this GPU at every load level.** At 64 users: 609 vs 384 output tok/s
  (fp16). Single-user decode speed was the same (20 ms per token), but SGLang's first token
  took about 2x longer and its throughput scaled less with batch size. Both used Triton
  attention, so the gap is elsewhere in the engine; I have not profiled it yet.
- **Continuous batching:** going from 1 to 64 concurrent users raised vLLM's total throughput
  12x (51 to 609 tok/s) while per-user decode slowed from 20 to 92 ms per token.
- **4-bit AWQ helps most at low load.** vLLM decode went from 20.0 to 8.5 ms per token (2.3x)
  for one user, because decode is memory-bandwidth bound and the weights shrink from 3.03 GB
  to 1.11 GB. At 64 users the gain is only 5% (609 to 637 tok/s): the weights are read once
  per step for the whole batch, so attention and KV-cache traffic dominate.
- Smaller weights also leave more room for KV cache: SGLang allocated 417K tokens of KV cache
  with AWQ vs 346K with fp16.
- **Quantized kernels matter as much as the format.** SGLang's AWQ run was slower than its
  fp16 run on the T4 (35.7 vs 20.1 ms per token for one user), while vLLM's AWQ run was 2.3x
  faster. Same checkpoint, same GPU: the gain depends on each engine's int4 kernels for that
  architecture.

**fp16**

| Users | Engine | Req/s | Output tok/s | TTFT p50 ms | TPOT p50 ms | GPU peak MiB |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | vLLM | 0.40 | 51 | 132.8 | 19.96 | 12,487 |
| 1 | SGLang | 0.35 | 45 | 255.6 | 20.07 | 13,399 |
| 4 | vLLM | 1.49 | 190 | 448.3 | 17.67 | 12,487 |
| 4 | SGLang | 1.28 | 164 | 683.4 | 19.15 | 13,399 |
| 16 | vLLM | 3.31 | 424 | 998.3 | 30.13 | 12,557 |
| 16 | SGLang | 1.98 | 253 | 2,037.3 | 44.94 | 13,421 |
| 64 | vLLM | 4.76 | 609 | 1,789.5 | 92.08 | 12,557 |
| 64 | SGLang | 3.00 | 384 | 7,417.1 | 108.54 | 13,497 |

**AWQ int4**

| Users | Engine | Req/s | Output tok/s | TTFT p50 ms | TPOT p50 ms | GPU peak MiB |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | vLLM | 0.82 | 106 | 132.0 | 8.51 | 12,553 |
| 1 | SGLang | 0.21 | 27 | 277.0 | 35.68 | 13,437 |
| 4 | vLLM | 2.41 | 308 | 472.8 | 9.36 | 12,553 |
| 4 | SGLang | 0.78 | 100 | 930.3 | 32.97 | 13,437 |
| 16 | vLLM | 4.18 | 535 | 1,044.0 | 21.85 | 12,623 |
| 16 | SGLang | 1.68 | 215 | 2,063.9 | 55.87 | 13,547 |
| 64 | vLLM | 4.98 | 637 | 1,873.3 | 87.67 | 12,623 |
| 64 | SGLang | 2.77 | 354 | 7,495.4 | 121.09 | 13,585 |

GPU peak memory is close to the configured 85% budget for both engines because they reserve
KV cache at startup, so it reflects the setting more than the model size.

TensorRT-LLM is not in these tables yet: it runs from NVIDIA's container, which Colab cannot
start. The suite runs it with `ENGINES=trtllm` inside that container on any NVIDIA machine.

## Run it

**Colab (vLLM and SGLang):** open `llm_serving_bench_colab.ipynb`, choose a T4 GPU runtime, Run all.

**Any NVIDIA machine:**

```bash
pip install -r requirements.txt
pip install vllm            # in its own venv
ENGINES=vllm MODEL=Qwen/Qwen2.5-1.5B-Instruct TAG=fp16 ./run_suite.sh

pip install "sglang[all]"   # in a second venv
ENGINES=sglang MODEL=Qwen/Qwen2.5-1.5B-Instruct TAG=fp16 ./run_suite.sh

# 4-bit quantized model, same engines
ENGINES="vllm" MODEL=Qwen/Qwen2.5-1.5B-Instruct-AWQ TAG=awq-int4 ./run_suite.sh
```

**TensorRT-LLM** runs in NVIDIA's container on an L4, A10G, A100 or H100:

```bash
docker run --rm -it --gpus all --ipc=host -v $PWD:/work -w /work \
  nvcr.io/nvidia/tensorrt-llm/release:latest \
  bash -c "pip install aiohttp && ENGINES=trtllm MODEL=Qwen/Qwen2.5-1.5B-Instruct TAG=fp16 ./run_suite.sh"
```

**Single engine, already running:**

```bash
python bench.py --url http://localhost:8000 --model Qwen/Qwen2.5-1.5B-Instruct \
  --engine vllm --concurrency 1 4 16 64 --out results/vllm-fp16.jsonl
python report.py results
```

**Without a GPU** (tests and a dry run of the whole pipeline against the mock server):

```bash
python -m pytest tests -q
ENGINES=mock MODEL=mock TOKENIZER_ARG= ./run_suite.sh
```

## Limitations and next steps

- Closed-loop load only. An open-loop Poisson arrival mode would show queueing behavior
  at a fixed request rate, which is closer to production traffic.
- One input/output shape (512/128). Long-context prefill (8K+) and long generation
  stress different parts of each engine and are worth separate sweeps.
- `nvidia-smi` sampling every 250 ms can miss short spikes; NVML or DCGM would be finer.
- Next: speculative decoding, FP8 on Hopper/Ada, and tensor parallelism across 2+ GPUs.
