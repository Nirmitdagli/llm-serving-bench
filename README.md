# LLM Serving Bench: vLLM vs SGLang vs TensorRT-LLM

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

## Results

Tables from `report.py` are added here after each GPU run, with the GPU name and engine versions.

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
