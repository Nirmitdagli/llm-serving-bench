#!/usr/bin/env bash
# Starts each serving engine with the SAME model, context length and GPU memory budget,
# runs the same load sweep against it, then shuts it down so the next engine gets a clean GPU.
#
#   ENGINES="vllm sglang" MODEL=Qwen/Qwen2.5-1.5B-Instruct TAG=fp16 ./run_suite.sh
#   ENGINES="vllm sglang" MODEL=Qwen/Qwen2.5-1.5B-Instruct-AWQ TAG=awq-int4 ./run_suite.sh
#   ENGINES="trtllm" ...   (run inside the TensorRT-LLM NGC container, see README)
#   ENGINES=mock MODEL=mock TOKENIZER_ARG= ./run_suite.sh   (dry run, no GPU)
set -euo pipefail

ENGINES=${ENGINES:-"vllm sglang trtllm"}
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
TAG=${TAG:-fp16}
DTYPE=${DTYPE:-half}              # T4 has no bf16; fp16 on every engine keeps the comparison fair
MAX_LEN=${MAX_LEN:-4096}
MEM=${MEM:-0.85}                  # fraction of GPU memory each engine may use for weights + KV cache
CONCURRENCY=${CONCURRENCY:-"1 4 16 64"}
NUM_PROMPTS=${NUM_PROMPTS:-128}
INPUT_LEN=${INPUT_LEN:-512}
OUTPUT_LEN=${OUTPUT_LEN:-128}
OUT_DIR=${OUT_DIR:-results}
# Extra engine flags, e.g. SGLANG_EXTRA="--attention-backend triton" on a T4.
# VLLM_EXTRA, SGLANG_EXTRA, TRTLLM_EXTRA are appended to that engine's command.
mkdir -p "$OUT_DIR" logs

# exec: the background PID becomes the server itself, so kill really frees the GPU.
start_engine() {
  case "$1" in
    vllm)   PORT=8000
            exec vllm serve "$MODEL" --port $PORT --dtype "$DTYPE" \
              --max-model-len "$MAX_LEN" --gpu-memory-utilization "$MEM" ${VLLM_EXTRA:-} ;;
    sglang) PORT=30000
            exec python -m sglang.launch_server --model-path "$MODEL" --port $PORT --dtype "$DTYPE" \
              --context-length "$MAX_LEN" --mem-fraction-static "$MEM" ${SGLANG_EXTRA:-} ;;
    trtllm) PORT=8001
            # Note: this fraction is of the memory left AFTER weights load, not of the whole GPU.
            exec trtllm-serve "$MODEL" --port $PORT --max_seq_len "$MAX_LEN" \
              --kv_cache_free_gpu_memory_fraction "$MEM" ${TRTLLM_EXTRA:-} ;;
    mock)   PORT=8099              # no GPU needed: dry-run the whole pipeline
            exec python mock_server.py --port $PORT ;;
    *) echo "unknown engine $1"; exit 1 ;;
  esac
}

port_of() { case "$1" in vllm) echo 8000;; sglang) echo 30000;; trtllm) echo 8001;; mock) echo 8099;; esac; }

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee "$OUT_DIR/gpu.txt" \
  || echo "no NVIDIA GPU" > "$OUT_DIR/gpu.txt"

for ENGINE in $ENGINES; do
  echo "=== $ENGINE | $MODEL | $TAG ==="
  start_engine "$ENGINE" > "logs/$ENGINE-$TAG.log" 2>&1 &
  PID=$!
  trap 'kill $PID 2>/dev/null || true' EXIT

  python bench.py --url "http://localhost:$(port_of "$ENGINE")" --model "$MODEL" \
    --engine "$ENGINE" --tag "$TAG" ${TOKENIZER_ARG---tokenizer $MODEL} \
    --concurrency $CONCURRENCY --num-prompts "$NUM_PROMPTS" \
    --input-len "$INPUT_LEN" --output-len "$OUTPUT_LEN" \
    --out "$OUT_DIR/$ENGINE-$TAG.jsonl" \
    || { echo "$ENGINE failed, last log lines:"; tail -n 30 "logs/$ENGINE-$TAG.log"; }

  kill $PID 2>/dev/null || true
  wait $PID 2>/dev/null || true
  sleep 5                          # let the driver release the GPU memory before the next engine
done

${REPORT_PY:-python} report.py "$OUT_DIR"
