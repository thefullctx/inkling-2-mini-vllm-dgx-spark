# Inkling-2-Mini on a Single NVIDIA DGX Spark (vLLM + DSpark)

Single-node recipe for running **[thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)** natively in **NVFP4 / MXFP8** with **DSpark speculative decoding** on a **single NVIDIA DGX Spark** using **vLLM**.

> **When to pick vLLM vs. SGLang on DGX Spark:** In our head-to-head DGX Spark benchmark, **vLLM** edged out SGLang by **+5.5% on short-context `c=1` decode (52.3 tok/s)** and **+3.7% on `c=4` concurrency (91.2 tok/s)**, while **SGLang** won overall on long-context prefill (**2.06× faster**), long-context decode (**+17.4%**), and lower unified RAM usage.
> Looking for our overall winner (SGLang)? See [inkling-2-mini-sglang-dgx-spark](https://github.com/thefullctx/inkling-2-mini-sglang-dgx-spark).

---

## Hardware & Model Overview

* **Hardware:** 1× NVIDIA DGX Spark (128 GB unified LPDDR5X, Blackwell native FP4)
* **Model:** `thinkingmachines/Inkling-2-Mini` — 26.4B total / 3.4B active (~5.4B effective across 3 looped middle-layer passes), multimodal (text + image in, text out), capped at 262K context on single DGX Spark, bundled DSpark speculator
* **Precision:** Native mixed **NVFP4 + MXFP8** (~20 GB checkpoint, zero requantization)
* **Image:** `vllm/vllm-openai:inkling-2-mini` (`linux/arm64` native)

---

## Head-to-Head DGX Spark Benchmarks (vLLM vs. SGLang)

Measured on a single NVIDIA DGX Spark (`driver 580.178.04`, 121.6 GiB usable unified RAM) using [`benchmarks/bench.py`](benchmarks/bench.py) over `/v1/chat/completions`:

| Metric | **vLLM + DSpark (this repo)** | SGLang + DSpark | Winner |
| :--- | :---: | :---: | :---: |
| **Smoke Tests** (Chat/Reasoning, Tools, Vision) | ✅ / ✅ / ✅ | ✅ / ✅ / ✅ | Tie |
| **`c=1` Short Decode (3-run avg)** | **52.30 tok/s** *(peak 62.8)* | 49.56 tok/s | **vLLM (+5.5%)** |
| **`c=1` Short TTFT** | **0.194s** | 0.221s | **vLLM** |
| **`c=1` 15.6K-Context Cold Prefill** | 9.23s *(~1,690 tok/s)* | **4.48s** *(~3,480 tok/s)* | SGLang (2.06×) |
| **`c=1` 15.6K-Context Warm Prefix TTFT** | **0.227s** | 0.254s | **vLLM** *(both ~0.2s)* |
| **`c=1` 15.6K-Context Decode** | 38.43 tok/s | **45.12 tok/s** | SGLang (+17.4%) |
| **`c=4` Concurrent Aggregate** | **91.17 tok/s** | 87.88 tok/s | **vLLM (+3.7%)** |
| **Unified RAM Used** | 105.5 GiB *(16.1 GiB free)* | **65.2 GiB** *(56.4 GiB free)* | SGLang |

Raw JSON outputs are included in [`benchmarks/results/`](benchmarks/results/).

---

## Quick Start

### Option 1: One-command launch (`start.sh` / Docker Compose)

```bash
git clone https://github.com/thefullctx/inkling-2-mini-vllm-dgx-spark.git
cd inkling-2-mini-vllm-dgx-spark
cp .env.example .env
./start.sh
```

Follow startup logs:

```bash
docker logs -f inkling2-mini-vllm
```

Stop the server:

```bash
./stop.sh
```

### Option 2: `sparkrun`

```bash
sparkrun run sparkrun.yaml
```

### Option 3: Persistent `systemd` user service

```bash
mkdir -p ~/.config/systemd/user
cp systemd/inkling2-mini-vllm.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now inkling2-mini-vllm.service
loginctl enable-linger "$USER"
```

---

## Verify & Benchmark

```bash
python3 benchmarks/bench.py --label vllm-dspark
```
