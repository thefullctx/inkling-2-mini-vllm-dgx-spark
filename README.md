# Inkling-2-Mini on a Single NVIDIA DGX Spark (vLLM + Humming MoE + DSpark)

Tuned single-node recipe for running **[thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)** natively in **NVFP4 / MXFP8** with **vLLM's Humming MoE backend**, **eager safetensors loading**, and **DSpark speculative decoding** on a **single NVIDIA DGX Spark**.

> **What's new in the tuned vLLM recipe:** Adding `--moe-backend humming`, `--safetensors-load-strategy eager`, persistent `~/.cache/vllm`, and `--gpu-memory-utilization 0.65` delivered **2.2× faster cold startup (`167s` vs `~360s`)**, **+7.9% `c=4` aggregate throughput (`98.41 tok/s`)**, **+8.6% faster deterministic `c=1` decode than SGLang (`49.03` vs `45.16 tok/s`)**, and **20.5 GiB lower unified RAM usage (`85.0 GiB` vs `105.5 GiB`)**.
> Looking for our SGLang recipe (best for 15K+ long-context prefill/decode & 1M context)? See [inkling-2-mini-sglang-dgx-spark](https://github.com/thefullctx/inkling-2-mini-sglang-dgx-spark).

---

## Hardware & Model Overview

* **Hardware:** 1× NVIDIA DGX Spark (128 GB unified LPDDR5X, Blackwell native FP4)
* **Model:** `thinkingmachines/Inkling-2-Mini` — 26.4B total / 3.4B active (~5.4B effective across 3 looped middle-layer passes), multimodal (text + image in, text out), 262K context on single DGX Spark, bundled DSpark speculator
* **Precision:** Native mixed **NVFP4** (layers 0–19 experts) + **MXFP8** (layers 20–23 experts) + **BF16** (attention, vision, draft head & KV cache)
* **Image:** `vllm/vllm-openai:inkling-2-mini` (`linux/arm64` native)

---

## DGX Spark Benchmarks: Stock vLLM vs. Tuned vLLM (Humming + Eager) vs. Tuned SGLang

Measured on a single NVIDIA DGX Spark (`driver 580.178.04`, 121.6 GiB usable unified RAM) using [`benchmarks/bench.py`](benchmarks/bench.py) over `/v1/chat/completions`:

| Metric | Stock vLLM (`0.85` util) | **Tuned vLLM (`humming` + `eager`, `0.65`)** | Tuned SGLang (`page=16` + Fused MoE) |
| :--- | :---: | :---: | :---: |
| **Smoke Tests** (Reasoning, Tools, Vision) | ✅ / ✅ / ✅ | ✅ / ✅ / ✅ | ✅ / ✅ / ✅ |
| **Cold Startup Time** | ~360s | **167s (2.2× faster)** *(~45s warm)* | ~372s |
| **`c=1` Short Decode (`temp=0.0`)** | — | **49.03 tok/s** *(peak 58.4)* | 45.16 tok/s |
| **`c=1` Short Decode (`temp=0.6`)** | **52.30 tok/s** | 49.96 tok/s *(peak 57.7)* | 49.56 tok/s |
| **15.6K Warm Prefix TTFT** | 0.23s | **0.21s** | **0.21s** |
| **15.6K Long-Context Decode** | 38.43 tok/s | 40.42 tok/s *(+5.2% vs stock)* | **47.75 tok/s** |
| **`c=4` Concurrent Aggregate (`temp=0.6`)** | 91.17 tok/s | **98.41 tok/s (+7.9%!)** | 90.44–94.17 tok/s |
| **Unified RAM Used** | 105.5 GiB *(16.1 GiB free)* | **85.0 GiB (-20.5 GiB saved!)** | **62.8 GiB** *(~59 GiB free)* |

Raw JSON outputs are in [`benchmarks/results/`](benchmarks/results/).

---

## Quick Start

### Option 1: One-command launch (`start.sh` / Docker Compose)

```bash
git clone https://github.com/thefullctx/inkling-2-mini-vllm-dgx-spark.git
cd inkling-2-mini-vllm-dgx-spark
cp .env.example .env
./start.sh
```

Check status / logs / stop:

```bash
./status.sh
./logs.sh
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

## Unified Memory Budget (1× DGX Spark — 121.6 GiB Usable)

Measured live on `NVIDIA GB10` (`--gpu-memory-utilization 0.65`, `--max-model-len 262144`):

| Component | Footprint |
| :--- | :---: |
| Main weights + DSpark draft head + KV cache + CUDA graphs & OS | ~85.0 GiB |
| **Free Headroom** | **~36.6 GiB** |

## Helper Scripts & Security

* `./status.sh` — Check container health and `/v1/models`
* `./logs.sh` — Tail live vLLM logs
* Inline overrides work without editing `.env` (e.g. `PORT=8001 ./start.sh`)
* Set `API_KEY=your-secret` or `HOST=127.0.0.1` in `.env` if you don't want an unauthenticated `0.0.0.0` bind.
