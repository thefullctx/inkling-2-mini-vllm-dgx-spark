#!/usr/bin/env python3
"""Reproducible smoke test + benchmark for Inkling-2-Mini (SGLang vs vLLM) on DGX Spark GB10.
Stdlib-only Python 3. Uses realistic prompts over /v1/chat/completions (streaming SSE)
so DSpark speculative decoding acceptance reflects real-world performance.
"""
import argparse
import base64
import concurrent.futures
import json
import os
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path


def make_solid_png_b64(width: int = 64, height: int = 64, rgb=(220, 30, 30)) -> str:
    def chunk(ctype: bytes, data: bytes) -> bytes:
        c = ctype + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    return base64.b64encode(png).decode("ascii")


def http_json(url: str, payload: dict | None = None, timeout: int = 120) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def stream_chat(base_url: str, model: str, messages: list, max_tokens: int = 512, temperature: float = 0.6) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    first_tok_t = None
    chunks = 0
    content_parts = []
    reasoning_parts = []
    usage = None

    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                evt = json.loads(body)
            except json.JSONDecodeError:
                continue
            if evt.get("usage"):
                usage = evt["usage"]
            choices = evt.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                c = delta.get("content") or ""
                r = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if c or r:
                    if first_tok_t is None:
                        first_tok_t = time.perf_counter()
                    chunks += 1
                    if c:
                        content_parts.append(c)
                    if r:
                        reasoning_parts.append(r)

    t1 = time.perf_counter()
    ttft = (first_tok_t - t0) if first_tok_t else (t1 - t0)
    gen_time = max(t1 - (first_tok_t or t0), 1e-6)
    total_time = max(t1 - t0, 1e-6)
    completion_tokens = (usage or {}).get("completion_tokens") or chunks
    prompt_tokens = (usage or {}).get("prompt_tokens") or 0
    decode_tps = (max(completion_tokens - 1, 1) / gen_time) if completion_tokens > 1 else (completion_tokens / total_time)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": round(ttft, 4),
        "total_s": round(total_time, 4),
        "decode_tps": round(decode_tps, 2),
        "e2e_tps": round(completion_tokens / total_time, 2),
        "has_reasoning": len("".join(reasoning_parts)) > 0,
        "preview": ("".join(content_parts) or "".join(reasoning_parts))[:120].replace("\n", " "),
    }


def wait_ready(base_url: str, timeout_s: int = 300) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            data = http_json(f"{base_url}/v1/models", timeout=5)
            models = data.get("data") or []
            if models:
                return models[0]["id"]
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"Server at {base_url} did not become ready within {timeout_s}s")


def mem_snapshot_mb() -> dict:
    out = subprocess.check_output(["free", "-m"], text=True)
    for line in out.splitlines():
        if line.startswith("Mem:"):
            parts = line.split()
            return {"total_mb": int(parts[1]), "used_mb": int(parts[2]), "available_mb": int(parts[6])}
    return {}


def run_smoke_tests(base_url: str, model: str) -> dict:
    results = {}

    # 1. Reasoning + chat
    r = http_json(
        f"{base_url}/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "What is 17 * 23? Answer briefly."}],
            "max_tokens": 256,
            "temperature": 0.2,
        },
    )
    msg = r["choices"][0]["message"]
    results["chat_reasoning"] = {
        "ok": "391" in (msg.get("content") or "") or "391" in (msg.get("reasoning_content") or ""),
        "has_reasoning_field": bool(msg.get("reasoning_content") or msg.get("reasoning")),
        "content_preview": (msg.get("content") or "")[:120],
    }

    # 2. Tool calling
    try:
        rt = http_json(
            f"{base_url}/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "Check the current weather in Tokyo using the get_weather tool."}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather for a city",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        },
                    }
                ],
                "max_tokens": 256,
                "temperature": 0.2,
            },
        )
        tmsg = rt["choices"][0]["message"]
        tcalls = tmsg.get("tool_calls") or []
        results["tool_calling"] = {
            "ok": len(tcalls) > 0 and tcalls[0]["function"]["name"] == "get_weather",
            "tool_calls": tcalls,
        }
    except Exception as e:
        results["tool_calling"] = {"ok": False, "error": str(e)}

    # 3. Vision
    try:
        png_b64 = make_solid_png_b64(64, 64, (220, 30, 30))
        rv = http_json(
            f"{base_url}/v1/chat/completions",
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What single primary color is this solid image? Reply in one word."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64}"}},
                        ],
                    }
                ],
                "max_tokens": 128,
                "temperature": 0.2,
            },
        )
        vmsg = rv["choices"][0]["message"]
        vtext = (vmsg.get("content") or "").lower()
        results["vision"] = {"ok": "red" in vtext, "content_preview": (vmsg.get("content") or "")[:120]}
    except Exception as e:
        results["vision"] = {"ok": False, "error": str(e)}

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "results"))
    args = ap.parse_args()

    model = wait_ready(args.base_url)
    print(f"Connected to {args.base_url} | model={model}")

    # Warmup
    print("Running warmup...")
    _ = stream_chat(args.base_url, model, [{"role": "user", "content": "Say hello in one sentence."}], max_tokens=64)

    # Smoke tests
    print("Running smoke tests (chat/reasoning, tool-calling, vision)...")
    smoke = run_smoke_tests(args.base_url, model)
    print(json.dumps(smoke, indent=2))

    # 1. c=1 short-prompt realistic coding/technical prose (3 runs)
    short_prompts = [
        "Write a clean, idiomatic Python async HTTP rate limiter using a token bucket algorithm with type hints and brief docstrings.",
        "Explain how Mixture-of-Experts routing, load balancing loss, and KV cache quantization work in modern LLM inference engines.",
        "Write a Rust function to parse a streaming Server-Sent Events (SSE) byte stream into complete event frames with unit tests.",
    ]
    print("Benchmark 1/3: c=1 short prompt decode (3 runs, max_tokens=512)...")
    c1_runs = [stream_chat(args.base_url, model, [{"role": "user", "content": p}], max_tokens=512) for p in short_prompts]
    c1_avg_tps = round(sum(r["decode_tps"] for r in c1_runs) / len(c1_runs), 2)
    c1_avg_ttft = round(sum(r["ttft_s"] for r in c1_runs) / len(c1_runs), 4)
    print(f"  c=1 avg decode: {c1_avg_tps} tok/s | avg TTFT: {c1_avg_ttft}s")

    # 2. c=1 long-context (~12K tokens) cold vs warm prefix
    para = (
        "In distributed systems engineering, consistent hashing maps both keys and nodes onto a ring so that "
        "adding or removing a node only remaps a small fraction of keys. Virtual nodes improve balance across "
        "heterogeneous hardware, while bounded-load variants prevent hot partitions during traffic spikes. "
    )
    long_doc = "\n".join(f"[Section {i:03d}] {para}" for i in range(1, 260))  # ~13K-15K tokens
    print("Benchmark 2/3: c=1 long-context (~14K tokens) cold vs warm prefix...")
    cold = stream_chat(
        args.base_url,
        model,
        [
            {"role": "system", "content": f"Reference document:\n{long_doc}"},
            {"role": "user", "content": "Summarize the three main concepts in the reference document in 3 bullet points."},
        ],
        max_tokens=256,
    )
    warm = stream_chat(
        args.base_url,
        model,
        [
            {"role": "system", "content": f"Reference document:\n{long_doc}"},
            {"role": "user", "content": "Now explain why virtual nodes help with heterogeneous hardware in 3 sentences."},
        ],
        max_tokens=256,
    )
    print(f"  Long ctx ({cold['prompt_tokens']} prompt tok): cold TTFT={cold['ttft_s']}s, warm TTFT={warm['ttft_s']}s, decode={warm['decode_tps']} tok/s")

    # 3. c=4 concurrent throughput
    print("Benchmark 3/3: c=4 concurrent decode (max_tokens=512 each)...")
    conc_prompts = short_prompts + [
        "Design a clean PostgreSQL schema and indexing strategy for a multi-tenant time-series metrics store with retention policies."
    ]
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        c4_runs = list(
            ex.map(lambda p: stream_chat(args.base_url, model, [{"role": "user", "content": p}], max_tokens=512), conc_prompts)
        )
    c4_wall = max(time.perf_counter() - t0, 1e-6)
    c4_total_tok = sum(r["completion_tokens"] for r in c4_runs)
    c4_agg_tps = round(c4_total_tok / c4_wall, 2)
    c4_per_user = round(sum(r["decode_tps"] for r in c4_runs) / len(c4_runs), 2)
    print(f"  c=4 aggregate: {c4_agg_tps} tok/s | avg per-user: {c4_per_user} tok/s")

    mem = mem_snapshot_mb()
    summary = {
        "label": args.label,
        "model": model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "memory_mb": mem,
        "smoke": smoke,
        "c1_short": {"avg_decode_tps": c1_avg_tps, "avg_ttft_s": c1_avg_ttft, "runs": c1_runs},
        "c1_long": {"cold": cold, "warm": warm},
        "c4_concurrent": {
            "wall_s": round(c4_wall, 2),
            "total_completion_tokens": c4_total_tok,
            "aggregate_tps": c4_agg_tps,
            "avg_per_user_decode_tps": c4_per_user,
            "runs": c4_runs,
        },
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.label}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
