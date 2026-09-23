#!/usr/bin/env python3
"""Quality check for the running qwen38-flash server (no GPU needed on this
side — it talks to the OpenAI-compatible API).

Two probes, both deterministic (temperature=0):

  1. Perplexity on fixed texts (EN prose / RU prose / code / JSON): uses
     vLLM's `prompt_logprobs: 0` on /v1/completions to get the NLL of every
     prompt token. PPL numbers are only comparable across configurations of
     THE SAME server stack (e.g. this hybrid vs the NVFP4 recipe) — run the
     script against each and diff the reports.
  2. Greedy few-shot sanity: short factual prompts with expected substrings;
     reports a pass-rate. Not an IQ test — it catches "gather/quant broke and
     the model babbles" regressions, the failure mode of a bad table read.

Usage:
  python3 tools/eval_quality.py            # both probes
  python3 tools/eval_quality.py ppl        # perplexity only
  python3 tools/eval_quality.py facts      # few-shot only

Env: BASE (default http://localhost:8000), MODEL (default qwen), API_KEY
(optional bearer token, matches serve.sh's API_KEY).
"""

import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("BASE", "http://localhost:8000")
MODEL = os.environ.get("MODEL", "qwen")
API_KEY = os.environ.get("API_KEY", "")


def _headers():
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["Authorization"] = "Bearer " + API_KEY
    return h


def _post(path, payload, timeout=600):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(), headers=_headers()
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


TEXTS = {
    "en-prose": (
        "The DGX Spark sits on the desk, a compact chassis hiding a unified "
        "memory pool that the CPU and GPU share without ceremony. Inference "
        "engines treat it neither as a big GPU nor as a small cluster, but as "
        "something in between: bandwidth-rich, capacity-rich, and utterly "
        "intolerant of oversubscription. The discipline it teaches is simple "
        "to state and hard to keep — size every pool explicitly, and never "
        "assume a graceful out-of-memory error will arrive to save you."
    ),
    "ru-prose": (
        "Зима в этом году пришла рано. К утру окна покрылись узорами, и город "
        "затих под первым снегом. В такие дни хочется горячего чая, толстой "
        "книги и чтобы никто не звонил. Дворник уже прошёл по тротуару, "
        "оставив за собой чистую дорожку, а из подъезда пахло морозом и "
        "свежей выпечкой из пекарни на углу."
    ),
    "code": (
        "def gather_rows(table, ids):\n"
        "    out = {}\n"
        "    for i in ids:\n"
        "        if i not in out:\n"
        "            out[i] = table[i]\n"
        "    return [out[i] for i in ids]\n\n"
        "if __name__ == '__main__':\n"
        "    print(gather_rows({0: 'a', 1: 'b'}, [0, 0, 1]))\n"
    ),
    "json": (
        '{"server": {"model": "qwen38-flash", "quant": "int4+int8+fp8", '
        '"decode_tok_s": 49, "mtp": 3, "prefix_cache": true}, '
        '"limits": {"ctx": 262144, "kv_bytes": "20g"}}'
    ),
}

FACTS = [
    ("The capital of France is", "paris"),
    ("The capital of Japan is", "tokyo"),
    ("2 + 2 =", "4"),
    ("12 * 12 =", "144"),
    ("The chemical formula of water is", "h2o"),
    ("The first president of the United States was", "washington"),
    ("How many days are in a leap year?", "366"),
    ("The largest planet in the solar system is", "jupiter"),
    ("Столица России —", "москва"),
    ("The speed of light in vacuum is approximately 3", "10^8"),
]


def perplexity() -> None:
    print("=== Perplexity (lower = better; compare across configs of the same stack) ===")
    for name, text in TEXTS.items():
        t0 = time.time()
        try:
            r = _post(
                "/v1/completions",
                {"model": MODEL, "prompt": text, "max_tokens": 1,
                 "temperature": 0, "prompt_logprobs": 0},
            )
        except urllib.error.HTTPError as e:
            print(f"  {name:9s} FAILED: {e.code} {e.read()[:120]!r} "
                  "(does this build support prompt_logprobs?)")
            continue
        pl = r.get("choices", [{}])[0].get("prompt_logprobs") or []
        nll = []
        for tok in pl:
            if not tok:
                continue  # first prompt token has no logprob
            lp = next(iter(tok.values())).get("logprob")
            if lp is not None:
                nll.append(-lp)
        if not nll:
            print(f"  {name:9s} no prompt_logprobs in response — skipped")
            continue
        ppl = math.exp(sum(nll) / len(nll))
        print(f"  {name:9s} tokens={len(nll):4d}  ppl={ppl:8.3f}  "
              f"({time.time() - t0:.1f}s)")


def facts() -> None:
    print("=== Greedy few-shot sanity (a babble detector, not an IQ test) ===")
    passed = 0
    for prompt, needle in FACTS:
        try:
            r = _post(
                "/v1/completions",
                {"model": MODEL, "prompt": prompt, "max_tokens": 24,
                 "temperature": 0},
            )
            text = r["choices"][0]["text"].strip()
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"  FAIL {prompt!r}: request error {e}")
            continue
        ok = needle.lower() in text.lower()
        passed += ok
        print(f"  {'PASS' if ok else 'MISS'} {prompt!r} -> {text[:48]!r} "
              f"(want {needle!r})")
    print(f"  pass-rate: {passed}/{len(FACTS)}")
    if passed < len(FACTS) // 2:
        print("  !! less than half passed — outputs look degraded; check the "
              "PLE table gather and the quantization config first")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("all", "ppl"):
        perplexity()
    if mode in ("all", "facts"):
        facts()


if __name__ == "__main__":
    main()
