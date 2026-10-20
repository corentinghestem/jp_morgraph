"""Mesure tokens, coût et latence d'un cycle d'analystes (vrai modèle, ≈ 0,01 $).

Usage :  uv run python scripts/measure_phase4.py [AS_OF]   (ex. 2026-10-09)
Prérequis : ANTHROPIC_API_KEY dans .env (et LANGSMITH_* pour retrouver la trace).
"""
import sys
import time

from dotenv import load_dotenv
from langchain_core.callbacks import UsageMetadataCallbackHandler

load_dotenv()

from jp_morgraph.phase4_graph import build_graph

# Tarifs Haiku 5.5 au 16/10/2026 ($ par million de tokens) — vois § 6.1 du guide.
PRICE_IN, PRICE_OUT = 0.10, 0.50

# Vraies données : load_market calcule le snapshot depuis data/*.parquet.
STATE_IN = {"as_of": sys.argv[1] if len(sys.argv) > 1 else "2026-10-09"}

if __name__ == "__main__":
    usage = UsageMetadataCallbackHandler()
    t0 = time.perf_counter()
    out = build_graph().invoke(STATE_IN, config={"callbacks": [usage], "recursion_limit": 40})
    dt = time.perf_counter() - t0

    tin = sum(u["input_tokens"] for u in usage.usage_metadata.values())
    tout = sum(u["output_tokens"] for u in usage.usage_metadata.values())
    cost = tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT

    print(f"rapports valides : {len(out['reports'])}/{2 * len(out['universe'])}")
    print(f"tokens           : {tin} in / {tout} out")
    print(f"coût estimé      : {cost:.4f} $")
    print(f"latence          : {dt:.1f} s (6 appels en parallèle)")
    for r in out["reports"]:
        print(f"  {r.ticker:5} {r.analyst:11} {r.signal:7} conv={r.conviction:.2f}")