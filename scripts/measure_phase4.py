import sys
import time

from dotenv import load_dotenv
from langchain_core.callbacks import UsageMetadataCallbackHandler

load_dotenv()

from jp_morgraph.phase4_graph import build_graph

PRICE_IN, PRICE_OUT = 0.10, 0.50

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