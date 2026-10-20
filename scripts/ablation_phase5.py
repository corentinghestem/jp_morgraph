import re
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

load_dotenv()

from jp_morgraph.agents.debate import format_reports
from jp_morgraph.agents.trader import trader
from jp_morgraph.data import MarketStore
from jp_morgraph.phase4_graph import DATA_DIR
from jp_morgraph.phase5_graph import build_graph

PRICE_IN, PRICE_OUT = 0.10, 0.50
ROOT = DATA_DIR.parent
MINUTES_DIR = DATA_DIR / "ablation"
CITATION = re.compile(r"\[[A-Z.]+/(technique|news_macro)\]|\bBull\b|\bBear\b")


def write_minutes(as_of: str, max_rounds: int, out: dict, sans_debat: list) -> Path:
    lines = [f"# Cycle du {as_of} ({max_rounds} aller-retour)", "",
             "## Rapports des analystes", "", format_reports(out["reports"]), "",
             "## Débat", ""]
    lines += [f"{msg}\n" for msg in out["debate"]]
    lines += ["## Ordres", "", "| Titre | Avec débat | Sans débat |", "|---|---|---|"]
    lines += [f"| {a.ticker} | {a.side} {a.conviction:.2f} x{a.quantity} | "
              f"{s.side} {s.conviction:.2f} x{s.quantity} |" for a, s in zip(out["orders"], sans_debat)]
    lines += ["", "## Justifications", ""]
    lines += [f"- **{a.ticker}** avec : {a.rationale}\n- **{s.ticker}** sans : {s.rationale}"
              for a, s in zip(out["orders"], sans_debat)]
    path = MINUTES_DIR / f"{as_of}_r{max_rounds}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def last_fridays(n: int) -> list[str]:
    last = MarketStore.load(DATA_DIR).prices.index.max()
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(end=last, periods=n, freq="W-FRI")]


if __name__ == "__main__":
    n_weeks = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    max_rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    graph = build_graph(max_rounds=max_rounds)
    usage = UsageMetadataCallbackHandler()
    config = {"callbacks": [usage], "recursion_limit": 40}

    rows, failures, t0 = [], [], time.perf_counter()
    for as_of in last_fridays(n_weeks):
        try:
            out = graph.invoke({"as_of": as_of}, config)
            sans_debat = trader({**out, "debate": []}, config)["orders"]
        except (ValidationError, OutputParserException, RuntimeError) as e:
            failures.append(f"{as_of} : {type(e).__name__}")
            print(f"{as_of} : échec ({type(e).__name__})")
            continue
        for avec, sans in zip(out["orders"], sans_debat):
            rows.append({
                "as_of": as_of, "ticker": avec.ticker,
                "avec": f"{avec.side} {avec.conviction:.2f} x{avec.quantity}",
                "sans": f"{sans.side} {sans.conviction:.2f} x{sans.quantity}",
                "sens_change": avec.side != sans.side,
                "delta_conv": round(avec.conviction - sans.conviction, 2),
                "cite_ses_sources": bool(CITATION.search(avec.rationale)),
            })
        path = write_minutes(as_of, max_rounds, out, sans_debat)
        print(f"{as_of} : {len(out['debate'])} messages de débat -> {path.relative_to(ROOT)}")

    if failures:
        print(f"\nsemaines en échec : {len(failures)}/{n_weeks} -> " + " ; ".join(failures))
    if not rows:
        sys.exit("aucune semaine exploitable")
    df = pd.DataFrame(rows)
    tin = sum(u["input_tokens"] for u in usage.usage_metadata.values())
    tout = sum(u["output_tokens"] for u in usage.usage_metadata.values())

    print()
    print(df.to_string(index=False))
    print()
    print(f"sens modifié par le débat   : {df.sens_change.sum()}/{len(df)} décisions")
    print(f"|Δ conviction| moyen        : {df.delta_conv.abs().mean():.2f}")
    print(f"ordres qui citent une source : {df.cite_ses_sources.sum()}/{len(df)}")
    print(f"coût total estimé           : {tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT:.4f} $"
          f"  ({tin} in / {tout} out, {time.perf_counter() - t0:.0f} s)")
