import json
import sys

from dotenv import load_dotenv
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv()

from jp_morgraph.graph import JOURNAL_PATH, SERDE, build_graph

PRICE_IN, PRICE_OUT = 0.10, 0.50


def ask(payload: dict):
    print(f"\nOrdres du {payload['as_of']} :")
    for o in payload["ordres"]:
        print(f"  {o['ticker']:5} {o['side']:4} conviction={o['conviction']:.2f} "
              f"quantité={o['quantity']:>6}  | {o['rationale']}")
    print(f"  Risque : {payload['avis_risque']}")
    if "erreur" in payload:
        print(f"  Erreur : {payload['erreur']}")
    answer = input("Valider ? [o / n / AAPL=50 / JPM:hold] > ").strip()
    if answer.lower() in ("", "o", "oui", "y"):
        return True
    if answer.lower() in ("n", "non"):
        return False
    decision = {"quantities": {}, "sides": {}}
    for token in answer.split():
        if "=" in token:
            t, q = token.split("=", 1)
            decision["quantities"][t.upper()] = int(q) if q.isdigit() else q
        elif ":" in token:
            t, s = token.split(":", 1)
            decision["sides"][t.upper()] = s.lower()
    return decision


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    as_of = args[0] if args else "2026-10-16"
    usage = UsageMetadataCallbackHandler()
    graph = build_graph(checkpointer=InMemorySaver(serde=SERDE))
    config = {"configurable": {"thread_id": f"cycle-{as_of}"}, "callbacks": [usage],
              "recursion_limit": 40}

    out = graph.invoke({"as_of": as_of, "auto_approve": "--auto" in sys.argv}, config)
    while interrupts := graph.get_state(config).interrupts:
        out = graph.invoke(Command(resume=ask(interrupts[0].value)), config)

    print("\nDébat :")
    for msg in out.get("debate", []):
        print(f"\n{msg}")
    print("\nRisque :")
    risk = out["risk"]
    print(f"{'accepté' if risk.approved else 'refusé'} après {out['risk_retries']} refus : {risk.comment}")
    for b in risk.breaches:
        print(f"  - {b}")
    print("\nExécution :")
    for f in out.get("fills", []):
        print(f"  {f['date']} {f['ticker']:5} {f['side']:4} {f['quantity']}/{f['requested']} "
              f"à {f.get('exec_price', '-')} ({f['source_prix']}) {f['note']}")
    print(f"Portefeuille : {json.dumps(out['portfolio'], ensure_ascii=False)}")

    tin = sum(u["input_tokens"] for u in usage.usage_metadata.values())
    tout = sum(u["output_tokens"] for u in usage.usage_metadata.values())
    print(f"\nCoût estimé : {tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT:.4f} $ "
          f"({tin} in / {tout} out), journal : {JOURNAL_PATH}")
