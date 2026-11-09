import json
import sys

from dotenv import load_dotenv
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv()

from jp_morgraph.costs import cost_of, tokens_of
from jp_morgraph.graph import JOURNAL_PATH, SERDE, build_graph
from jp_morgraph.manager import ask_in_terminal

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    as_of = args[0] if args else "2026-10-16"
    usage = UsageMetadataCallbackHandler()
    graph = build_graph(checkpointer=InMemorySaver(serde=SERDE))
    config = {"configurable": {"thread_id": f"cycle-{as_of}"}, "callbacks": [usage],
              "recursion_limit": 40}

    out = graph.invoke({"as_of": as_of, "auto_approve": "--auto" in sys.argv}, config)
    while interrupts := graph.get_state(config).interrupts:
        out = graph.invoke(Command(resume=ask_in_terminal(interrupts[0].value)), config)

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

    tin, tout = tokens_of(usage.usage_metadata)
    print(f"\nCoût estimé : {cost_of(usage.usage_metadata):.4f} $ "
          f"({tin} in / {tout} out), journal : {JOURNAL_PATH}")
