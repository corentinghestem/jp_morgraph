import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field, ValidationError

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from scenarios import AS_OF, SCENARIOS, Scenario

from jp_morgraph.agents import analysts, debate, risk_manager, trader
from jp_morgraph.agents.analysts import MODEL, AnalystTask
from jp_morgraph.agents.debate import format_reports
from jp_morgraph.config import MAX_RISK_RETRIES
from jp_morgraph.costs import cost_of
from jp_morgraph.phase4_graph import DATA_DIR

RESULTS_DIR = DATA_DIR / "evals"
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
LIST_MARKER = re.compile(r"^\s*\d+[.)]\s|\(tour \d+\)", re.MULTILINE)
RULE_NUMBERS = {0.0, 1.0, 40.0, 100.0}
INVALID = (ValidationError, OutputParserException, RuntimeError, ValueError, anthropic.APIError)


class JudgeScore(BaseModel):
    fidelite: int = Field(ge=1, le=5, description="5 : tout vient des données ; 1 : faits ou chiffres inventés")
    qualite: int = Field(ge=1, le=5, description="5 : arguments pertinents, clairs, cohérents avec la décision")
    commentaire: str = Field(description="une phrase")


JUDGE_PROMPT = """Tu évalues la réponse d'un agent dans une salle de marché simulée.

Données fournies à l'agent :
{data}

Réponse de l'agent :
{output}

Note de 1 à 5 :
- fidelite : la réponse n'utilise que des faits et des chiffres présents dans les données ;
- qualite : les arguments sont pertinents, clairs et cohérents avec la décision.
"""


def numbers_in(obj) -> set[float]:
    if isinstance(obj, BaseModel):
        return numbers_in(obj.model_dump())
    if isinstance(obj, dict):
        return set().union(set(), *(numbers_in(k) | numbers_in(v) for k, v in obj.items()))
    if isinstance(obj, list | tuple):
        return set().union(set(), *(numbers_in(x) for x in obj))
    if obj is None or isinstance(obj, bool):
        return set()
    if isinstance(obj, int | float):
        return {abs(float(obj))}
    return {float(n.replace(",", ".")) for n in NUMBER.findall(str(obj))}


def invented_numbers(text: str, data, own=()) -> list[str]:
    known = numbers_in(data) | RULE_NUMBERS | {round(float(x), 2) for x in own}
    return [raw for raw in NUMBER.findall(LIST_MARKER.sub(" ", text))
            if not any(abs(float(raw.replace(",", ".")) - k) <= max(0.051, 0.006 * k)
                       or float(raw.replace(",", ".")) in (round(k), round(k, 1)) for k in known)]


def desk_data(state: dict) -> dict:
    data = {"date": state["as_of"], "rapports": format_reports(state["reports"]),
            "portefeuille": state["portfolio"],
            "limites": "40 % maximum par titre, exposition totale <= 100 %, cash jamais négatif"}
    risk = state.get("risk")
    if risk is not None and not risk.approved:
        data["refus_precedent"] = {"violations": risk.breaches, "avis_risk_manager": risk.comment,
                                   "ordres_refuses": [(o.ticker, o.side, o.conviction, o.quantity)
                                                      for o in state.get("orders", [])]}
    return data


def rationales(orders) -> str:
    return "\n".join(f"{o.ticker} ({o.side}) : {o.rationale}" for o in orders)


def run_agent(s: Scenario, cfg: dict):
    i = s.inputs
    if s.agent in ("technique", "news_macro"):
        task = AnalystTask(ticker=i["ticker"], kind=s.agent, as_of=AS_OF, features=i["features"])
        report = analysts.analyst(task, cfg)["reports"][0]
        return report, "\n".join(report.arguments), i["features"], report.signal
    state = dict(i["state"])
    if s.agent == "trader":
        data = desk_data(state)
        orders = trader.trader(state, cfg)["orders"]
        return orders, rationales(orders), data, tuple(o.side for o in orders)
    if s.agent == "desk":
        data = desk_data(state)
        for proposal in range(1, MAX_RISK_RETRIES + 1):
            if proposal > 1:
                data = desk_data(state)
            state |= trader.trader(state, cfg)
            state |= risk_manager.risk_check(state, cfg)
            if state["risk"].approved:
                break
        out = {"orders": state["orders"], "risk": state["risk"], "state": state, "propositions": proposal}
        return out, rationales(state["orders"]), data, tuple(o.side for o in state["orders"])
    if s.agent == "risk":
        verdict = risk_manager.risk_check(state, cfg)["risk"]
        data = {"violations": verdict.breaches, "ordres": [o.model_dump() for o in state["orders"]],
                "portefeuille": state["portfolio"]}
        return verdict, verdict.comment, data, verdict.approved
    if s.agent == "bull":
        text = debate.bull(state, cfg)["debate"][0]
        return text, text, desk_data(state), None
    raise ValueError(f"agent inconnu : {s.agent}")


def evaluate(s: Scenario, repeat: int, judge, usage: dict) -> dict:
    runs = []
    for _ in range(repeat):
        start = time.perf_counter()
        try:
            output, text, data, decision = run_agent(s, {"callbacks": [usage["agents"]]})
        except INVALID as e:
            runs.append({"valide": False, "ok": False, "erreur": f"{type(e).__name__}: {e}"[:300],
                         "secondes": round(time.perf_counter() - start, 1)})
            continue
        own = [o.conviction for o in (output["orders"] if isinstance(output, dict) else output)
               if hasattr(o, "conviction")] if s.agent in ("trader", "desk") else []
        entry = {"valide": True, "ok": bool(s.check(output)), "decision": str(decision),
                 "sortie": text[:3000], "chiffres_inventes": invented_numbers(text, data, own),
                 "secondes": round(time.perf_counter() - start, 1)}
        if judge is not None and s.juge:
            prompt = JUDGE_PROMPT.format(data=json.dumps(data, ensure_ascii=False, indent=2, default=str),
                                         output=text)
            entry["juge"] = judge.invoke(prompt, {"callbacks": [usage["juge"]]}).model_dump()
        runs.append(entry)
    decisions = Counter(r["decision"] for r in runs if r["valide"])
    return {"name": s.name, "agent": s.agent, "attendu": s.attendu, "runs": runs,
            "reussite": sum(r["ok"] for r in runs) / len(runs),
            "stabilite": decisions.most_common(1)[0][1] / len(runs) if decisions else 0.0}


def mean(values) -> float | None:
    values = list(values)
    return round(statistics.mean(values), 3) if values else None


def totals(results: list[dict], usage: dict, seconds: float) -> dict:
    runs = [r for s in results for r in s["runs"]]
    valid = [r for r in runs if r["valide"]]
    judged = [r["juge"] for r in valid if "juge" in r]
    return {
        "scenarios": len(results),
        "scenarios_reussis": sum(s["reussite"] >= 0.5 for s in results),
        "reussite_moyenne": mean(s["reussite"] for s in results),
        "sorties_valides": mean(r["valide"] for r in runs),
        "sans_chiffre_invente": mean(not r["chiffres_inventes"] for r in valid),
        "fidelite_moyenne": mean(j["fidelite"] for j in judged),
        "qualite_moyenne": mean(j["qualite"] for j in judged),
        "stabilite_moyenne": mean(s["stabilite"] for s in results),
        "cout_agents": round(cost_of(usage["agents"].usage_metadata), 4),
        "cout_juge": round(cost_of(usage["juge"].usage_metadata), 4),
        "secondes": round(seconds, 1),
    }


def run_all(scenarios, repeat=1, judge=None, log=print) -> tuple[list[dict], dict]:
    usage = {"agents": UsageMetadataCallbackHandler(), "juge": UsageMetadataCallbackHandler()}
    start = time.perf_counter()
    results = []
    for s in scenarios:
        result = evaluate(s, repeat, judge, usage)
        results.append(result)
        log(line(result))
    return results, totals(results, usage, time.perf_counter() - start)


def line(r: dict) -> str:
    valid = [x for x in r["runs"] if x["valide"]]
    judged = [x["juge"] for x in valid if "juge" in x]
    juge = f"{mean(j['fidelite'] for j in judged)}/{mean(j['qualite'] for j in judged)}" if judged else "-"
    invented = sum(len(x["chiffres_inventes"]) for x in valid)
    return (f"{'OK ' if r['reussite'] >= 0.5 else 'KO '} {r['name']:<30} réussite {r['reussite']:.0%}  "
            f"valides {len(valid)}/{len(r['runs'])}  chiffres inventés {invented}  juge {juge}")


def compare(paths: list[str]) -> str:
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    labels = [r["label"] for r in runs]
    rows = [f"| | {' | '.join(labels)} |", f"|---|{'---|' * len(runs)}"]
    for key in runs[0]["totaux"]:
        rows.append(f"| {key} | {' | '.join(str(r['totaux'].get(key)) for r in runs)} |")
    for i, s in enumerate(runs[0]["resultats"]):
        rows.append(f"| {s['name']} | {' | '.join(f'{r['resultats'][i]['reussite']:.0%}' for r in runs)} |")
    return "\n".join(rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="uv run python evals/run_evals.py")
    parser.add_argument("--label")
    parser.add_argument("--trader-model", default=MODEL)
    parser.add_argument("--judge-model", default=MODEL)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--only", help="noms de scénarios séparés par des virgules")
    parser.add_argument("--compare", nargs="+", metavar="RESULTAT.json")
    args = parser.parse_args(argv)

    if args.compare:
        print(compare(args.compare))
        return
    trader.use_model(args.trader_model)
    judge = None if args.no_judge else init_chat_model(
        args.judge_model, effort="low", max_tokens=2000, timeout=120).with_structured_output(JudgeScore)
    scenarios = [s for s in SCENARIOS if not args.only or s.name in args.only.split(",")]
    results, summary = run_all(scenarios, args.repeat, judge)
    label = args.label or ("haiku" if args.trader_model == MODEL else
                           "trader-" + args.trader_model.split(":")[-1])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{label}.json"
    path.write_text(json.dumps({"label": label, "date": datetime.now().isoformat(timespec="seconds"),
                                "trader_model": args.trader_model, "judge_model": args.judge_model,
                                "repeat": args.repeat, "totaux": summary, "resultats": results},
                               ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"résultats : {path}")


if __name__ == "__main__":
    main()
