import argparse
import json
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from jp_morgraph import graph as desk_graph
from jp_morgraph.agents import trader as trader_agent
from jp_morgraph.config import CAPITAL, MAX_WEIGHT
from jp_morgraph.costs import cost_of, tokens_of
from jp_morgraph.data import BENCHMARK, MarketStore
from jp_morgraph.desk import active, append_journal, initial_portfolio
from jp_morgraph.manager import ask_in_terminal
from jp_morgraph.phase4_graph import DATA_DIR

CITATION = re.compile(r"\[[A-Z.]+/(technique|news_macro)\]|\bBull\b|\bBear\b")


def read_jsonl(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def list_runs(data_dir=DATA_DIR) -> list[str]:
    root = Path(data_dir) / "runs"
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "journal.jsonl").exists())


def decision_fridays(prices: pd.DataFrame, weeks: int) -> list[str]:
    last = prices.index.max() - pd.Timedelta(days=1)
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(end=last, periods=weeks, freq="W-FRI")]


def last_friday(prices: pd.DataFrame) -> str:
    return pd.date_range(end=prices.index.max(), periods=1, freq="W-FRI")[0].strftime("%Y-%m-%d")


class Run:
    def __init__(self, name: str, data_dir=DATA_DIR, interrupt_before=None):
        self.name = name
        self.dir = Path(data_dir) / "runs" / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.journal = self.dir / "journal.jsonl"
        self.costs = self.dir / "costs.jsonl"
        self.pending = self.dir / "pending.txt"
        conn = sqlite3.connect(self.dir / "checkpoints.db", check_same_thread=False)
        self.graph = desk_graph.build_graph(checkpointer=SqliteSaver(conn, serde=desk_graph.SERDE),
                                            interrupt_before=interrupt_before)

    def config(self, as_of: str, usage=None, journal=None) -> dict:
        cfg = {"configurable": {"thread_id": f"{self.name}/{as_of}",
                                "journal_path": str(journal or self.journal)},
               "recursion_limit": 60}
        if usage is not None:
            cfg["callbacks"] = [usage]
        return cfg

    def records(self) -> list[dict]:
        return read_jsonl(self.journal)

    def portfolio(self) -> dict:
        records = self.records()
        return records[-1]["portfolio"] if records else initial_portfolio()

    def cycle(self, as_of: str, auto_approve: bool, ask=None) -> dict:
        usage = UsageMetadataCallbackHandler()
        cfg = self.config(as_of, usage)
        start = time.perf_counter()
        if self.graph.get_state(cfg).next:
            out = self.graph.invoke(None, cfg)
        else:
            out = self.graph.invoke({"as_of": as_of, "portfolio": self.portfolio(),
                                     "auto_approve": auto_approve}, cfg)
        while ask is not None and (interrupts := self.graph.get_state(cfg).interrupts):
            out = self.graph.invoke(Command(resume=ask(interrupts[0].value)), cfg)
        self.log_cost(as_of, usage, start)
        return out

    def resume(self, as_of: str) -> dict:
        usage = UsageMetadataCallbackHandler()
        start = time.perf_counter()
        out = self.graph.invoke(None, self.config(as_of, usage))
        self.log_cost(as_of, usage, start)
        return out

    def log_cost(self, as_of: str, usage, start: float) -> None:
        tin, tout = tokens_of(usage.usage_metadata)
        append_journal(self.costs, {"as_of": as_of, "cost": round(cost_of(usage.usage_metadata), 6),
                                    "tokens_in": tin, "tokens_out": tout,
                                    "seconds": round(time.perf_counter() - start, 1),
                                    "models": sorted(usage.usage_metadata)})


def replay(name: str, weeks: int, data_dir=DATA_DIR, log=print) -> Run:
    run = Run(name, data_dir)
    done = {r["as_of"] for r in run.records()}
    todo = [d for d in decision_fridays(MarketStore.load(data_dir).prices, weeks) if d not in done]
    if done and todo and min(todo) < max(done):
        raise SystemExit(f"le run {name} s'arrête au {max(done)}, impossible d'ajouter des semaines "
                         f"avant cette date : choisis un autre nom de run")
    for as_of in todo:
        run.cycle(as_of, auto_approve=True)
        last = run.records()[-1]
        log(f"{as_of}  {last['statut']:<45} valeur {last['portfolio']['nav']:>12,.2f} $")
    return run


def live(name: str = "live", data_dir=DATA_DIR, ask=ask_in_terminal, log=print) -> Run:
    run = Run(name, data_dir, interrupt_before=["execute"])
    if run.pending.exists():
        as_of = run.pending.read_text().strip()
        if not settle(run, as_of, log):
            return run
    as_of = last_friday(MarketStore.load(data_dir).prices)
    if any(r["as_of"] == as_of for r in run.records()):
        log(f"cycle du {as_of} déjà décidé, rafraîchis les données la semaine prochaine")
        return run
    run.cycle(as_of, auto_approve=False, ask=ask)
    if run.graph.get_state(run.config(as_of)).next == ("execute",):
        run.pending.write_text(as_of)
        settle(run, as_of, log)
    else:
        log(f"{as_of} : {run.records()[-1]['statut']}")
    return run


def settle(run: Run, as_of: str, log=print) -> bool:
    state = run.graph.get_state(run.config(as_of))
    approved = state.values.get("approved_orders", [])
    tickers = sorted({o.ticker for o in approved} | set(state.values["portfolio"].get("positions", {})))
    date, _, _ = desk_graph.execution_prices(as_of, tickers)
    if date is None and active(approved):
        log(f"ordres du {as_of} validés, exécution à l'ouverture du lundi : "
            f"relance après avoir rafraîchi les données")
        return False
    run.resume(as_of)
    run.pending.unlink(missing_ok=True)
    log(f"{as_of} : {run.records()[-1]['statut']}")
    return True


def travel(name: str, as_of: str, instruction: str, data_dir=DATA_DIR) -> tuple[list, list]:
    run = Run(name, data_dir)
    history = list(run.graph.get_state_history(run.config(as_of)))
    before_trader = [s for s in history if s.next == ("trader",)]
    if not before_trader:
        raise SystemExit(f"aucun checkpoint avant le trader pour {name}/{as_of}")
    target = before_trader[-1]
    original = next((r["orders"] for r in run.records() if r["as_of"] == as_of), [])
    template = trader_agent.PROMPT_TEMPLATE
    extra = instruction.replace("{", "{{").replace("}", "}}")
    trader_agent.PROMPT_TEMPLATE = f"{template}\nConsigne du gérant : {extra}\n"
    try:
        fork = {"configurable": {**target.config["configurable"],
                                 "journal_path": str(run.dir / "time_travel.jsonl")},
                "recursion_limit": 60}
        out = run.graph.invoke(None, fork)
    finally:
        trader_agent.PROMPT_TEMPLATE = template
    return original, [o.model_dump() for o in out["orders"]]


def close_on(prices: pd.DataFrame, day: str, ticker: str) -> float | None:
    if ticker not in prices:
        return None
    s = prices[ticker].loc[:pd.Timestamp(day)].dropna()
    return float(s.iloc[-1]) if len(s) else None


def nav_table(records: list[dict], prices: pd.DataFrame) -> pd.DataFrame:
    days = [r["as_of"] for r in records] + [prices.index.max().strftime("%Y-%m-%d")]
    portfolios = [initial_portfolio()] + [r["portfolio"] for r in records]
    rows = []
    for day, pf in zip(days, portfolios):
        nav = pf["cash"] + sum(q * (close_on(prices, day, t) or 0) for t, q in pf["positions"].items())
        rows.append({"date": day, "nav": round(nav, 2), "spy": close_on(prices, day, BENCHMARK)})
    df = pd.DataFrame(rows).drop_duplicates("date", keep="last").reset_index(drop=True)
    df["spy"] = (CAPITAL * df["spy"] / df["spy"].iloc[0]).round(2)
    df["date"] = pd.to_datetime(df["date"])
    return df


def limit_violations(records: list[dict]) -> list[str]:
    found = []
    for r in records:
        pf = r["portfolio"]
        if pf["cash"] < 0:
            found.append(f"{r['as_of']} : cash négatif")
        found += [f"{r['as_of']} : {t} position négative" for t, q in pf["positions"].items() if q < 0]
        for f in r["fills"]:
            if f["side"] == "buy" and f["quantity"] > 0 and pf["nav"] > 0:
                weight = pf["positions"].get(f["ticker"], 0) * f["ref_price"] / pf["nav"]
                if weight > MAX_WEIGHT + 1e-9:
                    found.append(f"{r['as_of']} : {f['ticker']} à {weight:.1%}")
    return found


def summarize(name: str, data_dir=DATA_DIR) -> dict:
    run_dir = Path(data_dir) / "runs" / name
    records = read_jsonl(run_dir / "journal.jsonl")
    costs = pd.DataFrame(read_jsonl(run_dir / "costs.jsonl"))
    prices = MarketStore.load(data_dir).prices
    nav = nav_table(records, prices)
    proposed = [o for r in records for o in r["orders"] if o["side"] != "hold"]
    fills = [f for r in records for f in r["fills"] if f["quantity"] > 0]
    cycles = len(records)
    total_cost = float(costs["cost"].sum()) if len(costs) else 0.0
    return {
        "run": name,
        "records": records,
        "nav": nav,
        "costs": costs,
        "debut": records[0]["as_of"] if records else None,
        "fin": records[-1]["as_of"] if records else None,
        "cycles": cycles,
        "valeur_finale": float(nav["nav"].iloc[-1]),
        "rendement": float(nav["nav"].iloc[-1] / CAPITAL - 1),
        "rendement_spy": float(nav["spy"].iloc[-1] / CAPITAL - 1),
        "drawdown_max": float((nav["nav"] / nav["nav"].cummax() - 1).min()),
        "statuts": Counter(r["statut"].split(" (")[0] for r in records),
        "ordres_executes": len(fills),
        "ordres_reduits": sum(bool(f["note"]) for f in fills),
        "retours_au_trader": sum(r["risk_retries"] for r in records),
        "hors_limites": limit_violations(records),
        "citent_sources": (sum(bool(CITATION.search(o["rationale"])) for o in proposed) / len(proposed)
                           if proposed else None),
        "cout_total": total_cost,
        "cout_par_cycle": total_cost / cycles if cycles else 0.0,
    }


def pct(x: float | None) -> str:
    return "-" if x is None else f"{x:+.1%}".replace(".", ",")


def money(x: float) -> str:
    return f"{x:,.2f} $".replace(",", " ").replace(".", ",")


def report(name: str, data_dir=DATA_DIR) -> Path:
    s = summarize(name, data_dir)
    run_dir = Path(data_dir) / "runs" / name
    s["nav"].to_csv(run_dir / "nav.csv", index=False)
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_scatter(x=s["nav"]["date"], y=s["nav"]["nav"], name="Portefeuille")
    fig.add_scatter(x=s["nav"]["date"], y=s["nav"]["spy"], name="SPY (repère)")
    fig.update_layout(title=f"Valeur liquidative, run {name}", yaxis_title="$")
    fig.write_html(run_dir / "nav.html", include_plotlyjs="cdn")

    statuts = "\n".join(f"| {k} | {v} |" for k, v in s["statuts"].most_common())
    hors_limites = len(s["hors_limites"])
    cite = "-" if s["citent_sources"] is None else f"{s['citent_sources']:.0%}"
    text = f"""# Run {name}

Du {s['debut']} au {s['fin']}, {s['cycles']} cycles hebdomadaires.

| Indicateur | Valeur |
|---|---|
| Valeur finale | {money(s['valeur_finale'])} ({pct(s['rendement'])}) |
| SPY sur la période (repère) | {pct(s['rendement_spy'])} |
| Drawdown maximum | {pct(s['drawdown_max'])} |
| Ordres exécutés | {s['ordres_executes']} (dont {s['ordres_reduits']} réduits à l'exécution) |
| Retours au Trader après un refus | {s['retours_au_trader']} |
| Ordres exécutés hors limites | {hors_limites} |
| Ordres qui citent leurs sources | {cite} |
| Coût total | {money(s['cout_total'])} ({f"{s['cout_par_cycle']:.4f}".replace('.', ',')} $ par cycle) |

| Statut du cycle | Nombre |
|---|---|
{statuts}

Courbe : `nav.html`. Données : `nav.csv`, `journal.jsonl`, `costs.jsonl`.
"""
    path = run_dir / "report.md"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(prog="python -m jp_morgraph.simulate")
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay", help="rejoue les N derniers vendredis sans validation humaine")
    r.add_argument("--run", default="replay-01")
    r.add_argument("--weeks", type=int, default=52)
    r.add_argument("--trader-model")
    lv = sub.add_parser("live", help="cycle du dernier vendredi, avec ta validation")
    lv.add_argument("--run", default="live")
    rp = sub.add_parser("report", help="rapport d'un run")
    rp.add_argument("--run", default="replay-01")
    tt = sub.add_parser("travel", help="rejoue un cycle passé depuis le trader avec une consigne")
    tt.add_argument("--run", default="replay-01")
    tt.add_argument("--as-of", required=True)
    tt.add_argument("--consigne", required=True)
    args = parser.parse_args()

    if args.cmd == "replay":
        if args.trader_model:
            trader_agent.use_model(args.trader_model)
        replay(args.run, args.weeks)
        print(report(args.run))
    elif args.cmd == "live":
        live(args.run)
    elif args.cmd == "report":
        print(report(args.run).read_text(encoding="utf-8"))
    else:
        before, after = travel(args.run, args.as_of, args.consigne)
        print("avant :", json.dumps([(o["ticker"], o["side"], o["conviction"]) for o in before]))
        print("après :", json.dumps([(o["ticker"], o["side"], o["conviction"]) for o in after]))


if __name__ == "__main__":
    main()
