from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from jp_morgraph.agents.analysts import analyst, route_to_analysts
from jp_morgraph.agents.debate import bear, bull
from jp_morgraph.agents.risk_manager import risk_check
from jp_morgraph.agents.trader import trader
from jp_morgraph.config import MAX_RISK_RETRIES
from jp_morgraph.data import MarketStore, next_open
from jp_morgraph.desk import active, append_journal, execute_orders
from jp_morgraph.manager import pm_approval
from jp_morgraph.phase4_graph import DATA_DIR
from jp_morgraph.phase5_graph import load_market as load_market_and_portfolio
from jp_morgraph.state import AnalystReport, DeskState, OrderProposal, RiskVerdict

JOURNAL_PATH = DATA_DIR / "journal.jsonl"

SERDE = JsonPlusSerializer(allowed_msgpack_modules=[
    (cls.__module__, cls.__name__) for cls in (AnalystReport, OrderProposal, RiskVerdict)])


def load_market(state: dict) -> dict:
    return {**load_market_and_portfolio(state), "risk_retries": 0}


def risk_route(state: dict) -> str:
    if state["risk"].approved:
        return "pm_approval"
    return "trader" if state.get("risk_retries", 0) < MAX_RISK_RETRIES else "book"


def execution_prices(as_of: str, tickers: list[str]) -> tuple[str | None, dict, str]:
    return next_open(MarketStore.load(DATA_DIR), as_of, tickers)


def execute(state: dict) -> dict:
    orders = state.get("approved_orders", [])
    if not active(orders):
        return {"fills": []}
    tickers = sorted({o.ticker for o in orders} | set(state["portfolio"].get("positions", {})))
    date, prices, source = execution_prices(state["as_of"], tickers)
    fills, portfolio = execute_orders(orders, state["portfolio"], prices, date, source)
    return {"fills": fills, "portfolio": portfolio}


def cycle_status(state: dict) -> str:
    risk = state.get("risk")
    if risk is not None and not risk.approved:
        return f"refusé par le contrôle des risques ({state.get('risk_retries', 0)} propositions)"
    if not active(state.get("orders", [])):
        return "aucun ordre à exécuter"
    if not active(state.get("approved_orders", [])):
        return "rejeté par le gérant"
    if any(f["quantity"] > 0 for f in state.get("fills", [])):
        return "exécuté"
    return "non exécuté (voir fills)"


def book(state: dict, config: RunnableConfig) -> dict:
    risk = state.get("risk")
    record = {
        "as_of": state["as_of"],
        "statut": cycle_status(state),
        "portfolio": state["portfolio"],
        "fills": state.get("fills", []),
        "orders": [o.model_dump() for o in state.get("orders", [])],
        "approved_orders": [o.model_dump() for o in state.get("approved_orders", [])],
        "risk": risk.model_dump() if risk else None,
        "risk_retries": state.get("risk_retries", 0),
        "reports": [r.model_dump() for r in state.get("reports", [])],
        "debate": state.get("debate", []),
    }
    path = (config.get("configurable") or {}).get("journal_path", JOURNAL_PATH)
    append_journal(path, record)
    return {}


def build_graph(max_rounds: int = 1, checkpointer=None):
    def debate_route(state: dict) -> str:
        return "bull" if state.get("debate_round", 0) < max_rounds else "trader"

    b = StateGraph(DeskState)
    b.add_node("load_market", load_market)
    b.add_node("analyst", analyst)
    b.add_node("bull", bull)
    b.add_node("bear", bear)
    b.add_node("trader", trader)
    b.add_node("risk", risk_check)
    b.add_node("pm_approval", pm_approval)
    b.add_node("execute", execute)
    b.add_node("book", book)

    b.add_edge(START, "load_market")
    b.add_conditional_edges("load_market", route_to_analysts, ["analyst"])
    b.add_edge("analyst", "bull")
    b.add_edge("bull", "bear")
    b.add_conditional_edges("bear", debate_route, ["bull", "trader"])
    b.add_edge("trader", "risk")
    b.add_conditional_edges("risk", risk_route, ["pm_approval", "trader", "book"])
    b.add_edge("pm_approval", "execute")
    b.add_edge("execute", "book")
    b.add_edge("book", END)
    return b.compile(checkpointer=checkpointer)


graph = build_graph()
