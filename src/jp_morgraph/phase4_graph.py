from operator import add
from pathlib import Path
from typing import Annotated

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from jp_morgraph.agents.analysts import analyst, route_to_analysts
from jp_morgraph.data import MarketStore, _default_tickers, assert_no_future, snapshot
from jp_morgraph.state import AnalystReport


class Phase4State(TypedDict, total=False):
    as_of: str
    universe: list[str]
    snapshot: dict
    reports: Annotated[list[AnalystReport], add]
    n_join_runs: int


class Phase4StateNoReducer(TypedDict, total=False):
    as_of: str
    universe: list[str]
    snapshot: dict
    reports: list[AnalystReport]
    n_join_runs: int


DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def split_by_kind(snap: dict) -> dict:
    if "erreur" in snap:
        return snap
    return {
        "technique": {k: snap[k] for k in ("as_of", "last_price_date", "close",
                                           "indicateurs", "ret_12w_vs_spy_pts")},
        "news_macro": {k: snap[k] for k in ("as_of", "news", "macro")},
    }


def load_market(state: dict) -> dict:
    if state.get("snapshot"):
        return {}
    universe = state.get("universe") or _default_tickers()
    snap = snapshot(state["as_of"], MarketStore.load(DATA_DIR), universe)
    assert_no_future(snap, state["as_of"])
    return {"universe": universe, "snapshot": {t: split_by_kind(v) for t, v in snap.items()}}


def join(state: dict) -> dict:
    return {"n_join_runs": state.get("n_join_runs", 0) + 1}


def build_graph(with_reducer: bool = True, checkpointer=None):
    schema = Phase4State if with_reducer else Phase4StateNoReducer
    b = StateGraph(schema)
    b.add_node("load_market", load_market)
    b.add_node("analyst", analyst)
    b.add_node("join", join)
    b.add_edge(START, "load_market")
    b.add_conditional_edges("load_market", route_to_analysts, ["analyst"])
    b.add_edge("analyst", "join")
    b.add_edge("join", END)
    return b.compile(checkpointer=checkpointer)


graph = build_graph()