from langgraph.graph import END, START, StateGraph

from jp_morgraph.agents.analysts import analyst, route_to_analysts
from jp_morgraph.agents.debate import bear, bull
from jp_morgraph.agents.trader import trader
from jp_morgraph.desk import initial_portfolio
from jp_morgraph.phase4_graph import load_market as load_snapshot
from jp_morgraph.state import DeskState


def load_market(state: dict) -> dict:
    out = load_snapshot(state)
    out["debate_round"] = 0
    if not state.get("portfolio"):
        out["portfolio"] = initial_portfolio()
    return out


def build_graph(max_rounds: int = 1, with_debate: bool = True, stop_debate: bool = True,
                checkpointer=None):
    def debate_route(state: dict) -> str:
        if not stop_debate:
            return "bull"
        return "bull" if state.get("debate_round", 0) < max_rounds else "trader"

    b = StateGraph(DeskState)
    b.add_node("load_market", load_market)
    b.add_node("analyst", analyst)
    b.add_node("trader", trader)
    b.add_edge(START, "load_market")
    b.add_conditional_edges("load_market", route_to_analysts, ["analyst"])
    if with_debate:
        b.add_node("bull", bull)
        b.add_node("bear", bear)
        b.add_edge("analyst", "bull")
        b.add_edge("bull", "bear")
        b.add_conditional_edges("bear", debate_route, ["bull", "trader"])
    else:
        b.add_edge("analyst", "trader")
    b.add_edge("trader", END)
    return b.compile(checkpointer=checkpointer)


graph = build_graph()
