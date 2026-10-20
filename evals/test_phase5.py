import pytest
from langchain_core.messages import AIMessage
from langgraph.errors import GraphRecursionError

from jp_morgraph.agents import analysts, debate, trader
from jp_morgraph.agents.trader import TraderDecision, TraderView
from jp_morgraph.desk import initial_portfolio, size_order, sized
from jp_morgraph.phase5_graph import build_graph
from jp_morgraph.state import AnalystReport, OrderProposal

UNIVERSE = ["AAPL", "JPM", "XOM"]
STATE_IN = {
    "as_of": "2026-10-09",
    "universe": UNIVERSE,
    "snapshot": {t: {"technique": {"close": 100.0, "rsi": 55.0}, "news_macro": {"vix": 17.2}}
                 for t in UNIVERSE},
}


class FakeAnalystLLM:
    def invoke(self, prompt, config=None):
        return AnalystReport(analyst="technique", ticker="???", signal="achat",
                             conviction=0.6, arguments=["momentum positif"])


class FakeChatLLM:

    def __init__(self):
        self.prompts: list[str] = []

    def invoke(self, prompt, config=None):
        self.prompts.append(prompt)
        return AIMessage(content=f"argument n°{len(self.prompts)}")


class FakeTraderLLM:

    def __init__(self):
        self.prompts: list[str] = []

    def invoke(self, prompt, config=None):
        self.prompts.append(prompt)
        return TraderDecision(orders=[
            TraderView(ticker="aapl", side="buy", conviction=0.5, rationale="[AAPL/technique]"),
            TraderView(ticker="AAPL", side="sell", conviction=1.0, rationale="doublon"),
            TraderView(ticker="TSLA", side="buy", conviction=1.0, rationale="hors univers"),
            TraderView(ticker="JPM", side="hold", conviction=0.3, rationale="Bear"),
        ])


@pytest.fixture
def fakes(monkeypatch):
    chat, trade = FakeChatLLM(), FakeTraderLLM()
    monkeypatch.setattr(analysts, "get_structured_llm", lambda: FakeAnalystLLM())
    monkeypatch.setattr(debate, "get_llm", lambda: chat)
    monkeypatch.setattr(trader, "get_structured_llm", lambda: trade)
    return chat, trade


def test_un_aller_retour(fakes):
    chat, trade = fakes
    out = build_graph(max_rounds=1).invoke(STATE_IN)
    assert out["debate_round"] == 1
    assert [m.split(" :")[0] for m in out["debate"]] == ["Bull (tour 1)", "Bear (tour 1)"]
    assert len(chat.prompts) == 2
    assert "argument n°1" in chat.prompts[1]
    assert len(trade.prompts) == 1 and "Bear (tour 1)" in trade.prompts[0]


def test_deux_allers_retours(fakes):
    chat, _ = fakes
    out = build_graph(max_rounds=2).invoke(STATE_IN)
    assert out["debate_round"] == 2
    assert [m.split(" :")[0] for m in out["debate"]] == [
        "Bull (tour 1)", "Bear (tour 1)", "Bull (tour 2)", "Bear (tour 2)"]
    assert "argument n°2" in chat.prompts[2]


def test_debat_lit_les_six_rapports(fakes):
    chat, _ = fakes
    build_graph().invoke(STATE_IN)
    for t in UNIVERSE:
        for k in ("technique", "news_macro"):
            assert f"[{t}/{k}]" in chat.prompts[0]


def test_sans_condition_d_arret_graph_recursion_error(fakes):
    with pytest.raises(GraphRecursionError):
        build_graph(stop_debate=False).invoke(STATE_IN, {"recursion_limit": 15})


def test_ablation_sans_debat(fakes):
    chat, trade = fakes
    out = build_graph(with_debate=False).invoke(STATE_IN)
    assert not out.get("debate") and chat.prompts == []
    assert "Débat" not in trade.prompts[0]
    assert len(out["orders"]) == 3


def test_un_ordre_par_titre_de_l_univers(fakes):
    out = build_graph().invoke(STATE_IN)
    orders = {o.ticker: o for o in out["orders"]}
    assert list(orders) == UNIVERSE
    assert orders["AAPL"].side == "buy"
    assert orders["XOM"].side == "hold" and orders["XOM"].conviction == 0.0


def test_quantite_calculee_par_le_code(fakes):
    out = build_graph().invoke(STATE_IN)
    aapl = next(o for o in out["orders"] if o.ticker == "AAPL")
    assert aapl.quantity == 199
    assert "quantity" not in TraderView.model_fields


def test_size_order():
    assert size_order(0.5, 100_000, 0, 100) == 199
    assert size_order(1.0, 100_000, 40_000, 100) <= 0


def test_sized_achat_au_dela_de_la_cible_et_vente():
    snap = {"AAPL": {"technique": {"close": 100.0}}}
    pf = {"cash": 50_000, "positions": {"AAPL": 300}, "nav": 100_000}
    buy = OrderProposal(ticker="AAPL", side="buy", conviction=0.5, rationale="")
    sell = OrderProposal(ticker="AAPL", side="sell", conviction=0.5, rationale="")
    assert sized(buy, snap, pf).quantity == 0
    assert sized(sell, snap, pf).quantity == 150
    assert sized(sell, snap, initial_portfolio()).quantity == 0


def test_debat_vide_ou_coupe_leve_une_erreur(fakes, monkeypatch):
    class ThinkingOnlyLLM:
        def invoke(self, prompt, config=None):
            return AIMessage(content=[{"type": "thinking", "thinking": "", "signature": "x"}],
                             response_metadata={"stop_reason": "max_tokens"})

    monkeypatch.setattr(debate, "get_llm", lambda: ThinkingOnlyLLM())
    with pytest.raises(RuntimeError, match="max_tokens"):
        build_graph().invoke(STATE_IN)
