import json

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from jp_morgraph import graph as desk_graph
from jp_morgraph.agents import analysts, debate, risk_manager, trader
from jp_morgraph.agents.trader import TraderDecision, TraderView
from jp_morgraph.state import AnalystReport

UNIVERSE = ["AAPL", "JPM", "XOM"]
STATE_IN = {
    "as_of": "2026-10-09",
    "universe": UNIVERSE,
    "snapshot": {t: {"technique": {"close": 100.0}, "news_macro": {"vix": 17.2}} for t in UNIVERSE},
}
ALL_IN = {"AAPL": ("buy", 1.0), "JPM": ("buy", 1.0), "XOM": ("buy", 1.0)}
SAGE = {"AAPL": ("buy", 0.5), "JPM": ("hold", 0.2), "XOM": ("hold", 0.1)}


class FakeAnalystLLM:
    def invoke(self, prompt, config=None):
        return AnalystReport(analyst="technique", ticker="?", signal="achat", conviction=0.6,
                             arguments=["momentum positif"])


class FakeTextLLM:
    def __init__(self, text):
        self.text, self.prompts = text, []

    def invoke(self, prompt, config=None):
        self.prompts.append(prompt)
        return AIMessage(content=self.text)


class FakeTrader:

    def __init__(self, *decisions):
        self.decisions, self.prompts = list(decisions), []

    def invoke(self, prompt, config=None):
        self.prompts.append(prompt)
        d = self.decisions[min(len(self.prompts), len(self.decisions)) - 1]
        return TraderDecision(orders=[TraderView(ticker=t, side=s, conviction=c, rationale=f"[{t}/technique]")
                                      for t, (s, c) in d.items()])


@pytest.fixture
def desk(monkeypatch, tmp_path):
    risk_llm = FakeTextLLM("Baisse les convictions : trois achats à 40 % dépassent le cash.")
    monkeypatch.setattr(analysts, "get_structured_llm", lambda: FakeAnalystLLM())
    monkeypatch.setattr(debate, "get_llm", lambda: FakeTextLLM("argument"))
    monkeypatch.setattr(risk_manager, "get_llm", lambda: risk_llm)
    monkeypatch.setattr(desk_graph, "execution_prices",
                        lambda as_of, tickers: ("2026-10-12", {t: 101.0 for t in tickers}, "ouverture"))
    journal = tmp_path / "journal.jsonl"

    class Desk:
        risk = risk_llm

        def trader(self, *decisions):
            self.fake = FakeTrader(*decisions)
            monkeypatch.setattr(trader, "get_structured_llm", lambda: self.fake)

        def config(self, thread="t1"):
            return {"configurable": {"thread_id": thread, "journal_path": str(journal)}}

        def journal(self):
            return [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]

    return Desk()


def orders_by_ticker(out, key="orders"):
    return {o.ticker: o for o in out[key]}


def test_ordres_conformes_sans_appel_au_risk_manager(desk):
    desk.trader(SAGE)
    out = desk_graph.build_graph().invoke({**STATE_IN, "auto_approve": True}, desk.config())
    assert out["risk"].approved and out["risk_retries"] == 0
    assert desk.risk.prompts == []


def test_refus_puis_retour_au_trader(desk):
    desk.trader(ALL_IN, SAGE)
    out = desk_graph.build_graph().invoke({**STATE_IN, "auto_approve": True}, desk.config())
    assert len(desk.fake.prompts) == 2 and out["risk_retries"] == 1 and out["risk"].approved
    second = desk.fake.prompts[1]
    assert "REFUSÉE" in second and "cash négatif" in second
    assert "Baisse les convictions" in second
    assert "REFUSÉE" not in desk.fake.prompts[0]


def test_deux_refus_fin_du_cycle_sans_ordre(desk):
    desk.trader(ALL_IN)
    out = desk_graph.build_graph().invoke({**STATE_IN, "auto_approve": True}, desk.config())
    assert len(desk.fake.prompts) == 2
    assert not out["risk"].approved and not out.get("fills")
    assert out["portfolio"]["positions"] == {}
    [rec] = desk.journal()
    assert rec["statut"].startswith("refusé par le contrôle des risques")


def test_execution_lundi_et_journal(desk):
    desk.trader(SAGE)
    out = desk_graph.build_graph().invoke({**STATE_IN, "auto_approve": True}, desk.config())
    [fill] = out["fills"]
    assert fill["ticker"] == "AAPL" and fill["date"] == "2026-10-12" and fill["quantity"] == 199
    assert out["portfolio"]["positions"] == {"AAPL": 199}
    [rec] = desk.journal()
    assert rec["statut"] == "exécuté" and rec["fills"] == out["fills"]
    assert len(rec["reports"]) == 6 and len(rec["debate"]) == 2


def paused(desk, decisions=SAGE):
    desk.trader(decisions)
    g = desk_graph.build_graph(checkpointer=InMemorySaver(serde=desk_graph.SERDE))
    cfg = desk.config()
    g.invoke(STATE_IN, cfg)
    return g, cfg


def test_le_graphe_s_arrete_et_presente_les_ordres(desk):
    g, cfg = paused(desk)
    state = g.get_state(cfg)
    assert state.next == ("pm_approval",)
    [intr] = state.interrupts
    assert [o["ticker"] for o in intr.value["ordres"]] == UNIVERSE
    assert not state.values.get("fills")


def test_validation(desk):
    g, cfg = paused(desk)
    out = g.invoke(Command(resume=True), cfg)
    assert out["fills"][0]["quantity"] == 199 and desk.journal()[0]["statut"] == "exécuté"


def test_rejet(desk):
    g, cfg = paused(desk)
    out = g.invoke(Command(resume={"approve": False}), cfg)
    assert out["approved_orders"] == [] and out["fills"] == []
    assert desk.journal()[0]["statut"] == "rejeté par le gérant"


def test_correction_de_quantite(desk):
    g, cfg = paused(desk)
    out = g.invoke(Command(resume={"quantities": {"AAPL": 50}}), cfg)
    aapl = orders_by_ticker(out, "approved_orders")["AAPL"]
    assert aapl.quantity == 50 and "corrigé par le gérant" in aapl.rationale
    assert orders_by_ticker(out)["AAPL"].quantity == 199
    assert out["fills"][0]["quantity"] == 50
    assert g.get_state(cfg).values["approved_orders"][0].quantity == 50


def test_correction_hors_limites_redemandee(desk):
    g, cfg = paused(desk)
    g.invoke(Command(resume={"quantities": {"AAPL": 1000}}), cfg)
    [intr] = g.get_state(cfg).interrupts
    assert "règles de risque" in intr.value["erreur"]
    g.invoke(Command(resume={"quantities": {"JPM": "beaucoup"}}), cfg)
    assert "invalide" in g.get_state(cfg).interrupts[0].value["erreur"]
    out = g.invoke(Command(resume=True), cfg)
    assert out["fills"][0]["quantity"] == 199
