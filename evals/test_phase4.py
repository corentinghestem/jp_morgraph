"""Tests de la phase 4 : aucun réseau, aucun appel payant (faux modèle)."""
import pytest
from langgraph.errors import InvalidUpdateError

from jp_morgraph.agents import analysts
from jp_morgraph.phase4_graph import build_graph
from jp_morgraph.state import AnalystReport

UNIVERSE = ["AAPL", "JPM", "XOM"]
STATE_IN = {
    "as_of": "2026-10-09",
    "universe": UNIVERSE,
    "snapshot": {t: {"technique": {"rsi": 55.0}, "news_macro": {"vix": 17.2}} for t in UNIVERSE},
}


class FakeStructuredLLM:
    """Imite llm.with_structured_output(AnalystReport).invoke(prompt, config)."""

    def __init__(self):
        self.prompts: list[str] = []

    def invoke(self, prompt: str, config=None) -> AnalystReport:
        self.prompts.append(prompt)
        # Volontairement FAUX sur analyst/ticker : le code doit les écraser.
        return AnalystReport(
            analyst="technique", ticker="???", signal="neutre",
            conviction=0.5, arguments=["argument factice"],
        )


@pytest.fixture
def fake_llm(monkeypatch):
    fake = FakeStructuredLLM()
    monkeypatch.setattr(analysts, "get_structured_llm", lambda: fake)
    return fake


def test_six_rapports_et_join_une_seule_fois(fake_llm):
    out = build_graph().invoke(STATE_IN)
    assert len(out["reports"]) == 6
    assert len(fake_llm.prompts) == 6
    assert out["n_join_runs"] == 1  # carnet : le nœud suivant s'exécute 1 fois


def test_couples_titre_specialite_complets(fake_llm):
    out = build_graph().invoke(STATE_IN)
    couples = {(r.ticker, r.analyst) for r in out["reports"]}
    assert couples == {(t, k) for t in UNIVERSE for k in ("technique", "news_macro")}


def test_chaque_analyste_ne_voit_que_ses_donnees(fake_llm):
    build_graph().invoke(STATE_IN)
    technique = [p for p in fake_llm.prompts if "analyste technique" in p]
    news = [p for p in fake_llm.prompts if "analyste news_macro" in p]
    assert len(technique) == len(news) == 3
    assert all('"rsi"' in p and '"vix"' not in p for p in technique)
    assert all('"vix"' in p and '"rsi"' not in p for p in news)


def test_sans_reducer_invalid_update_error(fake_llm):
    with pytest.raises(InvalidUpdateError):
        build_graph(with_reducer=False).invoke(STATE_IN)