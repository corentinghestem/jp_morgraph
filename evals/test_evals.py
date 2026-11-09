import json

import pytest
import run_evals
from langchain_core.messages import AIMessage
from scenarios import SCENARIOS, TICKERS, tech

from jp_morgraph.agents import analysts, debate, risk_manager, trader
from jp_morgraph.agents.trader import TraderDecision, TraderView
from jp_morgraph.state import AnalystReport


class FakeAnalyst:
    def invoke(self, prompt, config=None):
        return AnalystReport(analyst="technique", ticker="?", signal="achat", conviction=0.9,
                             arguments=["RSI à 64", "objectif de cours 250"])


class FakeText:
    def invoke(self, prompt, config=None):
        return AIMessage(content="[ALFA/technique] haussier, [BETA/news_macro] rachat, "
                                 "[GAMA/news_macro] contrat. ALFA BETA GAMA. Baisse les convictions.")


class AllIn:
    def invoke(self, prompt, config=None):
        return TraderDecision(orders=[TraderView(ticker=t, side="buy", conviction=1.0, rationale="[ALFA/technique]")
                                      for t in TICKERS])


class FakeJudge:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt, config=None):
        self.prompts.append(prompt)
        return run_evals.JudgeScore(fidelite=4, qualite=3, commentaire="ok")


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setattr(analysts, "get_structured_llm", lambda: FakeAnalyst())
    monkeypatch.setattr(debate, "get_llm", lambda: FakeText())
    monkeypatch.setattr(risk_manager, "get_llm", lambda: FakeText())
    monkeypatch.setattr(trader, "get_structured_llm", lambda: AllIn())


def by_name(results):
    return {r["name"]: r for r in results}


def test_quinze_scenarios_figes():
    assert len(SCENARIOS) == 15
    assert len({s.name for s in SCENARIOS}) == 15
    assert {s.agent for s in SCENARIOS} == {"technique", "news_macro", "trader", "desk", "risk", "bull"}


def test_les_verifications_departagent_les_comportements(fakes):
    judge = FakeJudge()
    results, summary = run_evals.run_all(SCENARIOS, judge=judge, log=lambda *_: None)
    r = by_name(results)
    assert r["tech_tendance_haussiere"]["reussite"] == 1
    assert r["tech_tendance_baissiere"]["reussite"] == 0
    assert r["news_tres_negatives"]["reussite"] == 0
    assert r["trader_rapports_negatifs"]["reussite"] == 0
    assert r["trader_apres_refus"]["reussite"] == 0
    assert r["desk_tout_positif"]["reussite"] == 0
    assert r["risk_explique_et_propose"]["reussite"] == 1
    assert r["bull_cite_les_rapports"]["reussite"] == 1
    assert summary["scenarios"] == 15 and summary["sorties_valides"] == 1
    assert summary["fidelite_moyenne"] == 4 and len(judge.prompts) == 13
    assert r["tech_tendance_haussiere"]["runs"][0]["chiffres_inventes"] == ["250"]


def test_sortie_invalide_comptee(fakes, monkeypatch):
    class Broken:
        def invoke(self, prompt, config=None):
            raise RuntimeError("réponse vide")

    monkeypatch.setattr(analysts, "get_structured_llm", lambda: Broken())
    results, summary = run_evals.run_all(SCENARIOS[:2], repeat=2, log=lambda *_: None)
    assert summary["sorties_valides"] == 0 and results[0]["runs"][0]["erreur"].startswith("RuntimeError")


def test_stabilite(fakes):
    results, _ = run_evals.run_all(SCENARIOS[:1], repeat=3, log=lambda *_: None)
    assert results[0]["stabilite"] == 1


def test_chiffres_inventes():
    data = tech(120.0, 6.5, 18.2, 114.0, 98.0, 64.0, 18.0, -1.2)
    assert run_evals.invented_numbers("RSI à 64, hausse de 18,2 % sur 12 semaines, drawdown de 1,2 %", data) == []
    assert run_evals.invented_numbers("objectif à 250 $ et PER de 31,5", data) == ["250", "31,5"]


def test_comparaison(tmp_path):
    def result(label, ok):
        return {"label": label, "totaux": {"reussite_moyenne": ok, "cout_agents": 0.01},
                "resultats": [{"name": "s1", "reussite": ok}]}

    paths = []
    for label, ok in (("haiku", 0.5), ("trader-sonnet", 1.0)):
        p = tmp_path / f"{label}.json"
        p.write_text(json.dumps(result(label, ok)))
        paths.append(str(p))
    table = run_evals.compare(paths)
    assert "| | haiku | trader-sonnet |" in table and "| s1 | 50% | 100% |" in table


def test_numerotation_et_convictions_ne_sont_pas_inventees():
    text = "Bull (tour 1) : voici.\n1. ALFA monte\n2. BETA aussi, conviction 0,83"
    assert run_evals.invented_numbers(text, {}, own=[0.83]) == []
