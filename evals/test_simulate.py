import numpy as np
import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from jp_morgraph import graph as desk_graph
from jp_morgraph import phase5_graph, simulate
from jp_morgraph.agents import analysts, debate, risk_manager, trader
from jp_morgraph.agents.trader import TraderDecision, TraderView
from jp_morgraph.data import MarketStore, next_open
from jp_morgraph.state import AnalystReport

UNIVERSE = ["AAA", "BBB", "CCC"]


class Crash(Exception):
    pass


class FakeAnalyst:
    calls = 0

    def invoke(self, prompt, config=None):
        FakeAnalyst.calls += 1
        return AnalystReport(analyst="technique", ticker="?", signal="achat", conviction=0.6,
                             arguments=["[AAA/technique] momentum"])


class FakeText:
    def invoke(self, prompt, config=None):
        return AIMessage(content="argument")


class FakeTrader:
    def __init__(self):
        self.prompts, self.crash_on = [], None

    def invoke(self, prompt, config=None):
        self.prompts.append(prompt)
        if self.crash_on == len(self.prompts):
            raise Crash("coupure")
        return TraderDecision(orders=[
            TraderView(ticker="AAA", side="buy", conviction=0.5, rationale="[AAA/technique] tendance"),
            TraderView(ticker="BBB", side="buy", conviction=0.3, rationale="Bull"),
        ])


@pytest.fixture
def world(tmp_path, monkeypatch):
    days = pd.bdate_range("2026-01-02", "2026-03-31")
    rng = np.random.default_rng(1)
    cols = UNIVERSE + ["SPY"]
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(days), len(cols))), axis=0)),
                          index=days, columns=cols)
    store = MarketStore(prices=prices, news=pd.DataFrame(), macro={}, opens=prices * 0.999)
    prices.to_parquet(tmp_path / "prices.parquet")

    def load_snapshot(state):
        return {"universe": UNIVERSE,
                "snapshot": {t: {"technique": {"close": simulate.close_on(prices, state["as_of"], t)},
                                 "news_macro": {}} for t in UNIVERSE}}

    fake_trader = FakeTrader()
    FakeAnalyst.calls = 0
    monkeypatch.setattr(phase5_graph, "load_snapshot", load_snapshot)
    monkeypatch.setattr(desk_graph, "execution_prices", lambda as_of, t: next_open(store, as_of, t))
    monkeypatch.setattr(analysts, "get_structured_llm", lambda: FakeAnalyst())
    monkeypatch.setattr(debate, "get_llm", lambda: FakeText())
    monkeypatch.setattr(risk_manager, "get_llm", lambda: FakeText())
    monkeypatch.setattr(trader, "get_structured_llm", lambda: fake_trader)
    return tmp_path, fake_trader, store


def test_vendredis_de_decision():
    prices = pd.DataFrame({"X": 1.0}, index=pd.bdate_range("2026-01-02", "2026-03-30"))
    fridays = simulate.decision_fridays(prices, 3)
    assert fridays == ["2026-03-13", "2026-03-20", "2026-03-27"]
    assert simulate.last_friday(prices) == "2026-03-27"


def test_replay_enchaine_les_portefeuilles(world):
    data_dir, fake, _ = world
    run = simulate.replay("r1", 3, data_dir, log=lambda *_: None)
    records = run.records()
    assert [r["as_of"] for r in records] == simulate.decision_fridays(MarketStore.load(data_dir).prices, 3)
    assert records[0]["statut"] == "exécuté"
    assert records[0]["portfolio"]["positions"]["AAA"] > 0
    assert len(simulate.read_jsonl(run.costs)) == 3
    assert '"AAA": ' in fake.prompts[1]


def test_reprise_apres_interruption(world):
    data_dir, fake, _ = world
    fake.crash_on = 2
    with pytest.raises(Crash):
        simulate.replay("r2", 3, data_dir, log=lambda *_: None)
    run = simulate.Run("r2", data_dir)
    assert len(run.records()) == 1
    second = simulate.decision_fridays(MarketStore.load(data_dir).prices, 3)[1]
    assert run.graph.get_state(run.config(second)).next == ("trader",)
    analysts_before = FakeAnalyst.calls
    fake.crash_on = None
    simulate.replay("r2", 3, data_dir, log=lambda *_: None)
    assert len(simulate.Run("r2", data_dir).records()) == 3
    assert FakeAnalyst.calls - analysts_before == 6


def test_replay_refuse_de_remonter_le_temps(world):
    data_dir, _, _ = world
    simulate.replay("r3", 1, data_dir, log=lambda *_: None)
    with pytest.raises(SystemExit):
        simulate.replay("r3", 3, data_dir, log=lambda *_: None)


def test_relance_sans_doublon(world):
    data_dir, fake, _ = world
    simulate.replay("r4", 2, data_dir, log=lambda *_: None)
    calls = len(fake.prompts)
    simulate.replay("r4", 2, data_dir, log=lambda *_: None)
    assert len(fake.prompts) == calls and len(simulate.Run("r4", data_dir).records()) == 2


def test_rapport(world):
    data_dir, _, _ = world
    simulate.replay("r5", 4, data_dir, log=lambda *_: None)
    path = simulate.report("r5", data_dir)
    s = simulate.summarize("r5", data_dir)
    assert s["cycles"] == 4 and s["hors_limites"] == [] and s["citent_sources"] == 1.0
    assert len(s["nav"]) == 5 and s["nav"]["nav"].iloc[0] == 100_000
    assert s["nav"]["spy"].iloc[0] == 100_000
    assert "Ordres exécutés hors limites | 0" in path.read_text(encoding="utf-8")
    assert (path.parent / "nav.html").exists() and (path.parent / "nav.csv").exists()
    assert simulate.list_runs(data_dir) == ["r5"]


def test_limites_verifiees_dans_le_journal():
    bad = {"as_of": "d", "portfolio": {"cash": -1, "positions": {"A": 500}, "nav": 100_000},
           "fills": [{"side": "buy", "quantity": 1, "ticker": "A", "ref_price": 100.0}]}
    assert len(simulate.limit_violations([bad])) == 2


def test_live_attend_le_lundi_puis_execute(world, monkeypatch):
    data_dir, _, store = world
    as_of = simulate.last_friday(store.prices)
    monkeypatch.setattr(desk_graph, "execution_prices", lambda d, t: (None, {}, "pas encore de lundi"))
    run = simulate.live("live", data_dir, ask=lambda payload: True, log=lambda *_: None)
    assert run.pending.read_text() == as_of and run.records() == []
    monkeypatch.setattr(desk_graph, "execution_prices",
                        lambda d, t: ("2026-04-06", {x: 100.0 for x in t}, "ouverture"))
    run = simulate.live("live", data_dir, ask=lambda payload: True, log=lambda *_: None)
    [record] = run.records()
    assert record["as_of"] == as_of and record["statut"] == "exécuté"
    assert not run.pending.exists()


def test_live_avec_correction_du_gerant(world):
    data_dir, _, _ = world
    run = simulate.live("live2", data_dir, ask=lambda payload: {"quantities": {"AAA": 7}},
                        log=lambda *_: None)
    [record] = run.records()
    aaa = next(o for o in record["approved_orders"] if o["ticker"] == "AAA")
    assert aaa["quantity"] == 7


def test_time_travel(world):
    data_dir, fake, _ = world
    run = simulate.replay("r6", 2, data_dir, log=lambda *_: None)
    as_of = run.records()[0]["as_of"]
    analysts_before = FakeAnalyst.calls
    before, after = simulate.travel("r6", as_of, "Sois très prudent {ici}", data_dir)
    assert "Consigne du gérant : Sois très prudent {ici}" in fake.prompts[-1]
    assert FakeAnalyst.calls == analysts_before
    assert [o["ticker"] for o in before] == [o["ticker"] for o in after]
    assert len(simulate.read_jsonl(run.dir / "time_travel.jsonl")) == 1
    assert len(run.records()) == 2
    assert "Consigne" not in trader.PROMPT_TEMPLATE


def test_sortie_invalide_relancee(world, monkeypatch):
    data_dir, _, _ = world

    class Flaky(FakeTrader):
        attempts = 0

        def invoke(self, prompt, config=None):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("Trader : réponse vide")
            return super().invoke(prompt, config)

    flaky = Flaky()
    monkeypatch.setattr(trader, "get_structured_llm", lambda: flaky)
    run = simulate.replay("r7", 1, data_dir, log=lambda *_: None)
    assert len(run.records()) == 1 and flaky.attempts == 2


def test_dashboard(world, monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    data_dir, _, _ = world
    simulate.replay("r8", 3, data_dir, log=lambda *_: None)
    (data_dir / "evals").mkdir()
    (data_dir / "evals" / "haiku.json").write_text('{"label": "haiku", "totaux": {"reussite_moyenne": 0.9}}')
    monkeypatch.setenv("JP_DATA_DIR", str(data_dir))
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app" / "dashboard.py"), default_timeout=60).run()
    assert not app.exception
    assert app.metric[0].label == "Valeur liquidative"
    assert any("Dernier cycle" in h.value for h in app.subheader)


def test_dashboard_sans_run(tmp_path, monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("JP_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app" / "dashboard.py"), default_timeout=60).run()
    assert not app.exception and "Aucun run" in app.info[0].value
