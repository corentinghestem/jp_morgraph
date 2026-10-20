import json
import random

import pandas as pd

from jp_morgraph.config import MAX_WEIGHT
from jp_morgraph.data import MarketStore, _empty_news, next_open
from jp_morgraph.desk import (
    SLIPPAGE,
    append_journal,
    check_orders,
    execute_orders,
    initial_portfolio,
    size_order,
    trade,
)
from jp_morgraph.state import OrderProposal

PRICES = {"AAA": 100.0, "BBB": 50.0, "CCC": 20.0}


def order(ticker, side, qty, conviction=0.5):
    return OrderProposal(ticker=ticker, side=side, conviction=conviction, quantity=qty,
                         rationale="test")


def test_ordres_conformes():
    assert check_orders([order("AAA", "buy", 300), order("BBB", "hold", 0)],
                        initial_portfolio(), PRICES) == []


def test_vente_a_decouvert():
    pf = {"cash": 90_000, "positions": {"AAA": 10}, "nav": 91_000}
    [b] = check_orders([order("AAA", "sell", 11)], pf, PRICES)
    assert "découvert" in b


def test_poids_max_par_ligne():
    [b] = check_orders([order("AAA", "buy", 401)], initial_portfolio(), PRICES)
    assert "AAA : poids" in b


def test_cash_negatif_et_exposition():
    orders = [order(t, "buy", size_order(1.0, 100_000, 0, p)) for t, p in PRICES.items()]
    breaches = check_orders(orders, initial_portfolio(), PRICES)
    assert any("cash négatif" in b for b in breaches)
    assert any("exposition" in b for b in breaches)


def test_conviction_1_reste_sous_40_pct_apres_couts():
    qty = size_order(1.0, 100_000, 0, 100.0)
    assert check_orders([order("AAA", "buy", qty)], initial_portfolio(), PRICES) == []


def test_position_qui_a_derive_n_est_pas_une_violation():
    pf = {"cash": 50_000, "positions": {"AAA": 500}, "nav": 100_000}
    assert check_orders([order("AAA", "hold", 0), order("BBB", "buy", 100)], pf, PRICES) == []
    assert check_orders([order("AAA", "buy", 1)], pf, PRICES) != []


def test_prix_inconnu():
    [b] = check_orders([order("ZZZ", "buy", 1)], initial_portfolio(), PRICES)
    assert "aucun prix" in b


def test_slippage_toujours_defavorable():
    buy_px, buy_fees, buy_cash = trade("buy", 10, 100.0)
    sell_px, sell_fees, sell_cash = trade("sell", 10, 100.0)
    assert buy_px == 100.0 * (1 + SLIPPAGE) and sell_px == 100.0 * (1 - SLIPPAGE)
    assert buy_cash == -(10 * buy_px + buy_fees) and sell_cash == 10 * sell_px - sell_fees
    assert buy_cash + sell_cash < 0


def test_execution_et_registre_du_portefeuille():
    fills, pf = execute_orders([order("AAA", "buy", 100)], initial_portfolio(), {"AAA": 101.0},
                               "2026-10-12")
    [f] = fills
    assert f["quantity"] == 100 and f["exec_price"] == round(101.0 * (1 + SLIPPAGE), 4)
    assert pf["positions"] == {"AAA": 100}
    assert pf["cash"] == round(100_000 - f["amount"] - f["fees"], 2)
    assert pf["nav"] == round(pf["cash"] + 100 * 101.0, 2)


def test_gap_a_l_ouverture_reduit_l_achat():
    qty = size_order(1.0, 100_000, 0, 100.0)
    fills, pf = execute_orders([order("AAA", "buy", qty)], initial_portfolio(), {"AAA": 120.0},
                               "2026-10-12")
    assert fills[0]["quantity"] < qty and "réduit" in fills[0]["note"]
    assert pf["positions"]["AAA"] * 120.0 / pf["nav"] <= MAX_WEIGHT


def test_ventes_avant_achats():
    pf = {"cash": 0.0, "positions": {"AAA": 300}, "nav": 30_000}
    fills, after = execute_orders([order("BBB", "buy", 100), order("AAA", "sell", 100)], pf,
                                  {"AAA": 100.0, "BBB": 50.0}, "2026-10-12")
    assert [f["side"] for f in fills] == ["sell", "buy"]
    assert all(f["quantity"] == 100 for f in fills) and after["cash"] >= 0


def test_pas_de_prix_rien_ne_change():
    pf = initial_portfolio()
    fills, after = execute_orders([order("AAA", "buy", 10)], pf, {}, None, "cache trop court")
    assert fills[0]["quantity"] == 0 and after == pf


def test_aucun_ordre_execute_hors_limites():
    rng = random.Random(0)
    pf, executed = initial_portfolio(), 0
    for _ in range(500):
        close = {t: p * rng.uniform(0.7, 1.4) for t, p in PRICES.items()}
        nav = pf["cash"] + sum(q * close[t] for t, q in pf["positions"].items())
        pf = {**pf, "nav": nav}
        orders = []
        for t in PRICES:
            side = rng.choice(["buy", "sell", "hold"])
            held = pf["positions"].get(t, 0)
            qty = (max(0, size_order(rng.random(), nav, held * close[t], close[t]))
                   if side == "buy" else rng.randint(0, held) if side == "sell" else 0)
            orders.append(order(t, side, qty))
        if check_orders(orders, pf, close):
            continue
        opens = {t: p * rng.uniform(0.85, 1.15) for t, p in close.items()}
        fills, pf = execute_orders(orders, pf, opens, "j+1")
        executed += sum(f["quantity"] > 0 for f in fills)
        assert pf["cash"] >= 0
        assert all(q > 0 for q in pf["positions"].values())
        for f in fills:
            if f["side"] == "buy" and f["quantity"] > 0:
                assert pf["positions"][f["ticker"]] * opens[f["ticker"]] / pf["nav"] <= MAX_WEIGHT
    assert executed > 100


def test_journal_jsonl(tmp_path):
    path = tmp_path / "sous_dossier" / "journal.jsonl"
    append_journal(path, {"as_of": "2026-10-09", "statut": "exécuté"})
    append_journal(path, {"as_of": "2026-10-16", "statut": "rejeté par le gérant"})
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["as_of"] for r in lines] == ["2026-10-09", "2026-10-16"]
    assert len(pd.read_json(path, lines=True)) == 2


def _store(with_opens: bool) -> MarketStore:
    days = pd.to_datetime(["2026-10-08", "2026-10-09", "2026-10-12"])
    closes = pd.DataFrame({"AAA": [99.0, 100.0, 103.0]}, index=days)
    opens = pd.DataFrame({"AAA": [98.0, 99.5, 101.0]}, index=days) if with_opens else None
    return MarketStore(prices=closes, news=_empty_news(), macro={}, opens=opens)


def test_next_open_lundi_a_l_ouverture():
    assert next_open(_store(True), "2026-10-09", ["AAA"]) == (
        "2026-10-12", {"AAA": 101.0}, "ouverture")


def test_next_open_sans_ouvertures_prend_la_cloture():
    date, prices, source = next_open(_store(False), "2026-10-09", ["AAA"])
    assert (date, prices) == ("2026-10-12", {"AAA": 103.0}) and "clôture" in source


def test_next_open_apres_la_fin_du_cache():
    assert next_open(_store(True), "2026-10-12", ["AAA"])[0] is None


def test_les_agents_ne_voient_jamais_les_ouvertures():
    assert _store(True).truncated("2026-10-12").opens is None
