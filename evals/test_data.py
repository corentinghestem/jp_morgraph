"""Tests du Data Desk : aucune fuite du futur, aucun réseau, résultats reproductibles.

Les données sont synthétiques : les tests ne nécessitent ni clé API ni connexion.
"""

import copy
import json
import socket

import numpy as np
import pandas as pd
import pytest

from jp_morgraph.data import (
    MarketStore,
    assert_no_future,
    compute_indicators,
    snapshot,
    to_weekly,
)

UNIVERSE = ["AAA", "BBB", "CCC"]


def _make_store(seed: int = 0) -> MarketStore:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2023-06-01", "2026-10-23")
    cols = UNIVERSE + ["SPY"]
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, (len(days), len(cols))), axis=0)),
        index=days,
        columns=cols,
    )

    # une trentaine par titre et par trimestre, à des heures variées (UTC).
    rows = []
    for t in UNIVERSE:
        stamps = pd.to_datetime(
            rng.integers(
                pd.Timestamp("2023-07-01").value // 10**9,
                pd.Timestamp("2026-10-23").value // 10**9,
                400,
            ),
            unit="s",
            utc=True,
        )
        for i, ts in enumerate(stamps):
            rows.append({"ticker": t, "published_at": ts, "headline": f"{t} titre {i}", "source": "Test"})
    news = pd.DataFrame(rows).sort_values("published_at").reset_index(drop=True)

    # Macro : CPI mensuel publié ~45 jours après le mois, avec une révision ;
    # taux 10 ans quotidien publié J+1.
    months = pd.date_range("2023-01-01", "2026-09-01", freq="MS")
    cpi = pd.DataFrame(
        {
            "date": months,
            "available_from": months + pd.Timedelta(days=45),
            "value": 300 + np.arange(len(months)) * 0.7,
        }
    )
    rev = cpi.iloc[[-5]].copy()
    rev["available_from"] = rev["available_from"] + pd.Timedelta(days=30)
    rev["value"] += 5.0  # révision à la hausse, connue un mois plus tard
    cpi = pd.concat([cpi, rev], ignore_index=True)

    dgs = pd.DataFrame(
        {"date": days, "available_from": days + pd.Timedelta(days=1), "value": 4 + rng.normal(0, 0.05, len(days)).cumsum()}
    )
    return MarketStore(prices=prices, news=news, macro={"CPIAUCSL": cpi, "DGS10": dgs})


@pytest.fixture(scope="module")
def store() -> MarketStore:
    return _make_store()


@pytest.fixture(scope="module")
def fridays() -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.date_range("2024-07-05", "2026-10-16", freq="W-FRI")]


def _corrupt_future(store: MarketStore, as_of: str) -> MarketStore:
    """Copie du store où tout ce qui est postérieur à as_of est remplacé par des valeurs absurdes."""
    day = pd.Timestamp(as_of)
    s = copy.deepcopy(store)
    s.prices.loc[s.prices.index > day] = 1e9
    s.news.loc[s.news["published_at"] >= day.tz_localize("America/New_York") + pd.DateOffset(days=1), "headline"] = "FUTUR"
    for df in s.macro.values():
        df.loc[df["available_from"] > day, "value"] = -999.0
    return s


# --------------------------------------------------------------------------- #
# Tests anti-fuite
# --------------------------------------------------------------------------- #
def test_20_random_dates_have_no_future_data(store, fridays):
    rng = np.random.default_rng(42)
    for as_of in rng.choice(fridays, size=20, replace=False):
        snap = snapshot(str(as_of), store, UNIVERSE)
        assert_no_future(snap, str(as_of))  # lève en cas de fuite


def test_changing_the_future_does_not_change_the_snapshot(store, fridays):
    """Le test le plus parlant : on saccage tout ce qui suit as_of, rien ne doit bouger."""
    rng = np.random.default_rng(7)
    for as_of in rng.choice(fridays, size=10, replace=False):
        clean = snapshot(str(as_of), store, UNIVERSE)
        dirty = snapshot(str(as_of), _corrupt_future(store, str(as_of)), UNIVERSE)
        assert json.dumps(clean, sort_keys=True) == json.dumps(dirty, sort_keys=True)


def test_assert_no_future_catches_a_real_leak():
    leaky = {"AAA": {"last_price_date": "2026-10-19", "news": [{"date": "2026-10-10"}]}}
    with pytest.raises(AssertionError):
        assert_no_future(leaky, "2026-10-16")


def test_price_boundary_is_inclusive(store):
    as_of = "2026-10-16"
    snap = snapshot(as_of, store, UNIVERSE)
    assert snap["AAA"]["last_price_date"] == as_of  # le prix du vendredi est bien connu
    assert snapshot("2026-10-15", store, UNIVERSE)["AAA"]["last_price_date"] == "2026-10-15"


def test_news_cutoff_is_midnight_new_york(store):
    inside = pd.Timestamp("2026-10-16 23:30", tz="America/New_York").tz_convert("UTC")
    outside = pd.Timestamp("2026-10-17 00:30", tz="America/New_York").tz_convert("UTC")
    extra = pd.DataFrame(
        [
            {"ticker": "AAA", "published_at": inside, "headline": "DANS", "source": "t"},
            {"ticker": "AAA", "published_at": outside, "headline": "HORS", "source": "t"},
        ]
    )
    s = MarketStore(store.prices, pd.concat([store.news, extra], ignore_index=True), store.macro)
    titles = [n["titre"] for n in snapshot("2026-10-16", s, UNIVERSE)["AAA"]["news"]]
    assert "DANS" in titles
    assert "HORS" not in titles


def test_macro_is_only_visible_once_published(store):
    cpi = store.macro["CPIAUCSL"]
    last_obs = cpi.sort_values("available_from").iloc[-2]  # une observation récente, hors révision
    day_before = (last_obs["available_from"] - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    day_of = last_obs["available_from"].strftime("%Y-%m-%d")

    seen_before = snapshot(day_before, store, UNIVERSE)["AAA"]["macro"]["CPIAUCSL"]["obs_date"]
    seen_on_day = snapshot(day_of, store, UNIVERSE)["AAA"]["macro"]["CPIAUCSL"]["obs_date"]
    assert pd.Timestamp(seen_before) < last_obs["date"]
    assert pd.Timestamp(seen_on_day) >= last_obs["date"]


def test_macro_revision_is_used_only_after_its_release(store):
    cpi = store.macro["CPIAUCSL"]
    dup = cpi[cpi.duplicated("date", keep=False)].sort_values("available_from")
    first, second = dup.iloc[0], dup.iloc[1]
    assert second["value"] != first["value"]

    store_view = store.truncated((second["available_from"] - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    v = store_view.macro["CPIAUCSL"].set_index("date").loc[first["date"], "value"]
    assert v == first["value"]  # ancienne version tant que la révision n'est pas publiée

    store_view = store.truncated(second["available_from"].strftime("%Y-%m-%d"))
    v = store_view.macro["CPIAUCSL"].set_index("date").loc[first["date"], "value"]
    assert v == second["value"]


def test_incomplete_week_is_not_labelled_with_a_future_friday(store):
    wednesday = "2026-10-14"
    snap = snapshot(wednesday, store, UNIVERSE)
    assert snap["AAA"]["last_price_date"] == wednesday
    weekly = to_weekly(store.truncated(wednesday).prices)
    assert weekly.index[-1] == pd.Timestamp(wednesday)


# --------------------------------------------------------------------------- #
# Reproductibilité, réseau, taille, cas limites
# --------------------------------------------------------------------------- #
def test_same_date_same_snapshot(store):
    a = snapshot("2026-10-16", store, UNIVERSE)
    b = snapshot("2026-10-16", store, UNIVERSE)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_parquet_roundtrip_and_no_network(store, tmp_path, monkeypatch):
    store.save(tmp_path)

    def no_network(*args, **kwargs):
        raise AssertionError("Appel réseau pendant un snapshot")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    loaded = MarketStore.load(tmp_path)
    a = snapshot("2026-10-16", store, UNIVERSE)
    b = snapshot("2026-10-16", loaded, UNIVERSE)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_snapshot_stays_compact(store):
    snap = snapshot("2026-10-16", store, UNIVERSE)
    size = len(json.dumps(snap["AAA"], ensure_ascii=False))
    assert size < 2600, f"snapshot trop long : {size} caractères"


def test_short_history_gives_none_not_an_invented_number(store):
    early = snapshot("2023-07-14", store, UNIVERSE)["AAA"]["indicateurs"]
    assert early["sma40w"] is None
    assert early["ret_12w_pct"] is None or early["semaines_d_historique"] > 12


def test_indicators_on_known_series():
    close = pd.Series(np.arange(1, 61, dtype=float))  # tendance haussière parfaite
    ind = compute_indicators(close)
    assert ind["rsi14w"] == 100.0
    assert ind["drawdown_vs_plus_haut_pct"] == 0.0
    assert ind["ret_4w_pct"] == pytest.approx((60 / 56 - 1) * 100, abs=0.01)
    assert ind["sma10w"] == pytest.approx(np.mean(np.arange(51, 61)), abs=0.01)