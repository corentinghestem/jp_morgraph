"""Data Desk : données de marché, news et macro, sans fuite du futur.

Principe central : tout ce que voit un agent passe par ``snapshot(as_of, store)``.
Cette fonction commence par **tronquer** le store à la date de décision, puis
calcule tout sur les données tronquées. Une donnée postérieure à ``as_of`` ne
peut donc pas atteindre les indicateurs, par construction.

Séparation réseau / simulation :
- ``refresh_*`` : seules fonctions qui font des appels réseau (yfinance, Finnhub, FRED).
  À lancer à la main, jamais pendant une simulation.
- ``MarketStore`` + ``snapshot`` : lecture de fichiers Parquet uniquement.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BENCHMARK = "SPY"
NY = "America/New_York"

# Séries FRED suivies : identifiant -> libellé affiché aux agents.
MACRO_SERIES = {
    "DGS10": "Taux du Trésor US 10 ans (%)",
    "CPIAUCSL": "Indice des prix à la consommation (CPI)",
    "UNRATE": "Taux de chômage US (%)",
    "VIXCLS": "Indice de volatilité VIX",
}

# Délai de publication prudent (en jours), utilisé seulement si FRED ne fournit
# pas les dates de publication réelles (vintages).
FALLBACK_LAG_DAYS = {"DGS10": 1, "VIXCLS": 1, "UNRATE": 35, "CPIAUCSL": 45}

# Séries quotidiennes : jamais révisées, et trop de versions pour l'API des vintages
# (limite de 2000 dates). Elles utilisent directement le délai de publication.
DAILY_SERIES = {"DGS10", "VIXCLS"}

NEWS_MAX_ITEMS = 5
NEWS_WINDOW_DAYS = 14
HEADLINE_MAX_CHARS = 140


# --------------------------------------------------------------------------- #
# Store : les données brutes, rien que des DataFrames
# --------------------------------------------------------------------------- #
@dataclass
class MarketStore:
    """Données brutes en cache.

    prices : index = jours de bourse, colonnes = tickers (+ SPY), clôtures ajustées.
    news   : colonnes ticker, published_at (UTC, tz-aware), headline, source.
    macro  : série -> DataFrame[date, available_from, value] ; une ligne par
             observation *et par version publiée* (une révision = une ligne de plus).
    """

    prices: pd.DataFrame
    news: pd.DataFrame
    macro: dict[str, pd.DataFrame]

    def save(self, directory: str | Path = "data") -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.prices.to_parquet(d / "prices.parquet")
        self.news.to_parquet(d / "news.parquet")
        for sid, df in self.macro.items():
            df.to_parquet(d / f"macro_{sid}.parquet")

    @classmethod
    def load(cls, directory: str | Path = "data") -> MarketStore:
        d = Path(directory)
        prices = pd.read_parquet(d / "prices.parquet")
        news_path = d / "news.parquet"
        news = pd.read_parquet(news_path) if news_path.exists() else _empty_news()
        macro = {}
        for sid in MACRO_SERIES:
            p = d / f"macro_{sid}.parquet"
            if p.exists():
                macro[sid] = pd.read_parquet(p)
        return cls(prices=prices, news=news, macro=macro)

    def truncated(self, as_of: str | pd.Timestamp) -> MarketStore:
        """Ne garde que ce qui était connu à la clôture du jour ``as_of``."""
        day = _day(as_of)
        prices = self.prices.loc[self.prices.index <= day]

        # Une news est connue si publiée avant minuit (heure de New York) à l'issue du jour as_of.
        cutoff = day.tz_localize(NY) + pd.DateOffset(days=1)
        news = self.news[self.news["published_at"] < cutoff]

        # Macro : on ne garde que les versions déjà publiées, puis, pour chaque
        # observation, la version la plus récente connue à la date.
        macro = {}
        for sid, df in self.macro.items():
            known = df[df["available_from"] <= day]
            known = (
                known.sort_values(["date", "available_from"])
                .groupby("date", as_index=False)
                .last()
            )
            macro[sid] = known
        return MarketStore(prices=prices, news=news, macro=macro)


def _empty_news() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": pd.Series(dtype="string"),
            "published_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "headline": pd.Series(dtype="string"),
            "source": pd.Series(dtype="string"),
        }
    )


def _day(as_of: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(as_of).normalize().tz_localize(None)


# --------------------------------------------------------------------------- #
# Indicateurs (fonctions pures sur des clôtures déjà tronquées)
# --------------------------------------------------------------------------- #
def to_weekly(daily: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Clôtures hebdomadaires, indexées par le *dernier jour de bourse réel* de la semaine.

    On n'utilise pas ``resample("W-FRI")`` seul : il étiquetterait une semaine
    incomplète avec un vendredi futur.
    """
    grouper = pd.Grouper(freq="W-FRI")
    weekly = daily.groupby(grouper).last()
    last_days = daily.index.to_series().groupby(grouper).last()
    keep = last_days.notna()
    weekly = weekly[keep]
    weekly.index = pd.DatetimeIndex(last_days[keep])
    return weekly


def _rsi(close: pd.Series, n: int = 14) -> float | None:
    if len(close) < n + 1:
        return None
    delta = close.diff().dropna()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean().iloc[-1]
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean().iloc[-1]
    if down == 0:
        return 100.0
    return float(100 - 100 / (1 + up / down))


def compute_indicators(weekly_close: pd.Series) -> dict:
    """Indicateurs hebdomadaires. Une valeur non calculable vaut ``None``
    (l'analyste doit alors écrire « information insuffisante »)."""
    c = weekly_close.dropna()
    n = len(c)

    def ret(weeks: int) -> float | None:
        return round(float(c.iloc[-1] / c.iloc[-1 - weeks] - 1) * 100, 2) if n > weeks else None

    sma10 = float(c.iloc[-10:].mean()) if n >= 10 else None
    sma40 = float(c.iloc[-40:].mean()) if n >= 40 else None
    weekly_ret = c.pct_change().dropna()
    vol = (
        round(float(weekly_ret.iloc[-12:].std() * np.sqrt(52)) * 100, 2)
        if len(weekly_ret) >= 12
        else None
    )
    drawdown = round(float(c.iloc[-1] / c.cummax().iloc[-1] - 1) * 100, 2) if n else None
    rsi = _rsi(c)
    last = float(c.iloc[-1]) if n else None

    return {
        "ret_4w_pct": ret(4),
        "ret_12w_pct": ret(12),
        "sma10w": round(sma10, 2) if sma10 is not None else None,
        "sma40w": round(sma40, 2) if sma40 is not None else None,
        "close_vs_sma40w_pct": (
            round((last / sma40 - 1) * 100, 2) if sma40 is not None and last else None
        ),
        "rsi14w": round(rsi, 1) if rsi is not None else None,
        "vol_12w_annualisee_pct": vol,
        "drawdown_vs_plus_haut_pct": drawdown,
        "semaines_d_historique": n,
    }


# --------------------------------------------------------------------------- #
# Snapshot : ce que voient les agents
# --------------------------------------------------------------------------- #
def _macro_summary(macro: dict[str, pd.DataFrame]) -> dict:
    out = {}
    for sid, df in macro.items():
        if df.empty:
            continue
        last = df.sort_values("date").iloc[-1]
        item = {
            "libelle": MACRO_SERIES.get(sid, sid),
            "valeur": round(float(last["value"]), 2),
            "obs_date": last["date"].strftime("%Y-%m-%d"),
            "available_from": last["available_from"].strftime("%Y-%m-%d"),
        }
        if sid == "CPIAUCSL":
            # Inflation sur un an : comparaison avec l'observation d'il y a ~12 mois.
            target = last["date"] - pd.DateOffset(years=1)
            prior = df[(df["date"] - target).abs() <= pd.Timedelta(days=10)]
            if not prior.empty and float(prior.iloc[0]["value"]) != 0:
                item["variation_sur_1_an_pct"] = round(
                    (float(last["value"]) / float(prior.iloc[0]["value"]) - 1) * 100, 2
                )
        out[sid] = item
    return out


def _news_summary(news: pd.DataFrame, ticker: str, day: pd.Timestamp) -> list[dict]:
    n = news[news["ticker"] == ticker].copy()
    if n.empty:
        return []
    n["local"] = n["published_at"].dt.tz_convert(NY)
    n["date"] = n["local"].dt.tz_localize(None).dt.normalize()
    n = n[n["date"] > day - pd.Timedelta(days=NEWS_WINDOW_DAYS)]
    n = n.drop_duplicates("headline").sort_values("published_at", ascending=False)
    return [
        {
            "date": row.date.strftime("%Y-%m-%d"),
            "titre": str(row.headline)[:HEADLINE_MAX_CHARS],
            "source": str(row.source),
        }
        for row in n.head(NEWS_MAX_ITEMS).itertuples()
    ]


def snapshot(as_of: str, store: MarketStore, universe: list[str]) -> dict:
    """Résumé court par titre, limité à ce qui était connu le ``as_of`` après la clôture.

    Retourne ``{ticker: {...}}`` : c'est ce dictionnaire que reçoivent les agents
    (``state["snapshot"][ticker]``). Aucune requête réseau.
    """
    day = _day(as_of)
    known = store.truncated(day)
    weekly = to_weekly(known.prices)

    macro = _macro_summary(known.macro)
    spy_ret_12w = (
        compute_indicators(weekly[BENCHMARK])["ret_12w_pct"] if BENCHMARK in weekly else None
    )

    snap: dict[str, dict] = {}
    for t in universe:
        if t not in weekly:
            snap[t] = {"as_of": day.strftime("%Y-%m-%d"), "erreur": "aucune donnée de prix"}
            continue
        close = weekly[t].dropna()
        ind = compute_indicators(close)
        rel = (
            round(ind["ret_12w_pct"] - spy_ret_12w, 2)
            if ind["ret_12w_pct"] is not None and spy_ret_12w is not None
            else None
        )
        snap[t] = {
            "as_of": day.strftime("%Y-%m-%d"),
            "last_price_date": close.index[-1].strftime("%Y-%m-%d"),
            "close": round(float(close.iloc[-1]), 2),
            "indicateurs": ind,
            "ret_12w_vs_spy_pts": rel,
            "news": _news_summary(known.news, t, day),
            "macro": macro,
        }
    return snap


def assert_no_future(snap: dict, as_of: str) -> None:
    """Garde-fou exécutable à chaque cycle : lève si une date du snapshot dépasse ``as_of``.

    Vérifie toutes les clés qui contiennent « date » ou « available_from », à toute profondeur.
    """
    limit = _day(as_of)

    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, str) and ("date" in k or k == "available_from"):
                    if pd.Timestamp(v) > limit:
                        raise AssertionError(f"Fuite du futur : {path}/{k} = {v} > {as_of}")
                walk(v, f"{path}/{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(snap)


# --------------------------------------------------------------------------- #
# Réseau : à lancer à la main, jamais pendant une simulation
# --------------------------------------------------------------------------- #
def refresh_prices(tickers: list[str], directory: str | Path = "data", years: int = 3) -> pd.DataFrame:
    """Un seul téléchargement groupé (évite les erreurs 429), clôtures journalières ajustées.

    On télécharge du journalier et on agrège en hebdomadaire nous-mêmes (``to_weekly``) :
    les barres hebdomadaires de yfinance sont datées du lundi, ce qui prête à confusion.
    Limite connue : un prix ajusté tient compte des dividendes futurs. Les rendements
    restent corrects, mais les niveaux de prix passés sont légèrement décalés.
    """
    import yfinance as yf

    # Cache de fuseaux horaires propre au projet : le cache global partagé est la source
    # de l'erreur « database is locked » quand plusieurs processus yfinance tournent.
    cache_dir = Path(directory) / ".yfinance_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))

    symbols = sorted(set(tickers) | {BENCHMARK})
    start = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    # threads=False : le téléchargement parallèle de yfinance provoque des erreurs
    # « database is locked » (cache SQLite partagé entre threads).
    raw = yf.download(
        symbols, start=start, auto_adjust=True, progress=False, group_by="column", threads=False
    )
    prices = raw["Close"].copy()
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()
    prices = prices.dropna(how="all").sort_index()

    # On refuse d'écrire un cache incomplet : mieux vaut une erreur claire qu'un titre muet.
    missing = [t for t in symbols if t not in prices.columns or prices[t].dropna().empty]
    if missing:
        raise RuntimeError(
            f"Aucune donnée de prix pour : {missing}. "
            "Vérifie les symboles dans config.py (titre inexistant ou délisté ?) "
            "ou relance si l'erreur était temporaire. Cache non modifié."
        )
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(d / "prices.parquet")
    return prices


def refresh_news(tickers: list[str], directory: str | Path = "data", days_back: int = 365) -> pd.DataFrame:
    """News Finnhub par fenêtres de 30 jours. Le cache s'enrichit à chaque exécution."""
    import finnhub

    key = os.environ.get("FINNHUB_API_KEY")
    if not key:
        raise RuntimeError("FINNHUB_API_KEY manquante dans l'environnement (.env).")
    client = finnhub.Client(api_key=key)
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "news.parquet"
    old = pd.read_parquet(path) if path.exists() else _empty_news()

    today = pd.Timestamp.today().normalize()
    rows = []
    for t in tickers:
        end = today
        while end > today - pd.Timedelta(days=days_back):
            begin = end - pd.Timedelta(days=30)
            items = client.company_news(
                t, _from=begin.strftime("%Y-%m-%d"), to=end.strftime("%Y-%m-%d")
            )
            for it in items:
                if it.get("datetime") and it.get("headline"):
                    rows.append(
                        {
                            "ticker": t,
                            "published_at": pd.to_datetime(it["datetime"], unit="s", utc=True),
                            "headline": it["headline"],
                            "source": it.get("source", ""),
                        }
                    )
            end = begin
            time.sleep(1.1)  # ~60 appels/min en offre gratuite
    new = pd.DataFrame(rows) if rows else _empty_news()
    news = (
        pd.concat([old, new], ignore_index=True)
        .drop_duplicates(["ticker", "published_at", "headline"])
        .sort_values("published_at")
        .reset_index(drop=True)
    )
    news.to_parquet(path)
    return news


def refresh_macro(directory: str | Path = "data", years: int = 3) -> dict[str, pd.DataFrame]:
    """Séries FRED avec leurs *dates de publication réelles* (vintages ALFRED).

    ``get_series_all_releases`` renvoie une ligne par observation et par version publiée,
    avec ``realtime_start`` = jour où cette valeur est devenue publique. C'est ce qui permet
    de ne pas utiliser un CPI d'octobre en septembre. Si l'appel échoue pour une série,
    on retombe sur un délai de publication prudent (``FALLBACK_LAG_DAYS``).
    """
    from fredapi import Fred

    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY manquante dans l'environnement (.env).")
    fred = Fred(api_key=key)
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp.today() - pd.DateOffset(years=years + 1)

    out = {}
    for sid in MACRO_SERIES:
        df = None
        if sid not in DAILY_SERIES:  # séries mensuelles : révisables, on veut les vintages
            try:
                raw = fred.get_series_all_releases(sid)
                df = pd.DataFrame(
                    {
                        "date": pd.to_datetime(raw["date"]),
                        "available_from": pd.to_datetime(raw["realtime_start"]),
                        # errors="coerce" : une observation manquante (ex. CPI d'octobre 2025)
                        # devient NaN au lieu de faire planter la conversion.
                        "value": pd.to_numeric(raw["value"], errors="coerce"),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - repli documenté
                print(f"[macro] {sid} : vintages indisponibles ({exc}) → délai prudent.")
        if df is None:
            # Séries quotidiennes (taux, VIX) : jamais révisées, publiées avec ~1 jour de décalage.
            s = fred.get_series(sid, observation_start=start.strftime("%Y-%m-%d")).dropna()
            df = pd.DataFrame({"date": pd.to_datetime(s.index), "value": s.values.astype(float)})
            df["available_from"] = df["date"] + pd.Timedelta(days=FALLBACK_LAG_DAYS[sid])
        df = df.dropna(subset=["value"])
        df = df[df["date"] >= start].sort_values(["date", "available_from"]).reset_index(drop=True)
        df.to_parquet(d / f"macro_{sid}.parquet")
        out[sid] = df
    return out


# --------------------------------------------------------------------------- #
# Ligne de commande
# --------------------------------------------------------------------------- #
def _default_tickers() -> list[str]:
    """Titres du projet, lus dans config.py (seul endroit où data.py touche à la config)."""
    from jp_morgraph.config import tickers

    return [t for t in tickers if t != BENCHMARK]  # le SPY est ajouté à part comme repère


def _main() -> None:
    parser = argparse.ArgumentParser(description="Data Desk jp_morgraph")
    sub = parser.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("refresh", help="Télécharge prix, news et macro vers data/")
    r.add_argument("--tickers", nargs="+", help="par défaut : `tickers` de config.py")
    r.add_argument("--dir", default="data")

    s = sub.add_parser("snapshot", help="Affiche le snapshot à une date (lecture du cache)")
    s.add_argument("--as-of", required=True)
    s.add_argument("--tickers", nargs="+", help="par défaut : `tickers` de config.py")
    s.add_argument("--dir", default="data")

    args = parser.parse_args()
    tickers = args.tickers or _default_tickers()
    if args.cmd == "refresh":
        from dotenv import load_dotenv

        load_dotenv()
        refresh_prices(tickers, args.dir)
        refresh_news(tickers, args.dir)
        refresh_macro(args.dir)
        print("Cache mis à jour dans", args.dir)
    else:
        store = MarketStore.load(args.dir)
        snap = snapshot(args.as_of, store, tickers)
        assert_no_future(snap, args.as_of)
        print(json.dumps(snap, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _main()