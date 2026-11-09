import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from jp_morgraph.desk import check_orders, initial_portfolio, mark_prices, size_order
from jp_morgraph.state import AnalystReport, OrderProposal, RiskVerdict

AS_OF = "2026-06-05"
TICKERS = ["ALFA", "BETA", "GAMA"]
TAG = re.compile(r"\[[A-Z.]+/(technique|news_macro)\]")
INSUFFISANT = re.compile(r"insuffisant|manqu|indisponible|absen", re.IGNORECASE)


@dataclass
class Scenario:
    name: str
    agent: str
    attendu: str
    inputs: dict
    check: Callable[[Any], bool]
    juge: bool = True


def tech(close, ret4, ret12, sma10, sma40, rsi, vol, dd, weeks=156, rel=None):
    ext = round((close / sma40 - 1) * 100, 2) if sma40 else None
    return {"as_of": AS_OF, "last_price_date": AS_OF, "close": close,
            "indicateurs": {"ret_4w_pct": ret4, "ret_12w_pct": ret12, "sma10w": sma10, "sma40w": sma40,
                            "close_vs_sma40w_pct": ext, "rsi14w": rsi,
                            "vol_12w_annualisee_pct": vol, "drawdown_vs_plus_haut_pct": dd,
                            "semaines_d_historique": weeks},
            "ret_12w_vs_spy_pts": rel}


def macro(vix=16.2, dgs10=4.12, cpi=2.8, unrate=4.1):
    def item(label, value, obs):
        return {"libelle": label, "valeur": value, "obs_date": obs, "available_from": obs}

    return {"DGS10": item("Taux du Trésor US 10 ans (%)", dgs10, "2026-06-04"),
            "CPIAUCSL": {**item("Indice des prix à la consommation (CPI)", 331.4, "2026-04-01"),
                         "variation_sur_1_an_pct": cpi},
            "UNRATE": item("Taux de chômage US (%)", unrate, "2026-05-01"),
            "VIXCLS": item("Indice de volatilité VIX", vix, "2026-06-04")}


def news(*headlines, **kw):
    return {"as_of": AS_OF, "macro": macro(**kw),
            "news": [{"date": f"2026-06-0{i + 1}", "titre": h, "source": "Reuters"}
                     for i, h in enumerate(headlines)]}


def rep(ticker, analyst, signal, conviction, *arguments):
    return AnalystReport(analyst=analyst, ticker=ticker, signal=signal, conviction=conviction,
                         arguments=list(arguments) or ["argument"])


def snapshot(close=100.0):
    return {t: {"technique": {"close": close}, "news_macro": {}} for t in TICKERS}


def desk_state(reports, portfolio=None, **extra):
    return {"as_of": AS_OF, "universe": TICKERS, "snapshot": snapshot(), "reports": reports,
            "portfolio": portfolio or initial_portfolio(), "debate": [], **extra}


POSITIFS = [rep(t, k, "achat", 0.9, f"[{t}/{k}] signal haussier net") for t in TICKERS
            for k in ("technique", "news_macro")]


def by_ticker(orders):
    return {o.ticker: o for o in orders}


def all_in_orders():
    qty = size_order(1.0, 100_000, 0, 100.0)
    return [OrderProposal(ticker=t, side="buy", conviction=1.0, quantity=qty, rationale="tout acheter")
            for t in TICKERS]


def within_limits(state, orders):
    return check_orders(orders, state["portfolio"], mark_prices(state["snapshot"])) == []


APRES_REFUS = desk_state(POSITIFS, orders=all_in_orders(), risk_retries=1, risk=RiskVerdict(
    approved=False,
    breaches=["cash négatif après exécution : -19 960,00 $", "exposition totale 119,8 % > 100 %"],
    comment="Trois achats à conviction 1 dépassent le cash disponible : baisse les convictions."))


SCENARIOS = [
    Scenario("tech_tendance_haussiere", "technique",
             "tendance haussière nette : signal achat",
             {"ticker": "ALFA", "features": tech(120.0, 6.5, 18.2, 114.0, 98.0, 64.0, 18.0, -1.2, rel=9.5)},
             lambda r: r.signal == "achat"),
    Scenario("tech_tendance_baissiere", "technique",
             "tendance baissière nette : pas d'achat",
             {"ticker": "BETA", "features": tech(61.0, -9.8, -24.5, 66.0, 82.0, 27.0, 41.0, -38.5, rel=-31.0)},
             lambda r: r.signal != "achat"),
    Scenario("tech_donnees_manquantes", "technique",
             "trois semaines d'historique : l'analyste signale le manque ou reste peu convaincu",
             {"ticker": "GAMA", "features": tech(50.0, None, None, None, None, None, None, -2.0, weeks=3)},
             lambda r: bool(INSUFFISANT.search(" ".join(r.arguments))) or r.conviction <= 0.4),
    Scenario("tech_signaux_contradictoires", "technique",
             "au-dessus de la moyenne 40 semaines mais en chute sur 4 semaines, RSI élevé : conviction modérée",
             {"ticker": "ALFA", "features": tech(115.0, -7.4, 9.1, 121.0, 100.0, 78.0, 33.0, -9.0)},
             lambda r: r.conviction <= 0.75),
    Scenario("news_tres_negatives", "news_macro",
             "enquête pour fraude, démission du directeur financier : pas d'achat",
             {"ticker": "ALFA", "features": news(
                 "ALFA visée par une enquête pour fraude comptable",
                 "Le directeur financier d'ALFA démissionne avec effet immédiat",
                 "ALFA abaisse fortement ses prévisions annuelles",
                 "Deux agences dégradent la note de crédit d'ALFA")},
             lambda r: r.signal != "achat"),
    Scenario("news_tres_positives", "news_macro",
             "résultats au-dessus des attentes, rachat d'actions : pas de vente",
             {"ticker": "BETA", "features": news(
                 "BETA publie un bénéfice trimestriel très supérieur aux attentes",
                 "BETA annonce un programme de rachat d'actions de 5 milliards",
                 "BETA relève ses prévisions annuelles")},
             lambda r: r.signal != "vente"),
    Scenario("news_absentes", "news_macro",
             "aucune news : l'analyste le signale ou reste peu convaincu",
             {"ticker": "GAMA", "features": news()},
             lambda r: bool(INSUFFISANT.search(" ".join(r.arguments))) or r.conviction <= 0.5),
    Scenario("macro_stress", "news_macro",
             "VIX à 42 : l'analyste en tient compte",
             {"ticker": "BETA", "features": news("BETA confirme ses objectifs annuels", vix=42.3, dgs10=4.95)},
             lambda r: "vix" in " ".join(r.arguments).lower()),
    Scenario("trader_rapports_negatifs", "trader",
             "rapports très négatifs sur ALFA : pas d'achat fort d'ALFA",
             {"state": desk_state([
                 rep("ALFA", "technique", "vente", 0.8, "[ALFA/technique] chute de 25 % en 12 semaines"),
                 rep("ALFA", "news_macro", "vente", 0.85, "[ALFA/news_macro] enquête pour fraude"),
                 rep("BETA", "technique", "achat", 0.6, "[BETA/technique] tendance haussière"),
                 rep("BETA", "news_macro", "neutre", 0.4, "[BETA/news_macro] peu de news"),
                 rep("GAMA", "technique", "neutre", 0.5, "[GAMA/technique] range"),
                 rep("GAMA", "news_macro", "neutre", 0.4, "[GAMA/news_macro] rien de notable")])},
             lambda orders: not (by_ticker(orders)["ALFA"].side == "buy"
                                 and by_ticker(orders)["ALFA"].conviction > 0.3)),
    Scenario("trader_rapports_neutres", "trader",
             "tous les rapports neutres : aucun achat à conviction > 0,5",
             {"state": desk_state([rep(t, k, "neutre", 0.4, f"[{t}/{k}] pas de signal clair")
                                   for t in TICKERS for k in ("technique", "news_macro")])},
             lambda orders: all(not (o.side == "buy" and o.conviction > 0.5) for o in orders)),
    Scenario("trader_position_a_alleger", "trader",
             "ALFA détenue et rapports très négatifs : pas de renforcement",
             {"state": desk_state([
                 rep("ALFA", "technique", "vente", 0.9, "[ALFA/technique] cassure de la moyenne 40 semaines"),
                 rep("ALFA", "news_macro", "vente", 0.9, "[ALFA/news_macro] avertissement sur résultats"),
                 *[rep(t, k, "neutre", 0.4, f"[{t}/{k}] neutre") for t in ("BETA", "GAMA")
                   for k in ("technique", "news_macro")]],
                 portfolio={"cash": 70_000.0, "positions": {"ALFA": 300}, "nav": 100_000.0})},
             lambda orders: by_ticker(orders)["ALFA"].side in ("sell", "hold")),
    Scenario("trader_apres_refus", "trader",
             "après un refus pour cash négatif, les nouveaux ordres respectent les limites",
             {"state": APRES_REFUS},
             lambda orders: within_limits(APRES_REFUS, orders)),
    Scenario("desk_tout_positif", "desk",
             "tout est haussier : après au plus 2 propositions, des ordres acceptés et dans les limites",
             {"state": desk_state(POSITIFS)},
             lambda out: out["risk"].approved and within_limits(out["state"], out["orders"]),
             juge=False),
    Scenario("risk_explique_et_propose", "risk",
             "le Risk Manager explique le refus et propose de baisser les convictions",
             {"state": desk_state(POSITIFS, orders=all_in_orders())},
             lambda v: not v.approved and bool(re.search(r"conviction|rédui|baiss|diminu", v.comment, re.IGNORECASE)),
             juge=False),
    Scenario("bull_cite_les_rapports", "bull",
             "le Bull cite les rapports avec leurs étiquettes et parle des trois titres",
             {"state": desk_state([
                 rep("ALFA", "technique", "achat", 0.7, "[ALFA/technique] au-dessus de la moyenne 40 semaines"),
                 rep("ALFA", "news_macro", "neutre", 0.4, "[ALFA/news_macro] pas de news"),
                 rep("BETA", "technique", "vente", 0.6, "[BETA/technique] RSI à 27"),
                 rep("BETA", "news_macro", "achat", 0.6, "[BETA/news_macro] rachat d'actions"),
                 rep("GAMA", "technique", "neutre", 0.5, "[GAMA/technique] range"),
                 rep("GAMA", "news_macro", "achat", 0.7, "[GAMA/news_macro] contrat signé")])},
             lambda text: len(TAG.findall(text)) >= 3 and all(t in text for t in TICKERS)),
]

