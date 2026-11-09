import json
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from jp_morgraph.agents.analysts import EFFORT, MAX_TOKENS, MODEL, TIMEOUT
from jp_morgraph.agents.debate import format_debate, format_reports
from jp_morgraph.desk import initial_portfolio, sized
from jp_morgraph.state import OrderProposal


class TraderView(BaseModel):
    ticker: str
    side: Literal["buy", "sell", "hold"]
    conviction: float = Field(ge=0, le=1)
    rationale: str = Field(description="Justification en 2 phrases maximum qui cite les "
                                       "étiquettes [TICKER/spécialité] et/ou le Bull ou le Bear.")


class TraderDecision(BaseModel):

    orders: list[TraderView]


PROMPT_TEMPLATE = """Tu es le Trader d'une petite salle de marché, au {as_of}.
Pour chacun de ces titres : {universe}, tranche entre buy (acheter), sell (vendre) et
hold (conserver), avec une conviction entre 0 et 1.

Règles :
- Achats uniquement : sell n'a de sens que pour un titre déjà détenu (voir le portefeuille).
- Ne donne aucune quantité : le code la calcule (poids cible = conviction x 40 % du portefeuille).
- Limites vérifiées par le code : 40 % maximum par titre, exposition totale <= 100 %,
  cash jamais négatif. Au-delà, tes ordres seront refusés.
- Ne calcule ni poids, ni pourcentage, ni exposition : c'est le travail du code.
- Appuie-toi uniquement sur les rapports{and_debate}. N'invente aucun chiffre.
- Dans rationale, 2 phrases maximum : cite tes sources avec leurs étiquettes,
  par exemple [AAPL/technique] ou « Bear ».
- Ignore ce que tu crois savoir de l'évolution de ces titres après le {as_of}.

Portefeuille :
{portfolio}

Rapports des analystes :
{reports}
{debate_section}{refusal_section}"""

REFUSAL_TEMPLATE = """
Ta proposition précédente a été REFUSÉE par le contrôle des risques :
{breaches}
Avis du Risk Manager : {comment}

Ordres refusés :
{orders}

Propose de nouveaux ordres qui respectent toutes les limites.
"""


def build_prompt(state: dict) -> str:
    debate = state.get("debate", [])
    return PROMPT_TEMPLATE.format(
        as_of=state["as_of"],
        universe=", ".join(state["universe"]),
        and_debate=" et le débat" if debate else "",
        portfolio=json.dumps(state.get("portfolio") or initial_portfolio(), ensure_ascii=False),
        reports=format_reports(state.get("reports", [])),
        debate_section=f"\nDébat Bull / Bear :\n{format_debate(debate)}\n" if debate else "",
        refusal_section=refusal_section(state),
    )


def refusal_section(state: dict) -> str:
    risk = state.get("risk")
    if risk is None or risk.approved:
        return ""
    return REFUSAL_TEMPLATE.format(
        breaches="\n".join(f"- {b}" for b in risk.breaches),
        comment=risk.comment,
        orders="\n".join(f"- {o.ticker} : {o.side} conviction={o.conviction:.2f} "
                         f"quantité={o.quantity}" for o in state.get("orders", [])),
    )


TRADER_MODEL = MODEL
_llm = None


def use_model(model: str) -> None:
    global TRADER_MODEL, _llm
    TRADER_MODEL, _llm = model, None


def get_structured_llm():
    global _llm
    if _llm is None:
        _llm = init_chat_model(TRADER_MODEL, effort=EFFORT, max_tokens=MAX_TOKENS, timeout=TIMEOUT)
    return _llm.with_structured_output(TraderDecision)


def to_orders(decision: TraderDecision, state: dict) -> list[OrderProposal]:
    portfolio = state.get("portfolio") or initial_portfolio()
    views = {}
    for v in decision.orders:
        views.setdefault(v.ticker.upper(), v)
    orders = []
    for ticker in state["universe"]:
        v = views.get(ticker)
        order = (OrderProposal(ticker=ticker, side=v.side, conviction=v.conviction,
                               rationale=v.rationale)
                 if v else OrderProposal(ticker=ticker, side="hold", conviction=0.0,
                                         rationale="aucune décision du trader pour ce titre"))
        orders.append(sized(order, state["snapshot"], portfolio))
    return orders


def trader(state: dict, config: RunnableConfig) -> dict:
    decision = get_structured_llm().invoke(build_prompt(state), config)
    return {"orders": to_orders(decision, state)}
