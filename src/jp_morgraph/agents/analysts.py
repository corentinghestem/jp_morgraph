"""Phase 4 : deux analystes (technique, news & macro) lancés en parallèle avec l'API Send.

Principe : un seul nœud `analyst`, exécuté 6 fois (3 titres x 2 spécialités).
Le LLM rend un jugement ; le code impose l'identité du rapport (ticker, spécialité).
"""
import json
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langgraph.types import Send
from typing_extensions import TypedDict

from jp_morgraph.state import AnalystReport

KINDS = ("technique", "news_macro")

MODEL = "anthropic:claude-haiku-5-5"


class AnalystTask(TypedDict):
    """Entrée privée d'un appel du nœud `analyst` (ce n'est PAS l'état global)."""

    ticker: str
    kind: Literal["technique", "news_macro"]
    as_of: str
    features: dict


FOCUS = {
    "technique": (
        "Lis la tendance, le momentum et la volatilité à partir des indicateurs "
        "(rendements, moyennes mobiles, RSI, volatilité, drawdown)."
    ),
    "news_macro": (
        "Résume l'actualité du titre et le contexte macro (taux, inflation, chômage, VIX) "
        "et dis ce que cela change pour le titre."
    ),
}

PROMPT_TEMPLATE = """Tu es l'analyste {kind} d'une petite salle de marché.
Mission : évaluer {ticker} au {as_of}, à partir des seules données ci-dessous.
{focus}

Règles :
- N'utilise que les chiffres présents dans les données. N'en invente aucun.
- Si une information manque, écris « information insuffisante » au lieu de deviner.
- Ignore ce que tu crois savoir de l'évolution de ce titre après le {as_of}.
- Cinq arguments au maximum, du plus important au moins important.
- Réponds uniquement au format AnalystReport.

Données au {as_of} :
{features}
"""


def build_prompt(task: AnalystTask) -> str:
    return PROMPT_TEMPLATE.format(
        kind=task["kind"],
        ticker=task["ticker"],
        as_of=task["as_of"],
        focus=FOCUS[task["kind"]],
        features=json.dumps(task["features"], ensure_ascii=False, indent=2),
    )


_llm = None


def get_structured_llm():
    """Modèle créé à la demande : les tests n'ont besoin d'aucune clé API."""
    global _llm
    if _llm is None:
        _llm = init_chat_model(MODEL, max_tokens=800)
    return _llm.with_structured_output(AnalystReport)


def route_to_analysts(state: dict) -> list[Send]:  # `dict` : LangGraph ne déduit pas de schéma d'entrée
    """Fan-out : un Send par couple (titre, spécialité) = 3 x 2 = 6 appels parallèles."""
    return [
        Send(
            "analyst",
            AnalystTask(
                ticker=ticker,
                kind=kind,
                as_of=state["as_of"],
                # snapshot[ticker] peut être {"technique": {...}, "news_macro": {...}}
                # ou à plat : dans ce cas chaque analyste reçoit tout.
                features=state["snapshot"][ticker].get(kind, state["snapshot"][ticker]),
            ),
        )
        for ticker in state["universe"]
        for kind in KINDS
    ]


def analyst(task: AnalystTask, config: RunnableConfig) -> dict:
    """Un appel = un rapport. On passe `config` pour que les callbacks
    (LangSmith, comptage de tokens) suivent bien chaque appel parallèle."""
    report = get_structured_llm().invoke(build_prompt(task), config)
    # Le LLM juge ; le code décide de l'identité du rapport.
    report = report.model_copy(update={"analyst": task["kind"], "ticker": task["ticker"]})
    # Liste d'un élément : le reducer `add` de DeskState.reports les concatène.
    return {"reports": [report]}