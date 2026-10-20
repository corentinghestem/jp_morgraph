import json
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langgraph.types import Send
from typing_extensions import TypedDict

from jp_morgraph.state import AnalystReport

KINDS = ("technique", "news_macro")

MODEL = "anthropic:claude-haiku-5-5"
EFFORT = "low"
MAX_TOKENS = 4000


class AnalystTask(TypedDict):

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
    global _llm
    if _llm is None:
        _llm = init_chat_model(MODEL, effort=EFFORT, max_tokens=MAX_TOKENS)
    return _llm.with_structured_output(AnalystReport)


def route_to_analysts(state: dict) -> list[Send]:
    return [
        Send(
            "analyst",
            AnalystTask(
                ticker=ticker,
                kind=kind,
                as_of=state["as_of"],
                features=state["snapshot"][ticker].get(kind, state["snapshot"][ticker]),
            ),
        )
        for ticker in state["universe"]
        for kind in KINDS
    ]


def analyst(task: AnalystTask, config: RunnableConfig) -> dict:
    report = get_structured_llm().invoke(build_prompt(task), config)
    report = report.model_copy(update={"analyst": task["kind"], "ticker": task["ticker"]})
    return {"reports": [report]}