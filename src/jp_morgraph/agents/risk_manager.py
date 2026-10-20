import json

from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig

from jp_morgraph.agents.analysts import EFFORT, MAX_TOKENS, MODEL
from jp_morgraph.agents.debate import text_of
from jp_morgraph.desk import check_orders, mark_prices
from jp_morgraph.state import OrderProposal, RiskVerdict


def format_orders(orders: list[OrderProposal]) -> str:
    return "\n".join(f"- {o.ticker} : {o.side} conviction={o.conviction:.2f} quantité={o.quantity}"
                     for o in orders)


PROMPT_TEMPLATE = """Tu es le Risk Manager d'une petite salle de marché, au {as_of}.
Les règles de risque, appliquées par le code, ont REFUSÉ les ordres du Trader.
Ce n'est pas toi qui décides : ton rôle est d'expliquer clairement au Trader pourquoi,
et de lui proposer un ajustement concret.

Limites : 40 % maximum par titre acheté, exposition totale <= 100 %, cash jamais négatif,
pas de vente à découvert. La quantité d'un achat vaut conviction x 40 % du portefeuille :
pour réduire un achat, le Trader doit baisser sa conviction.

Violations détectées par le code :
{breaches}

Ordres proposés :
{orders}

Portefeuille avant les ordres :
{portfolio}

Réponds en 3 phrases maximum. N'invente aucun chiffre absent ci-dessus.
"""


def build_prompt(state: dict, breaches: list[str]) -> str:
    return PROMPT_TEMPLATE.format(
        as_of=state["as_of"],
        breaches="\n".join(f"- {b}" for b in breaches),
        orders=format_orders(state["orders"]),
        portfolio=json.dumps(state["portfolio"], ensure_ascii=False),
    )


_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = init_chat_model(MODEL, effort=EFFORT, max_tokens=MAX_TOKENS)
    return _llm


def risk_check(state: dict, config: RunnableConfig) -> dict:
    breaches = check_orders(state["orders"], state["portfolio"], mark_prices(state["snapshot"]))
    if not breaches:
        return {"risk": RiskVerdict(approved=True, breaches=[], comment="Aucune limite dépassée.")}
    comment = text_of(get_llm().invoke(build_prompt(state, breaches), config), "Risk Manager")
    return {"risk": RiskVerdict(approved=False, breaches=breaches, comment=comment),
            "risk_retries": state.get("risk_retries", 0) + 1}
