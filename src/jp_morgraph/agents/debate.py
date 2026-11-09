from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig

from jp_morgraph.agents.analysts import EFFORT, MAX_TOKENS, MODEL, TIMEOUT
from jp_morgraph.state import AnalystReport


def format_reports(reports: list[AnalystReport]) -> str:
    lines = []
    for r in sorted(reports, key=lambda r: (r.ticker, r.analyst)):
        args = " ; ".join(r.arguments) or "aucun argument"
        lines.append(f"- [{r.ticker}/{r.analyst}] signal={r.signal} "
                     f"conviction={r.conviction:.2f} : {args}")
    return "\n".join(lines)


def format_debate(debate: list[str]) -> str:
    return "\n\n".join(debate) if debate else "(le débat n'a pas encore commencé)"


ROLE = {
    "bull": ("Bull", "Défends la thèse HAUSSIÈRE pour chaque titre.",
             "Si le Bear a déjà parlé, réponds à ses objections une par une."),
    "bear": ("Bear", "Défends la thèse BAISSIÈRE pour chaque titre.",
             "Réponds point par point au dernier message du Bull."),
}

PROMPT_TEMPLATE = """Tu es le chercheur {name} d'une petite salle de marché, au {as_of}.
{mission}
{reply}

Règles :
- Appuie-toi uniquement sur les rapports des analystes ci-dessous. N'invente aucun chiffre.
- Cite tes sources avec leurs étiquettes, par exemple [AAPL/technique].
- Ignore ce que tu crois savoir de l'évolution de ces titres après le {as_of}.
- Trois arguments au maximum par titre, en texte court.

Rapports des analystes :
{reports}

Débat jusqu'ici :
{debate}
"""


def build_prompt(role: str, state: dict) -> str:
    name, mission, reply = ROLE[role]
    return PROMPT_TEMPLATE.format(
        name=name, mission=mission, reply=reply, as_of=state["as_of"],
        reports=format_reports(state.get("reports", [])),
        debate=format_debate(state.get("debate", [])),
    )


_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = init_chat_model(MODEL, effort=EFFORT, max_tokens=MAX_TOKENS, timeout=TIMEOUT,
                               thinking={"type": "disabled"})
    return _llm


def text_of(msg, who: str) -> str:
    stop = msg.response_metadata.get("stop_reason")
    if stop == "max_tokens" or not msg.text.strip():
        usage = msg.usage_metadata or {}
        raise RuntimeError(f"{who} : réponse vide ({stop}, {usage.get('output_tokens')} tokens)")
    return msg.text


def _speak(role: str, state: dict, config: RunnableConfig) -> str:
    text = text_of(get_llm().invoke(build_prompt(role, state), config), ROLE[role][0])
    tour = state.get("debate_round", 0) + 1
    return f"{ROLE[role][0]} (tour {tour}) : {text}"


def bull(state: dict, config: RunnableConfig) -> dict:
    return {"debate": [_speak("bull", state, config)]}


def bear(state: dict, config: RunnableConfig) -> dict:
    return {"debate": [_speak("bear", state, config)],
            "debate_round": state.get("debate_round", 0) + 1}
