# src/jp_morgraph/agents/technical.py
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from jp_morgraph.tools import get_price_history, compute_indicators
from jp_morgraph.state import AnalystReport

tools = [get_price_history, compute_indicators]
llm = init_chat_model("anthropic:claude-haiku-5-5", max_tokens=800)
llm_with_tools = llm.bind_tools(tools)


class TechState(MessagesState):
    report: AnalystReport  # le rapport validé, séparé de la conversation


def agent(state: TechState) -> dict:
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


from langchain_core.messages import ToolMessage


def report(state: TechState) -> dict:
    structured = llm.with_structured_output(AnalystReport)
    demande = state["messages"][0].content
    donnees = "\n".join(
        f"- {m.name} : {m.content}"
        for m in state["messages"] if isinstance(m, ToolMessage)
    )
    prompt = (
        f"Demande : {demande}\n\n"
        f"Résultats des outils :\n{donnees}\n\n"
        "Résume ton analyse au format AnalystReport. Analyste : technique. "
        "N'utilise que les chiffres ci-dessus. "
        "signal vaut exactement « achat », « neutre » ou « vente » ; "
        "conviction est comprise entre 0 et 1 ; cinq arguments au maximum."
    )
    return {"report": structured.invoke(prompt)}


builder = StateGraph(TechState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))
builder.add_node("report", report)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition,
                              {"tools": "tools", END: "report"})
builder.add_edge("tools", "agent")
builder.add_edge("report", END)

graph = builder.compile()