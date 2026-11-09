import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from jp_morgraph.data import MarketStore
from jp_morgraph.phase4_graph import DATA_DIR
from jp_morgraph.simulate import close_on, list_runs, summarize

data_dir = Path(os.environ.get("JP_DATA_DIR") or DATA_DIR)
st.set_page_config(page_title="jp_morgraph", layout="wide")
st.title("jp_morgraph")

runs = list_runs(data_dir)
if not runs:
    st.info("Aucun run pour l'instant : uv run python -m jp_morgraph.simulate replay --weeks 52")
    st.stop()

name = st.sidebar.selectbox("Run", runs)
s = summarize(name, data_dir)
st.caption(f"Du {s['debut']} au {s['fin']}, {s['cycles']} cycles. Simulation, aucun argent réel.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Valeur liquidative", f"{s['valeur_finale']:,.0f} $".replace(",", " "),
          f"{s['rendement']:+.1%}")
c2.metric("SPY sur la période", f"{s['rendement_spy']:+.1%}")
c3.metric("Drawdown max", f"{s['drawdown_max']:.1%}")
c4.metric("Coût LLM", f"{s['cout_total']:.2f} $", f"{s['cout_par_cycle']:.3f} $ / cycle",
          delta_color="off")

fig = go.Figure()
fig.add_scatter(x=s["nav"]["date"], y=s["nav"]["nav"], name="Portefeuille")
fig.add_scatter(x=s["nav"]["date"], y=s["nav"]["spy"], name="SPY (repère)", line={"dash": "dot"})
fig.update_layout(height=380, margin={"t": 20, "b": 20}, yaxis_title="$", legend={"orientation": "h"})
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
last = s["records"][-1]
prices = MarketStore.load(data_dir).prices
day = prices.index.max().strftime("%Y-%m-%d")
with left:
    st.subheader("Positions")
    pf = last["portfolio"]
    rows = [{"titre": t, "actions": q, "cours": close_on(prices, day, t)} for t, q in pf["positions"].items()]
    positions = pd.DataFrame(rows, columns=["titre", "actions", "cours"])
    positions["valeur"] = positions["actions"] * positions["cours"]
    nav = pf["cash"] + positions["valeur"].sum()
    positions["poids"] = (positions["valeur"] / nav).map("{:.1%}".format)
    st.dataframe(positions, hide_index=True, use_container_width=True)
    st.write(f"Cash : {pf['cash']:,.2f} $".replace(",", " "))

    st.subheader("Contrôles")
    hors_limites = len(s["hors_limites"])
    st.write(f"Ordres exécutés : {s['ordres_executes']} (dont {s['ordres_reduits']} réduits)")
    st.write(f"Retours au Trader après refus : {s['retours_au_trader']}")
    st.write(f"Ordres exécutés hors limites : {hors_limites}")
    if s["citent_sources"] is not None:
        st.write(f"Ordres qui citent leurs sources : {s['citent_sources']:.0%}")
    st.dataframe(pd.DataFrame(s["statuts"].most_common(), columns=["statut", "cycles"]),
                 hide_index=True, use_container_width=True)

with right:
    st.subheader(f"Dernier cycle : {last['as_of']}")
    st.write(last["statut"])
    orders = pd.DataFrame(last["orders"])
    if len(orders):
        st.dataframe(orders[["ticker", "side", "conviction", "quantity", "rationale"]],
                     hide_index=True, use_container_width=True)
    if last["fills"]:
        st.dataframe(pd.DataFrame(last["fills"]), hide_index=True, use_container_width=True)
    if last["risk"]:
        with st.expander("Risk Manager"):
            st.write(last["risk"]["comment"])
            for b in last["risk"]["breaches"]:
                st.write(f"- {b}")
    with st.expander("Débat Bull / Bear", expanded=True):
        for msg in last["debate"]:
            st.markdown(msg)
    with st.expander("Rapports des analystes"):
        st.dataframe(pd.DataFrame(last["reports"]), hide_index=True, use_container_width=True)

st.subheader("Coût par cycle")
if len(s["costs"]):
    costs = s["costs"].groupby("as_of", as_index=False)["cost"].sum()
    st.bar_chart(costs, x="as_of", y="cost")

results = sorted((data_dir / "evals").glob("*.json")) if (data_dir / "evals").exists() else []
if results:
    st.subheader("Évaluations")
    table = pd.DataFrame([{"configuration": r["label"], **r["totaux"]}
                          for r in (json.loads(p.read_text(encoding="utf-8")) for p in results)])
    st.dataframe(table, hide_index=True, use_container_width=True)
