# jp_morgraph

Une petite salle de marché simulée, pilotée par des agents LLM orchestrés avec LangGraph.

## Fonctionnement

Un cycle par semaine : décision le vendredi après la clôture, exécution le lundi à l'ouverture.

```
Data Desk → 2 analystes × 3 titres (en parallèle) → débat Bull / Bear → Trader
         → Risk Manager (refus : retour au Trader) → validation humaine → exécution → journal
```

- Les LLM donnent un avis et une conviction. Les prix, indicateurs, quantités et le P&L sont calculés par le code.
- Les limites de risque sont appliquées par du code, jamais par le LLM :
  40 % maximum par ligne, exposition ≤ 100 %, pas de levier, cash jamais négatif.
- Capital fictif de 100 000 $, achats uniquement, 5 points de base de slippage et 0,05 % de frais.

## Installation

```bash
uv sync
```

Créer un fichier `.env` :

```
ANTHROPIC_API_KEY=
FINNHUB_API_KEY=
FRED_API_KEY=
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=jp_morgraph
```

## Utilisation

```bash
uv run python -m jp_morgraph.data refresh          # télécharge prix, news et macro dans data/
uv run python scripts/cycle.py 2026-10-16          # un cycle, avec validation dans le terminal
uv run python scripts/cycle.py 2026-10-16 --auto   # un cycle sans pause
uv run langgraph dev                               # graphe dans LangGraph Studio
uv run pytest evals                                # tests, sans réseau ni appel payant
```

Chaque cycle est enregistré dans `data/journal.jsonl`.

## Structure

```
src/jp_morgraph/
  data.py          téléchargement, cache Parquet, snapshot(as_of) sans fuite du futur
  agents/          analystes, débat Bull / Bear, Trader, Risk Manager
  desk.py          taille des ordres, règles de risque, exécution, journal
  manager.py       validation humaine (interrupt)
  graph.py         le graphe complet
scripts/           cycle interactif, mesures, ablation du débat
evals/             tests
```

## Stack

Python 3.14, uv, LangGraph, LangChain, Claude Haiku 5.5, LangSmith, pandas.

## Limites

Rejouer le passé reste un banc d'essai technique : le modèle peut déjà connaître la suite des événements.
Seul un suivi semaine après semaine, en conditions réelles, mesure honnêtement la qualité des décisions.
