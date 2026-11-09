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

Copier `.env.example` en `.env` et remplir les clés (Anthropic, Finnhub, FRED, LangSmith).

## Utilisation

```bash
uv run python -m jp_morgraph.data refresh   # télécharge prix, news et macro dans data/
uv run python scripts/cycle.py 2026-10-16   # un cycle, avec validation dans le terminal
uv run python scripts/cycle.py 2026-10-16 --auto   # un cycle sans pause
uv run langgraph dev  #langmsith                         
uv run pytest evals   # tests
```

Chaque cycle est enregistré dans `data/journal.jsonl`.

### Simulation

```bash
uv run python -m jp_morgraph.simulate replay --weeks 52            # rejoue les 52 derniers vendredis
uv run python -m jp_morgraph.simulate report --run replay-01       # rapport : data/runs/replay-01/report.md
uv run python -m jp_morgraph.simulate live                         # cycle de la semaine, avec validation
uv run python -m jp_morgraph.simulate travel --as-of 2026-06-05 --consigne "Sois plus prudent"
uv run streamlit run app/dashboard.py                              # dashboard
```

Chaque cycle a son `thread_id` dans un checkpointer SQLite (`data/runs/<run>/checkpoints.db`) :
un replay interrompu reprend là où il s'est arrêté en relançant la même commande.
Le mode `live` met le cycle en pause après ta validation et l'exécute au prochain lancement,
une fois l'ouverture du lundi disponible.

### Évaluation

```bash
uv run python evals/run_evals.py                                          # 15 scénarios, tout en Haiku
uv run python evals/run_evals.py --trader-model anthropic:claude-sonnet-5-5
uv run python evals/run_evals.py --repeat 3                               # stabilité des décisions
uv run python evals/run_evals.py --compare data/evals/haiku.json data/evals/trader-claude-sonnet-5-5.json
```

Évaluateurs déterministes (comportement attendu, sorties valides, chiffres absents des données)
et LLM juge (fidélité aux données, qualité des arguments, de 1 à 5).

## Résultats d'évaluation

15 scénarios figés, sur des titres fictifs pour que le modèle ne puisse pas s'appuyer sur sa mémoire.
Juge : Claude Haiku 5.5. Une passe par configuration (novembre 2026).

| | Tout en Haiku 5.5 | Trader en Sonnet 5.5 |
|---|---|---|
| Scénarios réussis | 15 / 15 | 15 / 15 |
| Sorties structurées valides | 100 % | 100 % |
| Sorties sans chiffre absent des données | 87 % | 93 % |
| Fidélité aux données (juge, sur 5) | 4,9 | 4,9 |
| Qualité des arguments (juge, sur 5) | 4,2 | 4,2 |
| Coût des agents pour les 15 scénarios | 0,006 $ | 0,033 $ |

Le Trader en Sonnet coûte environ 5 fois plus cher pour un gain faible. Seule différence nette :
quand tous les signaux sont haussiers, il respecte les limites dès sa première proposition,
alors que le Trader en Haiku se fait d'abord refuser par le contrôle des risques.
Une première passe a montré que le Trader recalculait lui-même des poids de portefeuille ;
une consigne explicite a fait passer les sorties sans chiffre inventé de 60 % à 87 %.

## Structure

```
src/jp_morgraph/
  data.py          téléchargement, cache Parquet, snapshot(as_of) sans fuite du futur
  agents/          analystes, débat Bull / Bear, Trader, Risk Manager
  desk.py          taille des ordres, règles de risque, exécution, journal
  manager.py       validation humaine (interrupt)
  graph.py         le graphe complet
  simulate.py      replay, mode live, rapport, time travel
app/dashboard.py   dashboard Streamlit
scripts/           cycle interactif, mesures, ablation du débat
evals/             tests et scénarios d'évaluation
```

## Stack

Python 3.14, uv, LangGraph, LangChain, Claude Haiku 5.5, LangSmith, pandas.

