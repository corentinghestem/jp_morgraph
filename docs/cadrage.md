## Objectif

Apprendre et pratiquer l'IA agentique en construisant une petite salle de marché simulée, avec six agents orchestrés par LangGraph (analyste technique, analyste news & macro, analyste Bull, analyste Bear, trader, risk manager)


## Périmètre

- **Univers** : 3 actions américaines de secteurs différents, plus le SPY comme repère.
  - Technologie : AAPL
  - Banque : JPM 
  - Pétrole : XOM
  - Crypto : SOL
- **Rythme** : un cycle par semaine. Décision le vendredi après la clôture, exécution le lundi à l'ouverture.
- **Positions** : achats uniquement.
- **Capital** : 100 000 $ fictifs.
- **Coûts simulés** : slippage de 5 points de base, frais de 0,05 % du montant.
- **Stack** : Python, uv, LangGraph, LangChain, LangSmith, pandas, Streamlit.

## Hors périmètre

- Argent réel et conseil en investissement.
- Trading haute fréquence.
- Optimisation du rendement (retoucher les prompts pour améliorer la courbe).
- Vente à découvert et levier.

## Limites de risque

Appliquées par du code déterministe, jamais par le LLM :

| Règle | Valeur |
|---|---|
| Poids maximum par ligne | 40 % de la valeur du portefeuille |
| Exposition totale | ≤ 100 % |
| Levier | aucun |
| Cash | jamais négatif |

## Principes directeurs

1. Le LLM raisonne, le code calcule : chaque chiffre (prix, indicateurs, quantités, P&L) vient de Python.
2. Un graphe explicite plutôt qu'un superviseur libre.
3. Zéro argent réel.
4. Petit, puis grand : un agent et un titre avant six agents et trois titres.

## Indicateurs de réussite

| Indicateur | Cible |
|---|---|
| Sorties structurées valides | ≥ 95 % |
| Décisions qui ne citent que des données présentes | 100 % |
| Ordres exécutés hors limites | 0 |
| Coût par cycle (configuration économe) | ≈ 0,02 $ |
| Même snapshot, même décision | 2 fois sur 3 au moins |
| Décisions expliquées depuis la trace | toutes |

