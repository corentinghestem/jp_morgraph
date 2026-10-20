from langgraph.types import interrupt

from jp_morgraph.desk import check_orders, mark_prices
from jp_morgraph.state import OrderProposal

HELP = ('true = tout valider ; false = tout rejeter ; '
        '{"quantities": {"AAPL": 50}} = corriger (0 = annuler) ; {"sides": {"JPM": "hold"}}')


def apply_decision(decision, orders: list[OrderProposal]) -> list[OrderProposal]:
    if isinstance(decision, bool):
        decision = {"approve": decision}
    if not isinstance(decision, dict):
        raise TypeError(f"réponse attendue : {HELP}")
    if not decision.get("approve", True):
        return []
    quantities = decision.get("quantities", {})
    sides = decision.get("sides", {})
    unknown = (set(quantities) | set(sides)) - {o.ticker for o in orders}
    if unknown:
        raise ValueError(f"titres inconnus : {sorted(unknown)}")

    approved = []
    for o in orders:
        changes = {}
        if o.ticker in sides:
            if sides[o.ticker] not in ("buy", "sell", "hold"):
                raise ValueError(f"{o.ticker} : sens invalide {sides[o.ticker]!r}")
            changes["side"] = sides[o.ticker]
        if o.ticker in quantities:
            q = quantities[o.ticker]
            if isinstance(q, bool) or not isinstance(q, int | float) or q < 0 or q != int(q):
                raise ValueError(f"{o.ticker} : quantité invalide {q!r} (entier >= 0 attendu)")
            changes["quantity"] = int(q)
        if changes:
            new = o.model_copy(update=changes)
            changes["rationale"] = (f"{o.rationale} [corrigé par le gérant : {o.side} x{o.quantity}"
                                    f" -> {new.side} x{new.quantity}]")
            o = o.model_copy(update=changes)
        approved.append(o)
    return approved


def pm_approval(state: dict) -> dict:
    orders = state["orders"]
    if state.get("auto_approve"):
        return {"approved_orders": orders}

    payload = {
        "as_of": state["as_of"],
        "ordres": [o.model_dump() for o in orders],
        "avis_risque": state["risk"].comment,
        "portefeuille": state["portfolio"],
        "comment_repondre": HELP,
    }
    while True:
        decision = interrupt(payload)
        try:
            approved = apply_decision(decision, orders)
        except (ValueError, TypeError) as e:
            payload = {**payload, "erreur": f"Réponse invalide : {e}"}
            continue
        breaches = check_orders(approved, state["portfolio"], mark_prices(state["snapshot"]))
        if not breaches:
            return {"approved_orders": approved}
        payload = {**payload,
                   "erreur": "Correction refusée par les règles de risque : " + " ; ".join(breaches)}
