import json
from pathlib import Path

from jp_morgraph.config import CAPITAL, FEE_RATE, MAX_EXPOSURE, MAX_WEIGHT, SLIPPAGE_BPS
from jp_morgraph.state import OrderProposal

SLIPPAGE = SLIPPAGE_BPS / 10_000
COST_RATE = SLIPPAGE + FEE_RATE * (1 + SLIPPAGE)
BUFFER = 1 - 2 * COST_RATE
EPS = 1e-9


def initial_portfolio(cash: float = CAPITAL) -> dict:
    return {"cash": cash, "positions": {}, "nav": cash}


def price_of(snap_entry: dict) -> float | None:
    return snap_entry.get("technique", snap_entry).get("close")


def mark_prices(snapshot: dict) -> dict[str, float]:
    return {t: p for t, entry in snapshot.items() if (p := price_of(entry))}


def size_order(conviction: float, nav: float, held_value: float, price: float,
               max_weight: float = MAX_WEIGHT) -> int:
    if price <= 0:
        return 0
    return int((conviction * max_weight * nav * BUFFER - held_value) // price)


def sized(order: OrderProposal, snapshot: dict, portfolio: dict) -> OrderProposal:
    price = price_of(snapshot.get(order.ticker, {}))
    held = portfolio.get("positions", {}).get(order.ticker, 0)
    if order.side == "buy" and price:
        qty = max(0, size_order(order.conviction, portfolio["nav"], held * price, price))
    elif order.side == "sell":
        qty = int(order.conviction * held)
    else:
        qty = 0
    return order.model_copy(update={"quantity": qty})


def active(orders: list[OrderProposal]) -> list[OrderProposal]:
    todo = [o for o in orders if o.side in ("buy", "sell") and o.quantity > 0]
    return sorted(todo, key=lambda o: o.side != "sell")


def trade(side: str, qty: int, price: float) -> tuple[float, float, float]:
    exec_price = price * (1 + SLIPPAGE) if side == "buy" else price * (1 - SLIPPAGE)
    amount = qty * exec_price
    fees = amount * FEE_RATE
    cash_delta = -(amount + fees) if side == "buy" else amount - fees
    return exec_price, fees, cash_delta


def valued(cash: float, positions: dict[str, int], prices: dict[str, float]) -> dict:
    nav = cash + sum(q * prices[t] for t, q in positions.items())
    return {"cash": round(cash, 2), "positions": positions, "nav": round(nav, 2)}


def project(orders: list[OrderProposal], portfolio: dict, prices: dict[str, float]) -> dict:
    cash = portfolio["cash"]
    positions = dict(portfolio.get("positions", {}))
    for o in active(orders):
        cash += trade(o.side, o.quantity, prices[o.ticker])[2]
        positions[o.ticker] = positions.get(o.ticker, 0) + (
            o.quantity if o.side == "buy" else -o.quantity)
    return valued(cash, {t: q for t, q in positions.items() if q}, prices)


def check_orders(orders: list[OrderProposal], portfolio: dict,
                 prices: dict[str, float]) -> list[str]:
    held = portfolio.get("positions", {})
    breaches = []
    for o in orders:
        if o.quantity < 0:
            breaches.append(f"{o.ticker} : quantité négative ({o.quantity})")
        elif o.side in ("buy", "sell") and o.quantity > 0 and not prices.get(o.ticker):
            breaches.append(f"{o.ticker} : aucun prix connu, ordre impossible à évaluer")
        elif o.side == "sell" and o.quantity > held.get(o.ticker, 0):
            breaches.append(f"{o.ticker} : vente de {o.quantity} actions pour "
                            f"{held.get(o.ticker, 0)} détenues (vente à découvert interdite)")
    breaches += [f"{t} : position détenue sans prix connu" for t in held if not prices.get(t)]
    if breaches:
        return breaches

    after = project(orders, portfolio, prices)
    nav, cash = after["nav"], after["cash"]
    if cash < 0:
        breaches.append(f"cash négatif après exécution : {cash:,.2f} $")
    if nav <= 0:
        breaches.append(f"valeur du portefeuille négative après exécution : {nav:,.2f} $")
        return breaches
    exposure = (nav - cash) / nav
    if exposure > MAX_EXPOSURE + EPS:
        breaches.append(f"exposition totale {exposure:.1%} > {MAX_EXPOSURE:.0%}")
    for t in {o.ticker for o in active(orders) if o.side == "buy"}:
        weight = after["positions"].get(t, 0) * prices[t] / nav
        if weight > MAX_WEIGHT + EPS:
            breaches.append(f"{t} : poids {weight:.1%} > {MAX_WEIGHT:.0%} du portefeuille")
    return breaches


def execute_orders(orders: list[OrderProposal], portfolio: dict, prices: dict[str, float],
                   date: str | None, source: str = "ouverture") -> tuple[list[dict], dict]:
    if date is None:
        return [{"date": None, "ticker": o.ticker, "side": o.side, "requested": o.quantity,
                 "quantity": 0, "note": f"non exécuté : {source}"} for o in active(orders)], portfolio
    cash = portfolio["cash"]
    positions = dict(portfolio.get("positions", {}))
    known = {t: p for t, p in prices.items() if p}
    marks = {t: known.get(t) or 0.0 for t in positions}
    nav0 = valued(cash, positions, {**marks, **known})["nav"]
    fills = []
    for o in active(orders):
        fill = {"date": date, "ticker": o.ticker, "side": o.side, "requested": o.quantity,
                "quantity": 0, "ref_price": known.get(o.ticker), "source_prix": source}
        price = known.get(o.ticker)
        if date is None or not price:
            fills.append({**fill, "note": "non exécuté : aucun prix d'exécution connu"})
            continue
        held = positions.get(o.ticker, 0)
        if o.side == "sell":
            qty = min(o.quantity, held)
        else:
            unit_cost = price * (1 + SLIPPAGE) * (1 + FEE_RATE)
            room = MAX_WEIGHT * nav0 * BUFFER - held * price
            qty = max(0, min(o.quantity, int(cash // unit_cost), int(room // price)))
        if qty == 0:
            fills.append({**fill, "note": "non exécuté : hors limites au prix d'exécution"})
            continue
        exec_price, fees, cash_delta = trade(o.side, qty, price)
        cash += cash_delta
        positions[o.ticker] = held + (qty if o.side == "buy" else -qty)
        fills.append({**fill, "quantity": qty, "exec_price": round(exec_price, 4),
                      "amount": round(qty * exec_price, 2), "fees": round(fees, 2),
                      "note": "" if qty == o.quantity else "réduit pour rester dans les limites"})
    positions = {t: q for t, q in positions.items() if q}
    return fills, valued(cash, positions, {**marks, **known})


def append_journal(path: str | Path, record: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
