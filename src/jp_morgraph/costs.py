PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}


def price_of(model: str) -> tuple[float, float]:
    name = model.split(":")[-1]
    for prefix, price in PRICES.items():
        if name.startswith(prefix):
            return price
    return PRICES["claude-haiku-5-5"]


def tokens_of(usage_by_model: dict) -> tuple[int, int]:
    return (sum(u.get("input_tokens", 0) for u in usage_by_model.values()),
            sum(u.get("output_tokens", 0) for u in usage_by_model.values()))


def cost_of(usage_by_model: dict) -> float:
    total = 0.0
    for model, u in usage_by_model.items():
        pin, pout = price_of(model)
        total += u.get("input_tokens", 0) / 1e6 * pin + u.get("output_tokens", 0) / 1e6 * pout
    return total
