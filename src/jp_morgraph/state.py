from operator import add
from typing import Annotated, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class AnalystReport(BaseModel):
    analyst: Literal["technique", "news_macro"]
    ticker: str
    signal: Literal["achat", "neutre", "vente"]
    conviction: float = Field(ge=0, le=1)
    arguments: list[str] = Field(max_length=5)

class OrderProposal(BaseModel):
    ticker: str
    side: Literal["buy", "sell", "hold"]
    conviction: float = Field(ge=0, le=1)
    quantity: int = 0
    rationale: str
    
class RiskVerdict(BaseModel):
    approved: bool
    breaches: list[str]
    comment: str

class DeskState(TypedDict, total=False):
    as_of: str
    universe: list[str]
    snapshot: dict
    portfolio: dict
    reports: Annotated[list[AnalystReport], add]
    debate: Annotated[list[str], add]
    debate_round: int
    orders: list[OrderProposal]
    risk: RiskVerdict
    risk_retries: int
    auto_approve: bool
    approved_orders: list[OrderProposal]
    fills: list[dict]