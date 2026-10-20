# src/jp_morgraph/state.py
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
    quantity: int = 0 # remplie par le code, jamais par le LLM
    rationale: str
    
class RiskVerdict(BaseModel):
    approved: bool
    breaches: list[str] # règles violées, détectées par le code
    comment: str # explication du Risk Manager

class DeskState(TypedDict, total=False):
    as_of: str # vendredi de décision (AAAA-MM-JJ)
    universe: list[str] # 3 titres
    snapshot: dict # résumé produit par le Data Desk
    portfolio: dict # cash, positions, valeur liquidative
    reports: Annotated[list[AnalystReport], add] # écrits en parallèle
    debate: Annotated[list[str], add]
    debate_round: int
    orders: list[OrderProposal]
    risk: RiskVerdict
    risk_retries: int
    auto_approve: bool # True en simulation, False en mode interactif
    approved_orders: list[OrderProposal]
    fills: list[dict]