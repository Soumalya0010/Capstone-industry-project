from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from app.services.sequential_service import (
    analyze_sequential_experiment,
    simulate_peeking_problem
)

router = APIRouter(prefix="/api/sequential", tags=["Sequential Testing & Safe Peeking"])

class SequentialLookItem(BaseModel):
    look: int
    n_a: int
    n_b: int
    conv_a: Optional[int] = None
    conv_b: Optional[int] = None
    mean_a: Optional[float] = None
    mean_b: Optional[float] = None
    std_a: Optional[float] = 10.0
    std_b: Optional[float] = 10.0

class SequentialAnalysisRequest(BaseModel):
    looks_data: List[SequentialLookItem]
    alpha: float = Field(0.05, gt=0.0, lt=0.5)
    method: str = Field("obrien_fleming", description="'obrien_fleming' or 'pocock'")

class PeekingSimulationRequest(BaseModel):
    n_sims: int = Field(500, ge=50, le=2000)
    n_per_look: int = Field(200, ge=20, le=5000)
    n_looks: int = Field(10, ge=2, le=20)
    alpha: float = Field(0.05, gt=0.0, lt=0.5)

@router.post("/analyze")
def sequential_analyze_endpoint(req: SequentialAnalysisRequest):
    try:
        data = [item.model_dump() for item in req.looks_data]
        return analyze_sequential_experiment(
            looks_data=data,
            alpha=req.alpha,
            method=req.method
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/simulate-peeking")
def sequential_simulate_peeking_endpoint(req: PeekingSimulationRequest):
    try:
        return simulate_peeking_problem(
            n_sims=req.n_sims,
            n_per_look=req.n_per_look,
            n_looks=req.n_looks,
            alpha=req.alpha
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
