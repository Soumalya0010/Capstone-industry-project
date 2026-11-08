from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from app.services.bayesian_service import (
    run_bayesian_conversion_ab,
    run_bayesian_continuous_ab
)

router = APIRouter(prefix="/api/bayesian", tags=["Bayesian A/B Testing (PyMC)"])

class BayesianConversionRequest(BaseModel):
    control_conversions: int = Field(..., ge=0)
    control_total: int = Field(..., gt=0)
    treatment_conversions: int = Field(..., ge=0)
    treatment_total: int = Field(..., gt=0)
    prior_alpha_a: float = Field(1.0, gt=0)
    prior_beta_a: float = Field(1.0, gt=0)
    prior_alpha_b: float = Field(1.0, gt=0)
    prior_beta_b: float = Field(1.0, gt=0)
    draws: int = Field(1500, ge=500, le=5000)
    tune: int = Field(500, ge=200, le=2000)
    random_seed: int = Field(42)

class BayesianContinuousRequest(BaseModel):
    sample_a: List[float]
    sample_b: List[float]
    draws: int = Field(1000, ge=500, le=3000)
    tune: int = Field(400, ge=200, le=1000)
    random_seed: int = Field(42)

@router.post("/conversion")
def bayesian_conversion_endpoint(req: BayesianConversionRequest):
    try:
        return run_bayesian_conversion_ab(
            control_conversions=req.control_conversions,
            control_total=req.control_total,
            treatment_conversions=req.treatment_conversions,
            treatment_total=req.treatment_total,
            prior_alpha_a=req.prior_alpha_a,
            prior_beta_a=req.prior_beta_a,
            prior_alpha_b=req.prior_alpha_b,
            prior_beta_b=req.prior_beta_b,
            draws=req.draws,
            tune=req.tune,
            random_seed=req.random_seed
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/continuous")
def bayesian_continuous_endpoint(req: BayesianContinuousRequest):
    try:
        return run_bayesian_continuous_ab(
            sample_a=req.sample_a,
            sample_b=req.sample_b,
            draws=req.draws,
            tune=req.tune,
            random_seed=req.random_seed
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
