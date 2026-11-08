from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from app.services.calculator_service import calculate_proportion_sample_size, calculate_continuous_sample_size

router = APIRouter(prefix="/api/calculator", tags=["Power & Sample Size Calculator"])

class ProportionCalculatorRequest(BaseModel):
    baseline_rate: float = Field(..., gt=0.0, lt=1.0, description="Baseline conversion rate (e.g., 0.10 for 10%)")
    mde_relative: Optional[float] = Field(None, description="Relative lift desired (e.g., 0.15 for +15%)")
    mde_absolute: Optional[float] = Field(None, description="Absolute lift desired (e.g., 0.015 for +1.5%)")
    power: float = Field(0.80, ge=0.5, lt=1.0, description="Statistical power (default 0.80)")
    alpha: float = Field(0.05, gt=0.0, lt=0.5, description="Significance level alpha (default 0.05)")
    alternative: str = Field("two-sided", description="'two-sided' or 'larger'")
    ratio: float = Field(1.0, gt=0.1, le=10.0, description="Ratio of treatment to control sample size")
    daily_traffic: Optional[int] = Field(None, description="Estimated daily user traffic")

class ContinuousCalculatorRequest(BaseModel):
    baseline_mean: float = Field(..., description="Baseline mean (e.g., average revenue or duration)")
    baseline_std: float = Field(..., gt=0.0, description="Baseline standard deviation")
    mde_absolute: Optional[float] = Field(None, description="Absolute difference to detect")
    mde_relative: Optional[float] = Field(None, description="Relative difference as a fraction (e.g., 0.05 for +5%)")
    power: float = Field(0.80, ge=0.5, lt=1.0, description="Statistical power")
    alpha: float = Field(0.05, gt=0.0, lt=0.5, description="Significance level alpha")
    alternative: str = Field("two-sided", description="'two-sided' or 'larger'")
    ratio: float = Field(1.0, gt=0.1, le=10.0, description="Treatment to control ratio")
    daily_traffic: Optional[int] = Field(None, description="Estimated daily observations")

@router.post("/proportion")
def compute_proportion_power(req: ProportionCalculatorRequest):
    try:
        res = calculate_proportion_sample_size(
            baseline_rate=req.baseline_rate,
            mde_relative=req.mde_relative,
            mde_absolute=req.mde_absolute,
            power=req.power,
            alpha=req.alpha,
            alternative=req.alternative,
            ratio=req.ratio,
            daily_traffic=req.daily_traffic
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/continuous")
def compute_continuous_power(req: ContinuousCalculatorRequest):
    try:
        res = calculate_continuous_sample_size(
            baseline_mean=req.baseline_mean,
            baseline_std=req.baseline_std,
            mde_absolute=req.mde_absolute,
            mde_relative=req.mde_relative,
            power=req.power,
            alpha=req.alpha,
            alternative=req.alternative,
            ratio=req.ratio,
            daily_traffic=req.daily_traffic
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
