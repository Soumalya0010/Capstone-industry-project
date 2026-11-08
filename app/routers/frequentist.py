from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from app.services.frequentist_service import (
    simulate_ab_data,
    run_two_sample_ttest,
    run_chi_square_test,
    run_multiple_testing_adjustment
)

router = APIRouter(prefix="/api/ab", tags=["Frequentist A/B Testing & Simulation"])

class SimulateRequest(BaseModel):
    metric_type: str = Field("conversion", description="'conversion' or 'continuous'")
    n_a: int = Field(2500, ge=10, le=100000)
    n_b: int = Field(2500, ge=10, le=100000)
    baseline_val: float = Field(0.12, description="Baseline conversion rate or mean")
    relative_lift: float = Field(0.15, description="Relative lift (e.g., 0.15 for +15%)")
    std_dev: float = Field(25.0, description="Std dev for continuous metric")
    random_seed: int = Field(42)

class TTestRequest(BaseModel):
    sample_a: List[float]
    sample_b: List[float]
    alpha: float = Field(0.05, gt=0.0, lt=0.5)
    equal_var: bool = Field(False, description="Assume equal variance (Student's) vs unequal (Welch's)")

class ChiSquareRequest(BaseModel):
    control_conversions: int = Field(..., ge=0)
    control_total: int = Field(..., gt=0)
    treatment_conversions: int = Field(..., ge=0)
    treatment_total: int = Field(..., gt=0)
    alpha: float = Field(0.05, gt=0.0, lt=0.5)

class MetricTestItem(BaseModel):
    metric_name: str
    p_value: float = Field(..., ge=0.0, le=1.0)
    raw_diff: Optional[float] = None

class MultipleTestingRequest(BaseModel):
    test_items: List[MetricTestItem]
    alpha: float = Field(0.05, gt=0.0, lt=0.5)

@router.post("/simulate")
def simulate_data_endpoint(req: SimulateRequest):
    try:
        return simulate_ab_data(
            metric_type=req.metric_type,
            n_a=req.n_a,
            n_b=req.n_b,
            baseline_val=req.baseline_val,
            relative_lift=req.relative_lift,
            std_dev=req.std_dev,
            random_seed=req.random_seed
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/ttest")
def ttest_endpoint(req: TTestRequest):
    try:
        return run_two_sample_ttest(
            sample_a=req.sample_a,
            sample_b=req.sample_b,
            alpha=req.alpha,
            equal_var=req.equal_var
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/chisquare")
def chisquare_endpoint(req: ChiSquareRequest):
    try:
        return run_chi_square_test(
            control_conversions=req.control_conversions,
            control_total=req.control_total,
            treatment_conversions=req.treatment_conversions,
            treatment_total=req.treatment_total,
            alpha=req.alpha
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/multiple-testing")
def multiple_testing_endpoint(req: MultipleTestingRequest):
    try:
        items = [i.model_dump() for i in req.test_items]
        return run_multiple_testing_adjustment(test_items=items, alpha=req.alpha)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
