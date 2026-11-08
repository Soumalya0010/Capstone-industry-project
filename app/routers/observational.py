from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import io
import os
from app.core.config import DATASETS_DIR
from app.services.causal_service import (
    run_propensity_score_matching,
    run_difference_in_differences,
    run_synthetic_control,
    run_econml_dml
)

router = APIRouter(prefix="/api/observational", tags=["Observational Causal Inference (DoWhy & EconML)"])

class PSMRequest(BaseModel):
    dataset_name: Optional[str] = "lalonde_nsw.csv"
    custom_records: Optional[List[Dict[str, Any]]] = None
    treatment_col: str = "treat"
    outcome_col: str = "re78"
    confounder_cols: List[str] = ["age", "educ", "black", "hisp", "married"]

class DiDRequest(BaseModel):
    dataset_name: Optional[str] = "card_krueger_did.csv"
    custom_records: Optional[List[Dict[str, Any]]] = None
    outcome_col: str = "fte"
    treatment_col: str = "treatment"
    post_col: str = "post"

class SyntheticControlRequest(BaseModel):
    dataset_name: Optional[str] = "california_prop99.csv"
    custom_records: Optional[List[Dict[str, Any]]] = None
    unit_col: str = "state"
    time_col: str = "year"
    outcome_col: str = "cigsale"
    treated_unit: str = "California"
    treatment_year: int = 1989

class EconMLRequest(BaseModel):
    dataset_name: Optional[str] = "lalonde_nsw.csv"
    custom_records: Optional[List[Dict[str, Any]]] = None
    treatment_col: str = "treat"
    outcome_col: str = "re78"
    feature_cols: List[str] = ["age", "educ"]
    confounder_cols: List[str] = ["black", "hisp", "married"]

def _load_df(dataset_name: Optional[str], custom_records: Optional[List[Dict[str, Any]]]) -> pd.DataFrame:
    if custom_records and len(custom_records) > 0:
        return pd.DataFrame(custom_records)
    if dataset_name:
        path = os.path.join(DATASETS_DIR, dataset_name)
        if os.path.exists(path):
            return pd.read_csv(path)
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found.")
    raise HTTPException(status_code=400, detail="Must provide either dataset_name or custom_records.")

@router.post("/psm")
def psm_endpoint(req: PSMRequest):
    try:
        df = _load_df(req.dataset_name, req.custom_records)
        return run_propensity_score_matching(
            df=df,
            treatment_col=req.treatment_col,
            outcome_col=req.outcome_col,
            confounder_cols=req.confounder_cols
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/did")
def did_endpoint(req: DiDRequest):
    try:
        df = _load_df(req.dataset_name, req.custom_records)
        return run_difference_in_differences(
            df=df,
            outcome_col=req.outcome_col,
            treatment_col=req.treatment_col,
            post_col=req.post_col
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/synthetic-control")
def synthetic_control_endpoint(req: SyntheticControlRequest):
    try:
        df = _load_df(req.dataset_name, req.custom_records)
        return run_synthetic_control(
            df=df,
            unit_col=req.unit_col,
            time_col=req.time_col,
            outcome_col=req.outcome_col,
            treated_unit=req.treated_unit,
            treatment_year=req.treatment_year
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/econml-cate")
def econml_cate_endpoint(req: EconMLRequest):
    try:
        df = _load_df(req.dataset_name, req.custom_records)
        return run_econml_dml(
            df=df,
            treatment_col=req.treatment_col,
            outcome_col=req.outcome_col,
            feature_cols=req.feature_cols,
            confounder_cols=req.confounder_cols
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
