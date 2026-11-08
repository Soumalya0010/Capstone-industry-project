from fastapi import APIRouter, HTTPException, UploadFile, File
import os
import io
import pandas as pd
from app.core.config import DATASETS_DIR

router = APIRouter(prefix="/api/datasets", tags=["Dataset Management & Upload"])

BENCHMARK_METADATA = {
    "simulated_ab_ecommerce.csv": {
        "title": "Option 1: Simulated E-Commerce A/B Test",
        "description": "Simulated multi-metric A/B test with 10,000 users, conversions, revenue, and device segment.",
        "recommended_for": ["Frequentist t-test & Chi-Square", "PyMC Bayesian A/B", "Multiple Testing Correction"]
    },
    "cookie_cats.csv": {
        "title": "Option 2: Cookie Cats Mobile Game A/B Test",
        "description": "Classic mobile gaming dataset testing Gate 30 vs Gate 40 on 1-day and 7-day player retention.",
        "recommended_for": ["Chi-Square Test", "Sequential Testing", "PyMC Bayesian A/B"]
    },
    "card_krueger_did.csv": {
        "title": "Option 6: Card & Krueger (1994) Minimum Wage Study",
        "description": "Natural experiment comparing fast-food employment in New Jersey (treated) and Pennsylvania (control) before and after minimum wage hike.",
        "recommended_for": ["Difference-in-Differences (DiD)"]
    },
    "california_prop99.csv": {
        "title": "Option 7: California Proposition 99 Tobacco Control",
        "description": "Per-capita cigarette sales (1970-2000) across 39 US states to measure the causal impact of California's 1988 tobacco tax initiative.",
        "recommended_for": ["Synthetic Control Method (SCM)"]
    },
    "lalonde_nsw.csv": {
        "title": "Option 4: LaLonde (1986) National Supported Work",
        "description": "Observational job-training program data evaluating post-intervention earnings (re78) controlling for demographic confounders.",
        "recommended_for": ["DoWhy Propensity Score Matching", "EconML CATE Double Machine Learning"]
    }
}

@router.get("/list")
def list_benchmark_datasets():
    datasets = []
    if os.path.exists(DATASETS_DIR):
        for fname in os.listdir(DATASETS_DIR):
            if fname.endswith(".csv"):
                meta = BENCHMARK_METADATA.get(fname, {
                    "title": fname,
                    "description": "User benchmark dataset.",
                    "recommended_for": []
                })
                fpath = os.path.join(DATASETS_DIR, fname)
                df = pd.read_csv(fpath, nrows=5)
                full_len = sum(1 for _ in open(fpath, encoding="utf-8")) - 1
                datasets.append({
                    "filename": fname,
                    "title": meta["title"],
                    "description": meta["description"],
                    "recommended_for": meta["recommended_for"],
                    "total_rows": full_len,
                    "columns": list(df.columns)
                })
    return {"datasets": datasets}

@router.get("/preview/{filename}")
def preview_dataset(filename: str, rows: int = 10):
    fpath = os.path.join(DATASETS_DIR, filename)
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail="Dataset not found")
    df = pd.read_csv(fpath)
    return {
        "filename": filename,
        "total_rows": len(df),
        "columns": list(df.columns),
        "preview": df.head(rows).to_dict(orient="records")
    }

@router.post("/upload")
async def upload_custom_dataset(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        # Save to datasets dir for easy reuse
        save_path = os.path.join(DATASETS_DIR, file.filename)
        df.to_csv(save_path, index=False)
        return {
            "message": "File uploaded and parsed successfully",
            "filename": file.filename,
            "total_rows": len(df),
            "columns": list(df.columns),
            "preview": df.head(10).to_dict(orient="records"),
            "summary_stats": df.describe(include="all").fillna("").to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")
