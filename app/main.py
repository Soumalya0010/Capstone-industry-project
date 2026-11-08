import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.core.config import PROJECT_TITLE, VERSION, FRONTEND_DIR
from app.routers import (
    calculator,
    frequentist,
    sequential,
    bayesian,
    observational,
    datasets_router
)

app = FastAPI(
    title=PROJECT_TITLE,
    version=VERSION,
    description="Full-stack Data Science Experimentation & Causal Inference Platform: Power & Sample Size, Sequential Testing, Frequentist & Multiple Testing, Bayesian A/B (PyMC), and Observational Causal Inference (DoWhy & EconML)."
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(calculator.router)
app.include_router(frequentist.router)
app.include_router(sequential.router)
app.include_router(bayesian.router)
app.include_router(observational.router)
app.include_router(datasets_router.router)

# Mount Frontend static assets
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def serve_index():
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"status": "ok", "message": "API online. Frontend folder pending."}

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "project": PROJECT_TITLE,
        "version": VERSION,
        "modules": [
            "1. Statistical Foundations",
            "2. Power & Sample Size Calculator (Statsmodels)",
            "3. Frequentist A/B & Multiple Testing (Bonferroni, Benjamini-Hochberg)",
            "4. Sequential Testing & Safe Peeking (O'Brien-Fleming, Pocock, mSPRT)",
            "5. Bayesian A/B Analysis (PyMC MCMC)",
            "6. Observational Causal Inference (DoWhy PSM, DiD, Synthetic Control, EconML CATE)",
            "7. Interactive React UI"
        ]
    }
