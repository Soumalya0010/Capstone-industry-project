# Capstones & Industry Projects - Data Science
## A/B Testing, Sequential Analysis, Bayesian PyMC & Observational Causal Inference

This repository contains the complete implementation of the **Capstones & Industry Projects** Data Science platform developed as part of internship requirements. The project bridges classical statistical experimentation, modern sequential peeking techniques, Bayesian decision analysis, and advanced observational causal inference within an end-to-end Python FastAPI service and an interactive React frontend.

---

## 🛠️ Required Tech Stack
Strictly built using the specified technologies:
- **Statsmodels**: Power analysis (`NormalIndPower`, `TTestIndPower`), multiple-testing corrections (`multipletests`), OLS regression.
- **DoWhy**: 4-stage causal framework (Model DAG $\to$ Identify $\to$ Estimate PSM $\to$ Refute with Placebo & Random Common Cause).
- **EconML**: Double Machine Learning (`LinearDML`) for Conditional Average Treatment Effects (CATE) and heterogeneity estimation.
- **PyMC**: Markov Chain Monte Carlo (MCMC) Bayesian A/B analysis ($P(B > A)$, Credible Intervals, Expected Loss).
- **FastAPI**: High-performance RESTful API endpoints with Pydantic validation schemas.
- **React**: Interactive, modern single-page dashboard for data upload, parameter configuration, plain-English executive summaries, and interactive charts.

---

## 📂 Project Architecture

```
capstones_industry_projects/
├── README.md                           # Comprehensive documentation & theory
├── requirements.txt                    # Project dependencies
├── datasets/                           # Reference & benchmark datasets
│   ├── simulated_ab_ecommerce.csv     # Option 1: Multi-metric simulated e-commerce test (N=10,000)
│   ├── cookie_cats.csv                # Option 2: Cookie Cats Gate 30 vs Gate 40 (N=10,000)
│   ├── card_krueger_did.csv           # Option 6: Card & Krueger minimum wage study (N=800)
│   ├── california_prop99.csv          # Option 7: California Prop 99 tobacco sales (1970-2000, N=1,209)
│   └── lalonde_nsw.csv                # Option 4: LaLonde NSW job-training observational data (N=2,500)
├── docs/
│   └── 01_statistical_foundations.md   # Step 1: Learning notes on hypothesis testing, p-values, power
├── notebooks/
│   └── 01_statistical_foundations.ipynb# Executable Jupyter notebook with simulations
├── app/
│   ├── __init__.py
│   ├── main.py                         # FastAPI entrypoint & router aggregation
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py                   # Paths, titles, and configuration
│   ├── services/
│   │   ├── __init__.py
│   │   ├── calculator_service.py       # Step 2: Statsmodels sample size & power curves
│   │   ├── frequentist_service.py      # Step 3: Simulation, t-test, Chi-square, Bonferroni, BH
│   │   ├── sequential_service.py       # Step 4: O'Brien-Fleming, Pocock, mSPRT safe peeking
│   │   ├── bayesian_service.py         # Step 5: PyMC MCMC posterior, P(B>A), HDI, Expected Loss
│   │   └── causal_service.py           # Step 6: DoWhy PSM, DiD, Synthetic Control, EconML
│   └── routers/
│       ├── __init__.py
│       ├── calculator.py               # REST endpoints for power calculator
│       ├── frequentist.py              # REST endpoints for frequentist tests & simulation
│       ├── sequential.py               # REST endpoints for sequential testing
│       ├── bayesian.py                 # REST endpoints for PyMC Bayesian analysis
│       ├── observational.py            # REST endpoints for DoWhy, DiD, SCM, EconML
│       └── datasets_router.py          # Benchmark dataset listing, preview, and CSV upload
├── frontend/
│   ├── index.html                      # Single-page React mounting point (pure React 18 CDN)
│   ├── app.js                          # Modular React application with charts & plain-English summaries
│   ├── styles.css                      # Professional dark-mode dashboard styling
│   └── package.json                    # Frontend package metadata
├── scripts/
│   ├── create_datasets.py              # Dataset generation pipeline
│   └── generate_notebook.py            # Notebook builder
└── tests/
    └── test_all.py                     # Automated unit test suite across all 7 steps
```

---

## 🚀 Step-by-Step Implementation Details

### Step 1: Statistical Foundations
- Formalized hypothesis testing ($H_0$ vs $H_1$, Type I $\alpha$, Type II $\beta$).
- Detailed p-value behavior under the null hypothesis (uniform distribution on $[0, 1]$).
- Statistical power ($1 - \beta$) and effect size equations (Cohen's $d$, Cohen's $h$).
- Frequentist confidence interval interpretation vs Bayesian credible intervals.
- Documented in `docs/01_statistical_foundations.md` and simulated in `notebooks/01_statistical_foundations.ipynb`.

### Step 2: Power and Sample-Size Calculator (FastAPI + Statsmodels)
- **Proportion Test**: Uses `statsmodels.stats.power.NormalIndPower` to compute required sample size given baseline conversion rate $p_1$, target relative lift, significance level $\alpha$, and statistical power $1 - \beta$.
- **Continuous Test**: Uses `statsmodels.stats.power.TTestIndPower` with Cohen's $d = |\mu_2 - \mu_1| / \sigma$.
- Generates 20-point power curves (Power vs Sample Size) for interactive visualization.
- Calculates estimated test duration given daily traffic.
- Endpoint: `POST /api/calculator/proportion`, `POST /api/calculator/continuous`.

### Step 3: Simulation, Hypothesis Tests & Multiple-Testing Correction
- **Simulation**: Generates realistic synthetic A/B datasets with customizable true effect sizes and noise distributions.
- **Hypothesis Tests**:
  - Welch's / Student's two-sample t-test (`scipy.stats.ttest_ind`).
  - Pearson's Chi-square test of independence (`scipy.stats.chi2_contingency`) and two-proportion Z-test (`statsmodels.stats.proportion.proportions_ztest`).
- **Multiple Testing Correction**:
  - Addresses Family-Wise Error Rate inflation: $\text{FWER} = 1 - (1 - \alpha)^m$.
  - **Bonferroni**: FWER control at $\alpha/m$.
  - **Benjamini-Hochberg (FDR)**: False Discovery Rate control (`statsmodels.stats.multitest.multipletests`).
- Endpoints: `POST /api/ab/simulate`, `POST /api/ab/ttest`, `POST /api/ab/chisquare`, `POST /api/ab/multiple-testing`.

### Step 4: Sequential Testing & Safe Peeking
- **The Peeking Problem**: Simulates 500 A/A tests under $H_0$ to demonstrate how daily peeking inflates false positive rates from 5% to ~25%+.
- **Alpha Spending Boundaries**:
  - **O'Brien-Fleming**: Conservative stopping thresholds at early looks, relaxing toward 1.96 at final look.
  - **Pocock**: Constant boundary across all interim looks.
  - **mSPRT (Mixture Sequential Probability Ratio Test)**: Computes always-valid p-values.
- Decides whether to "Stop Early for Efficacy", "Stop for Futility", or "Continue Testing".
- Endpoints: `POST /api/sequential/analyze`, `POST /api/sequential/simulate-peeking`.

### Step 5: Bayesian A/B Analysis with PyMC
- Fits Bayesian hierarchical models using PyMC MCMC sampling:
  $$p_A \sim \text{Beta}(\alpha_A, \beta_A), \quad p_B \sim \text{Beta}(\alpha_B, \beta_B)$$
  $$\text{diff} = p_B - p_A, \quad \text{uplift} = (p_B - p_A) / p_A$$
- Outputs:
  - **$P(B > A)$**: Probability that Variant B beats Variant A.
  - **95% Highest Density Interval (HDI)**: Direct intuitive credible interval.
  - **Expected Loss / Risk**: $\mathbb{E}[\max(p_A - p_B, 0)]$ for risk-aware shipping decisions.
- Endpoints: `POST /api/bayesian/conversion`, `POST /api/bayesian/continuous`.

### Step 6: Observational Causal Inference (DoWhy & EconML)
- **Propensity Score Matching (PSM)** using DoWhy on `lalonde_nsw.csv`:
  - 4-step framework: Model graph $\to$ Identify estimand $\to$ Estimate PSM $\to$ Refute with Placebo Treatment & Random Common Cause.
  - Standardized Mean Differences (SMD) table verifying covariate balance post-matching.
- **Difference-in-Differences (DiD)** on `card_krueger_did.csv`:
  - Evaluates New Jersey vs Pennsylvania fast food employment before and after the 1992 minimum wage hike.
  - Parallel trends verification and counterfactual trajectory calculation.
- **Synthetic Control Method (SCM)** on `california_prop99.csv`:
  - Constrained optimization solving for non-negative donor state weights summing to 1.
  - Reconstructs Synthetic California cigarette sales trajectory (1970-2000) and measures post-1988 divergence.
- **Heterogeneous Treatment Effects (CATE)** with EconML:
  - Double Machine Learning (`LinearDML`) to uncover which customer sub-segments benefit most from treatment.
- Endpoints: `POST /api/observational/psm`, `POST /api/observational/did`, `POST /api/observational/synthetic-control`, `POST /api/observational/econml-cate`.

### Step 7: Interactive React Frontend
- Developed purely with React 18 (no Vite or extra bundlers).
- Unified dashboard featuring:
  - Benchmark dataset quick selector (Cookie Cats, Card & Krueger, Prop 99, LaLonde, Simulated E-Commerce).
  - Custom CSV drag-and-drop file upload with column preview.
  - Tabbed interface covering all 6 statistical & causal modules.
  - **Plain-English conclusions** on every method translating complex statistics into actionable business decisions.
  - Interactive SVG charts for power curves, distributions, peeking boundaries, Bayesian posteriors, and parallel trends.

---

## 🏃 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Benchmark Datasets
```bash
python scripts/create_datasets.py
```

### 3. Run Automated Tests
```bash
python -m unittest tests/test_all.py
```

### 4. Start the Application
```bash
python -m uvicorn app.main:app --reload --port 8000
```
Open your browser and navigate to:
```
http://localhost:8000
```
FastAPI automatically serves the interactive React frontend at the root URL, with API docs available at `http://localhost:8000/docs`.
