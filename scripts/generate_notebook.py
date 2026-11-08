import json
import os

nb = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Capstones & Industry Projects - Data Science\n",
    "## Notebook 01: Statistical Foundations of Experimentation\n",
    "### Step 1: Hypothesis Tests, p-Values, Power, and Confidence Intervals\n",
    "\n",
    "This notebook demonstrates the core frequentist experimentation foundations:\n",
    "1. Simulating Type I Error and uniform distribution of p-values under the Null ($H_0$)\n",
    "2. Simulating Statistical Power ($1 - \\beta$) under true alternative effects\n",
    "3. Confidence Interval coverage simulation (demonstrating frequentist coverage probability)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import scipy.stats as stats\n",
    "import statsmodels.api as sm\n",
    "from statsmodels.stats.power import NormalIndPower, TTestIndPower\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "np.random.seed(42)\n",
    "print('Libraries imported successfully!')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1. Behavior of p-Values Under the Null Hypothesis ($H_0$ is true)\n",
    "When there is NO true difference between Variant A and Variant B, the p-value is uniformly distributed across [0, 1]. Exactly 5% of p-values fall below $\\alpha = 0.05$ purely by chance!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Simulate 10,000 A/A tests where mean_A == mean_B == 100\n",
    "n_sims = 10000\n",
    "n_sample = 500\n",
    "p_values_null = []\n",
    "\n",
    "for _ in range(n_sims):\n",
    "    sample_a = np.random.normal(loc=100.0, scale=15.0, size=n_sample)\n",
    "    sample_b = np.random.normal(loc=100.0, scale=15.0, size=n_sample)\n",
    "    t_stat, p_val = stats.ttest_ind(sample_a, sample_b, equal_var=True)\n",
    "    p_values_null.append(p_val)\n",
    "\n",
    "empirical_alpha = np.mean(np.array(p_values_null) < 0.05)\n",
    "print(f'Empirical False Positive Rate (Nominal alpha=0.05): {empirical_alpha:.4f}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2. Statistical Power Simulation ($1 - \\beta$)\n",
    "Now let's introduce a true effect: $\\mu_A = 100$, $\\mu_B = 103$, $\\sigma = 15$ (Cohen's $d = 0.20$)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "power_calc = TTestIndPower()\n",
    "cohens_d = (103.0 - 100.0) / 15.0\n",
    "req_n = power_calc.solve_power(effect_size=cohens_d, power=0.80, alpha=0.05, ratio=1.0)\n",
    "print(f'Cohen\\'s d: {cohens_d:.2f}')\n",
    "print(f'Theoretical Required Sample Size per Variant (80% Power): {int(np.ceil(req_n))}')\n",
    "\n",
    "# Simulate 1,000 runs with n = req_n to verify empirical power\n",
    "n_test = int(np.ceil(req_n))\n",
    "p_values_alt = []\n",
    "for _ in range(1000):\n",
    "    s_a = np.random.normal(100.0, 15.0, n_test)\n",
    "    s_b = np.random.normal(103.0, 15.0, n_test)\n",
    "    _, p = stats.ttest_ind(s_a, s_b)\n",
    "    p_values_alt.append(p)\n",
    "\n",
    "empirical_power = np.mean(np.array(p_values_alt) < 0.05)\n",
    "print(f'Empirical Statistical Power: {empirical_power:.3f}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3. Confidence Interval Coverage Probability\n",
    "Let's verify the frequentist definition of a 95% Confidence Interval: out of 1,000 experiments, does the 95% CI contain the true difference $\\delta = 3.0$ roughly 95% of the time?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "true_diff = 3.0\n",
    "ci_contains_true = []\n",
    "\n",
    "for _ in range(1000):\n",
    "    s_a = np.random.normal(100.0, 15.0, 200)\n",
    "    s_b = np.random.normal(103.0, 15.0, 200)\n",
    "    diff_mean = np.mean(s_b) - np.mean(s_a)\n",
    "    se_diff = np.sqrt(np.var(s_a, ddof=1)/200 + np.var(s_b, ddof=1)/200)\n",
    "    ci_lower = diff_mean - 1.96 * se_diff\n",
    "    ci_upper = diff_mean + 1.96 * se_diff\n",
    "    ci_contains_true.append(ci_lower <= true_diff <= ci_upper)\n",
    "\n",
    "coverage = np.mean(ci_contains_true)\n",
    "print(f'Empirical 95% CI Coverage Probability: {coverage:.3f}')"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

os.makedirs(r"C:\Users\Soumlya\.gemini\antigravity\scratch\capstones_industry_projects\notebooks", exist_ok=True)
with open(r"C:\Users\Soumlya\.gemini\antigravity\scratch\capstones_industry_projects\notebooks\01_statistical_foundations.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Notebook 01_statistical_foundations.ipynb written successfully")
