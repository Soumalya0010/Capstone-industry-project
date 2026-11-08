import math
import numpy as np
import scipy.stats as stats
import pymc as pm

def run_bayesian_conversion_ab(
    control_conversions: int,
    control_total: int,
    treatment_conversions: int,
    treatment_total: int,
    prior_alpha_a: float = 1.0,
    prior_beta_a: float = 1.0,
    prior_alpha_b: float = 1.0,
    prior_beta_b: float = 1.0,
    draws: int = 2000,
    tune: int = 500,
    random_seed: int = 42
):
    """
    Bayesian A/B analysis for conversion rates using PyMC.
    Estimates the posterior distribution of p_A, p_B, the difference (p_B - p_A),
    and relative uplift ((p_B - p_A) / p_A).
    """
    if control_total <= 0 or treatment_total <= 0:
        raise ValueError("Sample sizes must be greater than zero.")
    if control_conversions > control_total or treatment_conversions > treatment_total:
        raise ValueError("Conversions cannot exceed total sample size.")

    # PyMC model
    with pm.Model() as model:
        # Priors (Beta distribution)
        p_a = pm.Beta("p_a", alpha=prior_alpha_a, beta=prior_beta_a)
        p_b = pm.Beta("p_b", alpha=prior_alpha_b, beta=prior_beta_b)

        # Likelihoods (Binomial)
        obs_a = pm.Binomial("obs_a", n=control_total, p=p_a, observed=control_conversions)
        obs_b = pm.Binomial("obs_b", n=treatment_total, p=p_b, observed=treatment_conversions)

        # Deterministic variables
        diff = pm.Deterministic("diff", p_b - p_a)
        rel_uplift = pm.Deterministic("rel_uplift", (p_b - p_a) / p_a)

        # Sample from posterior
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=1,
            random_seed=random_seed,
            progressbar=False
        )

    # Extract posterior samples
    samples_a = np.array(trace.posterior["p_a"].values).flatten()
    samples_b = np.array(trace.posterior["p_b"].values).flatten()
    samples_diff = np.array(trace.posterior["diff"].values).flatten()
    samples_uplift = np.array(trace.posterior["rel_uplift"].values).flatten()

    # Probability that B beats A
    prob_b_beats_a = float(np.mean(samples_b > samples_a))

    # Summary stats
    mean_a = float(np.mean(samples_a))
    mean_b = float(np.mean(samples_b))
    mean_diff = float(np.mean(samples_diff))
    mean_uplift_pct = float(np.mean(samples_uplift)) * 100

    # 95% Credible Interval (Highest Density / Percentile Interval)
    ci_diff_lower = float(np.percentile(samples_diff, 2.5))
    ci_diff_upper = float(np.percentile(samples_diff, 97.5))
    
    ci_uplift_lower = float(np.percentile(samples_uplift, 2.5)) * 100
    ci_uplift_upper = float(np.percentile(samples_uplift, 97.5)) * 100

    # Expected Loss (Risk)
    # Expected loss if we choose B: E[max(p_A - p_B, 0)]
    loss_b = float(np.mean(np.maximum(samples_a - samples_b, 0)))
    # Expected loss if we choose A: E[max(p_B - p_A, 0)]
    loss_a = float(np.mean(np.maximum(samples_b - samples_a, 0)))

    # Kernel density estimation for plotting in React
    grid_diff = np.linspace(min(samples_diff), max(samples_diff), 30)
    kde_diff = stats.gaussian_kde(samples_diff)
    density_diff = kde_diff(grid_diff)

    grid_rates = np.linspace(
        max(0.001, min(min(samples_a), min(samples_b)) * 0.8),
        min(0.999, max(max(samples_a), max(samples_b)) * 1.2),
        30
    )
    kde_a = stats.gaussian_kde(samples_a)
    kde_b = stats.gaussian_kde(samples_b)
    density_a = kde_a(grid_rates)
    density_b = kde_b(grid_rates)

    posterior_chart = [
        {
            "rate": round(float(grid_rates[i]), 4),
            "density_control": round(float(density_a[i]), 2),
            "density_treatment": round(float(density_b[i]), 2)
        }
        for i in range(len(grid_rates))
    ]

    difference_chart = [
        {
            "diff": round(float(grid_diff[i]), 4),
            "density": round(float(density_diff[i]), 2),
            "is_positive": bool(grid_diff[i] > 0)
        }
        for i in range(len(grid_diff))
    ]

    # Plain English Summary
    if prob_b_beats_a >= 0.95:
        recommendation = "Deploy Variant B"
        verdict = f"High confidence winner: There is a {prob_b_beats_a*100:.1f}% probability that Variant B beats Variant A."
    elif prob_b_beats_a <= 0.05:
        recommendation = "Keep Variant A (Control)"
        verdict = f"Variant B underperformed: Only a {prob_b_beats_a*100:.1f}% probability that Variant B beats Variant A."
    else:
        recommendation = "Inconclusive / Continue Testing"
        verdict = f"Inconclusive result: There is a {prob_b_beats_a*100:.1f}% probability that Variant B beats Variant A, which does not meet the standard 95% threshold."

    explanation = (
        f"{verdict} The posterior expected conversion rate is {mean_b*100:.2f}% for Treatment vs {mean_a*100:.2f}% for Control. "
        f"The expected conversion lift is {mean_diff*100:+.2f}% ({mean_uplift_pct:+.2f}% relative uplift), "
        f"with a 95% credible interval of [{ci_diff_lower*100:+.2f}%, {ci_diff_upper*100:+.2f}%]. "
        f"Choosing Variant B carries an expected loss (risk) of only {loss_b*100:.3f}%."
    )

    return {
        "model_type": "Beta-Binomial (PyMC MCMC)",
        "probability_b_beats_a": round(prob_b_beats_a, 4),
        "recommendation": recommendation,
        "control_mean": round(mean_a, 4),
        "treatment_mean": round(mean_b, 4),
        "expected_lift_absolute": round(mean_diff, 4),
        "expected_lift_relative_pct": round(mean_uplift_pct, 2),
        "ci_95_diff_lower": round(ci_diff_lower, 4),
        "ci_95_diff_upper": round(ci_diff_upper, 4),
        "ci_95_uplift_lower_pct": round(ci_uplift_lower, 2),
        "ci_95_uplift_upper_pct": round(ci_uplift_upper, 2),
        "expected_loss_choosing_treatment": round(loss_b, 6),
        "expected_loss_choosing_control": round(loss_a, 6),
        "posterior_chart": posterior_chart,
        "difference_chart": difference_chart,
        "plain_english_summary": explanation
    }

def run_bayesian_continuous_ab(
    sample_a: list,
    sample_b: list,
    draws: int = 1500,
    tune: int = 500,
    random_seed: int = 42
):
    """
    Bayesian A/B analysis for continuous metrics (revenue, duration) using PyMC.
    Estimates posterior means mu_A, mu_B, standard deviations sigma_A, sigma_B,
    and difference mu_B - mu_A.
    """
    a = np.array(sample_a, dtype=float)
    b = np.array(sample_b, dtype=float)
    
    mean_obs_a = float(np.mean(a))
    std_obs_a = float(np.std(a, ddof=1))
    mean_obs_b = float(np.mean(b))
    std_obs_b = float(np.std(b, ddof=1))
    
    # Standardize data for fast robust sampling, then transform back
    pooled_mean = float(np.mean(np.concatenate([a, b])))
    pooled_std = float(np.std(np.concatenate([a, b]), ddof=1)) or 1.0

    with pm.Model() as model:
        # Priors
        mu_a = pm.Normal("mu_a", mu=mean_obs_a, sigma=std_obs_a * 2)
        mu_b = pm.Normal("mu_b", mu=mean_obs_b, sigma=std_obs_b * 2)
        
        sigma_a = pm.HalfNormal("sigma_a", sigma=std_obs_a * 2)
        sigma_b = pm.HalfNormal("sigma_b", sigma=std_obs_b * 2)
        
        # Likelihood
        obs_a = pm.Normal("obs_a", mu=mu_a, sigma=sigma_a, observed=a)
        obs_b = pm.Normal("obs_b", mu=mu_b, sigma=sigma_b, observed=b)
        
        diff = pm.Deterministic("diff", mu_b - mu_a)
        rel_uplift = pm.Deterministic("rel_uplift", (mu_b - mu_a) / mu_a)
        
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=1,
            random_seed=random_seed,
            progressbar=False
        )
        
    samples_a = np.array(trace.posterior["mu_a"].values).flatten()
    samples_b = np.array(trace.posterior["mu_b"].values).flatten()
    samples_diff = np.array(trace.posterior["diff"].values).flatten()
    samples_uplift = np.array(trace.posterior["rel_uplift"].values).flatten()

    prob_b_beats_a = float(np.mean(samples_b > samples_a))
    mean_a = float(np.mean(samples_a))
    mean_b = float(np.mean(samples_b))
    mean_diff = float(np.mean(samples_diff))
    mean_uplift_pct = float(np.mean(samples_uplift)) * 100

    ci_diff_lower = float(np.percentile(samples_diff, 2.5))
    ci_diff_upper = float(np.percentile(samples_diff, 97.5))

    loss_b = float(np.mean(np.maximum(samples_a - samples_b, 0)))
    loss_a = float(np.mean(np.maximum(samples_b - samples_a, 0)))

    # Grid for diff chart
    grid_diff = np.linspace(min(samples_diff), max(samples_diff), 30)
    kde_diff = stats.gaussian_kde(samples_diff)
    density_diff = kde_diff(grid_diff)

    difference_chart = [
        {
            "diff": round(float(grid_diff[i]), 3),
            "density": round(float(density_diff[i]), 4),
            "is_positive": bool(grid_diff[i] > 0)
        }
        for i in range(len(grid_diff))
    ]

    recommendation = "Deploy Variant B" if prob_b_beats_a >= 0.95 else ("Keep Control A" if prob_b_beats_a <= 0.05 else "Inconclusive")
    explanation = (
        f"There is a {prob_b_beats_a*100:.1f}% probability that Variant B has a higher mean than Variant A. "
        f"The expected difference is {mean_diff:+.2f} units ({mean_uplift_pct:+.2f}%), with a 95% Credible Interval "
        f"of [{ci_diff_lower:+.2f}, {ci_diff_upper:+.2f}]. Choosing Variant B carries an expected loss of {loss_b:.3f} units."
    )

    return {
        "model_type": "Normal Hierarchical (PyMC MCMC)",
        "probability_b_beats_a": round(prob_b_beats_a, 4),
        "recommendation": recommendation,
        "control_mean": round(mean_a, 3),
        "treatment_mean": round(mean_b, 3),
        "expected_lift_absolute": round(mean_diff, 3),
        "expected_lift_relative_pct": round(mean_uplift_pct, 2),
        "ci_95_diff_lower": round(ci_diff_lower, 3),
        "ci_95_diff_upper": round(ci_diff_upper, 3),
        "expected_loss_choosing_treatment": round(loss_b, 4),
        "expected_loss_choosing_control": round(loss_a, 4),
        "difference_chart": difference_chart,
        "plain_english_summary": explanation
    }
