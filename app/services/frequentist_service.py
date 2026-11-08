import math
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep

def simulate_ab_data(
    metric_type: str = "conversion",
    n_a: int = 2500,
    n_b: int = 2500,
    baseline_val: float = 0.12,
    relative_lift: float = 0.15,
    std_dev: float = 25.0,
    random_seed: int = 42
):
    """
    Simulate A/B test data for continuous or conversion metrics.
    """
    rng = np.random.default_rng(random_seed)
    
    if metric_type == "conversion":
        p_a = baseline_val
        p_b = min(0.99, max(0.01, baseline_val * (1 + relative_lift)))
        data_a = rng.binomial(1, p_a, n_a)
        data_b = rng.binomial(1, p_b, n_b)
        
        summary = {
            "control_count": int(n_a),
            "control_conversions": int(np.sum(data_a)),
            "control_rate": round(float(np.mean(data_a)), 4),
            "treatment_count": int(n_b),
            "treatment_conversions": int(np.sum(data_b)),
            "treatment_rate": round(float(np.mean(data_b)), 4),
            "true_relative_lift_pct": round(relative_lift * 100, 2)
        }
    else:
        mu_a = baseline_val
        mu_b = baseline_val * (1 + relative_lift)
        data_a = rng.normal(mu_a, std_dev, n_a)
        data_b = rng.normal(mu_b, std_dev, n_b)
        
        summary = {
            "control_count": int(n_a),
            "control_mean": round(float(np.mean(data_a)), 4),
            "control_std": round(float(np.std(data_a, ddof=1)), 4),
            "treatment_count": int(n_b),
            "treatment_mean": round(float(np.mean(data_b)), 4),
            "treatment_std": round(float(np.std(data_b, ddof=1)), 4),
            "true_relative_lift_pct": round(relative_lift * 100, 2)
        }
        
    return {
        "metric_type": metric_type,
        "summary": summary,
        "sample_a_preview": [round(float(x), 2) for x in data_a[:20]],
        "sample_b_preview": [round(float(x), 2) for x in data_b[:20]],
        "raw_a": data_a.tolist(),
        "raw_b": data_b.tolist()
    }

def run_two_sample_ttest(
    sample_a: list,
    sample_b: list,
    alpha: float = 0.05,
    equal_var: bool = False
):
    """
    Two-sample t-test (Welch's t-test by default or Student's t-test).
    """
    a = np.array(sample_a, dtype=float)
    b = np.array(sample_b, dtype=float)
    
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        raise ValueError("Each sample must have at least 2 data points.")
        
    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    var_a, var_b = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    std_a, std_b = math.sqrt(var_a), math.sqrt(var_b)
    
    # Welch's or Student's t-test
    res = stats.ttest_ind(b, a, equal_var=equal_var)
    t_stat = float(res.statistic)
    p_val = float(res.pvalue)
    
    # Degrees of freedom & Standard Error
    if equal_var:
        df = n_a + n_b - 2
        s_pooled = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / df)
        se_diff = s_pooled * math.sqrt(1/n_a + 1/n_b)
    else:
        se_a = var_a / n_a
        se_b = var_b / n_b
        se_diff = math.sqrt(se_a + se_b)
        df = ((se_a + se_b) ** 2) / ((se_a ** 2) / (n_a - 1) + (se_b ** 2) / (n_b - 1))
        
    diff = mean_b - mean_a
    rel_diff_pct = (diff / mean_a) * 100 if mean_a != 0 else 0.0
    
    # Critical t for confidence interval
    t_crit = stats.t.ppf(1 - alpha/2, df)
    ci_lower = diff - t_crit * se_diff
    ci_upper = diff + t_crit * se_diff
    
    # Levene test for equality of variance
    levene_stat, levene_p = stats.levene(a, b)
    
    # Plain English Conclusion
    significant = p_val < alpha
    direction = "an increase" if diff > 0 else "a decrease"
    decision = "Reject Null Hypothesis (Statistically Significant)" if significant else "Fail to Reject Null (Not Statistically Significant)"
    
    summary_text = (
        f"Variant B showed {direction} of {abs(diff):.2f} units ({rel_diff_pct:+.2f}%) compared to Variant A "
        f"(Control Mean: {mean_a:.2f}, Treatment Mean: {mean_b:.2f}). "
        f"The two-sample Welch's t-test yielded t = {t_stat:.3f} and p = {p_val:.4f}. "
    )
    if significant:
        summary_text += (
            f"Because p < {alpha}, this difference is statistically significant at the {int((1-alpha)*100)}% confidence level. "
            f"We are 95% confident that the true difference is between {ci_lower:.2f} and {ci_upper:.2f} units."
        )
    else:
        summary_text += (
            f"Because p >= {alpha}, there is insufficient evidence to conclude that Variant B outperforms Variant A. "
            f"The observed difference could plausibly be due to random sampling variability."
        )

    # Histogram / Distribution Points for Charting
    hist_a, bin_edges = np.histogram(a, bins=15, density=True)
    hist_b, _ = np.histogram(b, bins=bin_edges, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    dist_chart = [
        {"bin": round(float(bin_centers[i]), 2), "density_control": round(float(hist_a[i]), 5), "density_treatment": round(float(hist_b[i]), 5)}
        for i in range(len(bin_centers))
    ]

    return {
        "test_name": "Welch's Two-Sample t-test" if not equal_var else "Student's Two-Sample t-test",
        "sample_size_control": n_a,
        "sample_size_treatment": n_b,
        "mean_control": round(mean_a, 4),
        "mean_treatment": round(mean_b, 4),
        "std_control": round(std_a, 4),
        "std_treatment": round(std_b, 4),
        "diff_means": round(diff, 4),
        "relative_diff_pct": round(rel_diff_pct, 2),
        "standard_error": round(se_diff, 4),
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_val, 6),
        "degrees_of_freedom": round(df, 2),
        "confidence_level": 1 - alpha,
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "statistically_significant": significant,
        "decision": decision,
        "variance_equality_levene_p": round(float(levene_p), 4),
        "plain_english_summary": summary_text,
        "distribution_chart": dist_chart
    }

def run_chi_square_test(
    control_conversions: int,
    control_total: int,
    treatment_conversions: int,
    treatment_total: int,
    alpha: float = 0.05
):
    """
    Chi-Square Test of Independence for Conversion Rates (2x2 contingency table)
    and Two-Proportion Z-Test.
    """
    if control_total <= 0 or treatment_total <= 0:
        raise ValueError("Sample sizes must be greater than zero.")
    if control_conversions > control_total or treatment_conversions > treatment_total:
        raise ValueError("Conversions cannot exceed total sample size.")
        
    ctrl_non = control_total - control_conversions
    treat_non = treatment_total - treatment_conversions
    
    # 2x2 Contingency Table
    # Rows: Control, Treatment
    # Cols: Converted, Not Converted
    table = np.array([
        [control_conversions, ctrl_non],
        [treatment_conversions, treat_non]
    ])
    
    chi2_stat, p_val, dof, expected = stats.chi2_contingency(table, correction=True)
    
    # Proportions
    p_ctrl = control_conversions / control_total
    p_treat = treatment_conversions / treatment_total
    diff = p_treat - p_ctrl
    rel_lift_pct = (diff / p_ctrl) * 100 if p_ctrl > 0 else 0.0
    
    # Two-proportion z-test via statsmodels
    count = np.array([treatment_conversions, control_conversions])
    nobs = np.array([treatment_total, control_total])
    z_stat, z_pval = proportions_ztest(count, nobs)
    
    # Confidence Interval for difference in proportions (Wilson score / Agresti-Caffo)
    low, upp = confint_proportions_2indep(
        treatment_conversions, treatment_total,
        control_conversions, control_total,
        method="agresti-caffo",
        compare="diff",
        alpha=alpha
    )
    
    # Odds Ratio & Relative Risk
    odds_ctrl = control_conversions / max(1, ctrl_non)
    odds_treat = treatment_conversions / max(1, treat_non)
    odds_ratio = odds_treat / odds_ctrl if odds_ctrl > 0 else 1.0
    relative_risk = p_treat / p_ctrl if p_ctrl > 0 else 1.0
    
    significant = p_val < alpha
    decision = "Reject Null Hypothesis (Statistically Significant)" if significant else "Fail to Reject Null"
    direction = "higher" if diff > 0 else "lower"
    
    summary_text = (
        f"Variant B conversion rate was {p_treat*100:.2f}% vs Variant A at {p_ctrl*100:.2f}% "
        f"({rel_lift_pct:+.2f}% relative change). "
        f"The Chi-Square test (with Yates' continuity correction) resulted in Chi2 = {chi2_stat:.3f} and p = {p_val:.4f} "
        f"(Two-proportion Z = {z_stat:.3f}). "
    )
    if significant:
        summary_text += (
            f"Because p < {alpha}, the difference in conversion rate is statistically significant. "
            f"We are {int((1-alpha)*100)}% confident that the true conversion lift is between {low*100:+.2f}% and {upp*100:+.2f}%."
        )
    else:
        summary_text += (
            f"Because p >= {alpha}, there is no statistically significant evidence of a true difference in conversion rates."
        )

    return {
        "test_name": "Pearson's Chi-Square Test & Two-Proportion Z-Test",
        "control_total": control_total,
        "control_conversions": control_conversions,
        "control_rate": round(p_ctrl, 4),
        "treatment_total": treatment_total,
        "treatment_conversions": treatment_conversions,
        "treatment_rate": round(p_treat, 4),
        "rate_difference": round(diff, 4),
        "relative_lift_pct": round(rel_lift_pct, 2),
        "chi2_statistic": round(float(chi2_stat), 4),
        "chi2_p_value": round(float(p_val), 6),
        "z_statistic": round(float(z_stat), 4),
        "z_p_value": round(float(z_pval), 6),
        "odds_ratio": round(float(odds_ratio), 4),
        "relative_risk": round(float(relative_risk), 4),
        "ci_lower_diff": round(float(low), 4),
        "ci_upper_diff": round(float(upp), 4),
        "statistically_significant": significant,
        "decision": decision,
        "plain_english_summary": summary_text
    }

def run_multiple_testing_adjustment(
    test_items: list,
    alpha: float = 0.05
):
    """
    Perform Bonferroni and Benjamini-Hochberg (FDR) corrections
    using statsmodels.stats.multitest.multipletests.
    
    test_items format:
    [
        {"metric_name": "Click-through Rate", "p_value": 0.012, "raw_diff": 0.02},
        {"metric_name": "Checkout Rate", "p_value": 0.045, "raw_diff": 0.015},
        ...
    ]
    """
    if not test_items:
        raise ValueError("Must provide at least one test item.")
        
    p_vals = [item["p_value"] for item in test_items]
    m = len(p_vals)
    
    # 1. Bonferroni Correction (FWER)
    reject_bonf, p_bonf, _, _ = multipletests(p_vals, alpha=alpha, method="bonferroni")
    
    # 2. Benjamini-Hochberg Correction (FDR)
    reject_bh, p_bh, _, _ = multipletests(p_vals, alpha=alpha, method="fdr_bh")
    
    # Expected family-wise error rate without correction
    uncorrected_fwer = 1.0 - (1.0 - alpha) ** m
    
    adjusted_results = []
    for i, item in enumerate(test_items):
        adjusted_results.append({
            "metric_name": item.get("metric_name", f"Metric_{i+1}"),
            "raw_p_value": round(float(p_vals[i]), 5),
            "raw_significant": bool(p_vals[i] < alpha),
            "bonferroni_p_value": round(float(p_bonf[i]), 5),
            "bonferroni_significant": bool(reject_bonf[i]),
            "benjamini_hochberg_p_value": round(float(p_bh[i]), 5),
            "benjamini_hochberg_significant": bool(reject_bh[i]),
            "raw_effect": item.get("raw_diff", None)
        })
        
    bonf_discoveries = int(np.sum(reject_bonf))
    bh_discoveries = int(np.sum(reject_bh))
    raw_discoveries = int(np.sum([p < alpha for p in p_vals]))
    
    explanation = (
        f"You evaluated {m} metrics simultaneously at nominal alpha = {alpha}. "
        f"Without multiple-testing adjustment, the family-wise false positive risk rises to {uncorrected_fwer*100:.1f}%. "
        f"Out of {m} tests, {raw_discoveries} were significant before adjustment. "
        f"Applying the conservative Bonferroni correction (alpha/m = {alpha/m:.4f}) retains {bonf_discoveries} discoveries. "
        f"Applying Benjamini-Hochberg False Discovery Rate (FDR) control retains {bh_discoveries} discoveries."
    )
    
    return {
        "num_tests": m,
        "nominal_alpha": alpha,
        "uncorrected_family_wise_error_rate_pct": round(uncorrected_fwer * 100, 2),
        "raw_discoveries": raw_discoveries,
        "bonferroni_discoveries": bonf_discoveries,
        "benjamini_hochberg_discoveries": bh_discoveries,
        "results": adjusted_results,
        "plain_english_summary": explanation
    }
