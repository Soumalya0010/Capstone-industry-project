import math
import numpy as np
from statsmodels.stats.power import NormalIndPower, TTestIndPower

def calculate_proportion_sample_size(
    baseline_rate: float,
    mde_relative: float = None,
    mde_absolute: float = None,
    power: float = 0.80,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    ratio: float = 1.0,
    daily_traffic: int = None
):
    """
    Calculate required sample size for two-sample proportion test using statsmodels NormalIndPower.
    """
    if baseline_rate <= 0 or baseline_rate >= 1:
        raise ValueError("Baseline conversion rate must be between 0 and 1.")
    
    if mde_absolute is not None:
        target_rate = baseline_rate + mde_absolute
        rel_lift = (mde_absolute / baseline_rate) * 100
    elif mde_relative is not None:
        target_rate = baseline_rate * (1 + mde_relative)
        rel_lift = mde_relative * 100
        mde_absolute = target_rate - baseline_rate
    else:
        raise ValueError("Either mde_relative or mde_absolute must be provided.")
        
    if target_rate <= 0 or target_rate >= 1:
        raise ValueError(f"Target conversion rate ({target_rate:.4f}) must be between 0 and 1.")
        
    # Cohen's h: 2 * (arcsin(sqrt(p2)) - arcsin(sqrt(p1)))
    effect_size = 2 * (math.asin(math.sqrt(target_rate)) - math.asin(math.sqrt(baseline_rate)))
    abs_effect_size = abs(effect_size)
    
    sm_power = NormalIndPower()
    n_per_variant = sm_power.solve_power(
        effect_size=abs_effect_size,
        alpha=alpha,
        power=power,
        ratio=ratio,
        alternative=alternative
    )
    
    n_a = int(math.ceil(n_per_variant))
    n_b = int(math.ceil(n_per_variant * ratio))
    total_sample = n_a + n_b
    
    # Generate Power Curve (Power vs Sample Size)
    sample_sizes = np.linspace(max(10, int(n_a * 0.1)), int(n_a * 2.0), 20).astype(int)
    power_curve = []
    for s in sample_sizes:
        pw = sm_power.power(
            effect_size=abs_effect_size,
            nobs1=s,
            alpha=alpha,
            ratio=ratio,
            alternative=alternative
        )
        power_curve.append({
            "sample_size_per_variant": int(s),
            "total_sample_size": int(s * (1 + ratio)),
            "power": round(float(pw), 4)
        })
        
    days_needed = None
    if daily_traffic and daily_traffic > 0:
        days_needed = math.ceil(total_sample / daily_traffic)

    # Plain English explanation
    explanation = (
        f"To detect a {rel_lift:+.2f}% relative change in conversion rate "
        f"(from {baseline_rate*100:.2f}% to {target_rate*100:.2f}%) with {int(power*100)}% statistical power "
        f"at a {alpha*100:.1f}% significance level (alpha={alpha}), you need at least "
        f"{n_a:,} users in Control and {n_b:,} users in Treatment (Total: {total_sample:,} users)."
    )
    if days_needed:
        explanation += f" At your estimated traffic of {daily_traffic:,} users/day, this experiment will need to run for {days_needed} days."

    return {
        "metric_type": "proportion",
        "baseline_rate": baseline_rate,
        "target_rate": target_rate,
        "mde_absolute": round(mde_absolute, 4),
        "mde_relative_pct": round(rel_lift, 2),
        "cohens_h": round(effect_size, 4),
        "alpha": alpha,
        "power": power,
        "alternative": alternative,
        "sample_size_control": n_a,
        "sample_size_treatment": n_b,
        "total_sample_size": total_sample,
        "estimated_days": days_needed,
        "power_curve": power_curve,
        "plain_english_summary": explanation
    }

def calculate_continuous_sample_size(
    baseline_mean: float,
    baseline_std: float,
    mde_absolute: float = None,
    mde_relative: float = None,
    power: float = 0.80,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    ratio: float = 1.0,
    daily_traffic: int = None
):
    """
    Calculate required sample size for two-sample t-test using statsmodels TTestIndPower.
    """
    if baseline_std <= 0:
        raise ValueError("Standard deviation must be strictly positive.")

    if mde_absolute is not None:
        target_mean = baseline_mean + mde_absolute
        rel_lift = (mde_absolute / baseline_mean) * 100 if baseline_mean != 0 else 0
    elif mde_relative is not None:
        target_mean = baseline_mean * (1 + mde_relative)
        mde_absolute = target_mean - baseline_mean
        rel_lift = mde_relative * 100
    else:
        raise ValueError("Either mde_relative or mde_absolute must be provided.")

    cohens_d = abs(mde_absolute) / baseline_std
    if cohens_d == 0:
        raise ValueError("Effect size Cohen's d cannot be zero.")

    sm_power = TTestIndPower()
    n_per_variant = sm_power.solve_power(
        effect_size=cohens_d,
        alpha=alpha,
        power=power,
        ratio=ratio,
        alternative=alternative
    )

    n_a = int(math.ceil(n_per_variant))
    n_b = int(math.ceil(n_per_variant * ratio))
    total_sample = n_a + n_b

    # Power Curve
    sample_sizes = np.linspace(max(10, int(n_a * 0.1)), int(n_a * 2.0), 20).astype(int)
    power_curve = []
    for s in sample_sizes:
        pw = sm_power.power(
            effect_size=cohens_d,
            nobs1=s,
            alpha=alpha,
            ratio=ratio,
            alternative=alternative
        )
        power_curve.append({
            "sample_size_per_variant": int(s),
            "total_sample_size": int(s * (1 + ratio)),
            "power": round(float(pw), 4)
        })

    days_needed = None
    if daily_traffic and daily_traffic > 0:
        days_needed = math.ceil(total_sample / daily_traffic)

    explanation = (
        f"To detect an effect of {mde_absolute:+.2f} units (from {baseline_mean:.2f} to {target_mean:.2f}, "
        f"Cohen's d={cohens_d:.3f}) with {int(power*100)}% power at alpha={alpha}, you need "
        f"{n_a:,} observations in Control and {n_b:,} observations in Treatment (Total: {total_sample:,})."
    )
    if days_needed:
        explanation += f" With {daily_traffic:,} observations/day, test duration is approximately {days_needed} days."

    return {
        "metric_type": "continuous",
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "target_mean": round(target_mean, 4),
        "mde_absolute": round(mde_absolute, 4),
        "mde_relative_pct": round(rel_lift, 2),
        "cohens_d": round(cohens_d, 4),
        "alpha": alpha,
        "power": power,
        "alternative": alternative,
        "sample_size_control": n_a,
        "sample_size_treatment": n_b,
        "total_sample_size": total_sample,
        "estimated_days": days_needed,
        "power_curve": power_curve,
        "plain_english_summary": explanation
    }
