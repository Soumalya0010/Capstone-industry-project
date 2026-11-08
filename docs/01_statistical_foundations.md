# Module 1: Statistical Foundations of Experimentation & A/B Testing

## 1. Hypothesis Testing Framework
In classical (Neyman-Pearson & Fisher) hypothesis testing, we formalize decision-making under uncertainty by comparing two competing claims:

- **Null Hypothesis ($H_0$)**: The default status quo assumption that there is *no effect*, *no difference*, or *no relationship* between variants (e.g., $\mu_B - \mu_A = 0$ or $p_B - p_A = 0$).
- **Alternative Hypothesis ($H_1$ or $H_A$)**: The research hypothesis that an actual effect exists (two-sided: $\mu_B \neq \mu_A$; one-sided: $\mu_B > \mu_A$).

### Decision Matrix & Error Types

| Reality \ Decision | Fail to Reject $H_0$ (Keep Status Quo) | Reject $H_0$ (Launch Variant B) |
| :--- | :--- | :--- |
| **$H_0$ is True** (No real effect) | Correct Decision ($1 - \alpha$) | **Type I Error ($\alpha$)** - False Positive |
| **$H_1$ is True** (Effect exists) | **Type II Error ($\beta$)** - False Negative | Correct Decision ($1 - \beta$) - **Power** |

- **Significance Level ($\alpha$)**: The probability of committing a Type I error (standard default: $\alpha = 0.05$).
- **Statistical Power ($1 - \beta$)**: The probability of correctly rejecting the null hypothesis when a true effect of size $\delta$ actually exists (standard target: $80\%$ or $90\%$).

---

## 2. p-Values: Definition and Critical Caveats

### Exact Definition
The **p-value** is the probability, assuming the null hypothesis ($H_0$) is strictly true, of observing a test statistic at least as extreme as the one calculated from the sample data:
$$p\text{-value} = P(\text{Data or more extreme} \mid H_0)$$

### Crucial Student & Practitioner Caveats
1. **$p$-value $\neq P(H_0 \mid \text{Data})$**: A p-value is **not** the probability that the null hypothesis is true. It measures compatibility of the data with $H_0$, not the Bayesian posterior probability of $H_0$.
2. **$1 - p \neq P(\text{Effect is real})$**: A p-value of 0.03 does not mean there is a 97% chance the new feature works.
3. **Statistical Significance $\neq$ Practical/Business Significance**: With $N = 10,000,000$, a microscopic, economically useless lift of +0.001% will yield $p < 0.0001$. Always examine the effect size and confidence intervals.
4. **Behavior under $H_0$**: When $H_0$ is strictly true and sample assumptions hold, the p-value is uniformly distributed across $[0, 1]$: $p \sim \text{Uniform}(0, 1)$.

---

## 3. Statistical Power & Minimum Detectable Effect (MDE)

Statistical power is determined by four interconnected parameters. Fixing any three determines the fourth:
1. **Significance Level ($\alpha$)**: Lower $\alpha$ (e.g., 0.01 vs 0.05) lowers power.
2. **Sample Size ($N$)**: Larger sample size reduces standard error $\text{SE} = \frac{\sigma}{\sqrt{N}}$, increasing power.
3. **Standard Deviation ($\sigma$)**: Higher metric variance reduces power.
4. **Effect Size ($\delta = \mu_B - \mu_A$)**: Larger effects are easier to detect.

### Standardized Effect Sizes
- **Continuous Metrics (Cohen's $d$)**:
  $$d = \frac{\mu_B - \mu_A}{\sigma_{\text{pooled}}}$$
- **Proportions / Conversion Rates (Cohen's $h$)**:
  $$h = 2 \arcsin(\sqrt{p_B}) - 2 \arcsin(\sqrt{p_A})$$

### Sample Size Formula (Two-Sample Z-Test for Proportions)
For equal sample sizes $N_A = N_B = N$:
$$N = \frac{\left(Z_{1 - \alpha/2} + Z_{1 - \beta}\right)^2 \left[ p_A(1 - p_A) + p_B(1 - p_B) \right]}{(p_B - p_A)^2}$$
where $Z_{0.975} \approx 1.96$ and $Z_{0.80} \approx 0.84$.

---

## 4. Confidence Intervals (Frequentist Interpretation)

A **$100(1 - \alpha)\%$ Confidence Interval** represents an interval estimation of the true parameter $\theta$:
$$\text{CI}_{1-\alpha} = \hat{\theta} \pm Z_{1 - \alpha/2} \cdot \text{SE}(\hat{\theta})$$

### Frequentist Interpretation
The true parameter $\theta$ is fixed, not random. If we were to repeat the experiment 1,000 times under identical conditions and compute a 95% confidence interval each time, approximately 950 of those intervals would enclose the true parameter $\theta$.

### Duality with Hypothesis Testing
A two-sided hypothesis test of $H_0: \theta = \theta_0$ at level $\alpha$ rejects $H_0$ if and only if the $(1 - \alpha)$ confidence interval does not contain $\theta_0$.
