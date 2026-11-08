import unittest
import pandas as pd
import numpy as np

from app.services.calculator_service import calculate_proportion_sample_size, calculate_continuous_sample_size
from app.services.frequentist_service import (
    simulate_ab_data,
    run_two_sample_ttest,
    run_chi_square_test,
    run_multiple_testing_adjustment
)
from app.services.sequential_service import (
    analyze_sequential_experiment,
    simulate_peeking_problem,
    get_obrien_fleming_boundaries
)
from app.services.bayesian_service import (
    run_bayesian_conversion_ab,
    run_bayesian_continuous_ab
)
from app.services.causal_service import (
    run_propensity_score_matching,
    run_difference_in_differences,
    run_synthetic_control,
    run_econml_dml
)

class TestCapstonePipeline(unittest.TestCase):

    def test_01_power_calculator_proportion(self):
        res = calculate_proportion_sample_size(
            baseline_rate=0.10,
            mde_relative=0.20,
            power=0.80,
            alpha=0.05
        )
        self.assertIn("sample_size_control", res)
        self.assertGreater(res["sample_size_control"], 1000)
        self.assertEqual(len(res["power_curve"]), 20)
        self.assertIn("plain_english_summary", res)

    def test_02_power_calculator_continuous(self):
        res = calculate_continuous_sample_size(
            baseline_mean=50.0,
            baseline_std=15.0,
            mde_absolute=2.5,
            power=0.80,
            alpha=0.05
        )
        self.assertIn("sample_size_control", res)
        self.assertGreater(res["sample_size_control"], 100)
        self.assertIn("cohens_d", res)

    def test_03_frequentist_simulation_and_tests(self):
        sim = simulate_ab_data(metric_type="conversion", n_a=1000, n_b=1000, baseline_val=0.10, relative_lift=0.30)
        self.assertEqual(sim["summary"]["control_count"], 1000)

        # Chi square
        chi = run_chi_square_test(
            control_conversions=100,
            control_total=1000,
            treatment_conversions=140,
            treatment_total=1000
        )
        self.assertIn("chi2_p_value", chi)
        self.assertTrue(chi["statistically_significant"])

        # t-test
        s_a = [10.0, 11.0, 12.0, 9.5, 10.5] * 20
        s_b = [14.0, 15.0, 13.5, 14.2, 15.1] * 20
        tt = run_two_sample_ttest(s_a, s_b)
        self.assertTrue(tt["statistically_significant"])
        self.assertLess(tt["p_value"], 0.001)

    def test_04_multiple_testing(self):
        tests = [
            {"metric_name": "Metric 1", "p_value": 0.005},
            {"metric_name": "Metric 2", "p_value": 0.035},
            {"metric_name": "Metric 3", "p_value": 0.048},
            {"metric_name": "Metric 4", "p_value": 0.150},
            {"metric_name": "Metric 5", "p_value": 0.850}
        ]
        res = run_multiple_testing_adjustment(tests, alpha=0.05)
        self.assertEqual(res["num_tests"], 5)
        self.assertEqual(res["raw_discoveries"], 3)
        self.assertLessEqual(res["bonferroni_discoveries"], res["raw_discoveries"])

    def test_05_sequential_testing(self):
        looks = [
            {"look": 1, "n_a": 500, "n_b": 500, "conv_a": 50, "conv_b": 52},
            {"look": 2, "n_a": 1000, "n_b": 1000, "conv_a": 98, "conv_b": 135},
            {"look": 3, "n_a": 1500, "n_b": 1500, "conv_a": 150, "conv_b": 220}
        ]
        seq = analyze_sequential_experiment(looks, alpha=0.05, method="obrien_fleming")
        self.assertEqual(len(seq["trajectory"]), 3)
        self.assertTrue(seq["stopped_early"])

    def test_06_bayesian_conversion(self):
        res = run_bayesian_conversion_ab(
            control_conversions=100,
            control_total=1000,
            treatment_conversions=140,
            treatment_total=1000,
            draws=500,
            tune=200
        )
        self.assertGreater(res["probability_b_beats_a"], 0.95)
        self.assertIn("ci_95_diff_lower", res)
        self.assertIn("plain_english_summary", res)

    def test_07_causal_observational(self):
        # 1. PSM with LaLonde
        df_lalonde = pd.read_csv("C:/Users/Soumlya/.gemini/antigravity/scratch/capstones_industry_projects/datasets/lalonde_nsw.csv")
        psm_res = run_propensity_score_matching(
            df=df_lalonde,
            treatment_col="treat",
            outcome_col="re78",
            confounder_cols=["age", "educ", "black", "hisp", "married"]
        )
        self.assertIn("estimated_ate", psm_res)
        self.assertIn("covariate_balance", psm_res)

        # 2. DiD with Card & Krueger
        df_did = pd.read_csv("C:/Users/Soumlya/.gemini/antigravity/scratch/capstones_industry_projects/datasets/card_krueger_did.csv")
        did_res = run_difference_in_differences(
            df=df_did,
            outcome_col="fte",
            treatment_col="treatment",
            post_col="post"
        )
        self.assertIn("did_estimate_delta", did_res)
        self.assertIn("means", did_res)

        # 3. Synthetic Control with California Prop 99
        df_prop99 = pd.read_csv("C:/Users/Soumlya/.gemini/antigravity/scratch/capstones_industry_projects/datasets/california_prop99.csv")
        sc_res = run_synthetic_control(
            df=df_prop99,
            unit_col="state",
            time_col="year",
            outcome_col="cigsale",
            treated_unit="California",
            treatment_year=1989
        )
        self.assertIn("pre_treatment_rmspe", sc_res)
        self.assertGreater(len(sc_res["donor_weights"]), 0)

        # 4. EconML LinearDML
        dml_res = run_econml_dml(
            df=df_lalonde,
            treatment_col="treat",
            outcome_col="re78",
            feature_cols=["age", "educ"],
            confounder_cols=["black", "hisp", "married"]
        )
        self.assertIn("overall_ate", dml_res)
        self.assertIn("cate_distribution", dml_res)

if __name__ == "__main__":
    unittest.main()
