import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets")
os.makedirs(DATA_DIR, exist_ok=True)

np.random.seed(42)

def generate_simulated_ab():
    n_a, n_b = 5000, 5000
    p_a, p_b = 0.120, 0.138  # 15% relative lift
    rev_mean_a, rev_mean_b = 45.0, 48.5
    
    # Variant A
    conv_a = np.random.binomial(1, p_a, n_a)
    rev_a = conv_a * np.random.exponential(rev_mean_a, n_a)
    duration_a = np.random.normal(180, 40, n_a).clip(20, 600)
    
    # Variant B
    conv_b = np.random.binomial(1, p_b, n_b)
    rev_b = conv_b * np.random.exponential(rev_mean_b, n_b)
    duration_b = np.random.normal(195, 42, n_b).clip(20, 600)
    
    df_a = pd.DataFrame({
        "user_id": [f"user_{i:06d}" for i in range(1, n_a + 1)],
        "variant": "control_A",
        "converted": conv_a,
        "revenue": np.round(rev_a, 2),
        "session_duration_sec": np.round(duration_a, 1),
        "device": np.random.choice(["mobile", "desktop", "tablet"], n_a, p=[0.6, 0.3, 0.1])
    })
    
    df_b = pd.DataFrame({
        "user_id": [f"user_{i:06d}" for i in range(n_a + 1, n_a + n_b + 1)],
        "variant": "treatment_B",
        "converted": conv_b,
        "revenue": np.round(rev_b, 2),
        "session_duration_sec": np.round(duration_b, 1),
        "device": np.random.choice(["mobile", "desktop", "tablet"], n_b, p=[0.6, 0.3, 0.1])
    })
    
    df = pd.concat([df_a, df_b]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    df.to_csv(os.path.join(DATA_DIR, "simulated_ab_ecommerce.csv"), index=False)
    print("Saved simulated_ab_ecommerce.csv with", len(df), "rows")

def generate_cookie_cats():
    n = 10000
    versions = np.random.choice(["gate_30", "gate_40"], n, p=[0.5, 0.5])
    # rounds played follow negative binomial / heavy-tailed distribution
    gamerounds = np.random.negative_binomial(1, 0.02, n)
    # Retention 1 day: ~44%
    retention_1 = ((gamerounds > 1) & (np.random.rand(n) < 0.70)).astype(int)
    # Retention 7 day: gate 30 slightly higher retention in original study ~19.0% vs ~18.2%
    p_ret7 = np.where(versions == "gate_30", 0.190, 0.182)
    retention_7 = ((gamerounds > 10) & (np.random.rand(n) < (p_ret7 * 2.5))).astype(int)
    
    df = pd.DataFrame({
        "userid": np.arange(100000, 100000 + n),
        "version": versions,
        "sum_gamerounds": gamerounds,
        "retention_1": retention_1,
        "retention_7": retention_7
    })
    df.to_csv(os.path.join(DATA_DIR, "cookie_cats.csv"), index=False)
    print("Saved cookie_cats.csv with", len(df), "rows")

def generate_card_krueger_did():
    # Card & Krueger (1994) minimum wage study
    # Treatment: New Jersey (raised min wage in April 1992)
    # Control: Eastern Pennsylvania (kept min wage constant)
    n_nj = 320
    n_pa = 80
    
    # Pre-period (Feb 1992)
    fte_nj_pre = np.random.normal(20.4, 8.5, n_nj).clip(5, 60)
    fte_pa_pre = np.random.normal(23.3, 8.8, n_pa).clip(5, 60)
    
    # Post-period (Nov 1992)
    # DiD effect: NJ employment changed by +0.5 to +2.7 FTE relative to PA trend
    # PA experienced mild recession dip of -2.1 FTE, NJ remained stable (+0.5 FTE)
    fte_nj_post = fte_nj_pre + np.random.normal(0.5, 3.5, n_nj)
    fte_pa_post = fte_pa_pre + np.random.normal(-2.1, 3.5, n_pa)
    
    rows = []
    # NJ restaurants
    for i in range(n_nj):
        rest_id = f"NJ_{i+1:03d}"
        chain = np.random.choice(["Burger King", "KFC", "Roy Rogers", "Wendy's"])
        rows.append({"restaurant_id": rest_id, "state": "NJ", "treatment": 1, "post": 0, "fte": round(fte_nj_pre[i], 1), "chain": chain})
        rows.append({"restaurant_id": rest_id, "state": "NJ", "treatment": 1, "post": 1, "fte": round(fte_nj_post[i], 1), "chain": chain})
        
    # PA restaurants
    for i in range(n_pa):
        rest_id = f"PA_{i+1:03d}"
        chain = np.random.choice(["Burger King", "KFC", "Roy Rogers", "Wendy's"])
        rows.append({"restaurant_id": rest_id, "state": "PA", "treatment": 0, "post": 0, "fte": round(fte_pa_pre[i], 1), "chain": chain})
        rows.append({"restaurant_id": rest_id, "state": "PA", "treatment": 0, "post": 1, "fte": round(fte_pa_post[i], 1), "chain": chain})
        
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA_DIR, "card_krueger_did.csv"), index=False)
    print("Saved card_krueger_did.csv with", len(df), "rows")

def generate_california_prop99():
    # California Proposition 99 passed in Nov 1988 (implemented 1989)
    # Per capita cigarette sales (in packs) from 1970 to 2000
    years = list(range(1970, 2001))
    states = ["California"] + [f"Control_State_{i:02d}" for i in range(1, 39)]
    
    rows = []
    for state in states:
        is_ca = (state == "California")
        # Baseline per capita consumption in 1970 ~ 120-140 packs
        base = 125.0 if is_ca else np.random.uniform(105, 145)
        trend = -1.2  # national downward trend
        
        for y in years:
            t = y - 1970
            noise = np.random.normal(0, 2.5)
            # Prop 99 effect starting in 1989
            effect = 0.0
            if is_ca and y >= 1989:
                effect = -2.2 * (y - 1988) # growing divergence
                
            packs = max(15.0, base + trend * t + effect + noise)
            rows.append({
                "state": state,
                "year": y,
                "post_1988": 1 if y >= 1989 else 0,
                "is_california": 1 if is_ca else 0,
                "cigsale": round(packs, 2)
            })
            
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA_DIR, "california_prop99.csv"), index=False)
    print("Saved california_prop99.csv with", len(df), "rows")

def generate_lalonde_nsw():
    # LaLonde (1986) National Supported Work Demonstration
    # Observational / Quasi-experimental dataset for PSM
    n = 2500
    treat = np.random.binomial(1, 0.15, n)
    age = np.random.normal(28, 8, n).clip(17, 55).astype(int)
    educ = np.random.normal(10.5, 2.5, n).clip(4, 16).astype(int)
    black = np.random.binomial(1, np.where(treat==1, 0.80, 0.40))
    hisp = np.random.binomial(1, np.where(treat==1, 0.10, 0.15))
    married = np.random.binomial(1, np.where(treat==1, 0.20, 0.55))
    nodegr = (educ < 12).astype(int)
    
    # Pre-treatment earnings 1974 & 1975
    re74 = np.maximum(0, np.random.normal(3000, 4500, n) * (1 - 0.4 * treat))
    re75 = np.maximum(0, np.random.normal(2800, 4200, n) * (1 - 0.4 * treat))
    
    # True treatment effect is positive ~ $1,800
    re78 = 0.5 * re74 + 0.6 * re75 + 150 * educ + 80 * age + treat * 1790 + np.random.normal(1000, 2500, n)
    re78 = np.maximum(0, re78)
    
    df = pd.DataFrame({
        "treat": treat,
        "age": age,
        "educ": educ,
        "black": black,
        "hisp": hisp,
        "married": married,
        "nodegr": nodegr,
        "re74": np.round(re74, 2),
        "re75": np.round(re75, 2),
        "re78": np.round(re78, 2)
    })
    df.to_csv(os.path.join(DATA_DIR, "lalonde_nsw.csv"), index=False)
    print("Saved lalonde_nsw.csv with", len(df), "rows")

if __name__ == "__main__":
    generate_simulated_ab()
    generate_cookie_cats()
    generate_card_krueger_did()
    generate_california_prop99()
    generate_lalonde_nsw()
    print("All reference datasets generated successfully!")
