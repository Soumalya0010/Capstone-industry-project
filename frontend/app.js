// Capstones & Industry Projects - Data Science Experimentation & Causal Inference Platform
// Built with React 18 (No Vite, purely vanilla React + Babel CDN)

const { useState, useEffect, useMemo } = React;

const API_BASE = "";

// Simple SVG Line Chart Component for clean visualization without external chart libs
function SvgLineChart({ data, xKey, lines, title, yDomain, referenceLine }) {
  if (!data || data.length === 0) return null;

  const width = 650;
  const height = 220;
  const padding = { top: 25, right: 30, bottom: 35, left: 55 };

  const xVals = data.map(d => d[xKey]);
  const allYVals = data.flatMap(d => lines.map(l => d[l.key])).filter(v => v !== undefined && !isNaN(v));
  
  const minY = yDomain ? yDomain[0] : Math.min(...allYVals);
  const maxY = yDomain ? yDomain[1] : Math.max(...allYVals);
  const ySpan = maxY - minY || 1;

  const getX = (idx) => padding.left + (idx / (data.length - 1)) * (width - padding.left - padding.right);
  const getY = (val) => height - padding.bottom - ((val - minY) / ySpan) * (height - padding.top - padding.bottom);

  return (
    <div className="chart-container">
      <div className="chart-header">
        <span>{title}</span>
        <div style={{ display: 'flex', gap: '1rem' }}>
          {lines.map(l => (
            <span key={l.key} style={{ color: l.color, fontSize: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
              <span style={{ display: 'inline-block', width: 10, height: 3, backgroundColor: l.color }}></span>
              {l.label}
            </span>
          ))}
        </div>
      </div>
      <svg className="svg-chart" viewBox={`0 0 ${width} ${height}`}>
        {/* Grid lines */}
        {[0, 0.25, 0.5, 0.75, 1].map((pct, i) => {
          const val = minY + pct * ySpan;
          const y = getY(val);
          return (
            <g key={i}>
              <line x1={padding.left} y1={y} x2={width - padding.right} y2={y} stroke="#334155" strokeDasharray="3 3" />
              <text x={padding.left - 8} y={y + 4} fill="#94a3b8" fontSize="10" textAnchor="end">
                {val.toFixed(2)}
              </text>
            </g>
          );
        })}

        {/* Reference horizontal line (e.g. Power 0.80) */}
        {referenceLine && referenceLine.val >= minY && referenceLine.val <= maxY && (
          <g>
            <line 
              x1={padding.left} 
              y1={getY(referenceLine.val)} 
              x2={width - padding.right} 
              y2={getY(referenceLine.val)} 
              stroke={referenceLine.color || "#f59e0b"} 
              strokeWidth="1.5" 
              strokeDasharray="4 4" 
            />
            <text x={width - padding.right} y={getY(referenceLine.val) - 4} fill={referenceLine.color || "#f59e0b"} fontSize="10" textAnchor="end">
              {referenceLine.label}
            </text>
          </g>
        )}

        {/* Data Lines */}
        {lines.map(l => {
          const points = data
            .map((d, idx) => {
              const val = d[l.key];
              if (val === undefined || isNaN(val)) return null;
              return `${getX(idx)},${getY(val)}`;
            })
            .filter(Boolean)
            .join(" ");

          return (
            <g key={l.key}>
              <polyline
                fill="none"
                stroke={l.color}
                strokeWidth={l.dashed ? "2" : "2.5"}
                strokeDasharray={l.dashed ? "4 4" : "none"}
                points={points}
              />
              {data.map((d, idx) => {
                const val = d[l.key];
                if (val === undefined || isNaN(val)) return null;
                return (
                  <circle
                    key={idx}
                    cx={getX(idx)}
                    cy={getY(val)}
                    r="3.5"
                    fill={l.color}
                  >
                    <title>{`${d[xKey]}: ${val}`}</title>
                  </circle>
                );
              })}
            </g>
          );
        })}

        {/* X Axis Labels */}
        {data.map((d, idx) => {
          if (idx % Math.ceil(data.length / 6) !== 0 && idx !== data.length - 1) return null;
          return (
            <text
              key={idx}
              x={getX(idx)}
              y={height - 10}
              fill="#94a3b8"
              fontSize="10"
              textAnchor="middle"
            >
              {String(d[xKey])}
            </text>
          );
        })}
      </svg>
    </div>
  );
}

// Main Application Component
function App() {
  const [activeTab, setActiveTab] = useState("calculator");
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState("simulated_ab_ecommerce.csv");
  const [datasetPreview, setDatasetPreview] = useState(null);
  const [loading, setLoading] = useState(false);

  // Tab 1: Calculator State
  const [calcMetric, setCalcMetric] = useState("proportion");
  const [calcForm, setCalcForm] = useState({
    baseline_rate: 0.10,
    mde_relative: 0.15,
    baseline_mean: 45.0,
    baseline_std: 15.0,
    mde_absolute: 2.0,
    power: 0.80,
    alpha: 0.05,
    daily_traffic: 5000
  });
  const [calcResult, setCalcResult] = useState(null);

  // Tab 2: Frequentist State
  const [freqForm, setFreqForm] = useState({
    metric_type: "conversion",
    n_a: 2500,
    n_b: 2500,
    baseline_val: 0.12,
    relative_lift: 0.15,
    std_dev: 20.0
  });
  const [freqSimResult, setFreqSimResult] = useState(null);
  const [freqTestResult, setFreqTestResult] = useState(null);
  const [multiTestItems, setMultiTestItems] = useState([
    { metric_name: "Click-through Rate", p_value: 0.008, raw_diff: 0.021 },
    { metric_name: "Add-to-Cart Rate", p_value: 0.034, raw_diff: 0.015 },
    { metric_name: "Checkout Completion", p_value: 0.046, raw_diff: 0.012 },
    { metric_name: "Average Order Value", p_value: 0.120, raw_diff: 1.45 },
    { metric_name: "Customer Retention", p_value: 0.450, raw_diff: -0.005 }
  ]);
  const [multiTestResult, setMultiTestResult] = useState(null);

  // Tab 3: Sequential State
  const [seqLooks, setSeqLooks] = useState([
    { look: 1, n_a: 500, n_b: 500, conv_a: 50, conv_b: 54 },
    { look: 2, n_a: 1000, n_b: 1000, conv_a: 102, conv_b: 125 },
    { look: 3, n_a: 1500, n_b: 1500, conv_a: 155, conv_b: 205 },
    { look: 4, n_a: 2000, n_b: 2000, conv_a: 210, conv_b: 285 },
    { look: 5, n_a: 2500, n_b: 2500, conv_a: 260, conv_b: 360 }
  ]);
  const [seqResult, setSeqResult] = useState(null);
  const [peekingSimResult, setPeekingSimResult] = useState(null);

  // Tab 4: Bayesian State
  const [bayesForm, setBayesForm] = useState({
    ctrl_conv: 120,
    ctrl_total: 1000,
    treat_conv: 155,
    treat_total: 1000,
    prior_alpha_a: 1.0,
    prior_beta_a: 1.0,
    prior_alpha_b: 1.0,
    prior_beta_b: 1.0
  });
  const [bayesResult, setBayesResult] = useState(null);

  // Tab 5: Observational State
  const [obsSubTab, setObsSubTab] = useState("psm");
  const [psmResult, setPsmResult] = useState(null);
  const [didResult, setDidResult] = useState(null);
  const [synthResult, setSynthResult] = useState(null);
  const [econmlResult, setEconmlResult] = useState(null);

  // Load benchmark datasets on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/datasets/list`)
      .then(res => res.json())
      .then(data => {
        if (data.datasets) setDatasets(data.datasets);
      })
      .catch(err => console.error("Could not load datasets:", err));
  }, []);

  // Fetch dataset preview when selected dataset changes
  useEffect(() => {
    if (selectedDataset) {
      fetch(`${API_BASE}/api/datasets/preview/${selectedDataset}`)
        .then(res => res.json())
        .then(data => setDatasetPreview(data))
        .catch(err => console.error("Error previewing dataset:", err));
    }
  }, [selectedDataset]);

  // Initial power calculation on mount
  useEffect(() => {
    handleCalculatePower();
  }, [calcMetric]);

  // Handle Calculator
  const handleCalculatePower = async () => {
    setLoading(true);
    try {
      let endpoint = `${API_BASE}/api/calculator/${calcMetric}`;
      let payload = calcMetric === "proportion" ? {
        baseline_rate: parseFloat(calcForm.baseline_rate),
        mde_relative: parseFloat(calcForm.mde_relative),
        power: parseFloat(calcForm.power),
        alpha: parseFloat(calcForm.alpha),
        daily_traffic: parseInt(calcForm.daily_traffic) || null
      } : {
        baseline_mean: parseFloat(calcForm.baseline_mean),
        baseline_std: parseFloat(calcForm.baseline_std),
        mde_absolute: parseFloat(calcForm.mde_absolute),
        power: parseFloat(calcForm.power),
        alpha: parseFloat(calcForm.alpha),
        daily_traffic: parseInt(calcForm.daily_traffic) || null
      };

      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      setCalcResult(data);
    } catch (e) {
      alert("Error calculating sample size: " + e.message);
    } finally {
      setLoading(false);
    }
  };

  // Handle Frequentist Simulation & Analysis
  const handleSimulateFreq = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/ab/simulate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(freqForm)
      });
      const data = await res.json();
      setFreqSimResult(data);

      // Immediately run t-test or chi-square on the simulated data
      if (freqForm.metric_type === "conversion") {
        const chiRes = await fetch(`${API_BASE}/api/ab/chisquare`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            control_conversions: data.summary.control_conversions,
            control_total: data.summary.control_count,
            treatment_conversions: data.summary.treatment_conversions,
            treatment_total: data.summary.treatment_count,
            alpha: 0.05
          })
        });
        setFreqTestResult(await chiRes.json());
      } else {
        const tRes = await fetch(`${API_BASE}/api/ab/ttest`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sample_a: data.raw_a,
            sample_b: data.raw_b,
            alpha: 0.05
          })
        });
        setFreqTestResult(await tRes.json());
      }
    } catch (e) {
      alert("Simulation error: " + e.message);
    } finally {
      setLoading(false);
    }
  };

  // Handle Multiple Testing
  const handleRunMultipleTesting = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/ab/multiple-testing`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ test_items: multiTestItems, alpha: 0.05 })
      });
      setMultiTestResult(await res.json());
    } catch (e) {
      alert("Multiple testing error: " + e.message);
    } finally {
      setLoading(false);
    }
  };

  // Handle Sequential Analysis
  const handleRunSequential = async (method = "obrien_fleming") => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/sequential/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ looks_data: seqLooks, alpha: 0.05, method })
      });
      setSeqResult(await res.json());
    } catch (e) {
      alert("Sequential analysis error: " + e.message);
    } finally {
      setLoading(false);
    }
  };

  // Handle Peeking Simulator
  const handleSimulatePeeking = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/sequential/simulate-peeking`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ n_sims: 500, n_per_look: 200, n_looks: 10, alpha: 0.05 })
      });
      setPeekingSimResult(await res.json());
    } catch (e) {
      alert("Peeking simulation error: " + e.message);
    } finally {
      setLoading(false);
    }
  };

  // Handle Bayesian PyMC
  const handleRunBayesian = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/bayesian/conversion`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          control_conversions: parseInt(bayesForm.ctrl_conv),
          control_total: parseInt(bayesForm.ctrl_total),
          treatment_conversions: parseInt(bayesForm.treat_conv),
          treatment_total: parseInt(bayesForm.treat_total),
          prior_alpha_a: parseFloat(bayesForm.prior_alpha_a),
          prior_beta_a: parseFloat(bayesForm.prior_beta_a),
          prior_alpha_b: parseFloat(bayesForm.prior_alpha_b),
          prior_beta_b: parseFloat(bayesForm.prior_beta_b),
          draws: 1500,
          tune: 400
        })
      });
      setBayesResult(await res.json());
    } catch (e) {
      alert("Bayesian analysis error: " + e.message);
    } finally {
      setLoading(false);
    }
  };

  // Handle Observational Causal Inference
  const handleRunObservational = async (type) => {
    setLoading(true);
    try {
      if (type === "psm") {
        const res = await fetch(`${API_BASE}/api/observational/psm`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            dataset_name: "lalonde_nsw.csv",
            treatment_col: "treat",
            outcome_col: "re78",
            confounder_cols: ["age", "educ", "black", "hisp", "married"]
          })
        });
        setPsmResult(await res.json());
      } else if (type === "did") {
        const res = await fetch(`${API_BASE}/api/observational/did`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            dataset_name: "card_krueger_did.csv",
            outcome_col: "fte",
            treatment_col: "treatment",
            post_col: "post"
          })
        });
        setDidResult(await res.json());
      } else if (type === "synth") {
        const res = await fetch(`${API_BASE}/api/observational/synthetic-control`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            dataset_name: "california_prop99.csv",
            unit_col: "state",
            time_col: "year",
            outcome_col: "cigsale",
            treated_unit: "California",
            treatment_year: 1989
          })
        });
        setSynthResult(await res.json());
      } else if (type === "econml") {
        const res = await fetch(`${API_BASE}/api/observational/econml-cate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            dataset_name: "lalonde_nsw.csv",
            treatment_col: "treat",
            outcome_col: "re78",
            feature_cols: ["age", "educ"],
            confounder_cols: ["black", "hisp", "married"]
          })
        });
        setEconmlResult(await res.json());
      }
    } catch (e) {
      alert("Causal estimation error: " + e.message);
    } finally {
      setLoading(false);
    }
  };

  // Upload custom CSV file
  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/datasets/upload`, {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      alert(`Uploaded ${data.filename} (${data.total_rows} rows).`);
      setSelectedDataset(data.filename);
    } catch (err) {
      alert("Upload failed: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="header-title-area">
          <h1>
            Capstones & Industry Projects
            <span className="header-badge">Data Science</span>
          </h1>
          <p className="header-subtitle">
            A/B Testing, Sequential Peeking, Bayesian Inference (PyMC) & Causal Inference (DoWhy, EconML)
          </p>
        </div>
        <div className="header-tech-stack">
          <span className="tech-tag">Statsmodels</span>
          <span className="tech-tag">DoWhy</span>
          <span className="tech-tag">EconML</span>
          <span className="tech-tag">PyMC</span>
          <span className="tech-tag">FastAPI</span>
          <span className="tech-tag">React</span>
        </div>
      </header>

      {/* Benchmark Datasets Selector Banner */}
      <div className="dataset-banner">
        <div className="dataset-picker-group">
          <span className="dataset-label">Active Dataset:</span>
          <select
            className="dataset-select"
            value={selectedDataset}
            onChange={(e) => setSelectedDataset(e.target.value)}
          >
            {datasets.map(ds => (
              <option key={ds.filename} value={ds.filename}>
                {ds.title} ({ds.total_rows.toLocaleString()} rows)
              </option>
            ))}
          </select>
          <label className="btn-upload">
            Upload Custom CSV
            <input type="file" accept=".csv" style={{ display: 'none' }} onChange={handleFileUpload} />
          </label>
        </div>
        {datasetPreview && (
          <div className="active-dataset-info">
            <span>Columns: {datasetPreview.columns.join(", ")}</span>
          </div>
        )}
      </div>

      {/* Navigation Tabs */}
      <nav className="nav-tabs">
        <button
          className={`tab-btn ${activeTab === 'calculator' ? 'active' : ''}`}
          onClick={() => setActiveTab('calculator')}
        >
          ⚡ 1. Power & Sample Size
        </button>
        <button
          className={`tab-btn ${activeTab === 'frequentist' ? 'active' : ''}`}
          onClick={() => setActiveTab('frequentist')}
        >
          📊 2. Frequentist & Multiple Testing
        </button>
        <button
          className={`tab-btn ${activeTab === 'sequential' ? 'active' : ''}`}
          onClick={() => setActiveTab('sequential')}
        >
          ⏱️ 3. Sequential Testing
        </button>
        <button
          className={`tab-btn ${activeTab === 'bayesian' ? 'active' : ''}`}
          onClick={() => setActiveTab('bayesian')}
        >
          🎲 4. Bayesian A/B (PyMC)
        </button>
        <button
          className={`tab-btn ${activeTab === 'observational' ? 'active' : ''}`}
          onClick={() => setActiveTab('observational')}
        >
          🔬 5. Observational Causal Inference
        </button>
        <button
          className={`tab-btn ${activeTab === 'foundations' ? 'active' : ''}`}
          onClick={() => setActiveTab('foundations')}
        >
          📖 6. Foundations Guide
        </button>
      </nav>

      {/* Main Content Area */}
      <main className="main-content">
        {/* ================= TAB 1: POWER & SAMPLE SIZE CALCULATOR ================= */}
        {activeTab === 'calculator' && (
          <div className="panel-grid">
            {/* Input Form */}
            <div className="card">
              <h2 className="card-title">Power Calculator Parameters</h2>
              <p className="card-desc">Calculate required sample size and generate power curves using Statsmodels.</p>
              
              <div className="form-group">
                <label className="form-label">Metric Type</label>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <button
                    className={`sub-btn ${calcMetric === 'proportion' ? 'active' : ''}`}
                    onClick={() => setCalcMetric('proportion')}
                  >
                    Proportions (Conversion Rate)
                  </button>
                  <button
                    className={`sub-btn ${calcMetric === 'continuous' ? 'active' : ''}`}
                    onClick={() => setCalcMetric('continuous')}
                  >
                    Continuous (Revenue / Time)
                  </button>
                </div>
              </div>

              {calcMetric === 'proportion' ? (
                <>
                  <div className="form-group">
                    <label className="form-label">Baseline Conversion Rate (e.g. 0.10 for 10%)</label>
                    <input
                      type="number"
                      step="0.01"
                      className="form-input"
                      value={calcForm.baseline_rate}
                      onChange={e => setCalcForm({ ...calcForm, baseline_rate: e.target.value })}
                    />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Minimum Detectable Effect (Relative Lift, e.g. 0.15 for +15%)</label>
                    <input
                      type="number"
                      step="0.01"
                      className="form-input"
                      value={calcForm.mde_relative}
                      onChange={e => setCalcForm({ ...calcForm, mde_relative: e.target.value })}
                    />
                  </div>
                </>
              ) : (
                <>
                  <div className="form-group">
                    <label className="form-label">Baseline Mean</label>
                    <input
                      type="number"
                      className="form-input"
                      value={calcForm.baseline_mean}
                      onChange={e => setCalcForm({ ...calcForm, baseline_mean: e.target.value })}
                    />
                  </div>
                  <div className="form-row">
                    <div className="form-group">
                      <label className="form-label">Baseline Std Dev</label>
                      <input
                        type="number"
                        className="form-input"
                        value={calcForm.baseline_std}
                        onChange={e => setCalcForm({ ...calcForm, baseline_std: e.target.value })}
                      />
                    </div>
                    <div className="form-group">
                      <label className="form-label">Absolute MDE</label>
                      <input
                        type="number"
                        className="form-input"
                        value={calcForm.mde_absolute}
                        onChange={e => setCalcForm({ ...calcForm, mde_absolute: e.target.value })}
                      />
                    </div>
                  </div>
                </>
              )}

              <div className="form-row">
                <div className="form-group">
                  <label className="form-label">Target Power (1 - Beta)</label>
                  <select
                    className="form-select"
                    value={calcForm.power}
                    onChange={e => setCalcForm({ ...calcForm, power: parseFloat(e.target.value) })}
                  >
                    <option value={0.80}>80% (Standard)</option>
                    <option value={0.90}>90% (High Confidence)</option>
                    <option value={0.95}>95% (Critical)</option>
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Alpha (Type I Error)</label>
                  <select
                    className="form-select"
                    value={calcForm.alpha}
                    onChange={e => setCalcForm({ ...calcForm, alpha: parseFloat(e.target.value) })}
                  >
                    <option value={0.05}>0.05 (Two-sided 95%)</option>
                    <option value={0.01}>0.01 (Two-sided 99%)</option>
                  </select>
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">Daily Traffic / Sample Rate (Optional)</label>
                <input
                  type="number"
                  className="form-input"
                  value={calcForm.daily_traffic}
                  onChange={e => setCalcForm({ ...calcForm, daily_traffic: e.target.value })}
                />
              </div>

              <button className="btn-primary" onClick={handleCalculatePower} disabled={loading}>
                {loading ? "Calculating..." : "Compute Sample Size & Power"}
              </button>
            </div>

            {/* Results Panel */}
            <div>
              {calcResult && (
                <>
                  <div className="conclusion-card">
                    <div className="conclusion-header">
                      <span className="conclusion-title">Sample Size Recommendation</span>
                      <span className="conclusion-badge">Statsmodels Engine</span>
                    </div>
                    <p className="conclusion-text">{calcResult.plain_english_summary}</p>
                  </div>

                  <div className="metrics-grid">
                    <div className="metric-stat-box">
                      <div className="stat-label">Sample Size per Group</div>
                      <div className="stat-value">{calcResult.sample_size_control.toLocaleString()}</div>
                      <div className="stat-subtext">Control (N_A) & Treatment (N_B)</div>
                    </div>
                    <div className="metric-stat-box">
                      <div className="stat-label">Total Sample Required</div>
                      <div className="stat-value">{calcResult.total_sample_size.toLocaleString()}</div>
                      <div className="stat-subtext">Across both variants</div>
                    </div>
                    <div className="metric-stat-box">
                      <div className="stat-label">Standardized Effect</div>
                      <div className="stat-value">
                        {calcMetric === 'proportion' ? `h = ${calcResult.cohens_h}` : `d = ${calcResult.cohens_d}`}
                      </div>
                      <div className="stat-subtext">{calcMetric === 'proportion' ? "Cohen's h" : "Cohen's d"}</div>
                    </div>
                    <div className="metric-stat-box">
                      <div className="stat-label">Estimated Test Duration</div>
                      <div className="stat-value">{calcResult.estimated_days ? `${calcResult.estimated_days} Days` : "N/A"}</div>
                      <div className="stat-subtext">At current daily traffic</div>
                    </div>
                  </div>

                  <SvgLineChart
                    data={calcResult.power_curve}
                    xKey="sample_size_per_variant"
                    lines={[
                      { key: "power", label: "Statistical Power", color: "#3b82f6" }
                    ]}
                    title="Statistical Power Curve (Power vs Sample Size per Variant)"
                    yDomain={[0, 1]}
                    referenceLine={{ val: calcResult.power, label: `Target Power (${calcResult.power * 100}%)`, color: "#10b981" }}
                  />
                </>
              )}
            </div>
          </div>
        )}

        {/* ================= TAB 2: FREQUENTIST & MULTIPLE TESTING ================= */}
        {activeTab === 'frequentist' && (
          <div>
            <div className="panel-grid" style={{ marginBottom: '2rem' }}>
              <div className="card">
                <h2 className="card-title">Simulate A/B Experiment</h2>
                <p className="card-desc">Generate synthetic experimental datasets and run hypothesis testing.</p>

                <div className="form-group">
                  <label className="form-label">Metric Type</label>
                  <select
                    className="form-select"
                    value={freqForm.metric_type}
                    onChange={e => setFreqForm({ ...freqForm, metric_type: e.target.value })}
                  >
                    <option value="conversion">Conversion Rate (0/1 Bernoulli)</option>
                    <option value="continuous">Continuous Metric (Revenue / Session Length)</option>
                  </select>
                </div>

                <div className="form-row">
                  <div className="form-group">
                    <label className="form-label">Control Size (N_A)</label>
                    <input
                      type="number"
                      className="form-input"
                      value={freqForm.n_a}
                      onChange={e => setFreqForm({ ...freqForm, n_a: parseInt(e.target.value) })}
                    />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Treatment Size (N_B)</label>
                    <input
                      type="number"
                      className="form-input"
                      value={freqForm.n_b}
                      onChange={e => setFreqForm({ ...freqForm, n_b: parseInt(e.target.value) })}
                    />
                  </div>
                </div>

                <div className="form-row">
                  <div className="form-group">
                    <label className="form-label">Baseline Value</label>
                    <input
                      type="number"
                      step="0.01"
                      className="form-input"
                      value={freqForm.baseline_val}
                      onChange={e => setFreqForm({ ...freqForm, baseline_val: parseFloat(e.target.value) })}
                    />
                  </div>
                  <div className="form-group">
                    <label className="form-label">True Relative Lift</label>
                    <input
                      type="number"
                      step="0.01"
                      className="form-input"
                      value={freqForm.relative_lift}
                      onChange={e => setFreqForm({ ...freqForm, relative_lift: parseFloat(e.target.value) })}
                    />
                  </div>
                </div>

                <button className="btn-primary" onClick={handleSimulateFreq} disabled={loading}>
                  {loading ? "Simulating..." : "Simulate & Run Test"}
                </button>
              </div>

              <div>
                {freqTestResult ? (
                  <>
                    <div className="conclusion-card">
                      <div className="conclusion-header">
                        <span className="conclusion-title">{freqTestResult.test_name}</span>
                        <span className={`conclusion-badge ${freqTestResult.statistically_significant ? '' : 'not-significant'}`}>
                          {freqTestResult.decision}
                        </span>
                      </div>
                      <p className="conclusion-text">{freqTestResult.plain_english_summary}</p>
                    </div>

                    <div className="metrics-grid">
                      <div className="metric-stat-box">
                        <div className="stat-label">p-Value</div>
                        <div className="stat-value">
                          {freqTestResult.p_value !== undefined ? freqTestResult.p_value.toFixed(4) : freqTestResult.chi2_p_value.toFixed(4)}
                        </div>
                        <div className="stat-subtext">Alpha = 0.05 threshold</div>
                      </div>
                      <div className="metric-stat-box">
                        <div className="stat-label">Observed Lift</div>
                        <div className="stat-value">
                          {freqTestResult.relative_lift_pct ? `${freqTestResult.relative_lift_pct > 0 ? '+' : ''}${freqTestResult.relative_lift_pct}%` : `${freqTestResult.relative_diff_pct > 0 ? '+' : ''}${freqTestResult.relative_diff_pct}%`}
                        </div>
                        <div className="stat-subtext">Relative percentage change</div>
                      </div>
                      <div className="metric-stat-box">
                        <div className="stat-label">Test Statistic</div>
                        <div className="stat-value">
                          {freqTestResult.t_statistic ? `t = ${freqTestResult.t_statistic}` : `Chi2 = ${freqTestResult.chi2_statistic}`}
                        </div>
                        <div className="stat-subtext">Standardized score</div>
                      </div>
                      <div className="metric-stat-box">
                        <div className="stat-label">95% Confidence Interval</div>
                        <div className="stat-value" style={{ fontSize: '1.1rem' }}>
                          [{freqTestResult.ci_lower || freqTestResult.ci_lower_diff}, {freqTestResult.ci_upper || freqTestResult.ci_upper_diff}]
                        </div>
                        <div className="stat-subtext">Difference in metrics</div>
                      </div>
                    </div>

                    {freqTestResult.distribution_chart && (
                      <SvgLineChart
                        data={freqTestResult.distribution_chart}
                        xKey="bin"
                        lines={[
                          { key: "density_control", label: "Control A", color: "#94a3b8" },
                          { key: "density_treatment", label: "Treatment B", color: "#3b82f6" }
                        ]}
                        title="Sample Metric Distribution Comparison"
                      />
                    )}
                  </>
                ) : (
                  <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
                    <p style={{ color: 'var(--text-secondary)' }}>Click "Simulate & Run Test" or load data to inspect t-test / Chi-square findings.</p>
                  </div>
                )}
              </div>
            </div>

            {/* Multiple Testing Correction Section */}
            <div className="card">
              <h2 className="card-title">Multiple Testing Correction (Bonferroni & Benjamini-Hochberg FDR)</h2>
              <p className="card-desc">
                When evaluating multiple metrics simultaneously, the Family-Wise Error Rate (FWER) inflates dramatically.
                Correct p-values using statsmodels.
              </p>

              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Metrics Tested in Experiment ({multiTestItems.length} metrics):</span>
                <button className="btn-secondary" onClick={handleRunMultipleTesting} disabled={loading}>
                  Run Corrections
                </button>
              </div>

              {multiTestResult && (
                <div className="conclusion-card" style={{ marginBottom: '1rem' }}>
                  <p className="conclusion-text">{multiTestResult.plain_english_summary}</p>
                </div>
              )}

              <div className="table-container">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Metric Name</th>
                      <th>Raw p-Value</th>
                      <th>Unadjusted Significant?</th>
                      <th>Bonferroni p-Value</th>
                      <th>Bonferroni Rejected?</th>
                      <th>Benjamini-Hochberg (FDR) p-Value</th>
                      <th>FDR Rejected?</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(multiTestResult ? multiTestResult.results : multiTestItems).map((item, i) => (
                      <tr key={i}>
                        <td><strong>{item.metric_name}</strong></td>
                        <td>{item.raw_p_value !== undefined ? item.raw_p_value : item.p_value}</td>
                        <td>
                          <span style={{ color: (item.raw_significant !== undefined ? item.raw_significant : item.p_value < 0.05) ? '#10b981' : '#94a3b8' }}>
                            {(item.raw_significant !== undefined ? item.raw_significant : item.p_value < 0.05) ? 'Yes (p < 0.05)' : 'No'}
                          </span>
                        </td>
                        <td>{item.bonferroni_p_value !== undefined ? item.bonferroni_p_value : "—"}</td>
                        <td>
                          {item.bonferroni_significant !== undefined ? (
                            <span style={{ color: item.bonferroni_significant ? '#10b981' : '#ef4444', fontWeight: 600 }}>
                              {item.bonferroni_significant ? 'Pass' : 'Rejected'}
                            </span>
                          ) : "—"}
                        </td>
                        <td>{item.benjamini_hochberg_p_value !== undefined ? item.benjamini_hochberg_p_value : "—"}</td>
                        <td>
                          {item.benjamini_hochberg_significant !== undefined ? (
                            <span style={{ color: item.benjamini_hochberg_significant ? '#10b981' : '#ef4444', fontWeight: 600 }}>
                              {item.benjamini_hochberg_significant ? 'Pass' : 'Rejected'}
                            </span>
                          ) : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ================= TAB 3: SEQUENTIAL TESTING ================= */}
        {activeTab === 'sequential' && (
          <div>
            <div className="panel-grid" style={{ marginBottom: '2rem' }}>
              <div className="card">
                <h2 className="card-title">Safe Interim Peeking</h2>
                <p className="card-desc">
                  Peeking at standard A/B test results daily inflates Type I error. 
                  Apply O'Brien-Fleming or Pocock alpha spending boundaries to peek safely.
                </p>

                <div className="form-group">
                  <label className="form-label">Sequential Boundary Method</label>
                  <div style={{ display: 'flex', gap: '0.5rem' }}>
                    <button className="sub-btn active" onClick={() => handleRunSequential("obrien_fleming")}>
                      O'Brien-Fleming (Conservative Early)
                    </button>
                    <button className="sub-btn" onClick={() => handleRunSequential("pocock")}>
                      Pocock (Constant Boundary)
                    </button>
                  </div>
                </div>

                <div style={{ marginTop: '1.5rem' }}>
                  <h3 style={{ fontSize: '0.9rem', color: '#fff', marginBottom: '0.5rem' }}>Simulate The Peeking Problem</h3>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
                    Simulate 500 A/A tests under null to compare unadjusted vs sequential false positive rates.
                  </p>
                  <button className="btn-secondary" style={{ width: '100%' }} onClick={handleSimulatePeeking} disabled={loading}>
                    Simulate Peeking Risk (500 Runs)
                  </button>
                </div>

                {peekingSimResult && (
                  <div style={{ marginTop: '1rem', background: '#0f172a', padding: '1rem', borderRadius: 8, border: '1px solid var(--border-color)' }}>
                    <p style={{ fontSize: '0.85rem', color: '#e2e8f0', lineHeight: 1.5 }}>
                      {peekingSimResult.plain_english_summary}
                    </p>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.75rem' }}>
                      <span style={{ fontSize: '0.8rem', color: '#ef4444' }}>
                        Naive Peeking False Positives: <strong>{peekingSimResult.naive_peeking_false_positive_rate_pct}%</strong>
                      </span>
                      <span style={{ fontSize: '0.8rem', color: '#10b981' }}>
                        O'Brien-Fleming: <strong>{peekingSimResult.sequential_boundary_false_positive_rate_pct}%</strong>
                      </span>
                    </div>
                  </div>
                )}
              </div>

              <div>
                {seqResult ? (
                  <>
                    <div className="conclusion-card">
                      <div className="conclusion-header">
                        <span className="conclusion-title">Sequential Decision Recommendation</span>
                        <span className={`conclusion-badge ${seqResult.stopped_early ? '' : 'not-significant'}`}>
                          {seqResult.stopped_early ? "Safe Early Stop" : "Continue"}
                        </span>
                      </div>
                      <p className="conclusion-text">{seqResult.plain_english_summary}</p>
                    </div>

                    <SvgLineChart
                      data={seqResult.trajectory}
                      xKey="look"
                      lines={[
                        { key: "z_statistic", label: "Observed Z-Statistic", color: "#3b82f6" },
                        { key: "obrien_fleming_boundary", label: "O'Brien-Fleming Critical Boundary", color: "#ef4444", dashed: true },
                        { key: "pocock_boundary", label: "Pocock Boundary", color: "#f59e0b", dashed: true }
                      ]}
                      title="Z-Statistic Trajectory vs Sequential Stopping Boundaries"
                      referenceLine={{ val: 1.96, label: "Naive Threshold (1.96)", color: "#94a3b8" }}
                    />

                    <div className="table-container" style={{ marginTop: '1.5rem' }}>
                      <table className="data-table">
                        <thead>
                          <tr>
                            <th>Look</th>
                            <th>Total N</th>
                            <th>Difference</th>
                            <th>Observed Z</th>
                            <th>OBF Boundary</th>
                            <th>Naive p-Value</th>
                            <th>Safe p-Value (mSPRT)</th>
                            <th>Decision</th>
                          </tr>
                        </thead>
                        <tbody>
                          {seqResult.trajectory.map((row, i) => (
                            <tr key={i}>
                              <td>Look {row.look}</td>
                              <td>{row.sample_size_total.toLocaleString()}</td>
                              <td>{row.difference > 0 ? `+${row.difference}` : row.difference}</td>
                              <td><strong>{row.z_statistic}</strong></td>
                              <td>{row.obrien_fleming_boundary}</td>
                              <td>{row.naive_p_value}</td>
                              <td><span style={{ color: '#14b8a6', fontWeight: 600 }}>{row.safe_p_value}</span></td>
                              <td>
                                <span style={{ color: row.decision.includes("STOP FOR EFFICACY") ? '#10b981' : '#94a3b8', fontWeight: 600 }}>
                                  {row.decision}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </>
                ) : (
                  <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
                    <p style={{ color: 'var(--text-secondary)' }}>Click "O'Brien-Fleming" or "Pocock" to compute interim sequential trajectories.</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ================= TAB 4: BAYESIAN A/B (PyMC) ================= */}
        {activeTab === 'bayesian' && (
          <div className="panel-grid">
            <div className="card">
              <h2 className="card-title">PyMC Bayesian A/B Model</h2>
              <p className="card-desc">
                Compute the exact posterior probability that Variant B beats Variant A, credible intervals, and expected loss.
              </p>

              <div className="form-group">
                <label className="form-label">Control Variant A (Conversions / Total)</label>
                <div className="form-row">
                  <input
                    type="number"
                    className="form-input"
                    placeholder="Conversions"
                    value={bayesForm.ctrl_conv}
                    onChange={e => setBayesForm({ ...bayesForm, ctrl_conv: e.target.value })}
                  />
                  <input
                    type="number"
                    className="form-input"
                    placeholder="Total N"
                    value={bayesForm.ctrl_total}
                    onChange={e => setBayesForm({ ...bayesForm, ctrl_total: e.target.value })}
                  />
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">Treatment Variant B (Conversions / Total)</label>
                <div className="form-row">
                  <input
                    type="number"
                    className="form-input"
                    placeholder="Conversions"
                    value={bayesForm.treat_conv}
                    onChange={e => setBayesForm({ ...bayesForm, treat_conv: e.target.value })}
                  />
                  <input
                    type="number"
                    className="form-input"
                    placeholder="Total N"
                    value={bayesForm.treat_total}
                    onChange={e => setBayesForm({ ...bayesForm, treat_total: e.target.value })}
                  />
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">Prior Beta Distribution: Beta(alpha, beta)</label>
                <div className="form-row">
                  <input
                    type="number"
                    step="0.5"
                    className="form-input"
                    placeholder="alpha=1"
                    value={bayesForm.prior_alpha_a}
                    onChange={e => setBayesForm({ ...bayesForm, prior_alpha_a: e.target.value, prior_alpha_b: e.target.value })}
                  />
                  <input
                    type="number"
                    step="0.5"
                    className="form-input"
                    placeholder="beta=1"
                    value={bayesForm.prior_beta_a}
                    onChange={e => setBayesForm({ ...bayesForm, prior_beta_a: e.target.value, prior_beta_b: e.target.value })}
                  />
                </div>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Default Beta(1, 1) is a flat, uninformative prior.</span>
              </div>

              <button className="btn-primary" onClick={handleRunBayesian} disabled={loading}>
                {loading ? "Sampling Posterior (PyMC NUTS)..." : "Run Bayesian MCMC Analysis"}
              </button>
            </div>

            <div>
              {bayesResult ? (
                <>
                  <div className="conclusion-card">
                    <div className="conclusion-header">
                      <span className="conclusion-title">Bayesian Decision Framework</span>
                      <span className={`conclusion-badge ${bayesResult.probability_b_beats_a >= 0.95 ? '' : 'not-significant'}`}>
                        {bayesResult.recommendation}
                      </span>
                    </div>
                    <p className="conclusion-text">{bayesResult.plain_english_summary}</p>
                  </div>

                  <div className="metrics-grid">
                    <div className="metric-stat-box">
                      <div className="stat-label">P(B beats A)</div>
                      <div className="stat-value" style={{ color: bayesResult.probability_b_beats_a >= 0.95 ? '#10b981' : '#f59e0b' }}>
                        {(bayesResult.probability_b_beats_a * 100).toFixed(1)}%
                      </div>
                      <div className="stat-subtext">Posterior probability</div>
                    </div>
                    <div className="metric-stat-box">
                      <div className="stat-label">Expected Uplift</div>
                      <div className="stat-value">
                        {bayesResult.expected_lift_relative_pct > 0 ? '+' : ''}{bayesResult.expected_lift_relative_pct}%
                      </div>
                      <div className="stat-subtext">Relative percentage lift</div>
                    </div>
                    <div className="metric-stat-box">
                      <div className="stat-label">95% Credible Interval (HDI)</div>
                      <div className="stat-value" style={{ fontSize: '1.05rem' }}>
                        [{(bayesResult.ci_95_diff_lower * 100).toFixed(2)}%, {(bayesResult.ci_95_diff_upper * 100).toFixed(2)}%]
                      </div>
                      <div className="stat-subtext">Absolute difference interval</div>
                    </div>
                    <div className="metric-stat-box">
                      <div className="stat-label">Expected Loss (Risk)</div>
                      <div className="stat-value">
                        {(bayesResult.expected_loss_choosing_treatment * 100).toFixed(3)}%
                      </div>
                      <div className="stat-subtext">Risk of launching B</div>
                    </div>
                  </div>

                  <SvgLineChart
                    data={bayesResult.posterior_chart}
                    xKey="rate"
                    lines={[
                      { key: "density_control", label: "Variant A (Control) Posterior", color: "#94a3b8" },
                      { key: "density_treatment", label: "Variant B (Treatment) Posterior", color: "#3b82f6" }
                    ]}
                    title="PyMC Posterior Conversion Rate Distributions"
                  />

                  <SvgLineChart
                    data={bayesResult.difference_chart}
                    xKey="diff"
                    lines={[
                      { key: "density", label: "Posterior Difference (p_B - p_A)", color: "#10b981" }
                    ]}
                    title="Posterior Distribution of Difference (P(Diff > 0) = Probability B Beats A)"
                    referenceLine={{ val: 0, label: "Null Difference (0)", color: "#ef4444" }}
                  />
                </>
              ) : (
                <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
                  <p style={{ color: 'var(--text-secondary)' }}>Click "Run Bayesian MCMC Analysis" to sample posterior distributions with PyMC.</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ================= TAB 5: OBSERVATIONAL CAUSAL INFERENCE ================= */}
        {activeTab === 'observational' && (
          <div>
            <div className="sub-nav">
              <button
                className={`sub-btn ${obsSubTab === 'psm' ? 'active' : ''}`}
                onClick={() => { setObsSubTab('psm'); handleRunObservational('psm'); }}
              >
                1. Propensity Score Matching (DoWhy)
              </button>
              <button
                className={`sub-btn ${obsSubTab === 'did' ? 'active' : ''}`}
                onClick={() => { setObsSubTab('did'); handleRunObservational('did'); }}
              >
                2. Difference-in-Differences (Card & Krueger)
              </button>
              <button
                className={`sub-btn ${obsSubTab === 'synth' ? 'active' : ''}`}
                onClick={() => { setObsSubTab('synth'); handleRunObservational('synth'); }}
              >
                3. Synthetic Control (California Prop 99)
              </button>
              <button
                className={`sub-btn ${obsSubTab === 'econml' ? 'active' : ''}`}
                onClick={() => { setObsSubTab('econml'); handleRunObservational('econml'); }}
              >
                4. Heterogeneous CATE (EconML LinearDML)
              </button>
            </div>

            {/* Sub-method: PSM */}
            {obsSubTab === 'psm' && (
              <div>
                {psmResult ? (
                  <>
                    <div className="conclusion-card">
                      <div className="conclusion-header">
                        <span className="conclusion-title">DoWhy Causal Analysis (Model → Identify → Estimate → Refute)</span>
                        <span className="conclusion-badge">DoWhy Pipeline</span>
                      </div>
                      <p className="conclusion-text">{psmResult.plain_english_summary}</p>
                    </div>

                    <div className="metrics-grid">
                      <div className="metric-stat-box">
                        <div className="stat-label">Estimated Causal ATE</div>
                        <div className="stat-value">{psmResult.estimated_ate.toFixed(2)}</div>
                        <div className="stat-subtext">Average Treatment Effect</div>
                      </div>
                      <div className="metric-stat-box">
                        <div className="stat-label">Placebo Refuter</div>
                        <div className="stat-value" style={{ color: psmResult.refutation_placebo_treatment.passed ? '#10b981' : '#ef4444' }}>
                          {psmResult.refutation_placebo_treatment.passed ? "PASSED" : "FAILED"}
                        </div>
                        <div className="stat-subtext">New effect: {psmResult.refutation_placebo_treatment.new_effect.toFixed(2)}</div>
                      </div>
                      <div className="metric-stat-box">
                        <div className="stat-label">Random Confounder Refuter</div>
                        <div className="stat-value" style={{ color: psmResult.refutation_random_common_cause.passed ? '#10b981' : '#ef4444' }}>
                          {psmResult.refutation_random_common_cause.passed ? "PASSED" : "FAILED"}
                        </div>
                        <div className="stat-subtext">New effect: {psmResult.refutation_random_common_cause.new_effect.toFixed(2)}</div>
                      </div>
                      <div className="metric-stat-box">
                        <div className="stat-label">Matched Cohort Size</div>
                        <div className="stat-value">{psmResult.sample_size_treated + psmResult.sample_size_control}</div>
                        <div className="stat-subtext">{psmResult.sample_size_treated} treated, {psmResult.sample_size_control} control</div>
                      </div>
                    </div>

                    <div className="card">
                      <h3 style={{ fontSize: '1rem', color: '#fff', marginBottom: '0.75rem' }}>
                        Covariate Balance Check (Standardized Mean Differences - SMD)
                      </h3>
                      <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                        SMD values below 0.10 (dashed threshold) indicate good covariate balance between treated and matched control.
                      </p>
                      <div className="table-container">
                        <table className="data-table">
                          <thead>
                            <tr>
                              <th>Confounder</th>
                              <th>Treated Mean</th>
                              <th>Control Mean</th>
                              <th>SMD (Before Matching)</th>
                              <th>SMD (After Matching)</th>
                              <th>Balance Status</th>
                            </tr>
                          </thead>
                          <tbody>
                            {psmResult.covariate_balance.map((row, i) => (
                              <tr key={i}>
                                <td><strong>{row.confounder}</strong></td>
                                <td>{row.mean_treated}</td>
                                <td>{row.mean_control}</td>
                                <td>{row.smd_before_matching}</td>
                                <td><strong style={{ color: '#60a5fa' }}>{row.smd_after_matching}</strong></td>
                                <td>
                                  <span style={{ color: row.is_balanced_post ? '#10b981' : '#f59e0b', fontWeight: 600 }}>
                                    {row.is_balanced_post ? "Balanced (SMD < 0.10)" : "Acceptable"}
                                  </span>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
                    <button className="btn-primary" style={{ maxWidth: 300, margin: '0 auto' }} onClick={() => handleRunObservational('psm')}>
                      Run DoWhy Propensity Score Matching
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* Sub-method: DiD */}
            {obsSubTab === 'did' && (
              <div>
                {didResult ? (
                  <>
                    <div className="conclusion-card">
                      <div className="conclusion-header">
                        <span className="conclusion-title">Card & Krueger (1994) Difference-in-Differences</span>
                        <span className={`conclusion-badge ${didResult.statistically_significant ? '' : 'not-significant'}`}>
                          {didResult.statistically_significant ? "Statistically Significant" : "Not Significant"}
                        </span>
                      </div>
                      <p className="conclusion-text">{didResult.plain_english_summary}</p>
                    </div>

                    <div className="metrics-grid">
                      <div className="metric-stat-box">
                        <div className="stat-label">DiD Estimate (delta)</div>
                        <div className="stat-value">{didResult.did_estimate_delta > 0 ? `+${didResult.did_estimate_delta}` : didResult.did_estimate_delta}</div>
                        <div className="stat-subtext">Causal impact on employment</div>
                      </div>
                      <div className="metric-stat-box">
                        <div className="stat-label">p-Value</div>
                        <div className="stat-value">{didResult.p_value.toFixed(4)}</div>
                        <div className="stat-subtext">t = {didResult.t_statistic}</div>
                      </div>
                      <div className="metric-stat-box">
                        <div className="stat-label">95% Confidence Interval</div>
                        <div className="stat-value" style={{ fontSize: '1.1rem' }}>
                          [{didResult.ci_lower}, {didResult.ci_upper}]
                        </div>
                        <div className="stat-subtext">Robust standard error: {didResult.standard_error}</div>
                      </div>
                      <div className="metric-stat-box">
                        <div className="stat-label">Counterfactual Post</div>
                        <div className="stat-value">{didResult.means.treatment_counterfactual_post}</div>
                        <div className="stat-subtext">Expected without treatment</div>
                      </div>
                    </div>

                    <SvgLineChart
                      data={didResult.chart_data}
                      xKey="period"
                      lines={[
                        { key: "control", label: "Pennsylvania (Control)", color: "#94a3b8" },
                        { key: "treatment", label: "New Jersey (Treated)", color: "#3b82f6" },
                        { key: "counterfactual", label: "New Jersey Counterfactual (Parallel Trend)", color: "#ef4444", dashed: true }
                      ]}
                      title="Difference-in-Differences Parallel Trends Chart"
                    />
                  </>
                ) : (
                  <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
                    <button className="btn-primary" style={{ maxWidth: 300, margin: '0 auto' }} onClick={() => handleRunObservational('did')}>
                      Run Difference-in-Differences
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* Sub-method: Synthetic Control */}
            {obsSubTab === 'synth' && (
              <div>
                {synthResult ? (
                  <>
                    <div className="conclusion-card">
                      <div className="conclusion-header">
                        <span className="conclusion-title">California Proposition 99 Synthetic Control</span>
                        <span className="conclusion-badge">Abadie et al. SCM</span>
                      </div>
                      <p className="conclusion-text">{synthResult.plain_english_summary}</p>
                    </div>

                    <div className="metrics-grid">
                      <div className="metric-stat-box">
                        <div className="stat-label">Avg Post-Treatment Effect</div>
                        <div className="stat-value">{synthResult.average_post_treatment_effect}</div>
                        <div className="stat-subtext">Packs per capita / year</div>
                      </div>
                      <div className="metric-stat-box">
                        <div className="stat-label">Pre-Treatment RMSPE</div>
                        <div className="stat-value">{synthResult.pre_treatment_rmspe}</div>
                        <div className="stat-subtext">Synthetic fit error (1970-1988)</div>
                      </div>
                      <div className="metric-stat-box">
                        <div className="stat-label">Intervention Year</div>
                        <div className="stat-value">{synthResult.treatment_year}</div>
                        <div className="stat-subtext">Proposition 99 enacted</div>
                      </div>
                      <div className="metric-stat-box">
                        <div className="stat-label">Contributing Donor States</div>
                        <div className="stat-value">{synthResult.donor_weights.length} States</div>
                        <div className="stat-subtext">Non-zero donor weights</div>
                      </div>
                    </div>

                    <SvgLineChart
                      data={synthResult.time_series}
                      xKey="year"
                      lines={[
                        { key: "actual", label: "California (Actual)", color: "#3b82f6" },
                        { key: "synthetic", label: "Synthetic California (Counterfactual)", color: "#f59e0b", dashed: true }
                      ]}
                      title="Per Capita Cigarette Sales: Actual California vs Synthetic California (1970 - 2000)"
                      referenceLine={{ val: 1989, label: "Prop 99 (1988)", color: "#ef4444" }}
                    />

                    <div className="card" style={{ marginTop: '1.5rem' }}>
                      <h3 style={{ fontSize: '1rem', color: '#fff', marginBottom: '0.75rem' }}>
                        Synthetic Control Donor Unit Weights
                      </h3>
                      <div className="table-container">
                        <table className="data-table">
                          <thead>
                            <tr>
                              <th>Donor Control State</th>
                              <th>Synthetic Weight</th>
                              <th>Percentage Contribution</th>
                            </tr>
                          </thead>
                          <tbody>
                            {synthResult.donor_weights.map((d, i) => (
                              <tr key={i}>
                                <td><strong>{d.donor}</strong></td>
                                <td>{d.weight}</td>
                                <td>{(d.weight * 100).toFixed(1)}%</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
                    <button className="btn-primary" style={{ maxWidth: 300, margin: '0 auto' }} onClick={() => handleRunObservational('synth')}>
                      Run Synthetic Control Method
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* Sub-method: EconML CATE */}
            {obsSubTab === 'econml' && (
              <div>
                {econmlResult ? (
                  <>
                    <div className="conclusion-card">
                      <div className="conclusion-header">
                        <span className="conclusion-title">EconML Double Machine Learning (LinearDML)</span>
                        <span className="conclusion-badge">EconML Engine</span>
                      </div>
                      <p className="conclusion-text">{econmlResult.plain_english_summary}</p>
                    </div>

                    <div className="metrics-grid">
                      <div className="metric-stat-box">
                        <div className="stat-label">Overall Causal ATE</div>
                        <div className="stat-value">{econmlResult.overall_ate > 0 ? `+${econmlResult.overall_ate.toFixed(2)}` : econmlResult.overall_ate.toFixed(2)}</div>
                        <div className="stat-subtext">Average Treatment Effect</div>
                      </div>
                      <div className="metric-stat-box">
                        <div className="stat-label">CATE 25th Percentile</div>
                        <div className="stat-value">{econmlResult.cate_distribution.p25}</div>
                        <div className="stat-subtext">Lower quartile response</div>
                      </div>
                      <div className="metric-stat-box">
                        <div className="stat-label">CATE Median</div>
                        <div className="stat-value">{econmlResult.cate_distribution.median}</div>
                        <div className="stat-subtext">Typical individual uplift</div>
                      </div>
                      <div className="metric-stat-box">
                        <div className="stat-label">CATE 75th Percentile</div>
                        <div className="stat-value">{econmlResult.cate_distribution.p75}</div>
                        <div className="stat-subtext">Upper quartile response</div>
                      </div>
                    </div>

                    <div className="card">
                      <h3 style={{ fontSize: '1rem', color: '#fff', marginBottom: '0.75rem' }}>
                        Heterogeneity Moderation Coefficients
                      </h3>
                      <div className="table-container">
                        <table className="data-table">
                          <thead>
                            <tr>
                              <th>Feature</th>
                              <th>Moderation Coefficient</th>
                              <th>Interpretation</th>
                            </tr>
                          </thead>
                          <tbody>
                            {econmlResult.feature_heterogeneity.map((row, i) => (
                              <tr key={i}>
                                <td><strong>{row.feature}</strong></td>
                                <td><strong style={{ color: '#38bdf8' }}>{row.effect_moderation_coef}</strong></td>
                                <td>
                                  {row.effect_moderation_coef > 0
                                    ? `Higher '${row.feature}' increases treatment responsiveness`
                                    : `Higher '${row.feature}' dampens treatment effect`}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
                    <button className="btn-primary" style={{ maxWidth: 300, margin: '0 auto' }} onClick={() => handleRunObservational('econml')}>
                      Run EconML Double Machine Learning
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ================= TAB 6: FOUNDATIONS GUIDE ================= */}
        {activeTab === 'foundations' && (
          <div className="notes-container">
            <h2>Step 1: Statistical Foundations of Experimentation</h2>
            <p>
              This section formalizes the core theoretical pillars mastered in this Capstone project: 
              hypothesis tests, p-values, power, and confidence intervals.
            </p>

            <h3>1. Hypothesis Testing Framework</h3>
            <p>
              In frequentist experimentation, we formulate decisions via mutually exclusive hypotheses:
            </p>
            <ul>
              <li><strong>Null Hypothesis (H0)</strong>: The default status quo that no difference exists between variants ($p_B - p_A = 0$ or $\mu_B - \mu_A = 0$).</li>
              <li><strong>Alternative Hypothesis (H1)</strong>: The claim that a meaningful effect exists.</li>
              <li><strong>Type I Error (Alpha)</strong>: Probability of falsely rejecting H0 when H0 is actually true (False Positive). Nominal standard is 5%.</li>
              <li><strong>Type II Error (Beta)</strong>: Probability of failing to reject H0 when H1 is true (False Negative). Nominal standard is 20%.</li>
            </ul>

            <h3>2. What a p-Value Actually Means (and Doesn't Mean)</h3>
            <p>
              A <code>p-value</code> is the probability of obtaining test results at least as extreme as the observed data, 
              assuming the null hypothesis is strictly true:
            </p>
            <pre>p = P(Observed Data or More Extreme | H0 is True)</pre>
            <p>
              <strong>Common Fallacy:</strong> The p-value is NOT the probability that the null hypothesis is true, 
              and $1 - p$ is NOT the probability that your feature works.
            </p>

            <h3>3. Statistical Power & Sample Size</h3>
            <p>
              Statistical power ($1 - \beta$) is the probability of detecting an effect when it truly exists. 
              Power increases with larger sample sizes ($N$), larger effect sizes ($\delta$), and higher significance thresholds ($\alpha$), 
              and decreases with metric variance ($\sigma^2$).
            </p>

            <h3>4. Confidence Intervals vs Bayesian Credible Intervals</h3>
            <p>
              A frequentist 95% Confidence Interval asserts: if this exact experiment were repeated 100 times, 
              approximately 95 of the computed intervals would cover the fixed, unknown parameter. 
              In contrast, PyMC Bayesian Credible Intervals directly declare: <em>"There is a 95% probability that the parameter lies within this range."</em>
            </p>
          </div>
        )}
      </main>
    </div>
  );
}

// Mount the React Application
const rootElement = document.getElementById("root");
ReactDOM.createRoot(rootElement).render(<App />);
