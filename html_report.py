"""
html_report.py — HTML Report Generator for ER Intake Triage Pipeline

Generates a self-contained HTML5 + CSS page from a validated Triage object.
All styles are inlined so the file renders independently in any browser.
"""

import logging
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from data_structures import RiskCalc, SymptomCluster, SymptomRead, Triage


def _severity_class(severity: int) -> str:
    """Return a CSS class name based on severity score."""
    if severity >= 4:
        return "severity-critical"
    if severity >= 2:
        return "severity-high"
    if severity >= 0:
        return "severity-moderate"
    return "severity-low"


def _risk_class(risk: float) -> str:
    """Return a CSS class name based on risk score."""
    if risk >= 0.8:
        return "risk-critical"
    if risk >= 0.5:
        return "risk-high"
    if risk >= 0.2:
        return "risk-moderate"
    return "risk-low"


def _severity_pct(severity: int) -> float:
    """Map severity -5..+5 to 0..100 percent."""
    return (severity + 5) / 10 * 100


def _render_symptom(sym: SymptomRead, index: int) -> str:
    cls = _severity_class(sym.severity)
    pct = _severity_pct(sym.severity)
    sign = "+" if sym.severity > 0 else ""
    return f"""
        <div class="symptom">
          <div class="symptom-header">
            <span class="symptom-index">{index}</span>
            <span class="symptom-clinical">{escape(sym.clinical)}</span>
          </div>
          <div class="severity-row">
            <span class="severity-label">Severity</span>
            <div class="bar-track">
              <div class="bar-fill {cls}" style="width:{pct:.0f}%"></div>
            </div>
            <span class="severity-value {cls}">{sign}{sym.severity}</span>
          </div>
        </div>"""


def _render_cluster(cluster: SymptomCluster, index: int) -> str:
    acute_badge = '<span class="badge badge-acute">ACUTE</span>' if cluster.acute else '<span class="badge badge-stable">Stable</span>'
    symptom_cards = "\n".join(_render_symptom(sym, i) for i, sym in enumerate(cluster.symptoms, 1))
    return f"""
      <div class="cluster {'cluster-acute' if cluster.acute else ''}">
        <div class="cluster-header">
          <h3>Cluster {index} {acute_badge}</h3>
          <span class="cluster-count">{len(cluster.symptoms)} symptom{"s" if len(cluster.symptoms) != 1 else ""}</span>
        </div>
        {symptom_cards}
      </div>"""


def _render_risk_calc(calc: RiskCalc, index: int) -> str:
    cls = _risk_class(calc.risk)
    pct = calc.risk * 100
    clusters_html = "\n".join(_render_cluster(cl, i) for i, cl in enumerate(calc.clusters, 1))
    return f"""
    <section class="risk-section">
      <div class="risk-header">
        <h2>Risk Dimension {index}</h2>
        <div class="risk-score-block">
          <div class="risk-bar-track">
            <div class="risk-bar-fill {cls}" style="width:{pct:.0f}%"></div>
          </div>
          <span class="risk-value {cls}">{calc.risk:.2f}</span>
        </div>
      </div>
      {clusters_html}
    </section>"""


def _judgment_class(judgment: str) -> str:
    return {
        "surgery": "judgment-surgery",
        "ICU": "judgment-icu",
        "observation": "judgment-observation",
        "discharge": "judgment-discharge",
    }.get(judgment, "")


_CSS = """\
:root {
  --bg: #0f1117;
  --surface: #181b24;
  --surface-2: #1e2230;
  --border: #2a2e3b;
  --text: #e0e2ea;
  --text-muted: #8b8fa3;
  --accent: #6c8aff;
  --green: #34d399;
  --amber: #fbbf24;
  --orange: #f97316;
  --red: #ef4444;
  --radius: 10px;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { font-size: 15px; }
body {
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.65;
  padding: 2rem 1rem;
  max-width: 960px;
  margin: 0 auto;
}
h1, h2, h3 { font-weight: 600; letter-spacing: -0.01em; }
h1 { font-size: 1.6rem; color: #fff; }
h2 { font-size: 1.15rem; color: var(--text); }
h3 { font-size: 0.95rem; color: var(--text-muted); }
header {
  text-align: center;
  padding: 2.5rem 1.5rem 2rem;
  background: linear-gradient(135deg, #1a1e2e 0%, #0f1117 100%);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  margin-bottom: 1.5rem;
}
header p { color: var(--text-muted); font-size: 0.85rem; margin-top: 0.3rem; }
.meta-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 0.75rem;
  margin-top: 1.5rem;
  text-align: left;
}
.meta-item {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.75rem 1rem;
}
.meta-item .label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted); }
.meta-item .value { font-size: 1rem; font-weight: 600; color: #fff; margin-top: 0.15rem; }

/* Summary cards */
.summary {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 0.75rem;
  margin-bottom: 1.5rem;
}
.stat-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem;
  text-align: center;
}
.stat-card .num { font-size: 1.8rem; font-weight: 700; color: var(--accent); }
.stat-card .desc { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.25rem; }

/* Risk sections */
.risk-section {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.25rem;
  margin-bottom: 1rem;
}
.risk-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1rem;
  gap: 1rem;
}
.risk-score-block {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  flex-shrink: 0;
}
.risk-bar-track {
  width: 140px;
  height: 8px;
  background: var(--surface-2);
  border-radius: 4px;
  overflow: hidden;
}
.risk-bar-fill {
  height: 100%;
  border-radius: 4px;
  transition: width 0.4s ease;
}
.risk-value { font-size: 1.05rem; font-weight: 700; min-width: 2.5rem; text-align: right; }

/* Clusters */
.cluster {
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem;
  margin-top: 0.75rem;
}
.cluster-acute { border-left: 3px solid var(--red); }
.cluster-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.75rem;
}
.cluster-count { font-size: 0.75rem; color: var(--text-muted); }
.badge {
  display: inline-block;
  font-size: 0.65rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  padding: 0.15rem 0.5rem;
  border-radius: 4px;
  vertical-align: middle;
  margin-left: 0.5rem;
}
.badge-acute { background: rgba(239,68,68,0.15); color: var(--red); }
.badge-stable { background: rgba(52,211,153,0.12); color: var(--green); }

/* Symptoms */
.symptom {
  padding: 0.6rem 0;
  border-bottom: 1px solid rgba(255,255,255,0.04);
}
.symptom:last-child { border-bottom: none; }
.symptom-header { display: flex; gap: 0.6rem; align-items: baseline; margin-bottom: 0.35rem; }
.symptom-index {
  font-size: 0.7rem;
  font-weight: 700;
  color: var(--accent);
  min-width: 1.4rem;
  text-align: center;
  background: rgba(108,138,255,0.1);
  padding: 0.1rem 0.35rem;
  border-radius: 4px;
}
.symptom-clinical { font-size: 0.88rem; color: var(--text); }
.severity-row { display: flex; align-items: center; gap: 0.6rem; margin-left: 2rem; }
.severity-label { font-size: 0.7rem; color: var(--text-muted); min-width: 4rem; }
.bar-track {
  flex: 1;
  max-width: 120px;
  height: 6px;
  background: var(--surface);
  border-radius: 3px;
  overflow: hidden;
}
.bar-fill {
  height: 100%;
  border-radius: 3px;
  transition: width 0.3s ease;
}
.severity-value { font-size: 0.8rem; font-weight: 700; min-width: 2rem; text-align: right; }

/* Colour coding */
.severity-critical, .risk-critical { color: var(--red); }
.severity-critical.bar-fill, .risk-critical.risk-bar-fill { background: var(--red); }
.severity-high, .risk-high { color: var(--orange); }
.severity-high.bar-fill, .risk-high.risk-bar-fill { background: var(--orange); }
.severity-moderate, .risk-moderate { color: var(--amber); }
.severity-moderate.bar-fill, .risk-moderate.risk-bar-fill { background: var(--amber); }
.severity-low, .risk-low { color: var(--green); }
.severity-low.bar-fill, .risk-low.risk-bar-fill { background: var(--green); }

/* Judgment */
.judgment-card {
  text-align: center;
  padding: 2rem 1.5rem;
  border-radius: var(--radius);
  border: 1px solid var(--border);
  margin: 1.5rem 0;
}
.judgment-card h2 { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-muted); margin-bottom: 0.75rem; }
.judgment-routing {
  font-size: 2rem;
  font-weight: 800;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.judgment-surgery { background: linear-gradient(135deg, rgba(239,68,68,0.08), rgba(239,68,68,0.02)); }
.judgment-surgery .judgment-routing { color: var(--red); }
.judgment-icu { background: linear-gradient(135deg, rgba(249,115,22,0.08), rgba(249,115,22,0.02)); }
.judgment-icu .judgment-routing { color: var(--orange); }
.judgment-observation { background: linear-gradient(135deg, rgba(251,191,36,0.08), rgba(251,191,36,0.02)); }
.judgment-observation .judgment-routing { color: var(--amber); }
.judgment-discharge { background: linear-gradient(135deg, rgba(52,211,153,0.08), rgba(52,211,153,0.02)); }
.judgment-discharge .judgment-routing { color: var(--green); }
.judgment-note { font-size: 0.8rem; color: var(--text-muted); margin-top: 1rem; max-width: 600px; margin-left: auto; margin-right: auto; }

/* Warnings */
.warnings {
  background: rgba(251,191,36,0.06);
  border: 1px solid rgba(251,191,36,0.2);
  border-radius: var(--radius);
  padding: 1rem 1.25rem;
  margin-bottom: 1.5rem;
}
.warnings h2 { color: var(--amber); font-size: 0.9rem; margin-bottom: 0.5rem; }
.warnings ul { list-style: none; padding: 0; }
.warnings li { font-size: 0.85rem; color: var(--text-muted); padding: 0.2rem 0; }
.warnings li::before { content: "\\26A0\\FE0F "; }

footer {
  text-align: center;
  padding: 1.5rem 0 0.5rem;
  font-size: 0.75rem;
  color: var(--text-muted);
  border-top: 1px solid var(--border);
  margin-top: 1.5rem;
}
"""


def _render_all_symptoms_section(triage: Triage) -> str:
    """Render Level 1: flat listing of every extracted SymptomRead."""
    all_symptoms: list[SymptomRead] = []
    for calc in triage.calculations:
        for cluster in calc.clusters:
            all_symptoms.extend(cluster.symptoms)

    cards = "\n".join(_render_symptom(sym, i) for i, sym in enumerate(all_symptoms, 1))
    return f"""
  <div class="level-section">
    <div class="level-header">
      <span class="level-badge">Level 1</span>
      <h2>Individual Symptom Extraction</h2>
    </div>
    <p class="level-description">
      Every symptom the LLM extracted from the intake transcripts, translated
      into clinical terminology and scored for severity. These are the atomic
      building blocks of the analysis.
    </p>
    <div class="symptom-list-section">
{cards}
    </div>
    <div class="level-count">Total symptoms extracted: {len(all_symptoms)}</div>
  </div>"""


def _render_all_clusters_section(triage: Triage) -> str:
    """Render Level 2: every SymptomCluster with its child symptoms."""
    all_clusters: list[SymptomCluster] = []
    for calc in triage.calculations:
        all_clusters.extend(calc.clusters)

    cluster_cards = "\n".join(_render_cluster(cl, i) for i, cl in enumerate(all_clusters, 1))
    acute_count = sum(1 for cl in all_clusters if cl.acute)
    stable_count = len(all_clusters) - acute_count
    return f"""
  <div class="level-section">
    <div class="level-header">
      <span class="level-badge">Level 2</span>
      <h2>Symptom Clusters</h2>
    </div>
    <p class="level-description">
      Symptoms grouped by shared pathophysiological mechanism or organ system.
      Each cluster is flagged as ACUTE (requires immediate intervention) or
      Stable. The same symptom may appear in multiple clusters when clinically
      relevant to more than one process.
    </p>
    <div class="cluster-list-section">
{cluster_cards}
    </div>
    <div class="level-count">
      Total clusters: {len(all_clusters)}
      ({acute_count} acute, {stable_count} stable)
    </div>
  </div>"""


def _render_all_risk_calcs_section(triage: Triage) -> str:
    """Render Level 3: every RiskCalc with its full cluster/symptom tree."""
    risk_sections = "\n".join(_render_risk_calc(calc, i) for i, calc in enumerate(triage.calculations, 1))
    return f"""
  <div class="level-section">
    <div class="level-header">
      <span class="level-badge">Level 3</span>
      <h2>Risk Calculations</h2>
    </div>
    <p class="level-description">
      Each risk dimension scores the overall danger to the patient across one
      or more symptom clusters, weighting both the plausibility of a serious
      diagnosis and the severity of its consequences. Clusters and their
      symptoms are shown again here in full so the reasoning chain is visible.
    </p>
{risk_sections}
  </div>"""


def generate_html_report(
    triage: Triage,
    model: str,
    warnings: list[str],
) -> str:
    """Return a complete self-contained HTML page for the triage report."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    total_calcs = len(triage.calculations)
    total_clusters = sum(len(c.clusters) for c in triage.calculations)
    total_symptoms = sum(len(cl.symptoms) for c in triage.calculations for cl in c.clusters)
    acute_clusters = sum(1 for c in triage.calculations for cl in c.clusters if cl.acute)
    max_risk = max((c.risk for c in triage.calculations), default=0.0)

    # Build the four level sections
    level1_html = _render_all_symptoms_section(triage)
    level2_html = _render_all_clusters_section(triage)
    level3_html = _render_all_risk_calcs_section(triage)

    warnings_html = ""
    if warnings:
        items = "\n".join(f"        <li>{escape(w)}</li>" for w in warnings)
        warnings_html = f"""
    <div class="warnings">
      <h2>Validation Warnings</h2>
      <ul>
{items}
      </ul>
    </div>"""

    j_cls = _judgment_class(triage.judgment)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ER Intake Triage Report</title>
  <style>
{_CSS}
  </style>
</head>
<body>
  <header>
    <h1>ER Intake Triage Analysis</h1>
    <p>Structured-Output LLM Pipeline Report</p>
    <div class="meta-grid">
      <div class="meta-item">
        <div class="label">Generated</div>
        <div class="value">{escape(timestamp)}</div>
      </div>
      <div class="meta-item">
        <div class="label">Model</div>
        <div class="value">{escape(model)}</div>
      </div>
      <div class="meta-item">
        <div class="label">Patient</div>
        <div class="value">58 y/o male &mdash; 24 transcripts</div>
      </div>
      <div class="meta-item">
        <div class="label">Transcript Period</div>
        <div class="value">Jan 14 &ndash; Mar 8, 2026</div>
      </div>
    </div>
  </header>

  <div class="summary">
    <div class="stat-card">
      <div class="num">{total_calcs}</div>
      <div class="desc">Risk Dimensions</div>
    </div>
    <div class="stat-card">
      <div class="num">{total_clusters}</div>
      <div class="desc">Symptom Clusters</div>
    </div>
    <div class="stat-card">
      <div class="num">{total_symptoms}</div>
      <div class="desc">Symptoms Extracted</div>
    </div>
    <div class="stat-card">
      <div class="num">{acute_clusters}</div>
      <div class="desc">Acute Clusters</div>
    </div>
    <div class="stat-card">
      <div class="num">{max_risk:.2f}</div>
      <div class="desc">Highest Risk</div>
    </div>
  </div>

{level1_html}

{level2_html}

{level3_html}

  <div class="level-section">
    <div class="level-header">
      <span class="level-badge">Level 4</span>
      <h2>Triage Decision</h2>
    </div>
    <div class="judgment-card {j_cls}">
      <h2>Final Triage Judgment</h2>
      <div class="judgment-routing">{escape(triage.judgment)}</div>
      <p class="judgment-note">
        Derived from the complete chain of clinical reasoning: individual symptom
        extraction (Level 1) from all intake transcripts, clustering by
        pathophysiological mechanism (Level 2), multi-dimensional risk
        calculation (Level 3), and synthesis into the single most appropriate
        triage destination (Level 4).
      </p>
    </div>
  </div>
{warnings_html}
  <footer>ER Intake Triage &mdash; Structured-Output LLM Pipeline</footer>
</body>
</html>
"""


def write_html_report(
    triage: Triage,
    model: str,
    warnings: list[str],
    filepath: Path,
    logger: logging.Logger,
) -> None:
    """Generate and write the HTML report to disk."""
    logger.info("Generating HTML report...")
    html = generate_html_report(triage, model, warnings)

    logger.info("Writing HTML report to: %s", filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    _ = filepath.write_text(html, encoding="utf-8")
    logger.info(
        "HTML report written: %d characters.",
        len(html),
    )
