"""
business_integration.py — Integration Showcase for ER Intake Triage Pipeline

Defines 8 business integration use-cases demonstrating how structured LLM
outputs (Triage objects) can feed real-world technology endpoints.  Generates
a self-contained HTML5 showcase page with live data from the Triage object.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape

from data_structures import Triage
from settings import build_integration_showcase_filepath

logger = logging.getLogger("er_triage.business_integration")


# ═══════════════════════════════════════════════════════════════════════
# Use-Case Data Model
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class IntegrationUseCase:
    """A single business integration use-case for the showcase."""

    title: str
    org_type: str
    technology: str
    description: str
    fields: list[str]
    code_snippet: str


# ═══════════════════════════════════════════════════════════════════════
# Use-Case Definitions
# ═══════════════════════════════════════════════════════════════════════

USE_CASES: list[IntegrationUseCase] = [
    # 1. Hospital EHR — FHIR Observation Resources
    IntegrationUseCase(
        title="Hospital EHR — FHIR Observation Resources",
        org_type="Healthcare",
        technology="REST / FHIR R4 API",
        description=("Push each extracted symptom as a FHIR Observation resource into the " "hospital's Electronic Health Records system, creating a structured " "clinical record from unstructured intake transcripts."),
        fields=["SymptomRead.clinical", "SymptomRead.severity"],
        code_snippet="""\
import requests, json

FHIR_BASE = "https://ehr.hospital.org/fhir/R4"

for calc in triage.calculations:
    for cluster in calc.clusters:
        for symptom in cluster.symptoms:
            observation = {
                "resourceType": "Observation",
                "status": "final",
                "category": [{
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "exam",
                        "display": "Exam"
                    }]
                }],
                "code": {
                    "text": symptom.clinical
                },
                "valueInteger": symptom.severity,
                "interpretation": [{
                    "text": f"Severity {symptom.severity}/5"
                }]
            }
            resp = requests.post(
                f"{FHIR_BASE}/Observation",
                json=observation,
                headers={"Content-Type": "application/fhir+json"},
                timeout=30,
            )
            resp.raise_for_status()""",
    ),
    # 2. CDC Syndromic Surveillance — MQTT Stream
    IntegrationUseCase(
        title="CDC Syndromic Surveillance — MQTT Stream",
        org_type="Government",
        technology="MQTT Message Broker",
        description=("Stream real-time acute cluster alerts to a public health syndromic " "surveillance system for early outbreak detection and situational awareness."),
        fields=["SymptomCluster.acute", "SymptomCluster.symptoms (count)"],
        code_snippet="""\
import paho.mqtt.client as mqtt
import json

client = mqtt.Client(client_id="er-triage-pipeline")
client.tls_set()  # Enable TLS for secure transport
client.connect("mqtt.cdc.gov", port=8883)

for calc in triage.calculations:
    for cluster in calc.clusters:
        if cluster.acute:
            payload = json.dumps({
                "alert_type": "acute_cluster",
                "symptom_count": len(cluster.symptoms),
                "symptoms": [s.clinical for s in cluster.symptoms],
                "acute": cluster.acute,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            client.publish(
                "cdc/syndromic/acute_alerts",
                payload=payload,
                qos=1,
            )

client.disconnect()""",
    ),
    # 3. Bed Management Dashboard — Redis Cache
    IntegrationUseCase(
        title="Bed Management Dashboard — Redis Cache",
        org_type="Hospital Operations",
        technology="Redis In-Memory Cache",
        description=("Instantly update a real-time bed-management dashboard with the triage " "routing decision so that charge nurses and bed coordinators can allocate " "resources before the patient arrives at the unit."),
        fields=["Triage.judgment"],
        code_snippet="""\
import redis, json

r = redis.Redis(host="redis.hospital.local", port=6379, db=0)

patient_id = "patient-20260310-001"
routing = {
    "judgment": triage.judgment,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "priority": "STAT" if triage.judgment in ("surgery", "ICU") else "routine",
}

# Cache routing decision with 2-hour TTL
r.set(
    f"triage:routing:{patient_id}",
    json.dumps(routing),
    ex=7200,
)

# Publish to real-time channel for dashboard subscribers
r.publish("triage:routing", json.dumps({
    "patient_id": patient_id,
    **routing,
}))""",
    ),
    # 4. Clinical Research Data Lake — Parquet/S3
    IntegrationUseCase(
        title="Clinical Research Data Lake — Parquet/S3",
        org_type="Clinical Research",
        technology="Apache Parquet / AWS S3 Data Lake",
        description=("Write flattened risk calculation records as columnar Parquet files to a " "research data lake for retrospective cohort analysis and outcomes research."),
        fields=["RiskCalc.risk", "RiskCalc.clusters (cluster count, acute count)"],
        code_snippet="""\
import pyarrow as pa
import pyarrow.parquet as pq

rows = []
for i, calc in enumerate(triage.calculations):
    acute_count = sum(1 for c in calc.clusters if c.acute)
    rows.append({
        "calc_index": i,
        "risk_score": calc.risk,
        "cluster_count": len(calc.clusters),
        "acute_cluster_count": acute_count,
        "judgment": triage.judgment,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

table = pa.Table.from_pylist(rows)
pq.write_to_dataset(
    table,
    root_path="s3://research-datalake/triage/risk_calcs",
    partition_cols=["judgment"],
)""",
    ),
    # 5. Insurance Pre-Authorization — SOAP/XML
    IntegrationUseCase(
        title="Insurance Pre-Authorization — SOAP/XML",
        org_type="Insurance",
        technology="SOAP / XML Web Service",
        description=("Submit the highest risk score and triage routing to a payer's " "pre-authorization service for automated approval of emergent procedures, " "reducing administrative delays in critical cases."),
        fields=["RiskCalc.risk", "Triage.judgment"],
        code_snippet="""\
from zeep import Client

wsdl = "https://payer.example.com/preauth?wsdl"
soap_client = Client(wsdl)

max_risk = max(calc.risk for calc in triage.calculations)
response = soap_client.service.SubmitPreAuthorization(
    PatientId="patient-20260310-001",
    RiskScore=max_risk,
    TriageRouting=triage.judgment,
    Urgency="emergent" if max_risk >= 0.8 else "urgent",
    RequestTimestamp=datetime.now(timezone.utc).isoformat(),
)

if response.Approved:
    auth_number = response.AuthorizationNumber
    print(f"Pre-auth approved: {auth_number}")""",
    ),
    # 6. Medical School Teaching Database — PostgreSQL
    IntegrationUseCase(
        title="Medical School Teaching Database — PostgreSQL",
        org_type="Academic",
        technology="PostgreSQL Relational Database",
        description=("Insert deduplicated symptom extractions into a normalized teaching database " "that medical students query to study symptom-to-diagnosis reasoning patterns."),
        fields=["SymptomRead.clinical", "SymptomRead.severity", "SymptomCluster.acute"],
        code_snippet="""\
import psycopg2

conn = psycopg2.connect(
    host="db.medschool.edu", dbname="teaching_cases",
    user="etl_service", password="***",
)
cur = conn.cursor()

try:
    for calc in triage.calculations:
        for cluster in calc.clusters:
            cur.execute(
                "INSERT INTO clusters (acute, symptom_count, created_at) "
                "VALUES (%s, %s, NOW()) RETURNING id",
                (cluster.acute, len(cluster.symptoms)),
            )
            cluster_id = cur.fetchone()[0]

            for symptom in cluster.symptoms:
                cur.execute(
                    "INSERT INTO symptoms (cluster_id, clinical, severity) "
                    "VALUES (%s, %s, %s)",
                    (cluster_id, symptom.clinical, symptom.severity),
                )
    conn.commit()
finally:
    cur.close()
    conn.close()""",
    ),
    # 7. Emergency Department Event Stream — Kafka
    IntegrationUseCase(
        title="Emergency Department Event Stream — Kafka",
        org_type="Emergency Services",
        technology="Apache Kafka Event Stream",
        description=("Publish triage routing events with risk metadata to a Kafka topic so that " "downstream systems (alerting, analytics, staffing) can react in real time " "to incoming patient acuity."),
        fields=["Triage.judgment", "RiskCalc.risk (max)"],
        code_snippet="""\
from confluent_kafka import Producer
import json, socket

producer = Producer({
    "bootstrap.servers": "kafka.hospital.local:9092",
    "client.id": socket.gethostname(),
})

max_risk = max(calc.risk for calc in triage.calculations)
event = json.dumps({
    "event_type": "triage_routing",
    "judgment": triage.judgment,
    "max_risk_score": max_risk,
    "risk_dimensions": len(triage.calculations),
    "timestamp": datetime.now(timezone.utc).isoformat(),
})

producer.produce(
    topic="ed.triage.events",
    key="patient-20260310-001",
    value=event,
)
producer.flush(timeout=10)""",
    ),
    # 8. State Health Department Analytics — GraphQL
    IntegrationUseCase(
        title="State Health Department Analytics — GraphQL",
        org_type="Public Health",
        technology="GraphQL API",
        description=("Report aggregated cluster-level data to a state health department's " "analytics platform for population health monitoring, resource planning, " "and epidemiological trend detection."),
        fields=[
            "SymptomCluster.symptoms (names)",
            "SymptomCluster.acute",
            "RiskCalc.risk",
        ],
        code_snippet='''\
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport

transport = RequestsHTTPTransport(
    url="https://analytics.health.state.gov/graphql",
    headers={"Authorization": "Bearer <token>"},
)
gql_client = Client(transport=transport, fetch_schema_from_transport=True)

mutation = gql("""
    mutation CreateTriageReport($input: TriageReportInput!) {
        createTriageReport(input: $input) {
            id
            status
        }
    }
""")

clusters_data = []
for calc in triage.calculations:
    for cluster in calc.clusters:
        clusters_data.append({
            "symptoms": [s.clinical for s in cluster.symptoms],
            "acute": cluster.acute,
            "symptomCount": len(cluster.symptoms),
        })

result = gql_client.execute(mutation, variable_values={
    "input": {
        "clusters": clusters_data,
        "riskScores": [c.risk for c in triage.calculations],
        "judgment": triage.judgment,
    }
})''',
    ),
]


# ═══════════════════════════════════════════════════════════════════════
# Accent colours per organization type
# ═══════════════════════════════════════════════════════════════════════

ORG_COLORS: dict[str, str] = {
    "Healthcare": "#0d9488",  # teal
    "Government": "#6366f1",  # indigo
    "Hospital Operations": "#f59e0b",  # amber
    "Clinical Research": "#8b5cf6",  # violet
    "Insurance": "#ef4444",  # red
    "Academic": "#d97706",  # amber-dark
    "Emergency Services": "#dc2626",  # red-bright
    "Public Health": "#059669",  # emerald
}

TECH_EMOJI: dict[str, str] = {
    "REST / FHIR R4 API": "\U0001f3e5",  # 🏥
    "MQTT Message Broker": "\U0001f4e1",  # 📡
    "Redis In-Memory Cache": "\u26a1",  # ⚡
    "Apache Parquet / AWS S3 Data Lake": "\U0001f4be",  # 💾
    "SOAP / XML Web Service": "\U0001f4e8",  # 📨
    "PostgreSQL Relational Database": "\U0001f5c4\ufe0f",  # 🗄️
    "Apache Kafka Event Stream": "\U0001f500",  # 🔀
    "GraphQL API": "\U0001f310",  # 🌐
}


# ═══════════════════════════════════════════════════════════════════════
# Live Data Extraction
# ═══════════════════════════════════════════════════════════════════════


def _extract_live_values(triage: Triage, uc: IntegrationUseCase) -> str:
    """Extract actual live values from the Triage object for a use-case's mapped fields."""
    parts: list[str] = []
    for field_path in uc.fields:
        if "SymptomRead.clinical" in field_path:
            symptoms: list[str] = []
            for calc in triage.calculations:
                for cluster in calc.clusters:
                    for s in cluster.symptoms:
                        if s.clinical not in symptoms:
                            symptoms.append(s.clinical)
            # Show first 5 with truncation
            display = symptoms[:5]
            extra = len(symptoms) - 5
            val = ", ".join(f'"{escape(c)}"' for c in display)
            if extra > 0:
                val += f" … and {extra} more"
            parts.append(f"<strong>SymptomRead.clinical</strong>: {val}")

        elif "SymptomRead.severity" in field_path:
            severities: list[int] = []
            for calc in triage.calculations:
                for cluster in calc.clusters:
                    for s in cluster.symptoms:
                        severities.append(s.severity)
            if severities:
                parts.append(f"<strong>SymptomRead.severity</strong>: range [{min(severities)}, {max(severities)}], mean {sum(severities) / len(severities):.1f}")

        elif "SymptomCluster.acute" in field_path:
            total = sum(len(c.clusters) for c in triage.calculations)
            acute = sum(1 for c in triage.calculations for cl in c.clusters if cl.acute)
            parts.append(f"<strong>SymptomCluster.acute</strong>: {acute} acute / {total} total clusters")

        elif "SymptomCluster.symptoms" in field_path:
            if "names" in field_path:
                names: list[str] = []
                for calc in triage.calculations:
                    for cluster in calc.clusters:
                        for s in cluster.symptoms:
                            if s.clinical not in names:
                                names.append(s.clinical)
                display_names = names[:6]
                extra_names = len(names) - 6
                val_n = ", ".join(f'"{escape(n)}"' for n in display_names)
                if extra_names > 0:
                    val_n += f" … and {extra_names} more"
                parts.append(f"<strong>SymptomCluster.symptoms</strong>: {val_n}")
            elif "count" in field_path:
                counts: list[int] = []
                for calc in triage.calculations:
                    for cluster in calc.clusters:
                        counts.append(len(cluster.symptoms))
                if counts:
                    parts.append(f"<strong>SymptomCluster.symptoms (count)</strong>: {counts} (total {sum(counts)})")

        elif "RiskCalc.risk" in field_path:
            risks = [c.risk for c in triage.calculations]
            if "max" in field_path:
                parts.append(f"<strong>RiskCalc.risk (max)</strong>: {max(risks):.2f}")
            else:
                parts.append(f"<strong>RiskCalc.risk</strong>: {', '.join(f'{r:.2f}' for r in risks)}")

        elif "RiskCalc.clusters" in field_path:
            for i, calc in enumerate(triage.calculations):
                acute_ct = sum(1 for cl in calc.clusters if cl.acute)
                parts.append(f"<strong>RiskCalc[{i}].clusters</strong>: {len(calc.clusters)} clusters ({acute_ct} acute)")

        elif "Triage.judgment" in field_path:
            parts.append(f'<strong>Triage.judgment</strong>: <span class="judgment-value">{escape(triage.judgment)}</span>')

    return "<br>".join(parts) if parts else "<em>No mapped data</em>"


# ═══════════════════════════════════════════════════════════════════════
# CSS Syntax Highlighting (static, no JS)
# ═══════════════════════════════════════════════════════════════════════


def _highlight_python(code: str) -> str:
    """Apply simple CSS-class-based syntax colouring to a Python snippet."""
    import re

    result = escape(code)

    # Comments (# ...)
    result = re.sub(
        r"(#[^\n]*)",
        r'<span class="py-comment">\1</span>',
        result,
    )

    # Strings — triple-quoted first, then single/double
    result = re.sub(
        r"(&quot;&quot;&quot;.*?&quot;&quot;&quot;|&#x27;&#x27;&#x27;.*?&#x27;&#x27;&#x27;)",
        r'<span class="py-string">\1</span>',
        result,
        flags=re.DOTALL,
    )
    result = re.sub(
        r"(&quot;[^&]*?&quot;|&#x27;[^&]*?&#x27;)",
        r'<span class="py-string">\1</span>',
        result,
    )

    # Keywords
    kw = r"\b(import|from|for|if|in|else|elif|try|except|finally|return|" + r"with|as|def|class|True|False|None|and|or|not|raise|await|async|print)\b"
    result = re.sub(kw, r'<span class="py-keyword">\1</span>', result)

    # f-string braces (already escaped, so look for { ... })
    result = re.sub(
        r"\{([^}]+)\}",
        r'<span class="py-fstring">{\1}</span>',
        result,
    )

    return result


# ═══════════════════════════════════════════════════════════════════════
# HTML Generation
# ═══════════════════════════════════════════════════════════════════════


def generate_integration_showcase(triage: Triage, model: str) -> str:
    """Generate the integration showcase HTML page and write it to disk.

    Returns the complete HTML string.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    filepath = build_integration_showcase_filepath(model)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Build nav links
    nav_links = "\n".join(f'        <a class="nav-link" href="#uc-{i + 1}">{i + 1}. {escape(uc.title.split("—")[0].strip())}</a>' for i, uc in enumerate(USE_CASES))

    # Build cards
    cards: list[str] = []
    for i, uc in enumerate(USE_CASES):
        num = i + 1
        accent = ORG_COLORS.get(uc.org_type, "#6b7280")
        emoji = TECH_EMOJI.get(uc.technology, "\U0001f50c")  # 🔌 default
        live_values = _extract_live_values(triage, uc)
        highlighted_code = _highlight_python(uc.code_snippet)

        card = f"""\
      <section class="card" id="uc-{num}" style="--accent: {accent}">
        <div class="card-header">
          <span class="card-num">{num}</span>
          <h2>{escape(uc.title)}</h2>
        </div>
        <div class="badges">
          <span class="badge badge-org" style="background: {accent}22; color: {accent}; border: 1px solid {accent}55">{escape(uc.org_type)}</span>
          <span class="badge badge-tech">{emoji} {escape(uc.technology)}</span>
        </div>
        <p class="description">{escape(uc.description)}</p>
        <div class="mapped-fields">
          <h3>Mapped Fields — Live Values</h3>
          <div class="field-values">{live_values}</div>
        </div>
        <div class="code-section">
          <h3>Sample Integration Code</h3>
          <pre><code>{highlighted_code}</code></pre>
        </div>
      </section>"""
        cards.append(card)

    cards_html = "\n\n".join(cards)

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Beyond the Chatbot: Building LLMs into Business Systems — Integration Showcase</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    :root {{
      --bg: #0f172a;
      --bg-card: #1e293b;
      --bg-code: #0f172a;
      --text: #e2e8f0;
      --text-muted: #94a3b8;
      --text-heading: #f1f5f9;
      --border: #334155;
      --shadow: rgba(0, 0, 0, 0.4);
    }}

    html {{ scroll-behavior: smooth; }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      min-height: 100vh;
    }}

    /* ── Sticky Nav ── */
    .top-nav {{
      position: sticky;
      top: 0;
      z-index: 100;
      background: rgba(15, 23, 42, 0.92);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border);
      padding: 0.6rem 1.5rem;
      display: flex;
      gap: 0.4rem;
      flex-wrap: wrap;
      align-items: center;
    }}
    .nav-brand {{
      font-weight: 700;
      color: var(--text-heading);
      margin-right: 1.2rem;
      font-size: 0.85rem;
      white-space: nowrap;
    }}
    .nav-link {{
      color: var(--text-muted);
      text-decoration: none;
      font-size: 0.78rem;
      padding: 0.3rem 0.6rem;
      border-radius: 6px;
      transition: background 0.2s, color 0.2s;
      white-space: nowrap;
    }}
    .nav-link:hover {{
      background: #334155;
      color: var(--text-heading);
    }}

    /* ── Header ── */
    .header {{
      text-align: center;
      padding: 3.5rem 1.5rem 2.5rem;
      background: linear-gradient(180deg, #1e293b 0%, var(--bg) 100%);
    }}
    .header h1 {{
      font-size: 2rem;
      color: var(--text-heading);
      margin-bottom: 0.5rem;
      font-weight: 800;
      letter-spacing: -0.02em;
    }}
    .header .subtitle {{
      font-size: 1.1rem;
      color: var(--text-muted);
      margin-bottom: 1.2rem;
    }}
    .header .meta {{
      font-size: 0.85rem;
      color: var(--text-muted);
      margin-bottom: 1rem;
    }}
    .header .meta span {{
      margin: 0 0.5rem;
    }}
    .header .intro {{
      max-width: 760px;
      margin: 0 auto;
      font-size: 0.95rem;
      color: var(--text);
      line-height: 1.7;
    }}

    /* ── Card Grid ── */
    .grid {{
      max-width: 900px;
      margin: 0 auto;
      padding: 2rem 1.5rem 3rem;
      display: grid;
      grid-template-columns: 1fr;
      gap: 1.5rem;
    }}

    /* ── Card ── */
    .card {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.5rem;
      box-shadow: 0 4px 24px var(--shadow);
      transition: transform 0.2s, box-shadow 0.2s;
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }}
    .card:hover {{
      transform: translateY(-3px);
      box-shadow: 0 8px 32px var(--shadow), 0 0 0 1px var(--accent, var(--border));
    }}

    .card-header {{
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }}
    .card-num {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 2rem;
      height: 2rem;
      border-radius: 50%;
      background: var(--accent, #6366f1);
      color: #fff;
      font-weight: 700;
      font-size: 0.9rem;
      flex-shrink: 0;
    }}
    .card-header h2 {{
      font-size: 1.15rem;
      font-weight: 700;
      color: var(--text-heading);
      line-height: 1.3;
    }}

    /* ── Badges ── */
    .badges {{
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
    }}
    .badge {{
      font-size: 0.75rem;
      padding: 0.25rem 0.65rem;
      border-radius: 9999px;
      font-weight: 600;
      white-space: nowrap;
    }}
    .badge-tech {{
      background: #33415522;
      color: var(--text-muted);
      border: 1px solid var(--border);
    }}

    .description {{
      font-size: 0.9rem;
      color: var(--text);
      line-height: 1.6;
    }}

    /* ── Mapped Fields ── */
    .mapped-fields {{
      background: rgba(0, 0, 0, 0.2);
      border-radius: 8px;
      padding: 1rem;
    }}
    .mapped-fields h3 {{
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--text-muted);
      margin-bottom: 0.6rem;
      font-weight: 600;
    }}
    .field-values {{
      font-size: 0.85rem;
      line-height: 1.8;
    }}
    .field-values strong {{
      color: var(--text-heading);
    }}
    .judgment-value {{
      display: inline-block;
      background: #dc262622;
      color: #fca5a5;
      padding: 0.1rem 0.5rem;
      border-radius: 4px;
      font-weight: 700;
      text-transform: uppercase;
    }}

    /* ── Code Block ── */
    .code-section h3 {{
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--text-muted);
      margin-bottom: 0.5rem;
      font-weight: 600;
    }}
    pre {{
      background: var(--bg-code);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1rem;
      overflow-x: auto;
      font-size: 0.8rem;
      line-height: 1.55;
    }}
    code {{
      font-family: "Cascadia Code", "Fira Code", "JetBrains Mono", "SF Mono", Consolas, monospace;
      color: var(--text);
    }}

    /* Syntax colours */
    .py-keyword {{ color: #c084fc; font-weight: 600; }}
    .py-string {{ color: #86efac; }}
    .py-comment {{ color: #64748b; font-style: italic; }}
    .py-fstring {{ color: #fbbf24; }}

    /* ── Footer ── */
    .footer {{
      text-align: center;
      padding: 2.5rem 1.5rem;
      border-top: 1px solid var(--border);
      color: var(--text-muted);
      font-size: 0.82rem;
      line-height: 1.6;
      max-width: 700px;
      margin: 0 auto;
    }}
  </style>
</head>
<body>

  <nav class="top-nav">
    <span class="nav-brand">Integration Showcase</span>
{nav_links}
  </nav>

  <header class="header">
    <h1>Beyond the Chatbot: Building LLMs into Business Systems</h1>
    <p class="subtitle">Demonstrating how structured LLM outputs feed real-world technology endpoints</p>
    <p class="meta">
      <span>Model: <strong>{escape(model)}</strong></span>
      <span>|</span>
      <span>Generated: {escape(timestamp)}</span>
      <span>|</span>
      <span>Triage Routing: <strong>{escape(triage.judgment.upper())}</strong></span>
    </p>
    <p class="intro">
      This showcase demonstrates eight concrete integration patterns where a validated,
      schema-constrained <code>Triage</code> object — produced by the ER Intake Triage
      structured-output LLM pipeline — feeds directly into real-world technology
      endpoints used by healthcare, government, academic, and insurance organizations.
      Each card below shows the target system, the specific Pydantic fields consumed,
      live values extracted from the current analysis, and realistic integration code.
    </p>
  </header>

  <main class="grid">

{cards_html}

  </main>

  <footer class="footer">
    <p>
      <strong>Educational Demonstration</strong> — This showcase is part of the
      "Beyond the Chatbot: Building LLMs into Business Systems" presentation.
      All integration endpoints shown are simulated for illustration purposes.
      The structured data values displayed above are real outputs from the
      ER Intake Triage pipeline using the <strong>{escape(model)}</strong> model.
    </p>
  </footer>

</body>
</html>"""

    _ = filepath.write_text(html, encoding="utf-8")
    logger.info("Integration showcase written to: %s", filepath)
    return html
