"""
data_structures.py — Pydantic Schema for ER Intake Triage Structured Outputs

Defines the four-layer class hierarchy that the LLM must populate:

    SymptomRead  →  SymptomCluster  →  RiskCalc  →  Triage

Each field description is written to serve as the LLM's primary (and
possibly only) guidance for constructing that field's value.  Descriptions
are deliberately thorough: they define expected content, value ranges,
edge-case handling, and domain-specific reasoning criteria.
"""

from typing import Literal

from pydantic import BaseModel, Field

# ── Layer 1: Individual symptom extraction ────────────────────────────


class SymptomRead(BaseModel):
    """
    A single canonical medical symptom or sign extracted from the intake
    transcripts.

    Intake transcripts are dictated by ER staff in informal, conversational
    language.  Patient complaints are described in layperson terms (e.g.,
    'burning feeling going up into his chest', 'heart is kind of racing',
    'ankles are puffy').  Each SymptomRead represents ONE distinct medical
    condition or finding — identified by its canonical clinical term — not
    one transcript mention.  If multiple transcripts describe the same
    underlying symptom in different layperson language (e.g., 'puffy
    ankles', 'swollen feet', 'legs are huge'), those descriptions all
    refer to the same medical phenomenon (peripheral edema) and MUST be
    consolidated into a single SymptomRead, not listed as separate entries.
    """

    clinical: str = Field(
        description=(
            "The canonical medical term for this symptom or sign, followed "
            "by a concise clinical description.  DEDUPLICATION RULE: use "
            "the single standard medical term that best captures the "
            "underlying condition.  If a patient describes the same "
            "phenomenon in multiple ways across different transcripts "
            "(e.g., 'puffy ankles', 'swollen feet', 'legs are huge', "
            "'shoes don't fit'), all of those map to ONE canonical symptom "
            "— 'Bilateral lower-extremity pitting edema' — and must be "
            "represented by ONE SymptomRead, not multiple entries.  "
            "Similarly, 'heart racing', 'fluttering in my chest', and "
            "'pulse feels fast' all map to ONE entry: 'Palpitations / "
            "tachyarrhythmia'.  Group by the suspected medical diagnosis, "
            "not by the layperson phrasing.\n"
            "After the canonical term, include laterality, timing, "
            "duration, trajectory across visits, and aggravating / "
            "alleviating factors when the transcripts mention them.  "
            "If the transcript is ambiguous about the exact nature of a "
            "symptom, state the most likely clinical interpretation "
            "followed by relevant differentials in parentheses.  "
            "Do NOT fabricate details absent from the transcript — only "
            "translate what is actually described."
        )
    )

    severity: int = Field(
        description=(
            "Integer severity score from -5 to +5 reflecting clinical "
            "concern level based on the transcript description.\n"
            "  -5 = Clearly benign or improving finding (e.g., patient "
            "reports symptom has resolved, wound is healing well).\n"
            "  -3 to -1 = Mildly reassuring or stable (e.g., pain reduced "
            "from prior visit, vitals trending toward normal).\n"
            "   0 = Clinically ambiguous — insufficient information to "
            "assess significance, or the finding could be benign or serious "
            "depending on context.\n"
            "  +1 to +2 = Mildly concerning (e.g., persistent fatigue, mild "
            "hypertension, occasional palpitations at rest).\n"
            "  +3 = Moderately concerning — warrants diagnostic workup (e.g., "
            "exertional chest pressure, new peripheral edema, orthopnea).\n"
            "  +4 = Seriously concerning — suggests active pathology (e.g., "
            "chest pain at rest lasting >5 minutes, cyanotic extremities, "
            "acute dyspnea with accessory muscle use).\n"
            "  +5 = Acutely life-threatening (e.g., ongoing crushing chest "
            "pain radiating to jaw/arms with diaphoresis, acute mental "
            "status change, hemodynamic instability).\n"
            "When severity changes across transcripts (e.g., symptom worsens "
            "from visit to visit), score based on the WORST presentation "
            "described for this particular symptom reading."
        )
    )


# ── Layer 2: Symptom clustering ──────────────────────────────────────


class SymptomCluster(BaseModel):
    """
    A clinically coherent group of related symptoms that share an
    underlying pathophysiological mechanism or organ system.

    Examples of valid clusters from cardiac intake transcripts:
      - Myocardial ischemia cluster: chest pressure, jaw pain, left arm
        numbness, diaphoresis, exertional dyspnea
      - Heart failure cluster: orthopnea, bilateral LE edema, paroxysmal
        nocturnal dyspnea, weight loss with anorexia, abdominal distension
      - Hemodynamic instability cluster: hypertension, tachycardia,
        near-syncope, peripheral cyanosis
      - Hepatic congestion cluster: jaundice/icterus, RUQ pain, dark urine

    Group symptoms by mechanism, not merely by body region.  A single
    symptom may appear in multiple clusters if it is clinically relevant
    to more than one pathological process.
    """

    symptoms: list[SymptomRead] = Field(
        description=(
            "The deduplicated set of canonical symptoms belonging to this "
            "clinical cluster.  Each entry represents one distinct medical "
            "finding identified by its standard clinical term — NOT one "
            "transcript mention.  Multiple layperson descriptions of the "
            "same underlying condition across different transcripts must "
            "already be merged into a single SymptomRead before being "
            "placed here.  Include every relevant canonical symptom from "
            "ALL transcripts that fits this cluster's pathophysiological "
            "theme.  Order symptoms from most clinically significant to "
            "least significant within the cluster.  Each cluster should "
            "contain at least 2 symptoms; a single isolated symptom should "
            "be grouped with the most closely related finding rather than "
            "forming a cluster of one."
        )
    )

    acute: bool = Field(
        description=(
            "True if this symptom cluster, taken as a whole, indicates a "
            "condition requiring immediate clinical intervention — meaning "
            "delay would risk irreversible organ damage, hemodynamic "
            "collapse, or death.  Criteria for True: any symptom in the "
            "cluster is rated +4 or +5 in severity, OR the combination of "
            "multiple moderate symptoms (+3) together constitutes an acute "
            "presentation (e.g., chest pressure + diaphoresis + radiation "
            "to jaw = acute coronary syndrome even if each individually "
            "might be moderate).  False if the cluster represents a chronic, "
            "stable, or slowly progressive condition that can be managed "
            "with scheduled follow-up rather than emergent action.  When in "
            "doubt and the patient's longitudinal trajectory shows clear "
            "worsening, err toward True."
        )
    )


# ── Layer 3: Risk calculation ─────────────────────────────────────────


class RiskCalc(BaseModel):
    """
    A risk assessment that scores the overall danger to the patient
    across one or more symptom clusters, weighting both the plausibility
    of a serious diagnosis and the severity of its consequences.
    """

    clusters: list[SymptomCluster] = Field(
        description=(
            "The symptom clusters that inform this risk calculation.  A "
            "single RiskCalc may encompass one cluster (e.g., isolated "
            "ischemia risk) or multiple interacting clusters (e.g., "
            "ischemia + heart failure + hepatic congestion as combined "
            "evidence of multi-organ impact from advanced cardiac disease).  "
            "Group clusters into RiskCalcs that represent distinct clinical "
            "risk dimensions — for example, one RiskCalc for acute coronary "
            "risk, another for decompensated heart failure risk, another "
            "for procedural/surgical risk."
        )
    )

    risk: float = Field(
        description=(
            "A composite risk score from 0.0 to 1.0 calculated as: "
            "plausibility × consequence.\n"
            "  Plausibility (0.0–1.0): How likely is the serious diagnosis "
            "given all transcript evidence?  Consider: number of supporting "
            "symptoms, consistency across visits, trajectory over time, "
            "known risk factors (smoking history, family cardiac history, "
            "age, hypertension), and diagnostic findings mentioned "
            "(abnormal stress test, EKG changes, catheterization referral).\n"
            "  Consequence (0.0–1.0): If the diagnosis is correct, how "
            "severe is the outcome without intervention?  Consider: "
            "lethality, speed of progression, reversibility, organ damage.\n"
            "  Multiply these to get the final score.  Examples:\n"
            "    0.0–0.1 = Very low risk (implausible diagnosis or trivial "
            "consequence, e.g., mild muscle strain in an otherwise healthy "
            "patient).\n"
            "    0.2–0.4 = Moderate risk (plausible but manageable, e.g., "
            "stable angina with outpatient management plan).\n"
            "    0.5–0.7 = High risk (probable serious condition, e.g., "
            "unstable angina with progressive symptoms).\n"
            "    0.8–0.9 = Very high risk (near-certain serious condition "
            "with life-threatening potential, e.g., multi-vessel CAD with "
            "decompensating heart failure).\n"
            "    0.95–1.0 = Critical (established life-threatening "
            "condition requiring emergent intervention)."
        )
    )


# ── Layer 4: Final triage routing ─────────────────────────────────────


TriageRouting = Literal[
    "surgery",  # Immediate or urgent surgical intervention required
    "discharge",  # Safe for discharge with self-care instructions
    "observation",  # Admitted overnight for monitoring and observation
    "ICU",  # Intensive Care Unit — critical-level monitoring and management
]


class Triage(BaseModel):
    """
    The top-level triage output — the final structured result of the
    entire intake analysis pipeline.

    The LLM must construct this object by working bottom-up through the
    full reasoning chain: (1) extract individual symptoms from ALL
    transcripts, (2) cluster them by clinical mechanism, (3) calculate
    risk scores for each clinical dimension, and (4) synthesize
    everything into a single routing judgment.

    This object represents the complete clinical picture of the patient
    across all visits, not just the most recent encounter.
    """

    calculations: list[RiskCalc] = Field(
        description=(
            "A list of independent risk calculations, each representing a "
            "distinct clinical risk dimension identified across the full "
            "set of intake transcripts.  Expected dimensions for a "
            "cardiac patient might include (but are not limited to): "
            "acute coronary syndrome / myocardial ischemia risk, "
            "congestive heart failure / volume overload risk, "
            "hemodynamic instability risk, hepatic / end-organ congestion "
            "risk, and procedural or surgical risk.  Adapt the risk "
            "dimensions to match the patient's actual presentation.  "
            "Each RiskCalc should contain the symptom clusters most "
            "relevant to that risk dimension.  Include at least 2 and "
            "ideally 3–5 RiskCalcs to demonstrate thorough "
            "multi-dimensional risk assessment."
        )
    )

    judgment: TriageRouting = Field(
        description=(
            "The single most appropriate triage routing decision for this "
            "patient, derived from the complete chain of reasoning: "
            "individual symptoms → symptom clusters → risk calculations → "
            "this final judgment.  Choose the option that matches the "
            "HIGHEST-acuity risk calculation — triage always routes to the "
            "most urgent need.  You MUST use exactly one of these four "
            "literals:\n"
            "  'surgery'     — The patient requires immediate or urgent "
            "surgical intervention (e.g., emergent valve repair or "
            "replacement, CABG, or any operative procedure that cannot "
            "be safely deferred).  Choose this when the clinical evidence "
            "shows a condition whose definitive treatment is operative.\n"
            "  'ICU'         — The patient requires intensive care unit "
            "admission for critical-level monitoring, aggressive medical "
            "management, or hemodynamic support, but surgery is not the "
            "primary indicated intervention.\n"
            "  'observation' — The patient is admitted overnight for "
            "continuous monitoring and observation but does not need "
            "ICU-level care or surgical intervention.  Covers telemetry, "
            "step-down, and short-stay observation scenarios.\n"
            "  'discharge'   — The patient is stable enough to be released "
            "with self-care instructions and scheduled outpatient "
            "follow-up.  Use only when no acute or high-risk condition "
            "requires inpatient management."
        )
    )
