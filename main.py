"""
main.py — ER Intake Triage: Structured-Output LLM Pipeline

Loads 24 intake transcripts, sends them to an LLM via the Groq API,
receives a validated Pydantic Triage object, and writes a presentation-
quality formatted text report.

Usage:
    # Store your Groq API key in src/.env or export it in the shell.
    # Adjust src/settings.py if you want to change MODEL or the
    # selected simulated data file.

    python src/main.py
"""

import asyncio
import logging
import os
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI, DefaultAioHttpClient

from data_structures import RiskCalc, SymptomCluster, SymptomRead, Triage
from groq_api_client import (
    MAX_ATTEMPTS,
    MAX_OUTPUT_TOKENS,
    REQUEST_TIMEOUT_SECONDS,
    RETRY_DELAY_SECONDS,
    GroqAPIClient,
)
from prompts import DEVELOPER_INSTRUCTIONS, build_prompt
from business_integration import generate_integration_showcase
from html_report import write_html_report
from settings import MODEL, SIMULATED_DATA_FILENAME, build_html_output_filepath, build_integration_showcase_filepath, build_output_filepath, get_runtime_defaults

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

RUNTIME_DEFAULTS = get_runtime_defaults()
ENV_FILE = RUNTIME_DEFAULTS.env_file
SIMULATED_DATA_FILE = RUNTIME_DEFAULTS.simulated_data_file
OUTPUT_FILE = build_output_filepath(MODEL)
HTML_OUTPUT_FILE = build_html_output_filepath(MODEL)
INTEGRATION_SHOWCASE_FILE = build_integration_showcase_filepath(MODEL)
GROQ_BASE_URL = RUNTIME_DEFAULTS.groq_base_url

# ═══════════════════════════════════════════════════════════════════════
# Logging setup
# ═══════════════════════════════════════════════════════════════════════


def configure_logging() -> logging.Logger:
    """Configure root and module loggers with timestamped console output."""
    log_format = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("er_triage")
    logger.setLevel(logging.INFO)
    return logger


def load_environment(logger: logging.Logger) -> None:
    """Load environment variables from src/.env when present."""
    if ENV_FILE.exists():
        _ = load_dotenv(ENV_FILE, override=False)
        logger.info("Loaded environment variables from: %s", ENV_FILE)
        return

    logger.warning(
        "Environment file not found at %s; using existing process environment.",
        ENV_FILE,
    )


# ═══════════════════════════════════════════════════════════════════════
# Step 1 — Load transcript data
# ═══════════════════════════════════════════════════════════════════════


def load_transcripts(filepath: Path, logger: logging.Logger) -> str:
    """
    Read the transcript file and return its full contents as a string.

    Raises:
        SystemExit: If the file is missing, empty, or unreadable.
    """
    logger.info("Loading transcript data from: %s", filepath)
    if not filepath.exists():
        logger.critical("Transcript file not found: %s", filepath.resolve())
        sys.exit(1)

    try:
        text = filepath.read_text(encoding="utf-8")
    except Exception as exc:
        logger.critical("Failed to read transcript file: %s", exc)
        sys.exit(1)

    if not text.strip():
        logger.critical("Transcript file is empty: %s", filepath)
        sys.exit(1)

    # Quick sanity check — count delimited transcripts
    count = text.count("=== BEGIN INTAKE TRANSCRIPT")
    logger.info(
        "Loaded %d characters containing %d intake transcripts.",
        len(text),
        count,
    )
    if count == 0:
        logger.warning("No transcript delimiters found — the file may be malformed.")

    return text


# ═══════════════════════════════════════════════════════════════════════
# Step 2 — Initialize the API client
# ═══════════════════════════════════════════════════════════════════════


def init_client(logger: logging.Logger) -> GroqAPIClient:
    """
    Create the AsyncOpenAI client pointed at Groq and wrap it in GroqAPIClient.

    Uses DefaultAioHttpClient (aiohttp) for reliable async HTTP transport.

    Raises:
        SystemExit: If GROQ_API_KEY is not set.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.critical(
            "GROQ_API_KEY environment variable is not set. " + "Add it to %s or export it in the shell before running.",
            ENV_FILE,
        )
        sys.exit(1)

    logger.info("Initializing AsyncOpenAI client with Groq base URL: %s", GROQ_BASE_URL)
    openai_client = AsyncOpenAI(
        api_key=api_key,
        base_url=GROQ_BASE_URL,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
        http_client=DefaultAioHttpClient(),
    )

    groq = GroqAPIClient(openai_client)
    logger.info("GroqAPIClient initialized successfully.")
    return groq


# ═══════════════════════════════════════════════════════════════════════
# Step 3 — Call the LLM
# ═══════════════════════════════════════════════════════════════════════


async def call_llm(
    groq: GroqAPIClient,
    transcript_data: str,
    model: str,
    logger: logging.Logger,
) -> Triage:
    """
    Build the prompt, call the Groq API, and return a validated Triage object.

    Raises:
        SystemExit: On unrecoverable API errors after all retries.
    """
    logger.info("Building prompt with sandwich pattern...")
    prompt = build_prompt(transcript_data)
    logger.info(
        "Prompt constructed: %d characters (transcript: %d, instructions: %d).",
        len(prompt),
        len(transcript_data),
        len(prompt) - len(transcript_data),
    )

    logger.info("Calling Groq API with model: %s", model)
    logger.info("Response model: Triage (4-layer nested Pydantic schema)")
    logger.info("This may take 30–120 seconds depending on model and load...")

    try:
        triage: Triage = await groq.call(
            developer_instructions=DEVELOPER_INSTRUCTIONS,
            prompt=prompt,
            response_model=Triage,
            model=model,
        )
    except KeyboardInterrupt:
        logger.warning("Received interrupt signal (KeyboardInterrupt/SIGINT) during LLM call. " + "This can be caused by Ctrl+C or by external terminal/task controls. " + "Exiting gracefully.")
        sys.exit(130)
    except ValueError as exc:
        logger.critical("Invalid model selection: %s", exc)
        sys.exit(1)
    except RuntimeError as exc:
        logger.critical("All API attempts exhausted: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.critical("Unexpected error during API call: %s — %s", type(exc).__name__, exc)
        sys.exit(1)

    logger.info("Received validated Triage object from API.")
    return triage


# ═══════════════════════════════════════════════════════════════════════
# Step 4 — Validate and inspect results
# ═══════════════════════════════════════════════════════════════════════


def validate_results(triage: Triage, logger: logging.Logger) -> list[str]:
    """
    Run post-hoc quality checks on the Triage object.

    Returns a list of warning strings (empty if everything looks good).
    """
    warnings: list[str] = []

    # Check that we have risk calculations
    n_calcs = len(triage.calculations)
    logger.info("Risk calculations returned: %d", n_calcs)
    if n_calcs < 2:
        w = f"Only {n_calcs} RiskCalc(s) returned — expected ≥2 for thorough analysis."
        warnings.append(w)
        logger.warning(w)

    # Check risk score ranges
    for i, calc in enumerate(triage.calculations):
        if not (0.0 <= calc.risk <= 1.0):
            w = f"RiskCalc[{i}] has out-of-range risk score: {calc.risk}"
            warnings.append(w)
            logger.warning(w)

    # Check for at least one high-acuity risk (given the data)
    max_risk = max((c.risk for c in triage.calculations), default=0.0)
    if max_risk < 0.7:
        w = f"Highest risk score is {max_risk:.2f} — expected ≥0.7 given " + f"the patient's trajectory toward emergent CABG."
        warnings.append(w)
        logger.warning(w)

    # Check that at least one cluster is marked acute
    any_acute = any(cluster.acute for calc in triage.calculations for cluster in calc.clusters)
    if not any_acute:
        w = "No symptom clusters marked as acute — expected at least one."
        warnings.append(w)
        logger.warning(w)

    # Count total symptoms extracted
    total_symptoms = sum(len(cluster.symptoms) for calc in triage.calculations for cluster in calc.clusters)
    logger.info("Total individual symptoms extracted: %d", total_symptoms)
    if total_symptoms < 10:
        w = f"Only {total_symptoms} symptoms extracted — may be incomplete."
        warnings.append(w)
        logger.warning(w)

    # Check severity score ranges
    out_of_range: list[str] = []
    for calc in triage.calculations:
        for cluster in calc.clusters:
            for sym in cluster.symptoms:
                if not (-5 <= sym.severity <= 5):
                    out_of_range.append(f"  '{sym.clinical[:50]}...' severity={sym.severity}")
    if out_of_range:
        w = f"{len(out_of_range)} symptom(s) with out-of-range severity:\n" + "\n".join(out_of_range)
        warnings.append(w)
        logger.warning(w)

    # Log the final judgment
    logger.info("Triage judgment: %s", triage.judgment)

    if not warnings:
        logger.info("All validation checks passed.")
    else:
        logger.warning("%d validation warning(s) detected.", len(warnings))

    return warnings


# ═══════════════════════════════════════════════════════════════════════
# Step 5 — Format and write output
# ═══════════════════════════════════════════════════════════════════════

LINE_WIDTH = 78
DIVIDER_HEAVY = "=" * LINE_WIDTH
DIVIDER_LIGHT = "-" * LINE_WIDTH
DIVIDER_DOT = "· " * (LINE_WIDTH // 2)


def _wrap(text: str, indent: int = 0) -> str:
    """Word-wrap text to LINE_WIDTH with the given indent."""
    prefix = " " * indent
    return textwrap.fill(
        text,
        width=LINE_WIDTH,
        initial_indent=prefix,
        subsequent_indent=prefix,
    )


def _severity_bar(score: int) -> str:
    """Create a visual severity indicator: -5 ▒▒▒▒▒░░░░░ +5"""
    # Map -5..+5 to 0..10
    pos = score + 5
    bar = "█" * pos + "░" * (10 - pos)
    return f"[{bar}] {score:+d}"


def _risk_bar(score: float) -> str:
    """Create a visual risk bar: 0.0 ▒▒▒▒▒▒▒░░░ 1.0"""
    filled = round(score * 20)
    bar = "█" * filled + "░" * (20 - filled)
    return f"[{bar}] {score:.2f}"


def format_symptom(sym: SymptomRead, index: int) -> str:
    """Format a single SymptomRead for the report."""
    lines = [
        f"    Symptom {index}:",
        _wrap(sym.clinical, indent=6),
        f"      Severity: {_severity_bar(sym.severity)}",
    ]
    return "\n".join(lines)


def format_cluster(cluster: SymptomCluster, index: int) -> str:
    """Format a single SymptomCluster for the report."""
    acute_label = "⚠  ACUTE — IMMEDIATE INTERVENTION REQUIRED" if cluster.acute else "Stable / Non-urgent"
    lines = [
        f"  Cluster {index}  [{acute_label}]",
        f"  {'─' * 60}",
    ]
    for i, sym in enumerate(cluster.symptoms, start=1):
        lines.append(format_symptom(sym, i))
        if i < len(cluster.symptoms):
            lines.append("")
    return "\n".join(lines)


def format_risk_calc(calc: RiskCalc, index: int) -> str:
    """Format a single RiskCalc for the report."""
    lines = [
        DIVIDER_LIGHT,
        f"RISK DIMENSION {index}",
        f"Composite Risk Score: {_risk_bar(calc.risk)}",
        DIVIDER_LIGHT,
    ]
    for i, cluster in enumerate(calc.clusters, start=1):
        lines.append("")
        lines.append(format_cluster(cluster, i))
    return "\n".join(lines)


def format_all_symptoms(triage: Triage) -> str:
    """Format Level 1: a flat listing of every extracted SymptomRead."""
    lines: list[str] = [
        "",
        DIVIDER_HEAVY,
        "LEVEL 1: INDIVIDUAL SYMPTOM EXTRACTION".center(LINE_WIDTH),
        DIVIDER_HEAVY,
        "",
        _wrap(
            "Every symptom the LLM extracted from the intake transcripts, " + "translated into clinical terminology and scored for severity. " + "These are the atomic building blocks of the analysis.",
            indent=2,
        ),
        "",
    ]
    idx = 1
    for calc in triage.calculations:
        for cluster in calc.clusters:
            for sym in cluster.symptoms:
                lines.append(format_symptom(sym, idx))
                lines.append("")
                idx += 1
    lines.append(f"  Total symptoms extracted: {idx - 1}")
    lines.append("")
    return "\n".join(lines)


def format_all_clusters(triage: Triage) -> str:
    """Format Level 2: every SymptomCluster with its child symptoms."""
    lines: list[str] = [
        "",
        DIVIDER_HEAVY,
        "LEVEL 2: SYMPTOM CLUSTERS".center(LINE_WIDTH),
        DIVIDER_HEAVY,
        "",
        _wrap(
            "Symptoms grouped by shared pathophysiological mechanism or " + "organ system. Each cluster is flagged as ACUTE (requires " + "immediate intervention) or Stable. The same symptom may " + "appear in multiple clusters when clinically relevant to " + "more than one process.",
            indent=2,
        ),
        "",
    ]
    idx = 1
    for calc in triage.calculations:
        for cluster in calc.clusters:
            lines.append(format_cluster(cluster, idx))
            lines.append("")
            idx += 1
    total_acute = sum(1 for c in triage.calculations for cl in c.clusters if cl.acute)
    lines.append(f"  Total clusters: {idx - 1}" + f"({total_acute} acute, {idx - 1 - total_acute} stable)")
    lines.append("")
    return "\n".join(lines)


def format_all_risk_calcs(triage: Triage) -> str:
    """Format Level 3: every RiskCalc with its full cluster/symptom tree."""
    lines: list[str] = [
        "",
        DIVIDER_HEAVY,
        "LEVEL 3: RISK CALCULATIONS".center(LINE_WIDTH),
        DIVIDER_HEAVY,
        "",
        _wrap(
            "Each risk dimension scores the overall danger to the patient " + "across one or more symptom clusters, weighting both the " + "plausibility of a serious diagnosis and the severity of its " + "consequences. Clusters and their symptoms are shown again " + "here in full so the reasoning chain is visible.",
            indent=2,
        ),
        "",
    ]
    for i, calc in enumerate(triage.calculations, start=1):
        lines.append("")
        lines.append(format_risk_calc(calc, i))
    lines.append("")
    return "\n".join(lines)


def format_triage_report(
    triage: Triage,
    model: str,
    warnings: list[str],
) -> str:
    """
    Produce the complete formatted text report from the Triage object.

    Shows all four levels of the Pydantic schema hierarchy:
      Level 1 — Individual symptom extraction (SymptomRead)
      Level 2 — Symptom clusters (SymptomCluster)
      Level 3 — Risk calculations (RiskCalc)
      Level 4 — Triage decision (Triage)
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Header
    header = "\n".join(
        [
            DIVIDER_HEAVY,
            "ER INTAKE TRIAGE ANALYSIS — STRUCTURED OUTPUT REPORT".center(LINE_WIDTH),
            DIVIDER_HEAVY,
            "",
            f"  Generated:  {timestamp}",
            f"  Model:      {model}",
            f"  Patient:    58 y/o male, 24 intake transcripts (Jan 14 – Mar 8, 2026)",
            f"  Judgment:   {triage.judgment.upper().replace('_', ' ')}",
            "",
            DIVIDER_HEAVY,
        ]
    )

    # Summary statistics
    total_calcs = len(triage.calculations)
    total_clusters = sum(len(c.clusters) for c in triage.calculations)
    total_symptoms = sum(len(cl.symptoms) for c in triage.calculations for cl in c.clusters)
    acute_clusters = sum(1 for c in triage.calculations for cl in c.clusters if cl.acute)
    max_risk = max((c.risk for c in triage.calculations), default=0.0)

    summary = "\n".join(
        [
            "",
            "ANALYSIS SUMMARY",
            DIVIDER_LIGHT,
            f"  Risk dimensions analyzed:    {total_calcs}",
            f"  Symptom clusters identified: {total_clusters} ({acute_clusters} acute)",
            f"  Individual symptoms extracted:{total_symptoms}",
            f"  Highest risk score:          {max_risk:.2f}",
            f"  Triage routing:              {triage.judgment}",
            "",
        ]
    )

    # Level 1 — All symptoms
    level1 = format_all_symptoms(triage)

    # Level 2 — All clusters
    level2 = format_all_clusters(triage)

    # Level 3 — All risk calculations
    level3 = format_all_risk_calcs(triage)

    # Level 4 — Final judgment
    judgment_section = "\n".join(
        [
            "",
            DIVIDER_HEAVY,
            "LEVEL 4: TRIAGE DECISION".center(LINE_WIDTH),
            DIVIDER_HEAVY,
            "",
            f"  Routing Decision:  >>> {triage.judgment.upper().replace('_', ' ')} <<<",
            "",
            _wrap(
                "This routing decision is derived from the complete chain of " + "clinical reasoning: individual symptom extraction (Level 1) " + "from all intake transcripts, clustering by pathophysiological " + "mechanism (Level 2), multi-dimensional risk calculation " + "(Level 3), and synthesis into the single most appropriate " + "triage destination (Level 4).",
                indent=2,
            ),
            "",
        ]
    )

    # Warnings
    warning_section = ""
    if warnings:
        warning_lines: list[str] = [
            "",
            DIVIDER_LIGHT,
            "VALIDATION WARNINGS",
            DIVIDER_LIGHT,
        ]
        for w in warnings:
            warning_lines.append(f"  ⚠  {w}")
        warning_lines.append("")
        warning_section = "\n".join(warning_lines)

    # Footer
    footer = "\n".join(
        [
            DIVIDER_HEAVY,
            "END OF REPORT".center(LINE_WIDTH),
            DIVIDER_HEAVY,
            "",
        ]
    )

    return "\n".join(
        [
            header,
            summary,
            level1,
            level2,
            level3,
            judgment_section,
            warning_section,
            footer,
        ]
    )


def write_output(
    triage: Triage,
    model: str,
    warnings: list[str],
    filepath: Path,
    logger: logging.Logger,
) -> None:
    """Format the Triage object and write it to a text file."""
    logger.info("Formatting triage report...")
    report = format_triage_report(triage, model, warnings)

    logger.info("Writing output to: %s", filepath)
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        _ = filepath.write_text(report, encoding="utf-8")
        logger.info(
            "Output written successfully: %d characters, %d lines.",
            len(report),
            report.count("\n") + 1,
        )
    except Exception as exc:
        logger.critical("Failed to write output file: %s", exc)
        sys.exit(1)


async def main() -> None:
    """Run the full triage pipeline."""
    logger = configure_logging()

    logger.info(DIVIDER_HEAVY)
    logger.info("ER INTAKE TRIAGE — STRUCTURED OUTPUT LLM PIPELINE")
    logger.info(DIVIDER_HEAVY)

    load_environment(logger)

    logger.info(
        "Configuration: model=%s, simulated_data=%s, output=%s",
        MODEL,
        SIMULATED_DATA_FILE,
        OUTPUT_FILE,
    )
    logger.info("Selected simulated data filename: %s", SIMULATED_DATA_FILENAME)
    logger.info(
        "API timing: request_timeout=%.1fs, max_output_tokens=%d, max_attempts=%d, retry_delay=%.1fs",
        REQUEST_TIMEOUT_SECONDS,
        MAX_OUTPUT_TOKENS,
        MAX_ATTEMPTS,
        RETRY_DELAY_SECONDS,
    )

    # Step 1 — Load transcripts
    logger.info(DIVIDER_LIGHT)
    logger.info("STEP 1: Loading transcript data")
    transcript_data = load_transcripts(SIMULATED_DATA_FILE, logger)

    # Step 2 — Initialize client
    logger.info(DIVIDER_LIGHT)
    logger.info("STEP 2: Initializing Groq API client")
    groq = init_client(logger)

    try:
        # Step 3 — Call LLM
        logger.info(DIVIDER_LIGHT)
        logger.info("STEP 3: Calling LLM for structured triage analysis")
        triage = await call_llm(groq, transcript_data, MODEL, logger)

        # Step 4 — Validate
        logger.info(DIVIDER_LIGHT)
        logger.info("STEP 4: Validating triage results")
        warnings = validate_results(triage, logger)

        # Step 5 — Write output
        logger.info(DIVIDER_LIGHT)
        logger.info("STEP 5: Writing formatted output report")
        write_output(triage, MODEL, warnings, OUTPUT_FILE, logger)

        # Step 6 — Write HTML report
        logger.info(DIVIDER_LIGHT)
        logger.info("STEP 6: Writing HTML report")
        write_html_report(triage, MODEL, warnings, HTML_OUTPUT_FILE, logger)

        # Step 7 — Generate integration showcase
        logger.info(DIVIDER_LIGHT)
        logger.info("STEP 7: Generating business integration showcase")
        _ = generate_integration_showcase(triage, MODEL)
        logger.info("Integration showcase output: %s", INTEGRATION_SHOWCASE_FILE)

        # Done
        logger.info(DIVIDER_HEAVY)
        logger.info("PIPELINE COMPLETE")
        logger.info("Triage judgment: %s", triage.judgment)
        logger.info("Text output: %s", OUTPUT_FILE)
        logger.info("HTML output: %s", HTML_OUTPUT_FILE)
        logger.info("Integration showcase: %s", INTEGRATION_SHOWCASE_FILE)
        logger.info(DIVIDER_HEAVY)
    finally:
        await groq.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.getLogger("er_triage").warning("Received interrupt signal (KeyboardInterrupt/SIGINT). " + "This can be caused by Ctrl+C or by external terminal/task controls. " + "Exiting gracefully.")
        sys.exit(130)
