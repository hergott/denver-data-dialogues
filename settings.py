"""Runtime settings for the ER Intake Triage pipeline."""

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from pathlib import Path

# User-edited runtime setting.
MODEL = "openai/gpt-oss-120b"
SIMULATED_DATA_FILENAME = "er_intake_mitral_emergency.txt"


@dataclass(frozen=True)
class RuntimeDefaults:
    """Internal default paths and service settings derived from this file."""

    project_root: Path
    src_dir: Path
    simulated_data_dir: Path
    env_file: Path
    simulated_data_file: Path
    results_dir: Path
    groq_base_url: str


def get_runtime_defaults() -> RuntimeDefaults:
    """Return internal default paths and service settings."""
    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent
    simulated_data_dir = project_root / "simulated_data"
    return RuntimeDefaults(
        project_root=project_root,
        src_dir=src_dir,
        simulated_data_dir=simulated_data_dir,
        env_file=src_dir / ".env",
        simulated_data_file=simulated_data_dir / SIMULATED_DATA_FILENAME,
        results_dir=project_root / "results",
        groq_base_url="https://api.groq.com/openai/v1",
    )


def build_output_filepath(model: str, now: datetime | None = None) -> Path:
    """Return the timestamped output filepath for a specific model run."""
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-") or "model"
    return get_runtime_defaults().results_dir / f"triage_{safe_model}_{timestamp}.txt"


def build_html_output_filepath(model: str, now: datetime | None = None) -> Path:
    """Return the timestamped HTML output filepath for a specific model run."""
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-") or "model"
    return get_runtime_defaults().results_dir / "html" / f"triage_{safe_model}_{timestamp}.html"


def build_integration_showcase_filepath(model: str, now: datetime | None = None) -> Path:
    """Return the timestamped path for the integration showcase HTML file."""
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-") or "model"
    return get_runtime_defaults().results_dir / "business_integration" / f"integration_showcase_{safe_model}_{timestamp}.html"
