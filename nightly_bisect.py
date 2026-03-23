"""Nightly bisection — identify when a regression was introduced.

Given a repro script that fails on the current nightly, this module installs
progressively older nightly builds and re-runs the script to find the first
version where the issue appears.  If the issue reproduces consistently across
all tested versions we report high confidence that it is a long-standing bug.

Strategy:
  1. Collect candidate nightly dates going back BISECT_LOOKBACK_DAYS.
  2. Binary-search (bisect) those dates: install the nightly for a date,
     run the repro, record pass/fail.
  3. Return a BisectResult with the regression window and confidence.
"""

import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from config import (
    BISECT_LOOKBACK_DAYS,
    BISECT_MAX_VERSIONS,
    BISECT_TIMEOUT_SECONDS,
    NIGHTLY_ENV,
)

logger = logging.getLogger(__name__)

NIGHTLY_INDEX_URL = "https://download.pytorch.org/whl/nightly/cu130"


@dataclass
class BisectResult:
    """Outcome of a nightly bisection run."""
    ran: bool = False
    oldest_fail_date: str = ""
    newest_pass_date: str = ""
    tested_versions: list[dict] = field(default_factory=list)
    confidence: str = "low"       # low | medium | high
    summary: str = ""


def _get_env_prefix() -> str | None:
    import json as _json
    r = subprocess.run(
        ["conda", "info", "--envs", "--json"],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        return None
    for p in _json.loads(r.stdout).get("envs", []):
        if p.endswith(f"/{NIGHTLY_ENV}"):
            return p
    return None


def _find_nightly_version_for_date(env_prefix: str, date_str: str) -> str | None:
    """Query pip for the nightly torch version matching a specific date.

    Returns a version string like ``2.7.0.dev20260320`` or None.
    """
    if date_str == "latest":
        return None
    pip = os.path.join(env_prefix, "bin", "pip")
    r = subprocess.run(
        [pip, "index", "versions", "torch", "--pre", "--index-url", NIGHTLY_INDEX_URL],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        return None
    for line in r.stdout.splitlines():
        if date_str in line:
            for token in line.replace(",", " ").replace("(", " ").replace(")", " ").split():
                if date_str in token and "dev" in token:
                    return token.strip()
    return None


def _install_nightly_for_date(env_prefix: str, date_str: str) -> bool:
    """Install a specific nightly build by date (YYYYMMDD).

    PyTorch nightlies are published daily.  We look up the version matching
    ``date_str`` via ``pip index versions`` and pin it explicitly.  For
    ``"latest"`` we install without pinning.
    """
    pip = os.path.join(env_prefix, "bin", "pip")
    logger.info("Bisect: installing nightly for date %s", date_str)

    torch_pkg = "torch"
    if date_str != "latest":
        pinned_version = _find_nightly_version_for_date(env_prefix, date_str)
        if pinned_version:
            torch_pkg = f"torch=={pinned_version}"
            logger.info("Bisect: pinning to %s", pinned_version)
        else:
            logger.warning("Bisect: could not find nightly version for %s", date_str)
            return False

    r = subprocess.run(
        [pip, "install", "--pre", "--force-reinstall",
         torch_pkg, "torchvision", "torchaudio",
         "--index-url", NIGHTLY_INDEX_URL],
        capture_output=True, text=True, timeout=600,
        env={**os.environ, "PIP_NO_CACHE_DIR": "1"},
    )
    if r.returncode != 0:
        logger.warning("Bisect: install failed for %s: %s", date_str, r.stderr[-300:])
        return False
    return True


def _get_torch_version(env_prefix: str) -> str:
    python = os.path.join(env_prefix, "bin", "python")
    r = subprocess.run(
        [python, "-c", "import torch; print(torch.__version__)"],
        capture_output=True, text=True, timeout=30,
    )
    return r.stdout.strip() if r.returncode == 0 else "unknown"


def _run_repro(env_prefix: str, repro_path: str) -> bool:
    """Run the repro script.  Returns True if the script FAILS (bug reproduces)."""
    python = os.path.join(env_prefix, "bin", "python")
    try:
        r = subprocess.run(
            [python, repro_path],
            capture_output=True, text=True,
            timeout=BISECT_TIMEOUT_SECONDS,
            env={**os.environ, "TORCH_INDUCTOR_DISABLE_CACHE": "1"},
        )
        return r.returncode != 0
    except subprocess.TimeoutExpired:
        logger.warning("Bisect: repro timed out")
        return False


def _candidate_dates(lookback_days: int, max_versions: int) -> list[str]:
    """Generate candidate nightly dates for bisection.

    We spread them roughly evenly over the lookback window so that the
    initial binary-search step covers a wide range.
    """
    today = datetime.now(timezone.utc).date()
    step = max(1, lookback_days // max_versions)
    dates = []
    for i in range(1, max_versions + 1):
        d = today - timedelta(days=i * step)
        dates.append(d.strftime("%Y%m%d"))
    return dates


def bisect_nightly(repro_path: str) -> BisectResult:
    """Run a binary-search style bisection across nightly builds.

    Parameters
    ----------
    repro_path : str
        Absolute path to the repro script (must exit non-zero to indicate bug).

    Returns
    -------
    BisectResult
    """
    result = BisectResult()
    env_prefix = _get_env_prefix()
    if not env_prefix:
        result.summary = "Could not locate pytorch-nightly conda env."
        return result

    current_version = _get_torch_version(env_prefix)
    logger.info("Bisect: current nightly is %s", current_version)

    current_fails = _run_repro(env_prefix, repro_path)
    if not current_fails:
        result.summary = "Bug does not reproduce on current nightly; bisection skipped."
        return result

    result.ran = True
    result.tested_versions.append({
        "version": current_version, "date": "current", "reproduces": True,
    })

    dates = _candidate_dates(BISECT_LOOKBACK_DAYS, BISECT_MAX_VERSIONS)
    lo, hi = 0, len(dates) - 1
    oldest_fail_date = "current"
    newest_pass_date = ""

    while lo <= hi:
        mid = (lo + hi) // 2
        date_str = dates[mid]

        if not _install_nightly_for_date(env_prefix, date_str):
            result.tested_versions.append({
                "version": "install_failed", "date": date_str, "reproduces": None,
            })
            hi = mid - 1
            continue

        version = _get_torch_version(env_prefix)
        fails = _run_repro(env_prefix, repro_path)
        result.tested_versions.append({
            "version": version, "date": date_str, "reproduces": fails,
        })
        logger.info("Bisect: %s (%s) → %s", date_str, version, "FAIL" if fails else "PASS")

        if fails:
            oldest_fail_date = date_str
            lo = mid + 1
        else:
            newest_pass_date = date_str
            hi = mid - 1

    result.oldest_fail_date = oldest_fail_date
    result.newest_pass_date = newest_pass_date

    all_fail = all(
        v["reproduces"] is True for v in result.tested_versions
    )
    if all_fail:
        result.confidence = "high"
        result.summary = (
            f"Bug reproduces on ALL tested nightlies back to {dates[-1]}. "
            "This is a long-standing issue, not a recent regression."
        )
    elif newest_pass_date:
        result.confidence = "medium"
        result.summary = (
            f"Regression window: passed on {newest_pass_date}, "
            f"first failure on {oldest_fail_date}."
        )
    else:
        result.confidence = "low"
        result.summary = "Bisection inconclusive — some installs failed."

    # Restore current nightly
    logger.info("Bisect: restoring current nightly...")
    _install_nightly_for_date(env_prefix, "latest")

    return result


def format_bisect_comment(br: BisectResult) -> str:
    """Format a BisectResult into a GitHub-comment markdown section."""
    if not br.ran:
        return ""

    lines = [
        "### 🔍 Nightly Bisection",
        "",
        f"**Confidence:** {br.confidence}",
        f"**Summary:** {br.summary}",
        "",
        "| Date | Version | Reproduces |",
        "|------|---------|------------|",
    ]
    for v in br.tested_versions:
        repro_str = {True: "✅ Yes", False: "❌ No", None: "⚠️ N/A"}.get(v["reproduces"], "?")
        lines.append(f"| {v['date']} | `{v['version']}` | {repro_str} |")

    return "\n".join(lines)
