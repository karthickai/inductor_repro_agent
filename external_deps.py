"""External dependency management — install and test with broader dep sets.

Issues often require third-party libraries (Hugging Face transformers, timm,
diffusers, etc.).  This module:

1. Scans the issue body / repro script for import markers that identify
   which external libraries are needed.
2. Installs those libraries into the pytorch-nightly conda env.
3. Provides helpers to verify the install succeeded.
"""

import logging
import os
import re
import subprocess

from config import NIGHTLY_ENV, EXTERNAL_DEPS, EXTERNAL_INSTALL_TIMEOUT

logger = logging.getLogger(__name__)


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


def detect_external_deps(text: str) -> list[str]:
    """Scan *text* (issue body, comments, or repro script) and return the
    keys of external dependency groups whose import markers appear."""
    needed: list[str] = []
    for group_name, info in EXTERNAL_DEPS.items():
        for marker in info["import_markers"]:
            if marker in text:
                needed.append(group_name)
                break
    return needed


def install_external_deps(groups: list[str]) -> dict[str, bool]:
    """Install the pip packages for each requested external-dep group.

    Returns a dict mapping group name → success bool.
    """
    env_prefix = _get_env_prefix()
    if not env_prefix:
        logger.error("Cannot find pytorch-nightly env for external dep install")
        return {g: False for g in groups}

    pip = os.path.join(env_prefix, "bin", "pip")
    results: dict[str, bool] = {}

    for group in groups:
        info = EXTERNAL_DEPS.get(group)
        if not info:
            results[group] = False
            continue

        packages = info["pip"]
        logger.info("Installing external deps [%s]: %s", group, packages)
        r = subprocess.run(
            [pip, "install", "--quiet"] + packages,
            capture_output=True, text=True, timeout=EXTERNAL_INSTALL_TIMEOUT,
        )
        ok = r.returncode == 0
        if not ok:
            logger.warning("Failed to install %s: %s", group, r.stderr[-300:])
        results[group] = ok

    return results


def verify_imports(groups: list[str]) -> dict[str, bool]:
    """Quick smoke-test: try to import the main package for each group."""
    env_prefix = _get_env_prefix()
    if not env_prefix:
        return {g: False for g in groups}

    python = os.path.join(env_prefix, "bin", "python")
    results: dict[str, bool] = {}

    for group in groups:
        info = EXTERNAL_DEPS.get(group)
        if not info:
            results[group] = False
            continue
        main_pkg = info["pip"][0].replace("-", "_")
        r = subprocess.run(
            [python, "-c", f"import {main_pkg}; print({main_pkg}.__version__)"],
            capture_output=True, text=True, timeout=30,
        )
        results[group] = r.returncode == 0
        if r.returncode == 0:
            logger.info("Verified %s: %s", group, r.stdout.strip())
        else:
            logger.warning("Import check failed for %s: %s", group, r.stderr[:200])

    return results


def ensure_external_deps(issue_text: str, repro_text: str = "") -> dict[str, bool]:
    """End-to-end: detect, install, verify external deps for an issue.

    Parameters
    ----------
    issue_text : str
        Combined issue body + comments text.
    repro_text : str
        Contents of the repro script (if any).

    Returns
    -------
    dict mapping group name → True if installed+verified, False otherwise.
    """
    combined = f"{issue_text}\n{repro_text}"
    groups = detect_external_deps(combined)
    if not groups:
        logger.info("No external dependencies detected")
        return {}

    logger.info("Detected external dependency groups: %s", groups)
    install_results = install_external_deps(groups)

    installed = [g for g, ok in install_results.items() if ok]
    if installed:
        verify_results = verify_imports(installed)
        for g in installed:
            install_results[g] = verify_results.get(g, False)

    return install_results


def format_external_deps_comment(deps: dict[str, bool]) -> str:
    """Format external dep install results into a markdown section."""
    if not deps:
        return ""

    lines = [
        "### 📦 External Dependencies",
        "",
        "| Library | Status |",
        "|---------|--------|",
    ]
    for group, ok in deps.items():
        status = "✅ Installed" if ok else "❌ Failed"
        packages = ", ".join(EXTERNAL_DEPS.get(group, {}).get("pip", []))
        lines.append(f"| {group} (`{packages}`) | {status} |")

    return "\n".join(lines)
