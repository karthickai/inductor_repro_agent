#!/usr/bin/env python3
"""
Inductor Agent — automated reproduction of PyTorch Inductor GitHub issues.

Usage:
    python main.py              # Poll once, process, exit
    python main.py --loop       # Poll continuously
    python main.py --issue 1    # Process a single issue
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

from codeowners import format_owner_tags
from config import (
    AUTO_CLOSE_AFTER_MINUTES,
    AUTO_CLOSE_CLASSIFICATIONS,
    AUTO_CLOSE_GRACE_MINUTES,
    BISECT_ENABLED,
    CLAUDE_TIMEOUT_SECONDS,
    GH_RETRY_ATTEMPTS,
    GITHUB_REPO,
    LOG_DIR,
    NIGHTLY_ENV,
    NIGHTLY_MAX_AGE_HOURS,
    POLL_INTERVAL_SECONDS,
    PROCESSING_LABEL,
    PROJECT_DIR,
    REPRO_FAIL_LABEL,
    REPRO_SUCCESS_LABEL,
    TRIGGER_LABEL,
    VALID_CLASSIFICATIONS,
    WORK_DIR,
    WORKDIR_CLEANUP_DAYS,
)
from nightly_bisect import bisect_nightly, format_bisect_comment
from external_deps import ensure_external_deps, format_external_deps_comment

SKIP_LABELS = {PROCESSING_LABEL, REPRO_SUCCESS_LABEL, REPRO_FAIL_LABEL}

collect_env_cache: str = ""

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(LOG_DIR, "main.log"))],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error_result(reason: str) -> dict:
    return {"classification": "ENV_ERROR", "reason": reason,
            "runs_failed": 0, "runs_total": 0, "error_output": "", "repro_script": ""}


def _gh(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    for attempt in range(GH_RETRY_ATTEMPTS):
        try:
            r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
            if r.returncode == 0:
                return r
            logger.warning("gh failed (%d/%d): %s", attempt + 1, GH_RETRY_ATTEMPTS, r.stderr[:200])
        except subprocess.TimeoutExpired:
            logger.warning("gh timed out (%d/%d)", attempt + 1, GH_RETRY_ATTEMPTS)
        if attempt < GH_RETRY_ATTEMPTS - 1:
            time.sleep(2)
    return subprocess.CompletedProcess(args=args, returncode=1, stdout="", stderr="retries exhausted")


# ---------------------------------------------------------------------------
# GitHub API wrappers
# ---------------------------------------------------------------------------

def gh_fetch_issues(label: str) -> list[dict]:
    r = _gh(["gh", "issue", "list", "--repo", GITHUB_REPO, "--label", label,
             "--state", "open", "--json", "number,title,labels", "--limit", "50"])
    return json.loads(r.stdout) if r.returncode == 0 else []


def gh_fetch_issue_data(issue_number: int) -> dict | None:
    r = _gh(["gh", "issue", "view", str(issue_number), "--repo", GITHUB_REPO,
             "--json", "title,body,labels,comments"])
    return json.loads(r.stdout) if r.returncode == 0 else None


def gh_get_labels(issue_number: int) -> set[str]:
    r = _gh(["gh", "issue", "view", str(issue_number), "--repo", GITHUB_REPO, "--json", "state,labels"])
    if r.returncode != 0:
        return set()
    data = json.loads(r.stdout)
    if data.get("state") == "CLOSED":
        return {"CLOSED"}
    return {lb["name"] for lb in data.get("labels", [])}


def gh_swap_label(issue_number: int, remove: str, add: str) -> bool:
    _gh(["gh", "api", f"repos/{GITHUB_REPO}/issues/{issue_number}/labels/{remove}", "-X", "DELETE"], timeout=15)
    r = _gh(["gh", "api", f"repos/{GITHUB_REPO}/issues/{issue_number}/labels",
             "--method", "POST", "-f", f"labels[]={add}"], timeout=15)
    return r.returncode == 0


def gh_post_comment(issue_number: int, body: str) -> None:
    _gh(["gh", "issue", "comment", str(issue_number), "--repo", GITHUB_REPO, "--body", body])


def gh_close_issue(issue_number: int) -> None:
    _gh(["gh", "issue", "close", str(issue_number), "--repo", GITHUB_REPO])


def should_skip(issue_number: int) -> bool:
    labels = gh_get_labels(issue_number)
    if "CLOSED" in labels:
        logger.info("Issue #%d is closed, skipping", issue_number)
        return True
    overlap = labels & SKIP_LABELS
    if overlap:
        logger.info("Issue #%d already has %s, skipping", issue_number, overlap)
        return True
    return False


# ---------------------------------------------------------------------------
# Nightly environment management
# ---------------------------------------------------------------------------

_last_update_check: dict[str, datetime] = {}


def _get_env_prefix() -> str | None:
    r = subprocess.run(["conda", "info", "--envs", "--json"], capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        return None
    for p in json.loads(r.stdout).get("envs", []):
        if p.endswith(f"/{NIGHTLY_ENV}"):
            return p
    return None


def _get_torch_version(env_prefix: str) -> str | None:
    r = subprocess.run([os.path.join(env_prefix, "bin", "python"), "-c", "import torch; print(torch.__version__)"],
                       capture_output=True, text=True, timeout=30)
    return r.stdout.strip() if r.returncode == 0 else None


def _is_nightly_stale(env_prefix: str) -> bool:
    last_check = _last_update_check.get(env_prefix)
    if last_check and (datetime.now(timezone.utc) - last_check) < timedelta(hours=NIGHTLY_MAX_AGE_HOURS):
        logger.info("pytorch-nightly: already checked recently, treating as fresh")
        return False
    version = _get_torch_version(env_prefix)
    if not version:
        return True
    try:
        dev_date = version.split("dev")[1].split("+")[0]
        age = datetime.now(timezone.utc) - datetime.strptime(dev_date, "%Y%m%d").replace(tzinfo=timezone.utc)
        stale = age > timedelta(hours=NIGHTLY_MAX_AGE_HOURS)
        logger.info("pytorch-nightly: %s (%s)", version, "stale" if stale else "fresh")
        return stale
    except (IndexError, ValueError):
        return True


def _install_nightly(env_prefix: str) -> bool:
    old_version = _get_torch_version(env_prefix)
    logger.info("Updating pytorch-nightly (current: %s)...", old_version)
    r = subprocess.run(
        [os.path.join(env_prefix, "bin", "pip"), "install", "--pre",
         "torch", "torchvision", "torchaudio",
         "--index-url", "https://download.pytorch.org/whl/nightly/cu130"],
        capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        logger.error("Nightly update failed: %s", r.stderr[-500:])
        return False
    new_version = _get_torch_version(env_prefix)
    logger.info("Updated to: %s", new_version)
    if new_version == old_version:
        logger.info("No newer nightly available, marking as fresh")
        _last_update_check[env_prefix] = datetime.now(timezone.utc)
    return True


def _create_nightly_env() -> bool:
    logger.info("Creating pytorch-nightly env...")
    r = subprocess.run(["conda", "create", "-y", "-n", NIGHTLY_ENV, "python=3.13"],
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        logger.error("Failed to create env: %s", r.stderr[-500:])
        return False
    env_prefix = _get_env_prefix()
    return _install_nightly(env_prefix) if env_prefix else False


def ensure_nightly_env() -> bool:
    global collect_env_cache
    env_prefix = _get_env_prefix()

    if env_prefix is None:
        if not _create_nightly_env():
            return False
        env_prefix = _get_env_prefix()
    elif _is_nightly_stale(env_prefix):
        if not _install_nightly(env_prefix):
            return False

    if not collect_env_cache and env_prefix:
        python_bin = os.path.join(env_prefix, "bin", "python")
        r = subprocess.run(
            f"curl -sL https://raw.githubusercontent.com/pytorch/pytorch/main/torch/utils/collect_env.py | {python_bin}",
            capture_output=True, text=True, timeout=60, shell=True)
        if r.returncode == 0:
            collect_env_cache = r.stdout.strip()
            logger.info("Cached collect_env output")

    return True


# ---------------------------------------------------------------------------
# Issue text analysis helpers
# ---------------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.+?)```", re.DOTALL)


def _extract_code_blocks(text: str) -> list[str]:
    """Extract Python code blocks from markdown text."""
    return _CODE_BLOCK_RE.findall(text)


def _has_clean_repro(issue_data: dict) -> bool:
    """Check if the issue has an explicit code block that looks like a repro."""
    body = issue_data.get("body", "")
    comments = issue_data.get("comments", [])
    all_text = body + "\n" + "\n".join(c.get("body", "") for c in comments)
    blocks = _extract_code_blocks(all_text)
    for block in blocks:
        if "import torch" in block or "torch.compile" in block or "torch._inductor" in block:
            return True
    return False


def _get_full_issue_text(issue_data: dict) -> str:
    """Combine body and all comment bodies into a single string."""
    body = issue_data.get("body", "")
    comments = issue_data.get("comments", [])
    parts = [body] + [c.get("body", "") for c in comments]
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Claude invocation
# ---------------------------------------------------------------------------

def invoke_claude(issue_number: int, issue_dir: str, synthesize_repro: bool = False) -> None:
    skill_path = os.path.join(PROJECT_DIR, ".claude", "skills", "inductor-repro", "SKILL.md")
    issue_file = os.path.join(issue_dir, "issue.json")
    log_file = os.path.join(issue_dir, "claude_output.log")

    prompt = (f"Read the skill file at {skill_path} and follow its instructions "
              f"step by step to reproduce GitHub issue #{issue_number}. "
              f"The issue data is at {issue_file}.")

    if synthesize_repro:
        prompt += (
            "\n\nIMPORTANT: This issue does NOT contain a clean reproduction script. "
            "You MUST attempt to synthesize a repro script from the description, "
            "error messages, and any code snippets mentioned in the issue. "
            "Follow the 'Synthesizing a repro' section in the skill file."
        )

    logger.info("Invoking Claude for issue #%d (synthesize=%s)", issue_number, synthesize_repro)
    with open(log_file, "w") as f:
        subprocess.run(["claude", "--print", "--dangerously-skip-permissions", prompt],
                       cwd=PROJECT_DIR, stdout=f, stderr=subprocess.STDOUT, timeout=CLAUDE_TIMEOUT_SECONDS)
    logger.info("Claude finished for issue #%d", issue_number)


# ---------------------------------------------------------------------------
# Result validation & comment building
# ---------------------------------------------------------------------------

def validate_result(raw: dict) -> dict:
    for key in ("classification", "reason", "runs_failed", "runs_total", "error_output", "repro_script"):
        if key not in raw:
            raw[key] = "" if key in ("reason", "error_output", "repro_script") else 0
    if raw["classification"] not in VALID_CLASSIFICATIONS:
        raw["classification"] = "ENV_ERROR"
    for key in ("runs_failed", "runs_total"):
        if not isinstance(raw[key], int):
            try:
                raw[key] = int(raw[key])
            except (TypeError, ValueError):
                raw[key] = 0
    return raw


def build_comment(
    result: dict,
    owner_tags: str = "",
    bisect_section: str = "",
    external_section: str = "",
) -> str:
    classification = result["classification"]
    runs = f"{result['runs_failed']}/{result['runs_total']} failed"
    error_output = result["error_output"][:3000]
    repro_script = result["repro_script"]
    synthesized = result.get("synthesized_repro", False)
    predefined_cat = result.get("predefined_category", "")

    if classification in ("REPRODUCED", "FLAKY"):
        emoji = "✅" if classification == "REPRODUCED" else "🔄"
        synth_note = " *(repro synthesized from description)*" if synthesized else ""
        comment = (
            f"🤖 **Inductor Agent — Reproduction Successful**{synth_note}\n\n"
            f"| Field | Value |\n|-------|-------|\n"
            f"| **Result** | {emoji} {classification} |\n"
            f"| **Runs** | {runs} |\n\n"
            f"<details>\n<summary>Error output (last run)</summary>\n\n```\n{error_output}\n```\n</details>\n\n"
            f"<details>\n<summary>Reproduction script</summary>\n\n```python\n{repro_script}\n```\n</details>"
        )
    elif classification == "PREDEFINED_MITIGATION":
        comment = (
            f"🤖 **Inductor Agent — Predefined Mitigation Applied**\n\n"
            f"| Field | Value |\n|-------|-------|\n"
            f"| **Result** | 📋 {classification} |\n"
            f"| **Category** | {predefined_cat} |\n"
            f"| **Reason** | {result['reason']} |\n"
            f"| **Runs** | {runs} |\n\n"
            f"<details>\n<summary>Reproduction script</summary>\n\n```python\n{repro_script}\n```\n</details>\n\n"
            f"<details>\n<summary>Output</summary>\n\n```\n{error_output}\n```\n</details>"
        )
    elif classification == "SYNTHESIZED_REPRO":
        comment = (
            f"🤖 **Inductor Agent — Synthesized Reproduction Attempt**\n\n"
            f"| Field | Value |\n|-------|-------|\n"
            f"| **Result** | 🔧 {classification} |\n"
            f"| **Reason** | {result['reason']} |\n"
            f"| **Runs** | {runs} |\n\n"
            f"*No clean reproduction script was provided in the issue. "
            f"The agent attempted to synthesize one from the description.*\n\n"
            f"<details>\n<summary>Synthesized script</summary>\n\n```python\n{repro_script}\n```\n</details>\n\n"
            f"<details>\n<summary>Output</summary>\n\n```\n{error_output}\n```\n</details>"
        )
    else:
        comment = (
            f"🤖 **Inductor Agent — Reproduction Failed**\n\n"
            f"| Field | Value |\n|-------|-------|\n"
            f"| **Result** | ❌ {classification} |\n"
            f"| **Reason** | {result['reason']} |\n"
            f"| **Runs** | {runs} |\n\n"
            f"<details>\n<summary>Output</summary>\n\n```\n{error_output}\n```\n</details>"
        )

    for section in (bisect_section, external_section):
        if section:
            comment += f"\n\n{section}"

    if collect_env_cache:
        comment += f"\n\n<details>\n<summary>Environment</summary>\n\n```\n{collect_env_cache}\n```\n</details>"
    if owner_tags:
        comment += f"\n\n{owner_tags}"
    return comment


# ---------------------------------------------------------------------------
# Core issue processing
# ---------------------------------------------------------------------------

def process_issue(issue_number: int) -> None:
    if should_skip(issue_number):
        return

    issue_data = gh_fetch_issue_data(issue_number)
    if not issue_data:
        logger.error("Could not fetch issue #%d", issue_number)
        return

    logger.info("Claiming issue #%d", issue_number)
    if not gh_swap_label(issue_number, TRIGGER_LABEL, PROCESSING_LABEL):
        logger.error("Failed to claim issue #%d", issue_number)
        return
    if PROCESSING_LABEL not in gh_get_labels(issue_number):
        logger.error("Race lost on issue #%d", issue_number)
        return

    result = None
    bisect_section = ""
    external_section = ""

    try:
        issue_dir = os.path.join(WORK_DIR, str(issue_number))
        os.makedirs(issue_dir, exist_ok=True)
        with open(os.path.join(issue_dir, "issue.json"), "w") as f:
            json.dump(issue_data, f, indent=2)

        full_text = _get_full_issue_text(issue_data)
        has_repro = _has_clean_repro(issue_data)
        synthesize = not has_repro

        if synthesize:
            logger.info("Issue #%d: no clean repro found, will attempt synthesis", issue_number)

        # --- External dependencies ---
        external_results = ensure_external_deps(full_text)
        if external_results:
            external_section = format_external_deps_comment(external_results)
            failed_deps = [g for g, ok in external_results.items() if not ok]
            if failed_deps:
                logger.warning("Issue #%d: some external deps failed: %s", issue_number, failed_deps)

        # --- Invoke Claude (handles repro, predefined checks, everything) ---
        try:
            invoke_claude(issue_number, issue_dir, synthesize_repro=synthesize)
        except subprocess.TimeoutExpired:
            logger.error("Claude timed out for issue #%d", issue_number)
            result = _error_result(f"Claude exceeded {CLAUDE_TIMEOUT_SECONDS}s limit.")
            result["classification"] = "TIMEOUT"

        if result is None:
            result_file = os.path.join(issue_dir, "result.json")
            if os.path.exists(result_file):
                try:
                    with open(result_file) as f:
                        result = validate_result(json.load(f))
                    logger.info("Result for #%d: %s", issue_number, result["classification"])
                except (json.JSONDecodeError, OSError) as e:
                    result = _error_result(f"Malformed result.json: {e}")

        if result is None:
            result = _error_result("Claude did not produce result.json.")

        # Tag synthesized repros
        if synthesize and result["classification"] in ("REPRODUCED", "FLAKY"):
            result["synthesized_repro"] = True

        # --- Nightly bisection ---
        repro_path = os.path.join(issue_dir, "repro.py")
        if (BISECT_ENABLED
                and result["classification"] in ("REPRODUCED", "FLAKY")
                and os.path.exists(repro_path)):
            logger.info("Issue #%d: running nightly bisection", issue_number)
            bisect_result = bisect_nightly(repro_path)
            if bisect_result.ran:
                bisect_section = format_bisect_comment(bisect_result)
                result["bisect_confidence"] = bisect_result.confidence
                result["bisect_summary"] = bisect_result.summary

    finally:
        if result is None:
            result = _error_result("Pipeline crashed.")

        final_label = (
            REPRO_SUCCESS_LABEL
            if result["classification"] in ("REPRODUCED", "FLAKY", "SYNTHESIZED_REPRO")
            else REPRO_FAIL_LABEL
        )
        gh_swap_label(issue_number, PROCESSING_LABEL, final_label)
        owner_tags = format_owner_tags(result.get("matched_areas", []))
        comment = build_comment(
            result,
            owner_tags=owner_tags,
            bisect_section=bisect_section,
            external_section=external_section,
        )
        gh_post_comment(issue_number, comment)

        # Save enriched result
        enriched_result_path = os.path.join(WORK_DIR, str(issue_number), "result.json")
        try:
            with open(enriched_result_path, "w") as f:
                json.dump(result, f, indent=2)
        except OSError:
            pass

        logger.info("Issue #%d: %s → %s", issue_number, result["classification"], final_label)


# ---------------------------------------------------------------------------
# Conservative auto-close
# ---------------------------------------------------------------------------

def auto_close_stale_repro_fails() -> None:
    """Auto-close only issues whose classification is in AUTO_CLOSE_CLASSIFICATIONS.

    This is more conservative than the previous approach: we skip
    DIFFERENT_ERROR, ENV_ERROR, and TIMEOUT since those may indicate
    infrastructure problems rather than invalid issues.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=AUTO_CLOSE_AFTER_MINUTES)
    grace_cutoff = datetime.now(timezone.utc) - timedelta(minutes=AUTO_CLOSE_GRACE_MINUTES)

    for issue in gh_fetch_issues(REPRO_FAIL_LABEL):
        num = issue["number"]
        data = gh_fetch_issue_data(num)
        if not data:
            continue

        comments = data.get("comments", [])

        # --- Conservative check: only auto-close proper non-repro cases ---
        classification = _extract_classification_from_comments(comments)
        if classification and classification not in AUTO_CLOSE_CLASSIFICATIONS:
            logger.info(
                "Issue #%d: classification=%s not eligible for auto-close, skipping",
                num, classification,
            )
            continue

        bot_time = None
        warning_time = None

        for c in reversed(comments):
            body = c.get("body", "")
            if warning_time is None and "Inductor Agent" in body and "Scheduled for Auto-close" in body:
                try:
                    warning_time = datetime.fromisoformat(c["createdAt"].replace("Z", "+00:00"))
                except (KeyError, ValueError):
                    pass
            if bot_time is None and "Inductor Agent" in body and ("Reproduction Failed" in body or "Predefined Mitigation Applied" in body):
                try:
                    bot_time = datetime.fromisoformat(c["createdAt"].replace("Z", "+00:00"))
                except (KeyError, ValueError):
                    pass
            if bot_time and warning_time:
                break

        if not bot_time or bot_time > cutoff:
            continue

        human_replied_after = bot_time
        if warning_time and warning_time > bot_time:
            human_replied_after = warning_time

        human_replied = any(
            "Inductor Agent" not in c.get("body", "")
            and datetime.fromisoformat(c.get("createdAt", "2000-01-01T00:00:00Z").replace("Z", "+00:00")) > human_replied_after
            for c in comments
        )
        if human_replied:
            continue

        if warning_time is None:
            logger.info("Posting auto-close warning on issue #%d", num)
            gh_post_comment(num, (
                "🤖 **Inductor Agent — Scheduled for Auto-close**\n\n"
                "This issue was marked as `NOT_REPRODUCED` and no response was received "
                f"within {AUTO_CLOSE_AFTER_MINUTES} minutes.\n\n"
                f"This issue will be **automatically closed in {AUTO_CLOSE_GRACE_MINUTES} minutes** "
                "unless a human responds.\n\n"
                "If this was closed in error, please reopen with additional reproduction details."
            ))
            continue

        if warning_time > grace_cutoff:
            continue

        logger.info("Auto-closing issue #%d (grace period expired)", num)
        gh_post_comment(num, (
            "🤖 **Inductor Agent — Auto-closing**\n\n"
            "The grace period has expired with no response. Closing this issue.\n\n"
            "If this was closed in error, please reopen with additional reproduction details."
        ))
        gh_close_issue(num)


def _extract_classification_from_comments(comments: list[dict]) -> str | None:
    """Extract the classification from the bot's result comment."""
    for c in reversed(comments):
        body = c.get("body", "")
        if "Inductor Agent" not in body:
            continue
        match = re.search(r"\*\*Result\*\*\s*\|\s*[^\|]*?\s+([\w_]+)\s*\|", body)
        if match:
            return match.group(1)
    return None


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------

def cleanup_old_workdirs() -> None:
    if not os.path.exists(WORK_DIR):
        return
    cutoff = time.time() - (WORKDIR_CLEANUP_DAYS * 86400)
    for entry in os.listdir(WORK_DIR):
        path = os.path.join(WORK_DIR, entry)
        if os.path.isdir(path) and os.path.getmtime(path) < cutoff:
            shutil.rmtree(path)
            logger.info("Cleaned up workdir: %s", entry)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def poll_once() -> int:
    if not ensure_nightly_env():
        logger.error("pytorch-nightly not ready, skipping cycle")
        return 0
    issues = gh_fetch_issues(TRIGGER_LABEL)
    for issue in issues:
        process_issue(issue["number"])
    auto_close_stale_repro_fails()
    return len(issues)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inductor Agent")
    parser.add_argument("--loop", action="store_true", help="Poll continuously")
    parser.add_argument("--issue", type=int, help="Process a single issue")
    args = parser.parse_args()

    r = subprocess.run(["gh", "auth", "status"], capture_output=True, timeout=15)
    if r.returncode != 0:
        logger.error("gh auth failed. Run: gh auth login")
        sys.exit(1)

    cleanup_old_workdirs()

    if args.issue:
        if not ensure_nightly_env():
            sys.exit(1)
        process_issue(args.issue)
    elif args.loop:
        logger.info("Starting loop (interval=%ds)", POLL_INTERVAL_SECONDS)
        while True:
            poll_once()
            time.sleep(POLL_INTERVAL_SECONDS)
    else:
        poll_once()


if __name__ == "__main__":
    main()
