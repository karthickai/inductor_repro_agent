"""Configuration for the inductor-agent."""

import os

# GitHub
GITHUB_REPO = os.environ.get("GITHUB_REPO", "karthickai/test-inuductor-agent")
TRIGGER_LABEL = "inductor:agent"
PROCESSING_LABEL = "inductor:agent:processing"
REPRO_SUCCESS_LABEL = "inductor:agent:repro_success"
REPRO_FAIL_LABEL = "inductor:agent:repro_fail"

# Polling
POLL_INTERVAL_SECONDS = int(os.environ.get("POLL_INTERVAL_SECONDS", "2"))

# Nightly env
NIGHTLY_ENV = "pytorch-nightly"
NIGHTLY_MAX_AGE_HOURS = int(os.environ.get("NIGHTLY_MAX_AGE_HOURS", "72"))

# Claude
CLAUDE_TIMEOUT_SECONDS = int(os.environ.get("CLAUDE_TIMEOUT_SECONDS", "2400"))

# GitHub retry
GH_RETRY_ATTEMPTS = 3

# Cleanup
WORKDIR_CLEANUP_DAYS = int(os.environ.get("WORKDIR_CLEANUP_DAYS", "7"))

VALID_CLASSIFICATIONS = {
    "REPRODUCED", "FLAKY", "NOT_REPRODUCED", "DIFFERENT_ERROR",
    "ENV_ERROR", "TIMEOUT",
    "PREDEFINED_MITIGATION",   # standard mitigation applied (e.g. numerical tolerance)
    "SYNTHESIZED_REPRO",       # repro was synthesized from description (no code in issue)
}

# Auto-close NOT_REPRODUCED issues after this period (default: 2 days)
AUTO_CLOSE_AFTER_MINUTES = int(os.environ.get("AUTO_CLOSE_AFTER_MINUTES", str(2 * 24 * 60)))

# Grace period: time between warning comment and actual close (default: 1 day)
AUTO_CLOSE_GRACE_MINUTES = int(os.environ.get("AUTO_CLOSE_GRACE_MINUTES", str(1 * 24 * 60)))

# Conservative auto-close: only these classifications are eligible for auto-close
AUTO_CLOSE_CLASSIFICATIONS = {"NOT_REPRODUCED", "PREDEFINED_MITIGATION"}

# Paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.path.join(PROJECT_DIR, "workdir")
LOG_DIR = os.path.join(PROJECT_DIR, "logs")

# ---------------------------------------------------------------------------
# Nightly bisection
# ---------------------------------------------------------------------------
BISECT_ENABLED = os.environ.get("BISECT_ENABLED", "1") == "1"
BISECT_MAX_VERSIONS = int(os.environ.get("BISECT_MAX_VERSIONS", "7"))
BISECT_LOOKBACK_DAYS = int(os.environ.get("BISECT_LOOKBACK_DAYS", "14"))
BISECT_TIMEOUT_SECONDS = int(os.environ.get("BISECT_TIMEOUT_SECONDS", "300"))

# ---------------------------------------------------------------------------
# External / broader dependency sets
# ---------------------------------------------------------------------------
EXTERNAL_DEPS = {
    "transformers": {
        "pip": ["transformers", "accelerate", "datasets"],
        "import_markers": ["transformers", "from transformers"],
    },
    "diffusers": {
        "pip": ["diffusers", "transformers", "accelerate"],
        "import_markers": ["diffusers", "from diffusers"],
    },
    "timm": {
        "pip": ["timm"],
        "import_markers": ["timm", "from timm"],
    },
}

# Max time to spend installing external deps
EXTERNAL_INSTALL_TIMEOUT = int(os.environ.get("EXTERNAL_INSTALL_TIMEOUT", "300"))
