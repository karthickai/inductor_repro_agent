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

VALID_CLASSIFICATIONS = {"REPRODUCED", "FLAKY", "NOT_REPRODUCED", "DIFFERENT_ERROR", "ENV_ERROR", "TIMEOUT"}

# Auto-close NOT_REPRODUCED issues after this period (default: 2 days)
AUTO_CLOSE_AFTER_MINUTES = int(os.environ.get("AUTO_CLOSE_AFTER_MINUTES", str(2 * 24 * 60)))

# Grace period: time between warning comment and actual close (default: 1 day)
AUTO_CLOSE_GRACE_MINUTES = int(os.environ.get("AUTO_CLOSE_GRACE_MINUTES", str(1 * 24 * 60))))

# Paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.path.join(PROJECT_DIR, "workdir")
LOG_DIR = os.path.join(PROJECT_DIR, "logs")
