# inductor-agent

Automated reproduction of PyTorch Inductor GitHub issues using Claude Code.

## Problem

When a PyTorch Inductor issue is filed, the engineer has to read and find code snippets, set up an environment, piece together a runnable repro script, run it multiple times to check for flakiness, and report back, all of which can be automated. This agent does exactly that, reducing a manual process, so the engineer can skip straight to debugging or trigger the debug agent.


## How it works

A cron job polls GitHub for `inductor:agent` labeled issues, fetches the issue body, hands it to Claude Code (headless, sandboxed, no network) which writes a repro script and runs it 3 times against `pytorch-nightly`, then updates the label and posts a structured comment with the results.

## Architecture
![Architecture](assets/overall.png)


## Label flow

![Label](assets/label.png)

## Classification outcomes

| Classification | Meaning | Label | Action |
|---|---|---|---|
| `REPRODUCED` | All 3 runs fail with the reported error | `repro_success` | Tag code owners |
| `FLAKY` | Some runs fail, some pass | `repro_success` | Tag code owners |
| `NOT_REPRODUCED` | All 3 runs pass | `repro_fail` | Auto-close after 2 days if no response |
| `DIFFERENT_ERROR` | Fails with a different error | `repro_fail` | On-call investigates |
| `ENV_ERROR` | Env broken or Claude failed | `repro_fail` | Check logs |
| `TIMEOUT` | Claude exceeded time limit | `repro_fail` | Check logs |


## Code owner tagging

Claude reads `codeowners.py` and identifies which inductor subsystems are affected based on the traceback and error type. Owners are tagged in the comment.


## Nightly env management

```
main.py starts
    │
    ├── pytorch-nightly env missing?
    │       → conda create + pip install nightly
    │
    ├── torch version date > 24h old?
    │       → pip install --upgrade nightly
    │
    └── fresh?
            → do nothing, proceed
            → cache collect_env output (once)
```

The `collect_env` output is captured once at startup via:
```bash
curl -sL https://raw.githubusercontent.com/pytorch/pytorch/main/torch/utils/collect_env.py | $ENV/bin/python
```
Included in every comment under a collapsible `<details>` dropdown.

## Setup

### Prerequisites

```bash
gh auth status
conda --version
claude --version
```

### Run

```bash
cd inductor-agent

# Poll once
python main.py

# Poll continuously or use cron
python main.py --loop

# Single issue
python main.py --issue 1
```


## Configuration

All in `config.py`, overridable via env vars:

| Variable | Default | Description |
|---|---|---|
| `GITHUB_REPO` | `karthickai/test-inuductor-agent` | Repo to monitor |
| `POLL_INTERVAL_SECONDS` | `120` | Loop poll interval |
| `NIGHTLY_MAX_AGE_HOURS` | `72` | Auto-update nightly if older |
| `CLAUDE_TIMEOUT_SECONDS` | `2400` | Hard timeout for Claude |
| `WORKDIR_CLEANUP_DAYS` | `7` | Auto-delete old workdirs |
| `AUTO_CLOSE_AFTER_MINUTES` | `2880` | Auto-close NOT_REPRODUCED after 2 days |
| `GH_RETRY_ATTEMPTS` | `3` | Retry count for gh calls |

## Project structure

```
inductor-agent/
├── main.py                 # Cron: poll → fetch → Claude → label → comment
├── config.py               # All configuration
├── codeowners.py           # Area-to-owner mapping (Claude reads this)
├── .claude/
│   ├── settings.local.json # Claude Code tool permissions
│   └── skills/
│       └── inductor-repro/
│           └── SKILL.md    # Claude skill: read issue → repro → result.json
├── workdir/
│   └── {issue_number}/
│       ├── issue.json          # Written by main.py
│       ├── repro.py            # Written by Claude
│       ├── result.json         # Written by Claude
│       └── claude_output.log   # Claude's full output
├── logs/
│   └── main.log
└── README.md
```
