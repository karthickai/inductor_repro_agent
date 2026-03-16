---
name: inductor-repro
description: Reproduce a PyTorch Inductor GitHub issue. Reads issue data from a local file, runs a repro script in the pytorch-nightly conda env, and writes a result.json file.
---

# Reproduce a PyTorch Inductor Issue

Read a GitHub issue from a local JSON file, write a repro script, run it 3 times, identify affected areas, and write a `result.json`.

**You have NO network access.** Everything you need is in local files.

## Inputs

- The **issue number** (e.g., `1`)
- The **path to `issue.json`** (e.g., `~/workspace/inductor-agent/workdir/1/issue.json`)

Replace `{N}` with the actual issue number in all commands.

## Steps

### 1. Read the issue

Read `issue.json`. Look at both `body` and `comments` for:
- Code snippets
- Stack traces

No code blocks at all -> skip to step 5, write `classification: "NOT_REPRODUCED"`.

### 2. Write the repro script

```bash
mkdir -p ~/workspace/inductor-agent/workdir/{N}
```

Write `~/workspace/inductor-agent/workdir/{N}/repro.py`:
- Complete, self-contained, all imports included
- Use the exact code from the issue
- Do NOT download model weights or datasets

### 3. Run 3 times

```bash
TORCH_INDUCTOR_DISABLE_CACHE=1 conda run -n pytorch-nightly python ~/workspace/inductor-agent/workdir/{N}/repro.py
```

Classify:

| Result | Condition |
|---|---|
| `REPRODUCED` | All 3 fail with the reported error |
| `FLAKY` | Some fail, some pass |
| `NOT_REPRODUCED` | All 3 pass |
| `DIFFERENT_ERROR` | Fails with a different error |
| `ENV_ERROR` | Env is broken |

If `DIFFERENT_ERROR` is fixable (missing import/package), fix and re-run (up to 3 attempts).

### 4. Identify affected areas

Read `~/workspace/inductor-agent/codeowners.py`. Based on the traceback and error type, pick matching areas. Include them in `result.json`.

### 5. Write result.json

Write `~/workspace/inductor-agent/workdir/{N}/result.json`:

```json
{
  "classification": "REPRODUCED",
  "reason": "All 3 runs failed with RuntimeError: ...",
  "runs_failed": 3,
  "runs_total": 3,
  "error_output": "stderr from last run (up to 3000 chars)",
  "repro_script": "contents of repro.py",
  "matched_areas": ["Dynamic Shapes", "Lowering"]
}
```

**This file is mandatory.** Always write it, even on failure.

### 6. Print summary

```
=== INDUCTOR AGENT RESULT ===
Issue:    #{N}
Result:   {classification}
Runs:     {runs_failed}/{runs_total} failed
Env:      pytorch-nightly
===========================
```

## Constraints

- **No network.** No `gh`, `curl`, `wget`.
- **Use `conda run -n pytorch-nightly`** for all commands.
- **Never modify the pytorch source tree.**
- **Always write result.json.**
