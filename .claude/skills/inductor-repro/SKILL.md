---
name: inductor-repro
description: "Reproduce a PyTorch Inductor GitHub issue. Fetches issue from GitHub (or reads from local file), sets up pytorch-nightly env, writes and runs a repro script, detects common issue categories, and writes a result.json."
user-invocable: true
model-invocable: true
allowed-tools: Read, Write, Bash, AskUserQuestion
---

# Reproduce a PyTorch Inductor Issue

Automatically reproduce a PyTorch Inductor GitHub issue: fetch it, set up the
environment, write a repro script, run it 3×, classify the result, and write
`result.json`.

## Inputs

The user or pipeline provides some combination of:

| Input | Example | Effect |
|-------|---------|--------|
| **Issue URL or number** | `https://github.com/pytorch/pytorch/issues/12345` or `12345` | Skill fetches issue via `gh` |
| **`issue.json` path** | `/tmp/workdir/12345/issue.json` | Skill reads directly, skips fetch |
| **`ENV_NAME`** | `pytorch-nightly` or `my-dev-env` | Skill uses this env, skips env discovery |
| **`WORK_DIR`** | `/tmp/inductor_repro_workdir` | Skill uses this directory |


## Step 0: Environment Setup

Before doing anything else, validate the environment. Each check is a gate —
if it fails, try to fix it or ask the user.

### 0a. Determine the work directory

```bash
WORK_DIR="${WORK_DIR:-/tmp/inductor_repro_workdir}"
mkdir -p "$WORK_DIR"
```

### 0b. Determine the conda environment

The skill needs a conda environment with PyTorch installed.

**If `ENV_NAME` is provided** (as input, environment variable, or by the user),
use it directly and jump to Step 0c.

**If `ENV_NAME` is NOT provided**, ask the user (**CRITICAL**):

> I need a conda environment with PyTorch to reproduce this issue. Options:
>
> 1. **Provide your env name** — if you already have one (e.g., `pytorch-nightly`, `my-dev-env`)
> 2. **I'll create one for you** — I'll create a `pytorch-nightly` conda env
>    and install the latest PyTorch nightly (auto-detects GPU/CPU)
> 3. **Create it yourself** — run these commands, then give me the env name:
>    ```bash
>    conda create -y -n pytorch-nightly python=3.12
>    # For GPU (check your CUDA version with nvidia-smi):
>    conda run -n pytorch-nightly pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
>    # For CPU only:
>    conda run -n pytorch-nightly pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
>    ```

**If user picks option 2 (auto-create):**

Detect hardware and create the env:

```bash
conda create -y -n pytorch-nightly python=3.12
```

```bash
# Detect GPU
nvidia-smi > /dev/null 2>&1
```

If GPU available — read CUDA version and map to the correct nightly index:
```bash
CUDA_MAJOR_MINOR=$(nvidia-smi | grep -oP 'CUDA Version: \K[\d.]+')
echo "Detected CUDA: $CUDA_MAJOR_MINOR"
```

| Driver CUDA Version | PyTorch index |
|---------------------|---------------|
| 12.6.x | `cu126` |
| 12.8.x | `cu128` |
| 13.0.x | `cu130` |

Pick the **highest `cuXYZ` that does not exceed** the driver CUDA version.
If unsure, default to `cu126` (widely compatible).

```bash
conda run -n pytorch-nightly pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu${CUDA_TAG}
```

If no GPU:
```bash
conda run -n pytorch-nightly pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cpu
```

Set `ENV_NAME=pytorch-nightly`.

### 0c. Validate the environment (smoke test)

Run this against `ENV_NAME`:

```bash
conda run -n ${ENV_NAME} python -c "
import sys, traceback

# Phase 1: torch import
try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available:  {torch.cuda.is_available()}')
except ImportError as e:
    print(f'FAIL_PHASE: torch_import')
    print(f'ERROR: {e}')
    sys.exit(1)

# Phase 2: CUDA setup (if available)
if torch.cuda.is_available():
    try:
        print(f'CUDA version:    {torch.version.cuda}')
        print(f'GPU:             {torch.cuda.get_device_name(0)}')
    except Exception as e:
        print(f'FAIL_PHASE: cuda_init')
        print(f'ERROR: {e}')
        traceback.print_exc()
        sys.exit(2)

    try:
        import triton
        print(f'Triton version:  {triton.__version__}')
    except ImportError:
        print(f'FAIL_PHASE: triton_import')
        sys.exit(3)

# Phase 3: torch.compile smoke test
device = 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    @torch.compile
    def _smoke(x):
        return x + 1
    _smoke(torch.randn(4, device=device))
    print(f'Inductor smoke test ({device}): PASS')
except Exception as e:
    print(f'FAIL_PHASE: inductor_compile')
    print(f'ERROR: {type(e).__name__}: {e}')
    traceback.print_exc()
    sys.exit(4)
" 2>&1
```

**Diagnose and fix based on exit code and error output:**

**Exit 0 — PASS:** Environment is ready. Proceed to Step 1.

**Exit 1 — `torch` import failed:**
Read the error. Common causes:
- `ModuleNotFoundError: No module named 'torch'` → torch not installed:
  ```bash
  conda run -n ${ENV_NAME} pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu126
  ```
- `ImportError: libcudart.so.XX: cannot open shared object file` → CUDA libs missing.
  Reinstall with correct CUDA version (check `nvidia-smi`).
- Other ImportError → read the message, it tells you what's missing.

Re-run smoke test after fix.

**Exit 2 — CUDA initialization failed:**
Read the error. Common causes:
- `CUDA error: no kernel image is available for execution on the device` →
  PyTorch CUDA version doesn't match driver. Compare:
  ```bash
  nvidia-smi | grep "CUDA Version"   # Driver supports up to this
  conda run -n ${ENV_NAME} python -c "import torch; print(torch.version.cuda)"  # PyTorch built for this
  ```
  Fix: reinstall torch with the correct CUDA index matching your driver.
- `CUDA error: out of memory` → GPU memory full. Run `nvidia-smi` to check,
  kill other processes, or ask user.
- `RuntimeError: CUDA unknown error` → driver issue. Tell user to check `nvidia-smi`.

**Exit 3 — Triton not installed:**
```bash
conda run -n ${ENV_NAME} pip install triton
```
Re-run smoke test.

**Exit 4 — `torch.compile` / inductor failed:**
This is the most informative failure. The traceback shows exactly what broke
in the inductor pipeline. **Read the full traceback carefully.** Common causes:

- `torch._dynamo.exc.BackendCompilerFailed` → inductor backend bug. Read the
  inner exception. This could be a genuine inductor bug in this nightly build.
- `triton.compiler.errors.CompilationError` → Triton codegen issue. May be a
  version mismatch between torch and triton.
- `CppCompileError` → C++ compiler issue. Check if `gcc`/`g++` is available.
- `subprocess.CalledProcessError` → compilation subprocess failed. Read stderr.

For inductor compile failures, ask the user:
> `torch.compile` failed on a trivial function. This indicates an issue with
> the inductor backend in this nightly build, NOT your issue's code.
>
> Error: `{type}: {message}`
> Full traceback: (show it)


**`conda` not found:**
Ask user:
> `conda` command not found. Options:
> 1. Install miniconda: `curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh | bash`
> 2. Provide an existing virtualenv with PyTorch nightly

**General principle:** Always read the actual error output. Classify which
phase failed, read the exception type and message, then apply the specific
fix. If you can fix it programmatically, do it and re-run. If you can't,
show the user the exact error with actionable options.

**Use `ENV_NAME` for ALL subsequent commands.**
From this point on, all commands use:
```bash
conda run -n ${ENV_NAME} python ...
```

### 0d. Install external dependencies (if needed)

After fetching the issue (Step 1), scan the issue text for import markers.
If any of these libraries are referenced, install them.

**CRITICAL: Record the torch version BEFORE installing any external package.**
Some packages (e.g., `transformers`, `accelerate`) can silently downgrade or
replace PyTorch with a stable release, breaking the nightly.

```bash
# Record current torch version BEFORE installing anything
TORCH_BEFORE=$(conda run -n ${ENV_NAME} python -c "import torch; print(torch.__version__)")
echo "Torch version before: $TORCH_BEFORE"
```

| Library | Import markers | Packages to install |
|---------|---------------|---------------------|
| **transformers** | `import transformers`, `from transformers` | `transformers accelerate datasets` |
| **diffusers** | `import diffusers`, `from diffusers` | `diffusers transformers accelerate` |
| **timm** | `import timm`, `from timm` | `timm` |

Install with `--no-deps` on torch-related packages to prevent PyTorch replacement,
then install remaining deps normally:

```bash
conda run -n ${ENV_NAME} pip install --quiet --no-deps <packages>
conda run -n ${ENV_NAME} pip install --quiet <packages> 2>&1 || true
```

**After install — verify torch was NOT replaced:**

```bash
TORCH_AFTER=$(conda run -n ${ENV_NAME} python -c "import torch; print(torch.__version__)")
echo "Torch version after: $TORCH_AFTER"
if [ "$TORCH_BEFORE" != "$TORCH_AFTER" ]; then
    echo "WARNING: PyTorch was changed from $TORCH_BEFORE to $TORCH_AFTER!"
    echo "Restoring nightly..."
    conda run -n ${ENV_NAME} pip install --pre --force-reinstall torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/nightly/cu126
fi
```

**Verify the external package imports correctly:**
```bash
conda run -n ${ENV_NAME} python -c "import <package>; print(<package>.__version__)"
```

**Note on issue-reported PyTorch version:** The issue may report a specific
torch version (e.g., "this happens on torch 2.6.0"). For now, **always use the
latest nightly**.

---

## Step 1: Fetch the issue

### Option A: GitHub URL or issue number provided

Try to fetch the issue using `gh`:

```bash
gh issue view {N} --repo pytorch/pytorch --json title,body,labels,comments,state,author
```

**If `gh` succeeds:** Save the output to `$WORK_DIR/{N}/issue.json`.

**If `gh` fails** (auth error, network restricted, `gh` not installed):

Ask the user:
> I couldn't fetch the issue from GitHub. This can happen if:
> - `gh` CLI is not installed or not authenticated
> - Network access is restricted
>
> You can either:
> 1. Run `gh auth login` to authenticate, then I'll retry
> 2. Paste the issue content directly here
> 3. Provide a path to a pre-fetched `issue.json` file
>
> To fetch it yourself:
> ```bash
> gh issue view {N} --repo pytorch/pytorch --json title,body,labels,comments > issue.json
> ```

### Option B: Local `issue.json` path provided

Read the file directly. This is the path used when main.py pipeline has
already fetched the issue.

### Save issue data

```bash
mkdir -p $WORK_DIR/{N}
```

Write or copy the issue data to `$WORK_DIR/{N}/issue.json`.

---

## Step 2: Analyze the issue and detect categories

Read the issue JSON. Look at both `body` and `comments` for:
- Code snippets (in ``` code blocks)
- Stack traces and error messages
- Descriptions of the problem

**Classify the issue content:**
- **Has code blocks with `import torch` / `torch.compile`** → proceed to Step 3 (standard repro)
- **Has code fragments but incomplete** → proceed to Step 3a (synthesize repro)
- **Has only descriptions / error messages, no code** → proceed to Step 3a (synthesize repro)

**Detect predefined categories** by scanning the issue text for keywords.
If 2+ keywords from any category below match, that category applies.
Multiple categories can apply; prioritize the one with the most keyword matches.

---

#### Category: `numerical_tolerance`

**Keywords:** allclose, atol, rtol, numerical, tolerance, accuracy, mismatch,
max diff, precision, incorrect output, wrong result

**What to do:** In your repro script, you MUST:
1. Compare eager vs compiled output with `torch.allclose()`
2. Test at multiple tolerances: `atol` in `[1e-4, 1e-5, 1e-6, 1e-7, 1e-8]`
3. Test across dtypes: `float32`, `float16`, `bfloat16`
4. Print the max diff and pass/fail at each tolerance
5. If ALL diffs are < **1e-6**, classify as `PREDEFINED_MITIGATION`
   with `predefined_category = "numerical_tolerance"`

**Auto-close eligible:** Yes

---

#### For other category

**Auto-close eligible:** No

---

**After analyzing:** Go back to Step 0d to install any external dependencies
detected in the issue text before writing the repro script.

---

## Step 3: Write the repro script (standard path)

Write `$WORK_DIR/{N}/repro.py`:
- Complete, self-contained, all imports included
- Use the exact code from the issue
- If a predefined category was detected, include the extra checks
  described in that category's "What to do" section

Proceed to Step 4.

## Step 3a: Synthesize a repro script (no clean code provided)

When the issue does NOT contain a clean repro script, you MUST attempt to
create one from the available information:

1. **Extract key information** from the issue:
   - What operation/model is mentioned? (e.g., "softmax", "attention", "conv2d")
   - What is the reported behavior? (e.g., "incorrect output", "crash", "slow")
   - Any tensor shapes, dtypes, or device info mentioned?
   - Any specific `torch.compile` options mentioned? (backend, mode, fullgraph)
   - Any error class mentioned? (RuntimeError, AssertionError, etc.)

2. **Build a minimal repro** using this template:
```python
import torch

# Synthesized from issue description
def test_fn(x):
    # <operation described in issue>
    return ...

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

# Create inputs based on described shapes/dtypes
x = torch.randn(<shape>, device=device)

# Eager execution
eager_out = test_fn(x)

# Compiled execution
compiled_fn = torch.compile(test_fn)
compiled_out = compiled_fn(x)

# Compare (for accuracy issues)
if isinstance(eager_out, torch.Tensor):
    max_diff = (eager_out - compiled_out).abs().max().item()
    print(f"Max diff: {max_diff}")
    if not torch.allclose(eager_out, compiled_out, atol=1e-5):
        raise RuntimeError(f"Mismatch: max_diff={max_diff}")
    print("PASSED")
```

3. If a predefined category was detected, include those extra checks too
4. Write the synthesized script to `$WORK_DIR/{N}/repro.py`
5. Proceed to Step 4

---

## Step 4: Run 3 times

```bash
TORCH_INDUCTOR_DISABLE_CACHE=1 conda run -n ${ENV_NAME} python $WORK_DIR/{N}/repro.py
```

Run 3 times. Classify:

| Result | Condition |
|---|---|
| `REPRODUCED` | All 3 fail with the reported error |
| `FLAKY` | Some fail, some pass |
| `NOT_REPRODUCED` | All 3 pass |
| `DIFFERENT_ERROR` | Fails with a different error |
| `ENV_ERROR` | Env is broken (missing package, etc.) |
| `PREDEFINED_MITIGATION` | Standard mitigation applies (see category rules above) |
| `SYNTHESIZED_REPRO` | Repro was synthesized and may not exactly match the reported issue |

If `DIFFERENT_ERROR` is fixable (missing import/package), fix and re-run (up to 3 attempts).

**For synthesized repros:** If the synthesized script does not reproduce the
exact error described but produces a *different* error on the same operation,
classify as `SYNTHESIZED_REPRO` rather than `DIFFERENT_ERROR`.

**For numerical/precision issues:** If the max difference is very small
(< 1e-6) and consistent across all 3 runs, classify as `PREDEFINED_MITIGATION`
with a clear reason explaining this is expected behavior.

---

## Step 5: Write result.json

Write `$WORK_DIR/{N}/result.json`:

```json
{
  "classification": "REPRODUCED",
  "reason": "All 3 runs failed with RuntimeError: ...",
  "runs_failed": 3,
  "runs_total": 3,
  "error_output": "stderr from last run (up to 3000 chars)",
  "repro_script": "contents of repro.py",
  "synthesized_repro": false,
  "external_deps_needed": [],
  "predefined_category": "",
  "torch_version": "<from torch.__version__>",
  "cuda_available": "<from torch.cuda.is_available()>",
  "gpu_name": "<from torch.cuda.get_device_name(0) or empty string>"
}
```

**Fields:**
- `classification` (str): one of the classification values above
- `reason` (str): human-readable explanation
- `runs_failed` / `runs_total` (int): run counts
- `error_output` (str): stderr/stdout from last run (up to 3000 chars)
- `repro_script` (str): full contents of the repro script
- `synthesized_repro` (bool): true if you created the repro from descriptions
- `external_deps_needed` (list[str]): e.g. `["transformers", "timm"]`
- `predefined_category` (str): e.g. `"numerical_tolerance"`, `"precision_cast"`, `""` if none
- `torch_version` (str): the PyTorch version used for reproduction
- `cuda_available` (bool): whether CUDA was available
- `gpu_name` (str): GPU name if CUDA available, else `""`

**This file is mandatory.** Always write it, even on failure.

---

## Step 6: Print summary

```
=== INDUCTOR AGENT RESULT ===
Issue:         #{N}
Result:        {classification}
Runs:          {runs_failed}/{runs_total} failed
Torch:         {torch_version}
GPU:           {gpu_name or "CPU-only"}
Synthesized:   {yes/no}
Category:      {predefined_category or none}
External deps: {list or none}
Work dir:      $WORK_DIR/{N}/
===========================
```

---

## Constraints

- **Use `conda run -n ${ENV_NAME}`** for all Python execution (`ENV_NAME` defaults to `pytorch-nightly`).
- **Never modify the pytorch source tree.**
- **Always write result.json** — even on failure.
- **Always attempt a repro** — even if there's no clean code, try to synthesize one.
- **Always check for predefined categories** — scan issue text for keywords.
- **Be intelligent about the environment** — detect hardware, install matching packages.
- **Ask the user when stuck** — don't fail silently. If something is wrong, explain what
  happened and offer options.
