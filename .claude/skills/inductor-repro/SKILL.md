---
name: inductor-repro
description: Reproduce a PyTorch Inductor GitHub issue. Reads issue data from a local file, runs a repro script in the pytorch-nightly conda env, and writes a result.json file. Handles issues with no clean repro by synthesizing scripts from descriptions.
---

# Reproduce a PyTorch Inductor Issue

Read a GitHub issue from a local JSON file, write a repro script, run it 3 times, identify affected areas, and write a `result.json`.

**You have NO network access.** Everything you need is in local files.

## Inputs

- The **issue number** (e.g., `1`)
- The **path to `issue.json`** (e.g., `workdir/1/issue.json`)

Replace `{N}` with the actual issue number in all commands.

## Steps

### 1. Read the issue

Read `issue.json`. Look at both `body` and `comments` for:
- Code snippets (in ``` code blocks)
- Stack traces
- Error messages
- Descriptions of the problem (even if no code is provided)

**Classify the issue content:**
- **Has code blocks with `import torch` / `torch.compile`** → proceed to Step 2 (standard repro)
- **Has code fragments but incomplete** → proceed to Step 2a (synthesize repro)
- **Has only descriptions / error messages, no code** → proceed to Step 2a (synthesize repro)

### 2. Write the repro script (standard path)

```bash
mkdir -p workdir/{N}
```

Write `workdir/{N}/repro.py`:
- Complete, self-contained, all imports included
- Use the exact code from the issue
- Do NOT download model weights or datasets

Proceed to Step 3.

### 2a. Synthesize a repro script (no clean code provided)

When the issue does NOT contain a clean repro script, you MUST attempt to
create one from the available information:

1. **Extract key information** from the issue:
   - What operation/model is mentioned? (e.g., "softmax", "attention", "conv2d")
   - What is the reported behavior? (e.g., "incorrect output", "crash", "slow")
   - Any tensor shapes, dtypes, or device info mentioned?
   - Any specific `torch.compile` options mentioned? (backend, mode, fullgraph)
   - Any error class mentioned? (RuntimeError, AssertionError, etc.)

2. **Check for common issue categories** (see Step 2b):
   - Numerical accuracy / tolerance issues
   - Precision / dtype issues
   - Dynamic shapes issues
   - Graph break issues

3. **Build a minimal repro** using this template:
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

4. Write the synthesized script to `workdir/{N}/repro.py`
5. Proceed to Step 3

### 2b. Predefined checks for common categories

If the issue matches one of these categories, run the appropriate predefined
check IN ADDITION to the standard repro:

#### Numerical / Tolerance Issues
**Triggers:** "allclose", "atol", "rtol", "numerical", "tolerance", "accuracy",
"mismatch", "max diff", "precision", "incorrect output", "wrong result"

**Action:** Test with multiple tolerance levels and precision modes:
```python
# Test at various tolerances
for atol in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
    close = torch.allclose(eager_out, compiled_out, atol=atol, rtol=1e-6)
    print(f"atol={atol}: {'PASS' if close else 'FAIL'}")
```

If the difference is very small (< 1e-6), note in the result that this may be
expected floating-point reordering behavior.

#### Precision Cast / Mixed Precision Issues
**Triggers:** "float16", "fp16", "bfloat16", "bf16", "mixed precision",
"autocast", "amp", "half"

**Action:** Test the operation in multiple dtypes and compare:
```python
for dtype in [torch.float32, torch.float16, torch.bfloat16]:
    x = torch.randn(shape, device=device, dtype=dtype)
    eager = fn(x)
    compiled = torch.compile(fn)(x)
    diff = (eager.float() - compiled.float()).abs().max().item()
    print(f"{dtype}: max_diff={diff}")
```

If all differences are < 1e-3, this is likely expected precision behavior.

#### External Library Issues
**Triggers:** imports from `transformers`, `diffusers`, `timm`

**Action:** Note in `result.json` which external libraries are required. The
orchestrator (main.py) will handle installing them. If an external library is
not available and causes an ImportError, classify as `ENV_ERROR` with reason
noting the missing dependency.

### 3. Run 3 times

```bash
TORCH_INDUCTOR_DISABLE_CACHE=1 conda run -n pytorch-nightly python workdir/{N}/repro.py
```

Classify:

| Result | Condition |
|---|---|
| `REPRODUCED` | All 3 fail with the reported error |
| `FLAKY` | Some fail, some pass |
| `NOT_REPRODUCED` | All 3 pass |
| `DIFFERENT_ERROR` | Fails with a different error |
| `ENV_ERROR` | Env is broken (missing package, etc.) |
| `PREDEFINED_MITIGATION` | Standard mitigation applies (small numerical diff, expected precision) |
| `SYNTHESIZED_REPRO` | Repro was synthesized and may not exactly match the reported issue |

If `DIFFERENT_ERROR` is fixable (missing import/package), fix and re-run (up to 3 attempts).

**For synthesized repros:** If the synthesized script does not reproduce the
exact error described but produces a *different* error on the same operation,
classify as `SYNTHESIZED_REPRO` rather than `DIFFERENT_ERROR`.

**For numerical/precision issues:** If the max difference is very small
(< 1e-6) and consistent across all 3 runs, classify as `PREDEFINED_MITIGATION`
with a clear reason explaining this is expected behavior.

### 4. Identify affected areas

Read `codeowners.py` (in the project root). Based on the traceback and error type, pick matching areas. Include them in `result.json`.

### 5. Write result.json

Write `workdir/{N}/result.json`:

```json
{
  "classification": "REPRODUCED",
  "reason": "All 3 runs failed with RuntimeError: ...",
  "runs_failed": 3,
  "runs_total": 3,
  "error_output": "stderr from last run (up to 3000 chars)",
  "repro_script": "contents of repro.py",
  "matched_areas": ["Dynamic Shapes", "Lowering"],
  "synthesized_repro": false,
  "partner_deps_needed": [],
  "predefined_category": ""
}
```

**Extra fields:**
- `synthesized_repro` (bool): true if you created the repro from descriptions
- `partner_deps_needed` (list[str]): e.g. `["transformers", "timm"]` — external libraries required
- `predefined_category` (str): e.g. `"numerical_tolerance"`, `"precision_cast"`

**This file is mandatory.** Always write it, even on failure.

### 6. Print summary

```
=== INDUCTOR AGENT RESULT ===
Issue:        #{N}
Result:       {classification}
Runs:         {runs_failed}/{runs_total} failed
Env:          pytorch-nightly
Synthesized:  {yes/no}
External deps: {list or none}
Predefined:   {category or none}
===========================
```

## Constraints

- **No network.** No `gh`, `curl`, `wget`.
- **Use `conda run -n pytorch-nightly`** for all commands.
- **Never modify the pytorch source tree.**
- **Always write result.json.**
- **Always attempt a repro** — even if there's no clean code, try to synthesize one.
