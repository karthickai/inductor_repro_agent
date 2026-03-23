---
name: inductor-repro
description: Reproduce a PyTorch Inductor GitHub issue. Reads issue data from a local file, writes and runs a repro script in the pytorch-nightly conda env, detects common issue categories, and writes a result.json file.
---

# Reproduce a PyTorch Inductor Issue

Read a GitHub issue from a local JSON file, write a repro script, run it 3 times, identify affected areas, and write a `result.json`.

**You have NO network access.** Everything you need is in local files.

## Inputs

- The **issue number** (e.g., `1`)
- The **path to `issue.json`** (e.g., `workdir/1/issue.json`)

Replace `{N}` with the actual issue number in all commands.

## Steps

### 1. Read the issue and detect category

Read `issue.json`. Look at both `body` and `comments` for:
- Code snippets (in ``` code blocks)
- Stack traces
- Error messages
- Descriptions of the problem (even if no code is provided)

**Classify the issue content:**
- **Has code blocks with `import torch` / `torch.compile`** → proceed to Step 2 (standard repro)
- **Has code fragments but incomplete** → proceed to Step 2a (synthesize repro)
- **Has only descriptions / error messages, no code** → proceed to Step 2a (synthesize repro)

**Detect predefined categories** by scanning the issue text for keywords.
If 2 or more keywords from any category below match, that category applies.
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

#### Category: `precision_cast`

**Keywords:** float16, fp16, bfloat16, bf16, half precision, mixed precision,
autocast, amp, precision loss, cast, dtype mismatch

**What to do:** In your repro script, you MUST:
1. Test the operation in `float32`, `float16`, and `bfloat16`
2. Compare eager vs compiled at each dtype
3. Also test precision-cast roundtrip: cast `fp32→low→fp32` in eager,
   compare against straight `fp32` eager to show inherent precision loss
4. Compare the compile diff against the cast-roundtrip diff
5. If compile diff is < **1e-6**, classify as `PREDEFINED_MITIGATION`
   with `predefined_category = "precision_cast"`

**Auto-close eligible:** Yes

---

#### Category: `dynamic_shapes_guard`

**Keywords:** guard, dynamic shape, symbolic, Unsupported: dynamic, SymInt,
data-dependent, GuardOnDataDependentSymNode

**What to do:** In your repro script, you MUST:
1. Test with both static and dynamic inputs
2. Try `torch._dynamo.mark_dynamic()` on varying dimensions
3. Check if the error changes with
   `torch._dynamo.config.capture_scalar_outputs = True`
4. Report which workarounds help (if any) in the result reason
5. Do NOT classify as `PREDEFINED_MITIGATION` — classify normally

**Auto-close eligible:** No

---

#### Category: `graph_break`

**Keywords:** graph break, graph_break, Unsupported:, skipping,
torch._dynamo.exc.Unsupported

**What to do:** In your repro script, you MUST:
1. Add `torch._dynamo.explain()` output
2. Try compiling with `fullgraph=False` and `fullgraph=True`
3. Report the graph break reason in the result
4. Do NOT classify as `PREDEFINED_MITIGATION` — classify normally

**Auto-close eligible:** No

---

### 2. Write the repro script (standard path)

```bash
mkdir -p workdir/{N}
```

Write `workdir/{N}/repro.py`:
- Complete, self-contained, all imports included
- Use the exact code from the issue
- Do NOT download model weights or datasets
- If a predefined category was detected, include the extra checks
  described in that category's "What to do" section

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
4. Write the synthesized script to `workdir/{N}/repro.py`
5. Proceed to Step 3

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
| `PREDEFINED_MITIGATION` | Standard mitigation applies (see category rules above) |
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
  "external_deps_needed": [],
  "predefined_category": ""
}
```

**Fields:**
- `classification` (str): one of the classification values above
- `reason` (str): human-readable explanation
- `runs_failed` / `runs_total` (int): run counts
- `error_output` (str): stderr/stdout from last run (up to 3000 chars)
- `repro_script` (str): full contents of the repro script
- `matched_areas` (list[str]): areas from codeowners.py
- `synthesized_repro` (bool): true if you created the repro from descriptions
- `external_deps_needed` (list[str]): e.g. `["transformers", "timm"]`
- `predefined_category` (str): e.g. `"numerical_tolerance"`, `"precision_cast"`, `""` if none

**This file is mandatory.** Always write it, even on failure.

### 6. Print summary

```
=== INDUCTOR AGENT RESULT ===
Issue:         #{N}
Result:        {classification}
Runs:          {runs_failed}/{runs_total} failed
Env:           pytorch-nightly
Synthesized:   {yes/no}
Category:      {predefined_category or none}
External deps: {list or none}
===========================
```

## Constraints

- **No network.** No `gh`, `curl`, `wget`.
- **Use `conda run -n pytorch-nightly`** for all commands.
- **Never modify the pytorch source tree.**
- **Always write result.json.**
- **Always attempt a repro** — even if there's no clean code, try to synthesize one.
- **Always check for predefined categories** — scan the issue text for keywords and follow the category instructions.
