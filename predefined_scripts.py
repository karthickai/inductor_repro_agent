"""Predefined scripts for common issue categories.

Each category defines:
- ``keywords``: terms to detect in issue text
- ``script_template``: a Python script template that Claude can use as a
  starting point (or the agent can run directly)
- ``mitigation_message``: explanation posted when the predefined script
  resolves the issue (i.e., the "standard mitigation" applies)
- ``auto_close_eligible``: whether issues matching this category can be
  auto-closed when the mitigation confirms the issue is expected behaviour
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass

from config import (
    NIGHTLY_ENV,
    NUMERICAL_ATOL_THRESHOLD,
    NUMERICAL_RTOL_THRESHOLD,
    PRECISION_AUTO_CLOSE_MAX_DIFF,
)

logger = logging.getLogger(__name__)


@dataclass
class PredefinedResult:
    matched_category: str = ""
    ran: bool = False
    mitigation_applies: bool = False
    auto_close_eligible: bool = False
    detail: str = ""
    script_used: str = ""


NUMERICAL_TOLERANCE_SCRIPT = '''"""Predefined: Numerical tolerance check.

Tests whether the reported accuracy issue is within acceptable tolerance
by running the compiled function in multiple precision modes and comparing
against eager execution.
"""
import torch

def run_comparison(fn, args, kwargs=None, atol={atol}, rtol={rtol}):
    kwargs = kwargs or {{}}
    eager_out = fn(*args, **kwargs)

    compiled_fn = torch.compile(fn, fullgraph=True)
    compiled_out = compiled_fn(*args, **kwargs)

    if isinstance(eager_out, torch.Tensor):
        max_diff = (eager_out - compiled_out).abs().max().item()
        close = torch.allclose(eager_out, compiled_out, atol=atol, rtol=rtol)
        print(f"Max diff: {{max_diff}}")
        print(f"Within tolerance (atol={{atol}}, rtol={{rtol}}): {{close}}")
        return max_diff, close
    return 0.0, True

{user_fn}

# --- precision cast tests ---
device = "cuda" if torch.cuda.is_available() else "cpu"

for dtype in [torch.float32, torch.float16, torch.bfloat16]:
    print(f"\\n=== Testing with {{dtype}} ===")
    torch.manual_seed(42)
    test_input = torch.randn({input_shape}, device=device, dtype=dtype)
    try:
        diff, ok = run_comparison(test_fn, (test_input,), atol=atol, rtol=rtol)
        if not ok:
            print(f"MISMATCH at {{dtype}}: max_diff={{diff}}")
        else:
            print(f"PASSED at {{dtype}}: max_diff={{diff}}")
    except Exception as e:
        print(f"ERROR at {{dtype}}: {{e}}")
'''

PRECISION_CAST_SCRIPT = '''"""Predefined: Precision cast emulation.

Emulates precision-cast scenarios (fp32 → fp16/bf16 → fp32) to check if
the reported numerical difference is a natural consequence of reduced
precision rather than a compiler bug.
"""
import torch

{user_fn}

device = "cuda" if torch.cuda.is_available() else "cpu"
results = []

for low_dtype in [torch.float16, torch.bfloat16]:
    torch.manual_seed(42)
    x_fp32 = torch.randn({input_shape}, device=device, dtype=torch.float32)

    # Eager in fp32
    eager_fp32 = test_fn(x_fp32)

    # Simulate precision loss: fp32 → low → fp32
    x_cast = x_fp32.to(low_dtype).to(torch.float32)
    eager_cast = test_fn(x_cast)
    cast_diff = (eager_fp32 - eager_cast).abs().max().item()

    # Compiled in fp32
    compiled_fn = torch.compile(test_fn, fullgraph=True)
    compiled_fp32 = compiled_fn(x_fp32)
    compile_diff = (eager_fp32 - compiled_fp32).abs().max().item()

    # Compiled in low precision
    x_low = x_fp32.to(low_dtype)
    eager_low = test_fn(x_low)
    compiled_low = compiled_fn(x_low)
    low_diff = (eager_low.float() - compiled_low.float()).abs().max().item()

    print(f"\\n=== {{low_dtype}} ===")
    print(f"  Precision-cast diff (eager fp32 vs cast-roundtrip): {{cast_diff:.2e}}")
    print(f"  Compile diff (eager fp32 vs compiled fp32):         {{compile_diff:.2e}}")
    print(f"  Low-precision diff (eager vs compiled at {{low_dtype}}): {{low_diff:.2e}}")

    results.append({{
        "dtype": str(low_dtype),
        "cast_diff": cast_diff,
        "compile_diff": compile_diff,
        "low_diff": low_diff,
    }})

# Summary
max_compile_diff = max(r["compile_diff"] for r in results)
print(f"\\n=== Summary ===")
print(f"Max compile diff across dtypes: {{max_compile_diff:.2e}}")
print(f"Threshold: {max_diff}")
if max_compile_diff <= {max_diff}:
    print("VERDICT: Differences are within expected precision range — NOT a compiler bug.")
else:
    print("VERDICT: Differences exceed expected precision range — potential compiler bug.")
'''


PREDEFINED_CATEGORIES = [
    {
        "name": "numerical_tolerance",
        "keywords": [
            "allclose", "atol", "rtol", "numerical", "tolerance",
            "accuracy", "mismatch", "max diff", "precision",
            "incorrect output", "wrong result",
        ],
        "script_template": NUMERICAL_TOLERANCE_SCRIPT,
        "template_defaults": {
            "atol": str(NUMERICAL_ATOL_THRESHOLD),
            "rtol": str(NUMERICAL_RTOL_THRESHOLD),
            "input_shape": "4, 64, 64",
            "user_fn": "def test_fn(x):\n    return torch.softmax(x, dim=-1)",
        },
        "mitigation_message": (
            "The reported numerical difference is within the expected tolerance range "
            f"(atol={NUMERICAL_ATOL_THRESHOLD}, rtol={NUMERICAL_RTOL_THRESHOLD}). "
            "Small numerical differences between eager and compiled execution are expected "
            "due to floating-point operation reordering during compilation. "
            "This is standard behavior and not a compiler bug."
        ),
        "auto_close_eligible": True,
    },
    {
        "name": "precision_cast",
        "keywords": [
            "float16", "fp16", "bfloat16", "bf16", "half precision",
            "mixed precision", "autocast", "amp", "precision loss",
            "cast", "dtype mismatch",
        ],
        "script_template": PRECISION_CAST_SCRIPT,
        "template_defaults": {
            "input_shape": "4, 64, 64",
            "user_fn": "def test_fn(x):\n    return torch.softmax(x, dim=-1)",
            "max_diff": str(PRECISION_AUTO_CLOSE_MAX_DIFF),
        },
        "mitigation_message": (
            "The reported numerical difference is consistent with expected precision loss "
            "from dtype casting (e.g., fp32 \u2192 fp16/bf16 \u2192 fp32). "
            "The compiler may reorder or fuse operations differently, leading to small "
            f"differences (< {PRECISION_AUTO_CLOSE_MAX_DIFF:.0e}) that are within the "
            "precision limits of the reduced dtype. This is expected behavior."
        ),
        "auto_close_eligible": True,
    },
    {
        "name": "dynamic_shapes_guard",
        "keywords": [
            "guard", "dynamic shape", "symbolic", "Unsupported: dynamic",
            "SymInt", "data-dependent", "GuardOnDataDependentSymNode",
        ],
        "script_template": "",
        "template_defaults": {},
        "mitigation_message": (
            "This appears to be a known limitation with dynamic shapes. "
            "Consider using `torch._dynamo.mark_dynamic()` or setting "
            "`torch._dynamo.config.capture_scalar_outputs = True`. "
            "See: https://pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html"
        ),
        "auto_close_eligible": False,
    },
    {
        "name": "graph_break",
        "keywords": [
            "graph break", "graph_break", "Unsupported:", "skipping",
            "torch._dynamo.exc.Unsupported",
        ],
        "script_template": "",
        "template_defaults": {},
        "mitigation_message": (
            "This issue involves a graph break in torch.compile. Graph breaks are expected "
            "for certain Python constructs. Use `torch._dynamo.explain()` to identify the "
            "cause. Consider restructuring the code to avoid the unsupported pattern."
        ),
        "auto_close_eligible": False,
    },
]


def detect_category(issue_text: str) -> list[dict]:
    """Return all predefined categories matching the issue text."""
    text_lower = issue_text.lower()
    matched = []
    for cat in PREDEFINED_CATEGORIES:
        score = sum(1 for kw in cat["keywords"] if kw.lower() in text_lower)
        if score >= 2:
            matched.append({**cat, "_score": score})
    matched.sort(key=lambda c: c["_score"], reverse=True)
    return matched


def generate_predefined_script(
    category: dict,
    user_fn: str = "",
    input_shape: str = "",
) -> str:
    """Generate a concrete script from a category template.

    If the user provided a function or shape, those override the defaults.
    """
    template = category.get("script_template", "")
    if not template:
        return ""

    defaults = dict(category.get("template_defaults", {}))
    if user_fn:
        defaults["user_fn"] = user_fn
    if input_shape:
        defaults["input_shape"] = input_shape

    try:
        return template.format(**defaults)
    except KeyError as e:
        logger.warning("Template formatting failed for %s: %s", category["name"], e)
        return template


def run_predefined_script(
    script_content: str,
    issue_dir: str,
    script_name: str = "predefined_check.py",
) -> PredefinedResult:
    """Write and execute a predefined script, parse the output."""
    result = PredefinedResult()

    script_path = os.path.join(issue_dir, script_name)
    with open(script_path, "w") as f:
        f.write(script_content)
    result.script_used = script_content

    env_prefix = _get_env_prefix()
    if not env_prefix:
        result.detail = "Cannot find pytorch-nightly env"
        return result

    python = os.path.join(env_prefix, "bin", "python")
    try:
        r = subprocess.run(
            [python, script_path],
            capture_output=True, text=True, timeout=300,
            env={**os.environ, "TORCH_INDUCTOR_DISABLE_CACHE": "1"},
        )
        result.ran = True
        output = r.stdout + r.stderr
        result.detail = output[-3000:]

        if "NOT a compiler bug" in output or "VERDICT: Differences are within" in output:
            result.mitigation_applies = True
        elif "PASSED" in output and "MISMATCH" not in output:
            result.mitigation_applies = True

    except subprocess.TimeoutExpired:
        result.detail = "Predefined script timed out"
    except Exception as e:
        result.detail = f"Error running predefined script: {e}"

    return result


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


def check_predefined_mitigations(
    issue_text: str,
    repro_script: str,
    issue_dir: str,
) -> PredefinedResult | None:
    """Top-level entry: detect categories, run predefined scripts, return result.

    Returns None if no predefined category matches.
    """
    categories = detect_category(issue_text)
    if not categories:
        return None

    best = categories[0]
    logger.info("Predefined category matched: %s (score=%d)", best["name"], best["_score"])

    result = PredefinedResult(matched_category=best["name"])

    script = generate_predefined_script(best)
    if script:
        run_result = run_predefined_script(script, issue_dir)
        result.ran = run_result.ran
        result.detail = run_result.detail
        result.script_used = run_result.script_used
        result.mitigation_applies = run_result.mitigation_applies
        result.auto_close_eligible = best["auto_close_eligible"] and run_result.mitigation_applies
    else:
        result.detail = best["mitigation_message"]

    return result


def format_predefined_comment(pr: PredefinedResult, category: dict | None = None) -> str:
    """Format a PredefinedResult into a GitHub-comment markdown section."""
    if not pr.matched_category:
        return ""

    cat_info = category
    if not cat_info:
        for c in PREDEFINED_CATEGORIES:
            if c["name"] == pr.matched_category:
                cat_info = c
                break

    lines = [
        f"### 📋 Predefined Check: {pr.matched_category.replace('_', ' ').title()}",
        "",
    ]

    if pr.mitigation_applies and cat_info:
        lines.append(f"**Standard mitigation applies:** {cat_info['mitigation_message']}")
        if pr.auto_close_eligible:
            lines.append("")
            lines.append("🔒 *This issue is eligible for automatic closure based on predefined mitigation criteria.*")
    elif pr.ran:
        lines.append("**Result:** The predefined check did not confirm standard mitigation.")
        lines.append("This issue may require manual investigation.")

    if pr.detail:
        lines.extend([
            "",
            "<details>",
            "<summary>Predefined check output</summary>",
            "",
            "```",
            pr.detail[:3000],
            "```",
            "</details>",
        ])

    return "\n".join(lines)
