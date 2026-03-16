"""Code owners for inductor subsystems.

Claude reads this file to determine who to tag on an issue.
Claude analyzes the error/traceback and picks the matching areas.
"""

# This is test codeowners (need to be updated)
CODEOWNERS = [
    {
        "area": "Pattern Matcher",
        "owners": ["karthickai"],
        "paths": ["torch/_inductor/pattern_matcher", "torch/_inductor/fx_passes/"],
        "description": "Pattern matching passes, joint/pre/post grad FX passes",
    },
    {
        "area": "Combo Kernels",
        "owners": ["karthickai"],
        "paths": ["torch/_inductor/combo_kernels"],
        "description": "Fused combo kernel generation",
    },
    {
        "area": "Numeric / Accuracy",
        "owners": ["karthickai"],
        "paths": [],
        "description": "Numerical accuracy issues, allclose failures, incorrect outputs from torch.compile",
    },
    {
        "area": "Codegen",
        "owners": [],
        "paths": ["torch/_inductor/codegen/", "torch/_inductor/codecache"],
        "description": "Triton/C++ code generation, kernel templates",
    },
    {
        "area": "Scheduler",
        "owners": [],
        "paths": ["torch/_inductor/scheduler"],
        "description": "Kernel scheduling, fusion decisions",
    },
    {
        "area": "Lowering",
        "owners": ["karthickai"],
        "paths": ["torch/_inductor/lowering", "torch/_inductor/decomposition"],
        "description": "ATen op lowering and decompositions",
    },
    {
        "area": "Dynamic Shapes",
        "owners": ["karthickai"],
        "paths": ["torch/_inductor/sizevars", "torch/fx/experimental/symbolic_shapes"],
        "description": "Symbolic shapes, guards, dynamic dimensions",
    },
    {
        "area": "CPP / AOTInductor",
        "owners": [],
        "paths": ["torch/_inductor/codegen/cpp", "torch/_inductor/codegen/aoti"],
        "description": "C++ backend, AOTInductor export",
    },
    {
        "area": "CUDA Graphs",
        "owners": [],
        "paths": ["torch/_inductor/cudagraph"],
        "description": "CUDA graph capture and replay",
    },
]


def format_owner_tags(matched_areas: list[str]) -> str:
    """Format matched area names into GitHub @mentions."""
    if not matched_areas:
        return ""

    area_map = {rule["area"]: rule["owners"] for rule in CODEOWNERS}
    lines = []
    for area in matched_areas:
        owners = area_map.get(area, [])
        if owners:
            lines.append(f"- **{area}**: {', '.join(f'@{o}' for o in owners)}")

    if not lines:
        return ""
    return "**Owners tagged:**\n" + "\n".join(lines)
