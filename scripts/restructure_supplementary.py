#!/usr/bin/env python3
"""Restructure supplementary.tex so figures appear in S1-S19 citation order."""
import re
from pathlib import Path

SUPP = Path("manuscript/elsevier_style/supplementary.tex")
content = SUPP.read_text()

# ── Extract all blocks (figures, tables, inputs) with their labels ──
fig_pat = re.compile(r'(\\begin\{figure\}\[H\].*?\\end\{figure\})', re.DOTALL)
tbl_pat = re.compile(r'(\\begin\{table\}\[H\].*?\\end\{table\})', re.DOTALL)

blocks = {}  # label → full LaTeX block
for pat in [fig_pat, tbl_pat]:
    for match in pat.finditer(content):
        block = match.group(1)
        lm = re.search(r'\\label\{([^}]+)\}', block)
        if lm:
            blocks[lm.group(1)] = block

# Extract \input blocks
for m in re.finditer(r'(\\input\{([^}]+)\})', content):
    blocks[f"input:{m.group(2)}"] = m.group(1)

# Extract \FloatBarrier and \subsection/\section commands near each block
# We'll just rebuild from the preamble + reordered content

# ── Preamble (everything before first \section) ──
preamble_end = content.find("\\section{")
preamble = content[:preamble_end]

# ── Build expected order ──
expected_order = [
    # Artefact Documentation (S1)
    ("\\section{Artefact Documentation}\n\n\\subsection{Artefact photographs}\n\n", None),
    ("fig:supp_ceramic_standard", None),  # contains both ceramic+steel photos in one figure

    # Experimental Setup
    ("\\section{Experimental Setup}\n\n", None),
    ("\\subsection{Rotation stage specifications}\n\n", None),
    ("tab:supp_stage_spec", "Table S1"),
    ("\\subsection{Environmental shielding and vibration isolation}\n\n", None),
    ("fig:supp_blanket", "S2"),  # contains both blanket+styrofoam in one figure

    # Phase Extraction (S3-S6)
    ("\\section{Image Processing --- Phase Extraction}\n\n", None),
    ("\\subsection{Single-image radial phase extraction}\n\n", None),
    ("fig:supp_ceramic_radial_phase", "S3"),
    ("fig:supp_steel_radial_phase", "S4"),
    ("\\subsection{Sequence processing steps}\n\n", None),
    ("fig:supp_ceramic_phase_steps", "S5"),
    ("fig:supp_steel_phase_steps", "S6"),

    # Results - Descriptor Comparisons (S7-S11 + Table S3)
    ("\\section{Results --- Descriptor Comparisons}\n\n", None),
    ("\\subsection{Descriptor comparison and uncertainty}\n\n", None),
    ("fig:supp_metric_comparison", "S7"),
    ("fig:supp_uncertainty", "S8"),
    ("\\subsection{Profile reconstruction}\n\n", None),
    ("fig:supp_ceramic_profile_reconstruction", "S9"),
    ("fig:supp_steel_profile_reconstruction", "S10"),
    ("\\subsection{Steel roundness --- polar representation}\n\n", None),
    ("fig:supp_steel_roundness_polar", "S11"),
    ("\\subsection{Per-sweep descriptor values}\n\n", None),
    ("input:per_sweep_table.tex", "Table S3"),

    # Validation Analyses (S12-S17)
    ("\\section{Validation Analyses}\n\n", None),
    ("\\subsection{Cross-instrument validation: optical versus tactile}\n\n", None),
    ("fig:supp_tactile_comparison", "S12"),
    ("\\subsection{Descriptor independence validation}\n\n", None),
    ("fig:supp_correlation", "S13"),
    ("fig:supp_psd", "S14"),
    ("fig:supp_cohens_d", "S15"),
    ("\\subsection{Critical descriptor assessment}\n\n", None),
    ("fig:supp_critical", "S16"),
    ("fig:supp_lp_convergence", "S17"),

    # Sensitivity and Flick Flat
    ("\\section{Sensitivity Analysis and Flick Flat Detection}\n\n", None),
    ("\\subsection{Descriptor sensitivity to synthetic defects}\n\n", None),
    ("tab:supp_descriptor_sensitivity", "Table S2"),
    ("fig:supp_descriptor_sensitivity", None),  # unnumbered in main refs
    ("\\subsection{Flick flat detection and measurement}\n\n", None),
    ("fig:supp_flick_flat_detection", None),  # unnumbered (main Fig. 3 is primary)
    ("fig:supp_flick_flat_measurement", "S18"),

    # Harmonic Decomposition (S19)
    ("\\section{Harmonic Decomposition}\n\n", None),
    ("fig:supp_harmonic_decomposition", "S19"),
]

# ── Build output ──
output_lines = [preamble.rstrip()]

# Find the sensitivity table's enumerated conclusions and footnote
# (they follow the table in the original source)
conclusions_block = ""
footnote_block = ""
# Extract the conclusions and footnote that follow tab:supp_descriptor_sensitivity
table_block = blocks.get("tab:supp_descriptor_sensitivity", "")
if table_block:
    # Find where this table ends in the original content
    tbl_start = content.find(table_block)
    tbl_end = tbl_start + len(table_block)
    after_table = content[tbl_end:]
    # Extract enumerated list and footnote
    enum_match = re.search(r'(The results lead to.*?\\end\{enumerate\})', after_table, re.DOTALL)
    fn_match = re.search(r'(\\textsuperscript\{\*\}.*?profiles\.)', after_table, re.DOTALL)
    if enum_match:
        conclusions_block = "\n" + enum_match.group(1) + "\n"
    if fn_match:
        footnote_block = "\n" + fn_match.group(1) + "\n"

for item in expected_order:
    label = item[0]
    if label.startswith("\\"):
        # Section/subsection header
        output_lines.append("")
        output_lines.append(label)
    elif label in blocks:
        output_lines.append("")
        output_lines.append(blocks[label])
        # Add conclusions + footnote after the sensitivity table
        if label == "tab:supp_descriptor_sensitivity":
            if conclusions_block:
                output_lines.append(conclusions_block)
            if footnote_block:
                output_lines.append(footnote_block)
    else:
        print(f"WARNING: block '{label}' not found!")

# Add bibliography
output_lines.append("")
output_lines.append("\\bibliography{bib,scopus-banach,scopus-interfero}")
output_lines.append("")
output_lines.append("\\end{document}")

# Write
result = "\n".join(output_lines)
# Clean up excessive blank lines
result = re.sub(r'\n{4,}', '\n\n\n', result)
SUPP.write_text(result)
print(f"Written {len(result.splitlines())} lines to {SUPP}")
print("Done. Verify with: pdflatex supplementary.tex")
