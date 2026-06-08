#!/usr/bin/env python3
"""Map supplementary figure/table order."""
import re
from pathlib import Path

supp = Path("manuscript/elsevier_style/supplementary.tex")
content = supp.read_text()

# Find all figure and table environments
fig_pattern = re.compile(r'(\\begin\{figure\}\[H\].*?\\end\{figure\})', re.DOTALL)
tbl_pattern = re.compile(r'(\\begin\{table\}\[H\].*?\\end\{table\})', re.DOTALL)

print("=== FIGURES (source order) ===")
for i, f in enumerate(fig_pattern.findall(content)):
    lm = re.search(r'\\label\{([^}]+)\}', f)
    cm = re.search(r'\\caption\{(.*?)\}\s*\\label', f, re.DOTALL)
    if not cm:
        cm = re.search(r'\\caption\{(.*?)\}\s*$', f, re.DOTALL)
    label = lm.group(1) if lm else "???"
    caption = cm.group(1)[:100].replace("\n"," ") if cm else "???"
    print(f"  {i+1}. {label}")
    print(f"     {caption}...")
    print()

print("=== TABLES (source order) ===")
for i, t in enumerate(tbl_pattern.findall(content)):
    lm = re.search(r'\\label\{([^}]+)\}', t)
    cm = re.search(r'\\caption\{(.*?)\}', t, re.DOTALL)
    label = lm.group(1) if lm else "???"
    caption = cm.group(1)[:100].replace("\n"," ") if cm else "???"
    print(f"  {i+1}. {label}")
    print(f"     {caption}...")
    print()

# Find \input
for m in re.finditer(r'\\input\{([^}]+)\}', content):
    print(f"INPUT: {m.group(1)}")
