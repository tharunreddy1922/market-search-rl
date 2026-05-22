"""
Definitive dashboard fix.
Run from your package folder:
    python fix_final.py
"""
import urllib.request
import re
import os

print("Step 1: Downloading Chart.js...")
urllib.request.urlretrieve(
    "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js",
    "chartjs.js"
)
print("  Done.")

print("Step 2: Reading files...")
with open("chartjs.js", "r", encoding="utf-8") as f:
    chartjs = f.read()

with open("dashboard_fixed.html", "r", encoding="utf-8") as f:
    html = f.read()

print("Step 3: Embedding Chart.js inline...")
# Replace the script tag - use regex to be robust
html = re.sub(
    r'<script src="[^"]*[Cc]hart[^"]*"[^>]*></script>',
    "CHARTJS_PLACEHOLDER",
    html
)
# Now insert Chart.js - split at placeholder to avoid any string issues
parts = html.split("CHARTJS_PLACEHOLDER")
if len(parts) == 2:
    html = parts[0] + "<script>" + chartjs + "</script>" + parts[1]
    print("  Chart.js embedded successfully.")
else:
    print("  Warning: could not find CDN script tag. Chart.js not embedded.")

print("Step 4: Fixing JavaScript string errors...")

# Fix 1: The significance table header - find exact line and fix it
old_sh = """let sh = '<tr><th>Pair</th><th>t-stat</th><th>p-value</th><th>Significant?</th><th>Cohen's d</th><th>Better</th></tr>';"""
new_sh = """let sh = "<tr><th>Pair</th><th>t-stat</th><th>p-value</th><th>Significant?</th><th>Cohens d</th><th>Better</th></tr>";"""
if old_sh in html:
    html = html.replace(old_sh, new_sh)
    print("  Fixed: Cohen's in significance header.")

# Fix 2: Already-partially-fixed version
old_sh2 = """let sh = "<tr><th>Pair</th><th>t-stat</th><th>p-value</th><th>Significant?</th><th>Cohens d</th><th>Better</th></tr>";"""
# This is already correct, leave it.

# Fix 3: Any remaining single-quoted JS string with Cohen's
html = html.replace(
    "'<tr><th>Pair</th><th>t-stat</th><th>p-value</th><th>Significant?</th><th>Cohen\\'s d</th><th>Better</th></tr>'",
    '"<tr><th>Pair</th><th>t-stat</th><th>p-value</th><th>Significant?</th><th>Cohens d</th><th>Better</th></tr>"'
)

print("Step 5: Writing output...")
out = os.path.abspath("dashboard_final2.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\nDone! File saved:")
print(f"  {out}")
print("\nJust double-click dashboard_final2.html to open it.")
print("No server needed.")