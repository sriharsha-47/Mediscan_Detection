"""Assemble reports/FINAL_REPORT_DRAFT.md from the template + auto-generated result files."""
import re
from common import *

def main():
    t = open(ROOT / "reports" / "final_report_template.md").read()
    def inc(m):
        f = ROOT / m.group(1)
        return f.read_text() if f.exists() else f"*[missing: {m.group(1)} - run the script that makes it]*"
    out = re.sub(r"\{\{include (.+?)\}\}", inc, t)
    open(ROOT / "reports" / "FINAL_REPORT_DRAFT.md", "w").write(out); print("wrote reports/FINAL_REPORT_DRAFT.md")

if __name__ == "__main__":
    main()
