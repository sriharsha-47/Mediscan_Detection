"""CEO arithmetic at camp prevalence: why '99% accuracy' is the wrong target."""
import json
import pandas as pd
from common import *


def row(name, sens, spec, n=1000, prev=CAMP_PREV):
    pos = n * prev; neg = n - pos; tp, fn = sens * pos, (1 - sens) * pos; tn, fp = spec * neg, (1 - spec) * neg
    return dict(scenario=name, sick_missed=round(fn, 1), healthy_flagged=round(fp, 1), accuracy=f"{100*(tp+tn)/n:.1f}%",
                sensitivity=f"{100*sens:.1f}%", specificity=f"{100*spec:.1f}%",
                PPV=f"{100*tp/(tp+fp):.1f}%" if tp + fp else "n/a", NPV=f"{100*tn/(tn+fn):.1f}%" if tn + fn else "n/a",
                cost=round(C_FN * fn + C_FP * fp, 1))


def main():
    th = json.load(open(RESULTS / "threshold.json")); t = th["test"]
    rows = [row("Always say NORMAL (does nothing)", 0.0, 1.0),
            row("Pilot model: ~70% of healthy flagged", 1.0, 0.30),
            row("Hypothetical '99% accuracy', all 10 errors = missed cases", 0.90, 1.0),
            row("Hypothetical '99% accuracy', all 10 errors = false alarms", 1.0, 1 - 10 / 900),
            row(f"ArogyaNet @ cost-optimal threshold {th['thr']:.3f} (locked test)", t["recall"], t["specificity"])]
    T = pd.DataFrame(rows)
    txt = (f"# Per 1,000 children at the camp (100 sick, 900 healthy); cost = {C_FN:g} per missed case + {C_FP:g} per false alarm\n\n"
           + T.to_markdown(index=False) +
           "\n\nTake-away: accuracy cannot distinguish the 2nd 99%-model (10 false alarms) from the 1st (10 missed sick children, cost 200). "
           "A do-nothing tool already scores 90%. The target must be a sensitivity floor (e.g. >= 95%) at an acceptable false-alarm rate, "
           "verified with confidence intervals on a patient-disjoint test set, plus PPV/NPV at 9:1 prevalence.\n")
    open(RESULTS / "ceo_arithmetic.md", "w").write(txt); print(txt)


if __name__ == "__main__":
    main()
