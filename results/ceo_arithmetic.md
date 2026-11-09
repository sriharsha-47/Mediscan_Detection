# Per 1,000 children at the camp (100 sick, 900 healthy); cost = 20 per missed case + 1 per false alarm

| scenario                                                  |   sick_missed |   healthy_flagged | accuracy   | sensitivity   | specificity   | PPV    | NPV    |   cost |
|:----------------------------------------------------------|--------------:|------------------:|:-----------|:--------------|:--------------|:-------|:-------|-------:|
| Always say NORMAL (does nothing)                          |         100   |               0   | 90.0%      | 0.0%          | 100.0%        | n/a    | 90.0%  | 2000   |
| Pilot model: ~70% of healthy flagged                      |           0   |             630   | 37.0%      | 100.0%        | 30.0%         | 13.7%  | 100.0% |  630   |
| Hypothetical '99% accuracy', all 10 errors = missed cases |          10   |               0   | 99.0%      | 90.0%         | 100.0%        | 100.0% | 98.9%  |  200   |
| Hypothetical '99% accuracy', all 10 errors = false alarms |           0   |              10   | 99.0%      | 100.0%        | 98.9%         | 90.9%  | 100.0% |   10   |
| ArogyaNet @ cost-optimal threshold 0.515 (locked test)    |           5.1 |              39.8 | 95.5%      | 94.9%         | 95.6%         | 70.4%  | 99.4%  |  141.3 |

Take-away: accuracy cannot distinguish the 2nd 99%-model (10 false alarms) from the 1st (10 missed sick children, cost 200). A do-nothing tool already scores 90%. The target must be a sensitivity floor (e.g. >= 95%) at an acceptable false-alarm rate, verified with confidence intervals on a patient-disjoint test set, plus PPV/NPV at 9:1 prevalence.
