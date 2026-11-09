# Shortcut hunt (trained on pool, evaluated on locked test)
| model                                           |   acc |   bal_acc |   recall |   specificity |   roc_auc |   pr_auc |
|:------------------------------------------------|------:|----------:|---------:|--------------:|----------:|---------:|
| majority class                                  | 0.73  |     0.5   |    1     |         0     |     0.5   |    0.73  |
| 16x16 pixels + LogReg                           | 0.925 |     0.901 |    0.953 |         0.85  |     0.975 |    0.991 |
| 16x16 pixels + RandForest                       | 0.912 |     0.871 |    0.959 |         0.783 |     0.968 |    0.987 |
| lung field masked (border only) + LogReg        | 0.888 |     0.844 |    0.939 |         0.748 |     0.923 |    0.959 |
| metadata only (mean, aspect, area) + LogReg     | 0.841 |     0.775 |    0.918 |         0.633 |     0.916 |    0.97  |
| metadata only (mean, aspect, area) + RandForest | 0.858 |     0.809 |    0.915 |         0.704 |     0.922 |    0.971 |

Test-set pneumonia share = 0.730. Chance balanced accuracy = 0.5, chance ROC-AUC = 0.5.
Read: if the crippled models are far above 0.5 balanced accuracy / AUC, brightness, framing or border information (not lung texture) already separates the classes.
