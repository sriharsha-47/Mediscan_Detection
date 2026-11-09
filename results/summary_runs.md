# All runs, mean ± std over seeds (thresholds fixed at 0.5 here; threshold work is in threshold_cost.md)

| tag                   |   n_seeds |   params | val_roc_auc   | val_pr_auc    | val_bal_acc   | val_recall    | val_specificity   | test_roc_auc   | test_pr_auc   | test_bal_acc   | test_recall   | test_specificity   |
|:----------------------|----------:|---------:|:--------------|:--------------|:--------------|:--------------|:------------------|:---------------|:--------------|:---------------|:--------------|:-------------------|
| arogyanet             |         3 |   242319 | 0.993 ± 0.002 | 0.996 ± 0.002 | 0.966 ± 0.006 | 0.982 ± 0.004 | 0.950 ± 0.014     | 0.988 ± 0.001  | 0.996 ± 0.000 | 0.945 ± 0.006  | 0.954 ± 0.012 | 0.937 ± 0.003      |
| arogyanet_aug_naive   |         3 |   242319 | 0.992 ± 0.001 | 0.995 ± 0.002 | 0.958 ± 0.005 | 0.984 ± 0.005 | 0.933 ± 0.015     | 0.988 ± 0.002  | 0.996 ± 0.001 | 0.946 ± 0.004  | 0.959 ± 0.007 | 0.934 ± 0.012      |
| arogyanet_aug_none    |         3 |   242319 | 0.991 ± 0.000 | 0.996 ± 0.000 | 0.952 ± 0.003 | 0.982 ± 0.002 | 0.922 ± 0.008     | 0.982 ± 0.001  | 0.993 ± 0.001 | 0.934 ± 0.006  | 0.969 ± 0.003 | 0.900 ± 0.009      |
| arogyanet_meta        |         3 |   242677 | 0.994 ± 0.001 | 0.997 ± 0.001 | 0.964 ± 0.003 | 0.986 ± 0.001 | 0.941 ± 0.006     | 0.990 ± 0.001  | 0.996 ± 0.001 | 0.949 ± 0.007  | 0.965 ± 0.004 | 0.934 ± 0.018      |
| arogyanet_nores       |         3 |   206159 | 0.991 ± 0.001 | 0.995 ± 0.001 | 0.947 ± 0.015 | 0.970 ± 0.008 | 0.924 ± 0.021     | 0.985 ± 0.001  | 0.994 ± 0.000 | 0.932 ± 0.003  | 0.952 ± 0.003 | 0.913 ± 0.007      |
| arogyanet_nose        |         3 |   217827 | 0.992 ± 0.000 | 0.996 ± 0.001 | 0.959 ± 0.001 | 0.980 ± 0.003 | 0.938 ± 0.004     | 0.988 ± 0.001  | 0.996 ± 0.000 | 0.944 ± 0.006  | 0.954 ± 0.008 | 0.934 ± 0.004      |
| arogyanet_res96       |         3 |   242319 | 0.992 ± 0.001 | 0.997 ± 0.000 | 0.961 ± 0.003 | 0.982 ± 0.002 | 0.940 ± 0.004     | 0.985 ± 0.003  | 0.994 ± 0.001 | 0.934 ± 0.003  | 0.959 ± 0.006 | 0.909 ± 0.007      |
| arogyanet_tuned       |         3 |   242319 | 0.993 ± 0.001 | 0.996 ± 0.000 | 0.969 ± 0.004 | 0.981 ± 0.001 | 0.957 ± 0.009     | 0.989 ± 0.001  | 0.996 ± 0.000 | 0.953 ± 0.003  | 0.956 ± 0.005 | 0.950 ± 0.007      |
| arogyanet_tuned_clahe |         3 |   242319 | 0.994 ± 0.001 | 0.997 ± 0.000 | 0.963 ± 0.004 | 0.983 ± 0.005 | 0.942 ± 0.012     | 0.989 ± 0.001  | 0.996 ± 0.001 | 0.950 ± 0.003  | 0.968 ± 0.006 | 0.932 ± 0.011      |
| arogyanet_tuned_cw    |         3 |   242319 | 0.990 ± 0.003 | 0.995 ± 0.001 | 0.955 ± 0.013 | 0.970 ± 0.008 | 0.940 ± 0.019     | 0.984 ± 0.004  | 0.994 ± 0.001 | 0.933 ± 0.014  | 0.939 ± 0.005 | 0.928 ± 0.026      |
| curve_reg             |         1 |   242319 | 0.994 ± 0.000 | 0.997 ± 0.000 | 0.965 ± 0.000 | 0.981 ± 0.000 | 0.949 ± 0.000     | 0.985 ± 0.000  | 0.994 ± 0.000 | 0.932 ± 0.000  | 0.948 ± 0.000 | 0.916 ± 0.000      |
| curve_unreg           |         1 |   242319 | 0.989 ± 0.000 | 0.995 ± 0.000 | 0.934 ± 0.000 | 0.975 ± 0.000 | 0.893 ± 0.000     | 0.980 ± 0.000  | 0.988 ± 0.000 | 0.931 ± 0.000  | 0.967 ± 0.000 | 0.894 ± 0.000      |
| cv_fold0              |         1 |   242319 | 0.993 ± 0.000 | 0.997 ± 0.000 | 0.958 ± 0.000 | 0.975 ± 0.000 | 0.941 ± 0.000     | 0.985 ± 0.000  | 0.994 ± 0.000 | 0.927 ± 0.000  | 0.961 ± 0.000 | 0.894 ± 0.000      |
| cv_fold1              |         1 |   242319 | 0.993 ± 0.000 | 0.997 ± 0.000 | 0.962 ± 0.000 | 0.973 ± 0.000 | 0.952 ± 0.000     | 0.986 ± 0.000  | 0.994 ± 0.000 | 0.945 ± 0.000  | 0.971 ± 0.000 | 0.920 ± 0.000      |
| cv_fold2              |         1 |   242319 | 0.993 ± 0.000 | 0.997 ± 0.000 | 0.964 ± 0.000 | 0.980 ± 0.000 | 0.948 ± 0.000     | 0.986 ± 0.000  | 0.995 ± 0.000 | 0.942 ± 0.000  | 0.954 ± 0.000 | 0.929 ± 0.000      |
| cv_fold3              |         1 |   242319 | 0.992 ± 0.000 | 0.997 ± 0.000 | 0.961 ± 0.000 | 0.981 ± 0.000 | 0.941 ± 0.000     | 0.988 ± 0.000  | 0.995 ± 0.000 | 0.940 ± 0.000  | 0.972 ± 0.000 | 0.907 ± 0.000      |
| cv_fold4              |         1 |   242319 | 0.991 ± 0.000 | 0.991 ± 0.000 | 0.959 ± 0.000 | 0.985 ± 0.000 | 0.934 ± 0.000     | 0.988 ± 0.000  | 0.996 ± 0.000 | 0.940 ± 0.000  | 0.956 ± 0.000 | 0.925 ± 0.000      |
| mobilenetv3           |         3 |   927585 | 0.997 ± 0.001 | 0.999 ± 0.000 | 0.966 ± 0.009 | 0.987 ± 0.003 | 0.946 ± 0.020     | 0.994 ± 0.001  | 0.998 ± 0.000 | 0.962 ± 0.007  | 0.981 ± 0.011 | 0.944 ± 0.024      |
| resnet18              |         3 | 11177025 | 0.998 ± 0.001 | 0.999 ± 0.000 | 0.977 ± 0.003 | 0.991 ± 0.005 | 0.963 ± 0.010     | 0.995 ± 0.000  | 0.998 ± 0.000 | 0.968 ± 0.007  | 0.981 ± 0.001 | 0.954 ± 0.016      |

# Ablations (variant minus reference). Rule: a difference is 'INSIDE NOISE' if |Δ| < sqrt(std_ref² + std_var²).

| ablation                                          | split   | metric   |   reference |   variant |   delta | spread   | verdict      |
|:--------------------------------------------------|:--------|:---------|------------:|----------:|--------:|:---------|:-------------|
| D1 residual connections REMOVED                   | val     | roc_auc  |       0.993 |     0.991 |  -0.002 | ±0.002   | worse        |
| D1 residual connections REMOVED                   | val     | pr_auc   |       0.996 |     0.995 |  -0     | ±0.002   | INSIDE NOISE |
| D1 residual connections REMOVED                   | val     | bal_acc  |       0.966 |     0.947 |  -0.019 | ±0.016   | worse        |
| D1 residual connections REMOVED                   | test    | roc_auc  |       0.988 |     0.985 |  -0.004 | ±0.001   | worse        |
| D1 residual connections REMOVED                   | test    | pr_auc   |       0.996 |     0.994 |  -0.001 | ±0.001   | worse        |
| D1 residual connections REMOVED                   | test    | bal_acc  |       0.945 |     0.932 |  -0.013 | ±0.007   | worse        |
| D2 squeeze-excitation REMOVED                     | val     | roc_auc  |       0.993 |     0.992 |  -0.001 | ±0.002   | INSIDE NOISE |
| D2 squeeze-excitation REMOVED                     | val     | pr_auc   |       0.996 |     0.996 |   0     | ±0.002   | INSIDE NOISE |
| D2 squeeze-excitation REMOVED                     | val     | bal_acc  |       0.966 |     0.959 |  -0.007 | ±0.006   | worse        |
| D2 squeeze-excitation REMOVED                     | test    | roc_auc  |       0.988 |     0.988 |   0     | ±0.001   | INSIDE NOISE |
| D2 squeeze-excitation REMOVED                     | test    | pr_auc   |       0.996 |     0.996 |   0     | ±0.000   | INSIDE NOISE |
| D2 squeeze-excitation REMOVED                     | test    | bal_acc  |       0.945 |     0.944 |  -0.001 | ±0.009   | INSIDE NOISE |
| metadata branch ADDED (two-branch)                | val     | roc_auc  |       0.993 |     0.994 |   0.001 | ±0.002   | INSIDE NOISE |
| metadata branch ADDED (two-branch)                | val     | pr_auc   |       0.996 |     0.997 |   0.001 | ±0.002   | INSIDE NOISE |
| metadata branch ADDED (two-branch)                | val     | bal_acc  |       0.966 |     0.964 |  -0.002 | ±0.007   | INSIDE NOISE |
| metadata branch ADDED (two-branch)                | test    | roc_auc  |       0.988 |     0.99  |   0.002 | ±0.002   | INSIDE NOISE |
| metadata branch ADDED (two-branch)                | test    | pr_auc   |       0.996 |     0.996 |   0.001 | ±0.001   | INSIDE NOISE |
| metadata branch ADDED (two-branch)                | test    | bal_acc  |       0.945 |     0.949 |   0.004 | ±0.009   | INSIDE NOISE |
| input resolution 224 -> 96                        | val     | roc_auc  |       0.993 |     0.992 |  -0.001 | ±0.002   | INSIDE NOISE |
| input resolution 224 -> 96                        | val     | pr_auc   |       0.996 |     0.997 |   0.001 | ±0.002   | INSIDE NOISE |
| input resolution 224 -> 96                        | val     | bal_acc  |       0.966 |     0.961 |  -0.005 | ±0.007   | INSIDE NOISE |
| input resolution 224 -> 96                        | test    | roc_auc  |       0.988 |     0.985 |  -0.003 | ±0.003   | worse        |
| input resolution 224 -> 96                        | test    | pr_auc   |       0.996 |     0.994 |  -0.001 | ±0.001   | INSIDE NOISE |
| input resolution 224 -> 96                        | test    | bal_acc  |       0.945 |     0.934 |  -0.012 | ±0.006   | worse        |
| augmentation safe -> none                         | val     | roc_auc  |       0.993 |     0.991 |  -0.002 | ±0.002   | INSIDE NOISE |
| augmentation safe -> none                         | val     | pr_auc   |       0.996 |     0.996 |   0.001 | ±0.002   | INSIDE NOISE |
| augmentation safe -> none                         | val     | bal_acc  |       0.966 |     0.952 |  -0.014 | ±0.007   | worse        |
| augmentation safe -> none                         | test    | roc_auc  |       0.988 |     0.982 |  -0.006 | ±0.001   | worse        |
| augmentation safe -> none                         | test    | pr_auc   |       0.996 |     0.993 |  -0.003 | ±0.001   | worse        |
| augmentation safe -> none                         | test    | bal_acc  |       0.945 |     0.934 |  -0.011 | ±0.008   | worse        |
| augmentation safe -> naive (with horizontal flip) | val     | roc_auc  |       0.993 |     0.992 |  -0.001 | ±0.002   | INSIDE NOISE |
| augmentation safe -> naive (with horizontal flip) | val     | pr_auc   |       0.996 |     0.995 |  -0     | ±0.002   | INSIDE NOISE |
| augmentation safe -> naive (with horizontal flip) | val     | bal_acc  |       0.966 |     0.958 |  -0.008 | ±0.008   | INSIDE NOISE |
| augmentation safe -> naive (with horizontal flip) | test    | roc_auc  |       0.988 |     0.988 |   0     | ±0.002   | INSIDE NOISE |
| augmentation safe -> naive (with horizontal flip) | test    | pr_auc   |       0.996 |     0.996 |   0     | ±0.001   | INSIDE NOISE |
| augmentation safe -> naive (with horizontal flip) | test    | bal_acc  |       0.945 |     0.946 |   0.001 | ±0.007   | INSIDE NOISE |
| ResNet18 (pretrained) vs ArogyaNet                | val     | roc_auc  |       0.998 |     0.993 |  -0.005 | ±0.002   | worse        |
| ResNet18 (pretrained) vs ArogyaNet                | val     | pr_auc   |       0.999 |     0.996 |  -0.004 | ±0.002   | worse        |
| ResNet18 (pretrained) vs ArogyaNet                | val     | bal_acc  |       0.977 |     0.966 |  -0.011 | ±0.007   | worse        |
| ResNet18 (pretrained) vs ArogyaNet                | test    | roc_auc  |       0.995 |     0.988 |  -0.007 | ±0.001   | worse        |
| ResNet18 (pretrained) vs ArogyaNet                | test    | pr_auc   |       0.998 |     0.996 |  -0.003 | ±0.000   | worse        |
| ResNet18 (pretrained) vs ArogyaNet                | test    | bal_acc  |       0.968 |     0.945 |  -0.022 | ±0.009   | worse        |
| class weights (training-side treatment)           | val     | roc_auc  |       0.993 |     0.99  |  -0.003 | ±0.003   | INSIDE NOISE |
| class weights (training-side treatment)           | val     | pr_auc   |       0.996 |     0.995 |  -0.001 | ±0.001   | INSIDE NOISE |
| class weights (training-side treatment)           | val     | bal_acc  |       0.969 |     0.955 |  -0.014 | ±0.014   | worse        |
| class weights (training-side treatment)           | test    | roc_auc  |       0.989 |     0.984 |  -0.005 | ±0.004   | worse        |
| class weights (training-side treatment)           | test    | pr_auc   |       0.996 |     0.994 |  -0.002 | ±0.001   | worse        |
| class weights (training-side treatment)           | test    | bal_acc  |       0.953 |     0.933 |  -0.02  | ±0.014   | worse        |
| CLAHE preprocessing (internal data)               | val     | roc_auc  |       0.993 |     0.994 |   0.001 | ±0.001   | better       |
| CLAHE preprocessing (internal data)               | val     | pr_auc   |       0.996 |     0.997 |   0.001 | ±0.000   | better       |
| CLAHE preprocessing (internal data)               | val     | bal_acc  |       0.969 |     0.963 |  -0.006 | ±0.006   | worse        |
| CLAHE preprocessing (internal data)               | test    | roc_auc  |       0.989 |     0.989 |   0     | ±0.001   | INSIDE NOISE |
| CLAHE preprocessing (internal data)               | test    | pr_auc   |       0.996 |     0.996 |  -0     | ±0.001   | INSIDE NOISE |
| CLAHE preprocessing (internal data)               | test    | bal_acc  |       0.953 |     0.95  |  -0.003 | ±0.004   | INSIDE NOISE |