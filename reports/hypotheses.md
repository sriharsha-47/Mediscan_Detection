# Hypotheses — COMMIT THIS FILE (git commit) BEFORE running any training. The commit timestamp is your proof.

**D1 – Residual connections.** They keep gradients flowing through the five stacked blocks, so I expect training to converge faster and
removing them to cost roughly 0–1 pp of validation ROC-AUC — likely inside seed noise because the network is shallow.

**D2 – Squeeze-and-excitation.** It re-weights channels using global image context, so channels that respond to diffuse opacity should be
boosted over background channels; I expect removing it to lower PR-AUC / specificity by about 1 pp, possibly inside the noise.

**Transfer baseline.** I expect ImageNet low-level filters (edges, blobs, textures) to transfer to radiographs but high-level object features
not to; so a pretrained ResNet18/MobileNetV3 should converge in fewer epochs and be at least as good as ArogyaNet in AUC, but not by a wide margin.

**Metadata branch (task 8).** I expect it to *help* on the internal split (brightness / framing differ between classes — see the shortcut hunt)
and to *hurt* at the camp, where exposure and image size are set by different hardware.

**Augmentation.** Horizontal flip is the medically wrong one (heart/stomach-bubble laterality and L/R markers are anatomically meaningful);
I expect "naive" ≈ "safe" internally (the shortcut dominates) but it teaches the model something that never occurs in real data.
