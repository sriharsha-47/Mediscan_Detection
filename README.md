# MediScan Audit — code package

## Setup
```
python -m venv .venv && source .venv/bin/activate      # Windows: use Git Bash or WSL for run_all.sh
pip install -r requirements.txt
```
Data layout (Kaggle downloads, unzipped):
```
data/chest_xray/{train,test,val}/{NORMAL,PNEUMONIA}/*.jpeg      # Chest X-Ray Images (Pneumonia)
data/covid/Normal/images/*.png  data/covid/"Viral Pneumonia"/images/*.png   # COVID-19 Radiography Database (external, Part 2.6 ONLY)
```
Set `MEDISCAN_DATA=/other/path` to move the data folder.

## Run
```
make smoke        # 5-minute pipeline check on synthetic images (numbers meaningless)
bash run_all.sh   # everything, real data.  Knobs: EPOCHS=20 RES=224 SEEDS="0 1 2" TUNE_N=12 QUICK=1
```
**Compute warning:** the full pipeline trains ~30 networks at 224 px. On a CPU-only laptop use `RES=128 EPOCHS=12`, or run on a free Colab/Kaggle GPU
(upload the folder, `!pip install -r requirements.txt`, `!bash run_all.sh`). Final report must state the settings you used.

## Order of scripts / which file feeds which report section
audit.py → split.py → shortcut.py → baselines.py → arch_report.py → train.py (many tags) → aggregate.py → tune.py → threshold_cost.py → ceo_numbers.py →
cv_eval.py → ensemble.py → embeddings.py → make_final.py → reject_test.py → **external_eval.py (once; creates results/.external_done lock)** → build_report.py.
Run `python app.py` for the Gradio demo, `python predict.py image.png` for the CLI. Put a dog photo at `tests/probes/dog.jpg` then rerun `python reject_test.py`.

## Before you start (to score well)
1. `git init && git add reports/hypotheses.md && git commit` — your hypotheses must pre-date the results. Edit them if you disagree.
2. Don't open the external dataset before external_eval.py. It refuses to run twice.
3. Write the prose (`WRITE:` markers in reports/Part1_audit.md, CEO_report.md, final_report_template.md); `python build_report.py` pulls in all tables.
4. After installing, `pip freeze > requirements.lock` and keep it in the repo.

## Honest status
Non-deep-learning scripts (audit, split, shortcut, baselines) were executed on synthetic data and ran cleanly. The PyTorch scripts (train, arogyanet, tune,
threshold_cost, cv_eval, ensemble, embeddings, make_final, predict, external_eval) were written and syntax-checked but NOT executed (no torch in the build
sandbox) — run `make smoke` first and send me any traceback. Dataset-specific assumptions to verify: patient-ID parsing of file names, the external folder names.
