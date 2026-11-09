"""Windows-friendly equivalent of run_all.sh.   Usage:  python run_all.py            (real run)
                                                       python run_all.py --smoke    (fake data, 2 epochs)
Env knobs (optional): EPOCHS, RES, SEEDS="0 1 2", TUNE_N"""
import os, sys, subprocess, time
os.environ["PYTHONUTF8"] = "1"   # Windows: write/read all files as UTF-8
smoke = "--smoke" in sys.argv
if smoke: os.environ.update(MEDISCAN_DATA="data_fake", QUICK="1", RES="64", EPOCHS="2")
PY = sys.executable
seeds = os.environ.get("SEEDS", "0 1 2").split(); epochs = os.environ.get("EPOCHS", "20"); res = os.environ.get("RES", "224")
tune_n = os.environ.get("TUNE_N", "12"); q = ["--quick"] if os.environ.get("QUICK") == "1" else []
if q: tune_n = "3"
if smoke and not os.path.isdir("data_fake"): subprocess.check_call([PY, "tools/make_fake_data.py"])
C = ["--seeds", *seeds, "--epochs", epochs, "--res", res, *q]

def run(*a):
    print("\n>>>", " ".join(a), flush=True); t = time.time()
    if subprocess.call([PY, *a]) != 0:
        sys.exit(f"\nFAILED: {' '.join(a)}\nFix this (or send me the error), then rerun. Finished steps are cached.")
    print(f"    ({time.time()-t:.0f}s)")

def train(model, tag, *extra, base=C): run("train.py", "--model", model, "--tag", tag, *base, *extra)

for s in ["audit.py", "split.py", "shortcut.py", "baselines.py", "arch_report.py"]: run(s)
train("resnet18", "resnet18"); train("mobilenetv3", "mobilenetv3")
train("arogyanet", "arogyanet"); train("arogyanet_nores", "arogyanet_nores"); train("arogyanet_nose", "arogyanet_nose")
train("arogyanet_meta", "arogyanet_meta")
train("arogyanet", "arogyanet_res96", base=["--seeds", *seeds, "--epochs", epochs, "--res", "96", *q])
train("arogyanet", "arogyanet_aug_none", "--aug", "none"); train("arogyanet", "arogyanet_aug_naive", "--aug", "naive")
run("aggregate.py")
run("tune.py", "--n", tune_n, "--res", res, *q)
hp = ["--hparams", "results/best_hparams.json"]
train("arogyanet", "arogyanet_tuned", *hp); train("arogyanet", "arogyanet_tuned_cw", *hp, "--pos_weight", "auto")
train("arogyanet", "arogyanet_tuned_clahe", *hp, "--preproc", "clahe")
for s in [["threshold_cost.py"], ["ceo_numbers.py"], ["cv_eval.py", "--epochs", epochs, "--res", res, *q],
          ["ensemble.py"], ["embeddings.py"], ["make_final.py"], ["reject_test.py"], ["aggregate.py"],
          ["external_eval.py"], ["build_report.py"]]: run(*s)
print("\nDONE. See results/, figures/, reports/")