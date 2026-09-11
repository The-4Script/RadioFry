"""Run the three RadioML 2018.01A experiments on the GPU, then evaluate them.

  linear_probe  V3 trunk frozen, new 24-class head  -> do V3's features transfer?
  finetune      V3 trunk trainable, new head        -> the candidate
  scratch       random init                         -> control

The frozen production checkpoint is never written to; candidates go to new filenames.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402

from radiofry.datasets.radioml2018 import CLASSES  # noqa: E402
from radiofry.evaluation.radioml_benchmark import (  # noqa: E402
    evaluate_checkpoint, write_report,
)
from radiofry.training.device import verify_cuda  # noqa: E402
from radiofry.training.train_radioml import RadioMLTrainingConfig, train  # noqa: E402

DATASET = r"C:\Users\Kaustubh Bhoir\Documents\RadioFry\dataset 2"
PRODUCTION = "models_saved/modulation_cnn_v3_spsaug.pt"
OUT = Path("models_saved")

parser = argparse.ArgumentParser()
parser.add_argument("--per-config-train", type=int, default=600)
parser.add_argument("--per-config-val", type=int, default=100)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch-size", type=int, default=1024)
parser.add_argument("--eval-per-config", type=int, default=120)
args = parser.parse_args()

print("=" * 96)
print("0. CUDA VERIFICATION")
print("=" * 96)
report = verify_cuda()
for line in report.checks:
    print(f"  [ok] {line}")
device = torch.device("cuda")
results = {"device": report.as_dict()}

MODES = [
    ("linear_probe", PRODUCTION, "modulation_cnn_radioml_linearprobe.pt",
     "V3 trunk FROZEN, new 24-class head"),
    ("finetune", PRODUCTION, "modulation_cnn_radioml_finetune.pt",
     "V3 trunk trainable, new 24-class head"),
    ("scratch", None, "modulation_cnn_radioml_scratch.pt",
     "random initialisation, control"),
]

for mode, initial, filename, description in MODES:
    print("\n" + "=" * 96)
    print(f"TRAIN: {mode}  -  {description}")
    print("=" * 96)
    started = time.perf_counter()
    config = RadioMLTrainingConfig(
        mode=mode, dataset_root=DATASET, initial_checkpoint=initial,
        output=str(OUT / filename), label=description,
        per_config_train=args.per_config_train,
        per_config_validation=args.per_config_val,
        epochs=args.epochs, batch_size=args.batch_size)
    metrics = train(config)
    entry = {k: v for k, v in metrics.items() if k != "history"}
    entry["history_tail"] = metrics["history"][-5:]
    results[f"train_{mode}"] = entry
    print(f"  [{(time.perf_counter()-started)/60:.1f} min total]")

print("\n" + "=" * 96)
print("EVALUATION ON THE SEALED TEST SPLIT")
print("=" * 96)
for mode, _, filename, _ in MODES:
    path = OUT / filename
    if not path.is_file():
        continue
    mappable = evaluate_checkpoint(
        path, DATASET, split="test", classes=("BPSK", "QPSK", "8PSK", "16QAM", "64QAM"),
        per_config=args.eval_per_config, seed=99, device=device, truth_space="dataset")
    full = evaluate_checkpoint(
        path, DATASET, split="test", classes=CLASSES,
        per_config=max(20, args.eval_per_config // 5), seed=97,
        device=device, truth_space="dataset")
    print(f"\n  {mode}")
    print(f"    5 mappable classes (free over 24): {mappable.overall_accuracy:>7.2%}  "
          f"({mappable.frames:,} frames)")
    print(f"    full 24-class task:                {full.overall_accuracy:>7.2%}  "
          f"({full.frames:,} frames)")
    print(f"    calibration: {json.dumps(mappable.calibration)}")
    results[f"eval_{mode}_mappable"] = mappable
    results[f"eval_{mode}_full24"] = full

path = write_report(results, "reports/radioml2018_experiments.json")
print(f"\nwrote {path}")
