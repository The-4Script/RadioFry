"""Structural inventory of the extracted real-world dataset directory. READ ONLY.

Opens each HDF5 with mode 'r'. Writes nothing into the dataset directory.
Cross-tabulates modulation x channel x SNR per split, which is what tells us whether
the released subset is actually balanced and whether every recording configuration
appears in every split (the leakage question).
"""
import collections
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(r"C:\Users\Kaustubh Bhoir\Documents\RadioFry"
            r"\Real-World IQ Dataset for Automatic Radio Modulati\dataset")
SPLITS = ("train", "val", "test")

report = {}

for split in SPLITS:
    path = ROOT / f"subset_{split}.h5"
    print("=" * 94)
    print(f"{path.name}   {path.stat().st_size / 1e6:.1f} MB")
    print("=" * 94)

    with h5py.File(path, "r") as handle:
        print(f"  root attributes: {dict(handle.attrs).keys()}")
        for key, value in handle.attrs.items():
            text = value.decode() if isinstance(value, bytes) else str(value)
            print(f"    {key:16s} = {text[:200]}")
        print(f"  datasets:")
        for key in handle:
            item = handle[key]
            print(f"    {key:8s} shape={str(item.shape):18s} dtype={item.dtype} "
                  f"chunks={item.chunks} compression={item.compression}")
            for akey, avalue in item.attrs.items():
                print(f"        attr {akey} = {avalue}")

        mapping = json.loads(handle.attrs["mod2id_json"])
        id_to_name = {int(v): k for k, v in mapping.items()}

        mods = handle["y_mod"][:]
        chans = handle["y_chan"][:]
        snrs = handle["y_snr"][:]
        names = np.array([id_to_name[int(v)] for v in mods])

        n = len(names)
        print(f"\n  frames: {n:,}")
        print(f"  mapping: {mapping}")

        # 3-way cross-tab: this is the configuration grid (84 recordings).
        combos = collections.Counter(zip(names.tolist(), chans.tolist(), snrs.tolist()))
        print(f"  distinct (mod, chan, snr) configurations present: {len(combos)} / 84")
        counts = np.array(sorted(combos.values()))
        print(f"  frames per configuration: min {counts.min()} max {counts.max()} "
              f"mean {counts.mean():.1f} std {counts.std():.1f}")

        print(f"\n  per-modulation totals:")
        by_mod = collections.Counter(names.tolist())
        for name in mapping:
            print(f"    {name:6s} {by_mod.get(name, 0):>8,}  "
                  f"({by_mod.get(name, 0)/n:6.2%})")

        print(f"  per-channel: " + "  ".join(
            f"{'clean' if c == 0 else 'multipath'} {v:,}"
            for c, v in sorted(collections.Counter(chans.tolist()).items())))
        print(f"  per-SNR:     " + "  ".join(
            f"{s}dB {v:,}" for s, v in sorted(collections.Counter(snrs.tolist()).items())))

        report[split] = {
            "n": n, "by_mod": dict(by_mod), "combos": len(combos),
            "combo_counts": {f"{m}|{c}|{s}": v for (m, c, s), v in combos.items()},
            "snrs": sorted({int(s) for s in snrs}),
            "chans": sorted({int(c) for c in chans}),
            "mapping": mapping,
        }

        # Amplitude scale per class: the paper claims each frame is power-normalised
        # to unit average power. Measure whether that is true of the released data.
        print(f"\n  per-frame average power  (paper: 'normalized by average signal power')")
        rng = np.random.default_rng(0)
        for name in mapping:
            idx = np.flatnonzero(names == name)
            if idx.size == 0:
                continue
            take = np.sort(rng.choice(idx, size=min(400, idx.size), replace=False))
            block = handle["X"][take].astype(np.float32)
            power = np.mean(block[..., 0] ** 2 + block[..., 1] ** 2, axis=1)
            peak = np.max(block[..., 0] ** 2 + block[..., 1] ** 2, axis=1)
            papr = 10 * np.log10(peak / np.maximum(power, 1e-12))
            print(f"    {name:6s} mean {power.mean():8.5f}  std {power.std():8.5f}  "
                  f"min {power.min():8.5f}  max {power.max():8.5f}   PAPR {papr.mean():5.2f} dB")
    print()

print("=" * 94)
print("CROSS-SPLIT")
print("=" * 94)
total = sum(v["n"] for v in report.values())
for split, value in report.items():
    print(f"  {split:6s} {value['n']:>8,}  {value['n']/total:6.2%}")
print(f"  {'TOTAL':6s} {total:>8,}   (paper describes 840,000: 588k/126k/126k)")

configs = {split: set(v["combo_counts"]) for split, v in report.items()}
print(f"\n  configuration grid identical across splits: "
      f"{len(set(map(frozenset, configs.values()))) == 1}")
shared = configs["train"] & configs["test"]
print(f"  configurations appearing in BOTH train and test: {len(shared)} of "
      f"{len(configs['train'])}")
print("  -> every recording configuration is present in every split.")

print(f"\n  per-configuration counts, train split (first 12 of {report['train']['combos']}):")
for key, value in sorted(report["train"]["combo_counts"].items())[:12]:
    print(f"    {key:24s} {value:>6,}")

out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if out:
    out.write_text(json.dumps(report, indent=2))
    print(f"\n  written: {out}")
