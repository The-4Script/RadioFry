"""Command-line entry point for Synthetic Dataset V1 generation."""

import argparse
from typing import Sequence

from .config import DEFAULT_FSK_MODULATION_INDICES, DEFAULT_SNR_SWEEP_DB, MODULATIONS
from .generator import DatasetSpec, generate_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m radiofry.synthetic_gen.v1",
        description="Generate the controlled Synthetic Dataset V1 with full ground truth.",
    )
    parser.add_argument("--output", required=True, help="dataset output directory")
    parser.add_argument("--modulations", nargs="+", default=list(MODULATIONS), choices=list(MODULATIONS))
    parser.add_argument("--snr-db", nargs="+", type=float, default=list(DEFAULT_SNR_SWEEP_DB))
    parser.add_argument("--include-noiseless", action="store_true", help="also emit a noise-free reference capture")
    parser.add_argument(
        "--fsk-modulation-index", nargs="+", type=float, default=list(DEFAULT_FSK_MODULATION_INDICES),
        help="FSK modulation indices h=2*deviation/Rs to sweep (default: 0.5 standard, 1.0 known-hard)",
    )
    parser.add_argument("--captures-per-condition", type=int, default=1)
    parser.add_argument("--num-symbols", type=int, default=4096)
    parser.add_argument("--samples-per-symbol", type=int, default=None, help="default 8 unless both rates are given")
    parser.add_argument("--sample-rate", type=float, default=None, help="default 200000 unless --symbol-rate is given")
    parser.add_argument("--symbol-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--formats", nargs="+", default=["iq", "wav"], choices=["iq", "wav"])
    parser.add_argument("--iq-dtype", default="int16", choices=["int16", "float32"])
    parser.add_argument("--iq-byte-order", default="little", choices=["little", "big"])
    parser.add_argument("--wav-dtype", default="int16", choices=["int16", "float32"])
    parser.add_argument("--full-scale-fraction", type=float, default=0.95)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sweep: list[float | None] = list(args.snr_db)
    if args.include_noiseless:
        sweep.append(None)
    sample_rate, symbol_rate, sps = args.sample_rate, args.symbol_rate, args.samples_per_symbol
    if sample_rate is None and symbol_rate is None:
        sample_rate = 200_000.0
    if sps is None and not (sample_rate is not None and symbol_rate is not None):
        sps = 8
    spec = DatasetSpec(
        output_dir=args.output,
        modulations=tuple(args.modulations),
        snr_sweep_db=tuple(sweep),
        fsk_modulation_indices=tuple(args.fsk_modulation_index),
        captures_per_condition=args.captures_per_condition,
        num_symbols=args.num_symbols,
        samples_per_symbol=sps,
        sample_rate_hz=sample_rate,
        symbol_rate_hz=symbol_rate,
        seed=args.seed,
        formats=tuple(args.formats),
        iq_dtype=args.iq_dtype,
        iq_byte_order=args.iq_byte_order,
        wav_dtype=args.wav_dtype,
        full_scale_fraction=args.full_scale_fraction,
    )
    manifest = generate_dataset(spec)
    from .config import MODULATIONS
    per_modulation = sum(
        len(spec.fsk_modulation_indices) if MODULATIONS[m].family == "fsk" else 1
        for m in spec.modulations
    )
    captures = per_modulation * len(spec.snr_sweep_db) * spec.captures_per_condition
    print(f"Wrote {captures} captures; manifest: {manifest}")
    return 0
