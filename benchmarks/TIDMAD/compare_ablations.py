"""
Read ablation CSVLogger logs and print a ranked comparison table.
Prints best val/loss (min over epochs) for each variant.

Usage:
    python benchmarks/TIDMAD/compare_ablations.py
    python benchmarks/TIDMAD/compare_ablations.py --log_dir logs/ablation --model conv
"""

import argparse
import os
import glob
import csv


def read_best_val_loss(metrics_csv: str) -> float | None:
    """Return the minimum val/loss from a Lightning CSVLogger metrics.csv."""
    best = None
    try:
        with open(metrics_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                v = row.get('val/loss', '').strip()
                if v:
                    try:
                        val = float(v)
                        if best is None or val < best:
                            best = val
                    except ValueError:
                        pass
    except FileNotFoundError:
        pass
    return best


def collect_results(log_dir: str, model_filter: str | None) -> list[dict]:
    """Scan log_dir for version_*/metrics.csv and return sorted results."""
    results = []
    pattern = os.path.join(log_dir, '*', 'version_*', 'metrics.csv')
    for csv_path in sorted(glob.glob(pattern)):
        variant = csv_path.split(os.sep)[-3]   # e.g. conv_s, linoss_m
        if model_filter and not variant.startswith(model_filter):
            continue
        best = read_best_val_loss(csv_path)
        results.append({'variant': variant, 'best_val_loss': best, 'path': csv_path})

    results.sort(key=lambda r: (r['best_val_loss'] is None, r['best_val_loss'] or 0))
    return results


def print_table(results: list[dict]):
    if not results:
        print('No results found.')
        return

    col_w = max(len(r['variant']) for r in results) + 2
    header = f"{'Rank':<6}{'Variant':<{col_w}}{'Best val/loss':>14}  Log path"
    print(header)
    print('-' * len(header))
    for rank, r in enumerate(results, 1):
        loss_str = f"{r['best_val_loss']:.6f}" if r['best_val_loss'] is not None else 'N/A (still running?)'
        print(f"{rank:<6}{r['variant']:<{col_w}}{loss_str:>14}  {r['path']}")

    best = results[0]
    if best['best_val_loss'] is not None:
        print(f"\nBest variant: {best['variant']}  (val/loss = {best['best_val_loss']:.6f})")
        # Suggest the corresponding full-training config
        model = best['variant'].split('_abl_')[0]   # conv or linoss
        tag   = best['variant'].split('_abl_')[1]   # s, m, l, w, d
        full_cfg = f"configs/TIDMAD/train_tidmad_{model}_denoising.yaml"
        abl_cfg  = f"configs/TIDMAD/ablation/{best['variant']}.yaml"
        print(f"To promote to full training, copy hyperparams from:")
        print(f"  {abl_cfg}")
        print(f"into:")
        print(f"  {full_cfg}")
        print(f"then run:  bash cluster/run_tidmad_pipeline.sh")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='logs/ablation')
    parser.add_argument('--model',   default=None, help='Filter by model prefix (conv or linoss)')
    args = parser.parse_args()

    print(f'Scanning: {args.log_dir}\n')

    if args.model:
        results = collect_results(args.log_dir, args.model)
        print_table(results)
    else:
        for model in ['conv', 'linoss']:
            print(f'=== {model.upper()} ===')
            results = collect_results(args.log_dir, model)
            print_table(results)
            print()


if __name__ == '__main__':
    main()
