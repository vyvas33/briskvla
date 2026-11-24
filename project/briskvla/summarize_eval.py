"""Print LIBERO success for every run folder written by run_eval.sh.

Expected layout: <root>/<name>_K<steps>/<suite>/eval_info.json

Example:
    uv run python project/briskvla/summarize_eval.py --root outputs/eval
"""

import argparse
import json
from pathlib import Path

from common import SUITE_SHORT, SUITES

SHORT = SUITE_SHORT


def load_run(run_dir: Path) -> dict:
    results = {}
    for suite in SUITES:
        path = run_dir / suite / "eval_info.json"
        if path.exists():
            results[suite] = json.loads(path.read_text())["overall"]
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=Path("outputs/eval"))
    args = parser.parse_args()

    runs = (
        {d.name: load_run(d) for d in sorted(args.root.iterdir()) if d.is_dir()} if args.root.exists() else {}
    )
    runs = {name: r for name, r in runs.items() if r}
    if not runs:
        print(f"No results under {args.root}")
        return

    print(f"\nSuccess rate (%), 95% interval in brackets. Results under {args.root}\n")
    header = f"{'run':<24}" + "".join(f"{SHORT[s]:>18}" for s in SUITES) + f"{'Avg':>8}"
    print(header)
    print("-" * len(header))
    summary = {}
    for name, results in runs.items():
        cells = []
        for suite in SUITES:
            o = results.get(suite)
            if o is None:
                cells.append(f"{'--':>18}")
                continue
            lo, hi = o["pc_success_ci95"]
            cells.append(f"{o['pc_success']:>8.1f} [{lo:4.0f}-{hi:3.0f}]")
        done = [results[s]["pc_success"] for s in SUITES if s in results]
        avg = f"{sum(done) / len(done):>8.1f}" if len(done) == len(SUITES) else f"{'--':>8}"
        print(f"{name:<24}" + "".join(cells) + avg)
        summary[name] = results

    out = args.root / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
