"""Run local image/folder inspection; no arguments opens the desktop interface."""
from pathlib import Path
import argparse
import json

from inspection.engine import Options, DEFAULT_CAMPAIGN, run
from inspection.registry import available_methods


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path)
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'data/inspection_runs')
    p.add_argument('--mode', choices=['display', 'raw', 'normalized'], default='display')
    p.add_argument('--pixel-size-mm', type=float)
    p.add_argument('--method', choices=['cnn', 'median', 'thresholding', 'all'], default='cnn')
    p.add_argument('--recursive', action='store_true')
    p.add_argument('--convert-colour', action='store_true')
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--campaign', type=Path, default=DEFAULT_CAMPAIGN)
    args = p.parse_args()
    if args.input is None:
        from inspection.gui import main as gui
        gui()
        return 0
    try:
        available = available_methods(args.campaign)
        arms = tuple(a for a in ('cnn', 'median', 'thresholding') if a in available) if args.method == 'all' else (args.method,)
        if any(a not in available for a in arms):
            raise ValueError('Selected method is unavailable in this frozen model.')
        result = run(args.input, args.output, Options(args.mode, args.pixel_size_mm, args.convert_colour,
                     args.recursive, arms), campaign=args.campaign, validate_only=args.check_only)
        print(json.dumps(dict(output=result['output'], records=len(result['records']),
                             errors=sum(r['status'] == 'error' for r in result['records']))))
        return 1 if any(r['status'] == 'error' for r in result['records']) else 0
    except Exception as exc:
        p.exit(2, f'Input error: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
