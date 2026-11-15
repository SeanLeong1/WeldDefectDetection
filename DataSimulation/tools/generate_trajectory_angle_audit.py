from __future__ import annotations

import argparse
import sys
from pathlib import Path

DATA_SIMULATION_ROOT = Path(__file__).resolve().parents[1]
if str(DATA_SIMULATION_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_SIMULATION_ROOT))

from bse_renderer.validation.casino_emission_fit import write_trajectory_angle_audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute alpha, phi, and varphi for one raw CASINO trajectory."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--nominal-tilt-deg", required=True, type=float)
    parser.add_argument("--repeat-id", required=True, type=int)
    parser.add_argument("--trajectory-id", required=True, type=int)
    parser.add_argument("--scan-point-id", type=int, default=0)
    parser.add_argument("--b-value", type=float)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    audit = write_trajectory_angle_audit(
        output_dir=args.output_dir,
        summary_path=args.summary,
        run_id=args.run_id,
        nominal_tilt_deg=args.nominal_tilt_deg,
        repeat_id=args.repeat_id,
        trajectory_id=args.trajectory_id,
        scan_point_id=args.scan_point_id,
        b_value=args.b_value,
    )
    angles = audit["angles"]
    print(f"alpha  = {angles['alpha']['degrees']:.9f} deg")
    print(f"phi    = {angles['phi']['degrees']:.9f} deg")
    print(f"varphi = {angles['varphi']['degrees']:.9f} deg")
    print(f"wrote  = {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
