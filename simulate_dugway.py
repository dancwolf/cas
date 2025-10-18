from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

from atmosphere import AtmosphereProfile
from rocket import (
    AerodynamicProperties,
    SimulationResult,
    build_default_rocket,
    default_guidance,
    load_atmosphere_profile,
    simulate_flight,
)


def _load_atmosphere(path: Path | None) -> Tuple[AtmosphereProfile, Path | None]:
    """Load the requested atmosphere profile and report the source path used."""

    if path is not None:
        return load_atmosphere_profile(path), path

    candidate = Path("trajectory_dugway.csv")
    if candidate.exists():
        return load_atmosphere_profile(candidate), candidate

    return load_atmosphere_profile(None), None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate the Dugway launch trajectory using an atmosphere profile. "
            "If --atmosphere is omitted the script will look for trajectory_dugway.csv "
            "in the working directory before falling back to the bundled sounding."
        )
    )
    parser.add_argument(
        "--atmosphere",
        type=Path,
        default=None,
        help="CSV file describing the atmospheric state versus altitude.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("trajectory_dugway_simulation.csv"),
        help="Where to write the simulated trajectory report.",
    )
    parser.add_argument(
        "--time-step",
        type=float,
        default=0.5,
        help="Integrator time step in seconds.",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=600.0,
        help="Maximum simulated time in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    atmosphere, atmosphere_path = _load_atmosphere(args.atmosphere)
    rocket = build_default_rocket()

    result: SimulationResult = simulate_flight(
        rocket,
        atmosphere,
        default_guidance,
        AerodynamicProperties(drag_coefficient=0.2, reference_area_m2=3.0),
        max_time_s=args.max_time,
        time_step_s=args.time_step,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output)

    apogee = max(result.altitude)
    max_q = max(result.dynamic_pressure)

    if atmosphere_path is None:
        print("Atmosphere: bundled Dugway September 21, 2021 12Z sounding")
    else:
        print(f"Atmosphere: {atmosphere_path}")

    print(f"Trajectory written to {args.output}")
    print(f"Apogee: {apogee/1000:.2f} km")
    print(f"Max dynamic pressure: {max_q/1000:.1f} kPa")
    print(f"Final altitude logged: {result.altitude[-1]:.1f} m")


if __name__ == "__main__":
    main()
