from __future__ import annotations

import csv
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass
class AtmosphericSample:
    altitude_m: float
    temperature_k: float
    pressure_pa: float
    density_kg_m3: float
    wind_u_ms: float
    wind_v_ms: float
    wind_w_ms: float


class AtmosphereProfile:
    """Interpolates atmospheric state from tabulated sounding data."""

    def __init__(self, samples: List[AtmosphericSample]):
        if len(samples) < 2:
            raise ValueError("Atmosphere profile requires at least two samples for interpolation.")
        # Sort by altitude to be safe
        self._samples = sorted(samples, key=lambda s: s.altitude_m)
        self._altitudes = [s.altitude_m for s in self._samples]
        self._temperature = [s.temperature_k for s in self._samples]
        self._pressure = [s.pressure_pa for s in self._samples]
        self._density = [s.density_kg_m3 for s in self._samples]
        self._wind_u = [s.wind_u_ms for s in self._samples]
        self._wind_v = [s.wind_v_ms for s in self._samples]
        self._wind_w = [s.wind_w_ms for s in self._samples]

    @classmethod
    def from_csv(cls, path: Path) -> "AtmosphereProfile":
        samples: List[AtmosphericSample] = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                samples.append(
                    AtmosphericSample(
                        altitude_m=float(row["altitude_m"]),
                        temperature_k=float(row["temperature_k"]),
                        pressure_pa=float(row["pressure_pa"]),
                        density_kg_m3=float(row["density_kg_m3"]),
                        wind_u_ms=float(row.get("wind_u_ms", 0.0)),
                        wind_v_ms=float(row.get("wind_v_ms", 0.0)),
                        wind_w_ms=float(row.get("wind_w_ms", 0.0)),
                    )
                )
        return cls(samples)

    def _interp(self, values: List[float], altitude_m: float) -> float:
        if altitude_m <= self._altitudes[0]:
            return float(values[0])
        if altitude_m >= self._altitudes[-1]:
            return float(values[-1])
        idx = bisect_left(self._altitudes, altitude_m)
        low_idx = max(idx - 1, 0)
        high_idx = min(idx, len(self._altitudes) - 1)
        low_alt = self._altitudes[low_idx]
        high_alt = self._altitudes[high_idx]
        if high_alt == low_alt:
            return float(values[low_idx])
        fraction = (altitude_m - low_alt) / (high_alt - low_alt)
        return float(values[low_idx] + fraction * (values[high_idx] - values[low_idx]))

    def temperature(self, altitude_m: float) -> float:
        return self._interp(self._temperature, altitude_m)

    def pressure(self, altitude_m: float) -> float:
        return self._interp(self._pressure, altitude_m)

    def density(self, altitude_m: float) -> float:
        return self._interp(self._density, altitude_m)

    def wind_vector(self, altitude_m: float) -> Tuple[float, float, float]:
        u = self._interp(self._wind_u, altitude_m)
        v = self._interp(self._wind_v, altitude_m)
        w = self._interp(self._wind_w, altitude_m)
        return (u, v, w)


def load_dugway_default_profile() -> AtmosphereProfile:
    """Load the bundled September 21, 2021 12Z Dugway sounding."""

    data_path = Path(__file__).with_name("data").joinpath("dugway_2021-09-21_12z_profile.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Expected bundled atmosphere profile at {data_path!s}")
    return AtmosphereProfile.from_csv(data_path)
