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

            def _normalize(name: str) -> str:
                return name.strip().lstrip("\ufeff").lower()

            required = ["altitude_m", "temperature_k", "pressure_pa", "density_kg_m3"]
            fieldnames = reader.fieldnames or []
            normalized_map = {_normalize(name): name for name in fieldnames if name}

            missing_columns = [name for name in required if name not in normalized_map]
            if missing_columns:
                raise KeyError(
                    "Atmosphere CSV is missing required columns "
                    f"{missing_columns!r} at {path!s}"
                )

            for line_number, row in enumerate(reader, start=2):
                # Skip completely blank lines that the CSV reader returns with ``None`` keys.
                if not row or all(value in (None, "") for value in row.values()):
                    continue

                normalized_row = {
                    _normalize(key): value
                    for key, value in row.items()
                    if key is not None
                }

                try:
                    altitude_raw = normalized_row.get("altitude_m")
                    temperature_raw = normalized_row.get("temperature_k")
                    pressure_raw = normalized_row.get("pressure_pa")
                    density_raw = normalized_row.get("density_kg_m3")
                    wind_u_raw = normalized_row.get("wind_u_ms", 0.0)
                    wind_v_raw = normalized_row.get("wind_v_ms", 0.0)
                    wind_w_raw = normalized_row.get("wind_w_ms", 0.0)

                    if altitude_raw is None or temperature_raw is None or pressure_raw is None or density_raw is None:
                        missing = [
                            name
                            for name, raw in (
                                ("altitude_m", altitude_raw),
                                ("temperature_k", temperature_raw),
                                ("pressure_pa", pressure_raw),
                                ("density_kg_m3", density_raw),
                            )
                            if raw is None
                        ]
                        raise KeyError(
                            "Missing columns "
                            f"{missing!r} at {path!s} line {line_number}"
                        )

                    samples.append(
                        AtmosphericSample(
                            altitude_m=float(altitude_raw),
                            temperature_k=float(temperature_raw),
                            pressure_pa=float(pressure_raw),
                            density_kg_m3=float(density_raw),
                            wind_u_ms=float(wind_u_raw or 0.0),
                            wind_v_ms=float(wind_v_raw or 0.0),
                            wind_w_ms=float(wind_w_raw or 0.0),
                        )
                    )
                except KeyError as exc:  # pragma: no cover - defensive guard
                    raise KeyError(
                        f"Missing column {exc!s} at {path!s} line {line_number}"
                    ) from exc
                except ValueError as exc:  # pragma: no cover - defensive guard
                    raise ValueError(
                        f"Invalid numeric value in {path!s} at line {line_number}: {exc!s}"
                    ) from exc
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
