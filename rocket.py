from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple


@dataclass
class RocketStage:
    burn_time: float
    thrust_newton: Callable[[float], float]
    mass_full: float
    mass_empty: float
    drag_coefficient: float
    reference_area_m2: float
    jettison_on_burnout: bool = True

    def mass(self, time_in_stage: float) -> float:
        if time_in_stage <= 0.0:
            return self.mass_full
        if time_in_stage >= self.burn_time:
            return self.mass_empty
        propellant_burned = (time_in_stage / self.burn_time) * (self.mass_full - self.mass_empty)
        return self.mass_full - propellant_burned

    def thrust(self, time_in_stage: float) -> float:
        if time_in_stage < 0.0:
            time_clamped = 0.0
        elif time_in_stage > self.burn_time:
            time_clamped = self.burn_time
        else:
            time_clamped = time_in_stage
        return float(self.thrust_newton(time_clamped))


class Rocket:
    def __init__(self, stages: Sequence[RocketStage], payload_mass: float) -> None:
        if not stages:
            raise ValueError("A rocket requires at least one stage.")
        self._stages = list(stages)
        self._payload_mass = payload_mass

    @property
    def stages(self) -> Sequence[RocketStage]:
        return self._stages

    @property
    def payload_mass(self) -> float:
        return self._payload_mass

    def initial_mass(self) -> float:
        total = self._payload_mass
        for stage in self._stages:
            total += stage.mass_full
        return total

    def mass_and_thrust(self, t: float) -> Tuple[float, float, RocketStage | None]:
        """Return the instantaneous mass, thrust, and active stage at time ``t``."""

        elapsed = 0.0
        mass = self._payload_mass
        for idx, stage in enumerate(self._stages):
            burn_end = elapsed + stage.burn_time
            if t < burn_end:
                time_in_stage = t - elapsed
                mass += stage.mass(time_in_stage)
                for future_stage in self._stages[idx + 1 :]:
                    mass += future_stage.mass_full
                thrust = stage.thrust(time_in_stage)
                return mass, thrust, stage
            else:
                # Stage consumed; add dry mass if it remains attached
                if not stage.jettison_on_burnout:
                    mass += stage.mass_empty
            elapsed = burn_end
        # After final burnout only payload plus any non-jettisoned dry masses remain
        return mass, 0.0, None
