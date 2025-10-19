from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

from atmosphere import AtmosphereProfile, load_dugway_default_profile


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


Vector = Tuple[float, float, float]

EARTH_ROTATION_RATE = 7.2921159e-5  # rad/s
EARTH_RADIUS_EQUATOR = 6_378_137.0  # m
LAUNCH_LATITUDE_DEG = 40.2
LAUNCH_LONGITUDE_DEG = -112.8
LAUNCH_SITE_ALTITUDE_M = 1_324.0  # Dugway Proving Ground approximate elevation


def vec_add(a: Vector, b: Vector) -> Vector:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec_sub(a: Vector, b: Vector) -> Vector:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec_scale(a: Vector, scalar: float) -> Vector:
    return (a[0] * scalar, a[1] * scalar, a[2] * scalar)


def vec_norm(a: Vector) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def vec_cross(a: Vector, b: Vector) -> Vector:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def vec_linear_comb(terms: List[Tuple[float, Vector]]) -> Vector:
    x = y = z = 0.0
    for coeff, vec in terms:
        x += coeff * vec[0]
        y += coeff * vec[1]
        z += coeff * vec[2]
    return (x, y, z)


def unit_vector(vector: Vector) -> Vector:
    norm = vec_norm(vector)
    if norm == 0.0:
        return (0.0, 0.0, 0.0)
    return vec_scale(vector, 1.0 / norm)


def gravity_acceleration(altitude_m: float) -> Vector:
    g0 = 9.80665
    g = g0 * (EARTH_RADIUS_EQUATOR / (EARTH_RADIUS_EQUATOR + altitude_m)) ** 2
    return (0.0, 0.0, -g)


def coriolis_acceleration(latitude_deg: float, velocity_enu_m_s: Vector) -> Vector:
    lat_rad = math.radians(latitude_deg)
    omega = (
        EARTH_ROTATION_RATE * math.cos(lat_rad),
        0.0,
        EARTH_ROTATION_RATE * math.sin(lat_rad),
    )
    return vec_scale(vec_cross(omega, velocity_enu_m_s), -2.0)


def centrifugal_acceleration(latitude_deg: float, altitude_m: float) -> Vector:
    lat_rad = math.radians(latitude_deg)
    omega = (
        EARTH_ROTATION_RATE * math.cos(lat_rad),
        0.0,
        EARTH_ROTATION_RATE * math.sin(lat_rad),
    )
    r = (0.0, 0.0, EARTH_RADIUS_EQUATOR + altitude_m)
    return vec_scale(vec_cross(omega, vec_cross(omega, r)), -1.0)


@dataclass
class AerodynamicProperties:
    drag_coefficient: float
    reference_area_m2: float


@dataclass
class VehicleState:
    time_s: float
    position_enu_m: Vector
    velocity_enu_m_s: Vector
    altitude_m: float
    mass_kg: float
    dynamic_pressure_pa: float
    mach_number: float


GuidanceLaw = Callable[[VehicleState], Vector]


@dataclass
class SimulationSettings:
    rocket: Rocket
    atmosphere: AtmosphereProfile
    guidance: GuidanceLaw
    payload_aero: AerodynamicProperties
    max_time_s: float = 600.0
    time_step_s: float = 0.1


@dataclass
class SimulationResult:
    time: List[float]
    altitude: List[float]
    downrange: List[float]
    crossrange: List[float]
    velocity: List[float]
    flight_path_angle_deg: List[float]
    mass: List[float]
    dynamic_pressure: List[float]
    mach_number: List[float]

    def to_csv(self, path: Path) -> None:
        import csv

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "time_s",
                    "altitude_m",
                    "downrange_m",
                    "crossrange_m",
                    "speed_m_s",
                    "flight_path_angle_deg",
                    "mass_kg",
                    "dynamic_pressure_pa",
                    "mach_number",
                ]
            )
            for row in zip(
                self.time,
                self.altitude,
                self.downrange,
                self.crossrange,
                self.velocity,
                self.flight_path_angle_deg,
                self.mass,
                self.dynamic_pressure,
                self.mach_number,
            ):
                writer.writerow(row)


class TrajectorySimulator:
    def __init__(self, settings: SimulationSettings) -> None:
        self.settings = settings

    def run(self) -> SimulationResult:
        s = self.settings
        t = 0.0
        position: Vector = (0.0, 0.0, 0.0)
        velocity: Vector = (0.0, 0.0, 0.0)
        result = SimulationResult([], [], [], [], [], [], [], [], [])

        while t <= s.max_time_s:
            mass, _, _ = s.rocket.mass_and_thrust(t)
            altitude = LAUNCH_SITE_ALTITUDE_M + position[2]
            if altitude < 0.0 and t > 0.0:
                break

            density = s.atmosphere.density(max(0.0, altitude))
            temperature = s.atmosphere.temperature(max(0.0, altitude))
            wind_tuple = s.atmosphere.wind_vector(max(0.0, altitude))
            wind = (float(wind_tuple[0]), float(wind_tuple[1]), float(wind_tuple[2]))
            air_velocity = vec_sub(velocity, wind)
            speed_air = vec_norm(air_velocity)
            speed_ground = vec_norm(velocity)

            dynamic_pressure = 0.5 * density * speed_air * speed_air
            gamma = math.degrees(math.atan2(velocity[2], math.hypot(velocity[0], velocity[1]))) if speed_ground > 1e-6 else 0.0

            sound_speed = math.sqrt(1.4 * 287.05 * temperature)
            mach = speed_air / sound_speed if sound_speed > 0.0 else 0.0

            result.time.append(t)
            result.altitude.append(altitude)
            result.downrange.append(position[0])
            result.crossrange.append(position[1])
            result.velocity.append(speed_ground)
            result.flight_path_angle_deg.append(gamma)
            result.mass.append(mass)
            result.dynamic_pressure.append(dynamic_pressure)
            result.mach_number.append(mach)

            position, velocity = self._rk4_step(position, velocity, t, s.time_step_s)

            t += s.time_step_s

        return result

    def _rk4_step(
        self,
        position: Vector,
        velocity: Vector,
        time: float,
        time_step: float,
    ) -> Tuple[Vector, Vector]:
        s = self.settings

        def derivatives(pos: Vector, vel: Vector, t_local: float) -> Tuple[Vector, Vector]:
            altitude = LAUNCH_SITE_ALTITUDE_M + pos[2]
            density = s.atmosphere.density(max(0.0, altitude))
            temperature = s.atmosphere.temperature(max(0.0, altitude))
            wind_tuple = s.atmosphere.wind_vector(max(0.0, altitude))
            wind = (float(wind_tuple[0]), float(wind_tuple[1]), float(wind_tuple[2]))
            air_vel = vec_sub(vel, wind)
            air_speed = vec_norm(air_vel)

            mass_local, thrust_local, stage_local = s.rocket.mass_and_thrust(t_local)
            if stage_local is None:
                aero = s.payload_aero
            else:
                aero = AerodynamicProperties(stage_local.drag_coefficient, stage_local.reference_area_m2)

            drag_vec = (0.0, 0.0, 0.0)
            if air_speed > 1e-3:
                drag_mag = 0.5 * density * aero.drag_coefficient * aero.reference_area_m2 * air_speed * air_speed
                drag_vec = vec_scale(unit_vector(air_vel), -drag_mag)

            sound_speed = math.sqrt(1.4 * 287.05 * temperature)
            mach = air_speed / sound_speed if sound_speed > 0.0 else 0.0

            vehicle_state = VehicleState(
                time_s=t_local,
                position_enu_m=pos,
                velocity_enu_m_s=vel,
                altitude_m=altitude,
                mass_kg=mass_local,
                dynamic_pressure_pa=0.5 * density * air_speed * air_speed,
                mach_number=mach,
            )

            thrust_dir = unit_vector(s.guidance(vehicle_state))
            thrust_vec = vec_scale(thrust_dir, thrust_local)

            accel = vec_scale(vec_add(thrust_vec, drag_vec), 1.0 / mass_local)
            accel = vec_add(accel, gravity_acceleration(altitude))
            accel = vec_add(accel, coriolis_acceleration(LAUNCH_LATITUDE_DEG, vel))
            accel = vec_add(accel, centrifugal_acceleration(LAUNCH_LATITUDE_DEG, altitude))

            return vel, accel

        k1_pos, k1_vel = derivatives(position, velocity, time)
        k2_pos, k2_vel = derivatives(
            vec_add(position, vec_scale(k1_pos, 0.5 * time_step)),
            vec_add(velocity, vec_scale(k1_vel, 0.5 * time_step)),
            time + 0.5 * time_step,
        )
        k3_pos, k3_vel = derivatives(
            vec_add(position, vec_scale(k2_pos, 0.5 * time_step)),
            vec_add(velocity, vec_scale(k2_vel, 0.5 * time_step)),
            time + 0.5 * time_step,
        )
        k4_pos, k4_vel = derivatives(
            vec_add(position, vec_scale(k3_pos, time_step)),
            vec_add(velocity, vec_scale(k3_vel, time_step)),
            time + time_step,
        )

        sum_pos = vec_linear_comb([(1.0, k1_pos), (2.0, k2_pos), (2.0, k3_pos), (1.0, k4_pos)])
        sum_vel = vec_linear_comb([(1.0, k1_vel), (2.0, k2_vel), (2.0, k3_vel), (1.0, k4_vel)])

        new_position = vec_add(position, vec_scale(sum_pos, time_step / 6.0))
        new_velocity = vec_add(velocity, vec_scale(sum_vel, time_step / 6.0))

        return new_position, new_velocity


def default_guidance(state: VehicleState) -> Vector:
    """Pitch program: vertical ascent followed by gravity turn eastward."""

    if state.time_s < 10.0:
        pitch_deg = 90.0
    elif state.altitude_m < 5000.0:
        pitch_deg = 90.0 - 0.01 * (state.altitude_m - 5000.0)
    else:
        pitch_deg = max(25.0, 90.0 - 0.002 * (state.altitude_m - 5000.0))

    yaw_deg = 90.0  # due east in ENU frame

    pitch_rad = math.radians(pitch_deg)
    yaw_rad = math.radians(yaw_deg)

    direction = (
        math.cos(pitch_rad) * math.cos(yaw_rad),
        math.cos(pitch_rad) * math.sin(yaw_rad),
        math.sin(pitch_rad),
    )
    return direction


def _default_atmosphere_paths() -> List[Path]:
    """Candidate locations for the Dugway trajectory profile."""

    module_path = Path(__file__).resolve()
    module_dir = module_path.parent
    return [
        Path.cwd() / "trajectory_dugway.csv",
        module_dir / "trajectory_dugway.csv",
    ]


def find_default_atmosphere_path() -> Path | None:
    for candidate in _default_atmosphere_paths():
        if candidate.exists():
            return candidate
    return None


def load_atmosphere_profile(path: Path | None = None) -> AtmosphereProfile:
    """Load an atmosphere profile from ``path`` or fall back to the bundled sounding."""

    if path is not None:
        resolved = path.expanduser()
        if not resolved.exists():
            raise FileNotFoundError(f"Atmosphere profile {resolved!s} does not exist")
        return AtmosphereProfile.from_csv(resolved)

    default_path = find_default_atmosphere_path()
    if default_path is not None:
        return AtmosphereProfile.from_csv(default_path)
    return load_dugway_default_profile()


def simulate_flight(
    rocket: Rocket,
    atmosphere: AtmosphereProfile,
    guidance: GuidanceLaw,
    payload_aero: AerodynamicProperties,
    *,
    max_time_s: float = 600.0,
    time_step_s: float = 0.5,
) -> SimulationResult:
    settings = SimulationSettings(
        rocket=rocket,
        atmosphere=atmosphere,
        guidance=guidance,
        payload_aero=payload_aero,
        max_time_s=max_time_s,
        time_step_s=time_step_s,
    )
    simulator = TrajectorySimulator(settings)
    return simulator.run()


def build_default_rocket() -> Rocket:
    def first_stage_thrust(_: float) -> float:
        return 1_900_000.0  # N

    def second_stage_thrust(_: float) -> float:
        return 220_000.0

    first_stage = RocketStage(
        burn_time=120.0,
        thrust_newton=first_stage_thrust,
        mass_full=120_000.0,
        mass_empty=30_000.0,
        drag_coefficient=0.35,
        reference_area_m2=10.0,
        jettison_on_burnout=True,
    )

    second_stage = RocketStage(
        burn_time=380.0,
        thrust_newton=second_stage_thrust,
        mass_full=30_000.0,
        mass_empty=8_000.0,
        drag_coefficient=0.25,
        reference_area_m2=5.0,
        jettison_on_burnout=False,
    )

    payload_mass = 4_500.0

    return Rocket([first_stage, second_stage], payload_mass)


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Simulate the default rocket using an atmosphere profile."
    )
    parser.add_argument(
        "--atmosphere",
        type=Path,
        default=None,
        help=(
            "Path to an atmosphere CSV. Defaults to trajectory_dugway.csv if present, "
            "otherwise uses the bundled Dugway sounding."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("trajectory_dugway_simulation.csv"),
        help="CSV file to write the simulated trajectory.",
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

    args = parser.parse_args()

    atmosphere = load_atmosphere_profile(args.atmosphere)
    rocket = build_default_rocket()
    result = simulate_flight(
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

    print(f"Trajectory written to {args.output}")
    print(f"Apogee: {apogee/1000:.2f} km")
    print(f"Max dynamic pressure: {max_q/1000:.1f} kPa")
    print(f"Final altitude logged: {result.altitude[-1]:.1f} m")


if __name__ == "__main__":
    _cli()
