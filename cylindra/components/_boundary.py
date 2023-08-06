from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any
from typing_extensions import TypeGuard

from cylindra.utils import roundint


@dataclass(frozen=True)
class CylindricParameters:
    skew_tilt_angle: float
    rise_angle: float
    spacing: float
    radius: float
    npf: int

    @property
    def perimeter(self) -> float:
        return 2 * math.pi * self.radius

    @property
    def tan_skew_tilt(self) -> float:
        return math.tan(math.radians(self.skew_tilt_angle))

    @property
    def tan_rise(self) -> float:
        return math.tan(math.radians(self.rise_angle))

    @property
    def start(self) -> int:
        start_f = (
            self.perimeter
            / self.spacing
            / (self.tan_skew_tilt * self.tan_rise + 1)
            * self.tan_rise
        )
        return roundint(start_f)

    @property
    def skew_angle(self) -> float:
        return math.degrees(self.skew_angle_rad)

    @property
    def skew_angle_rad(self) -> float:
        return self.tan_skew_tilt * 2 * self.spacing / self.radius

    @property
    def rise_angle_rad(self) -> float:
        return math.radians(self.rise_angle)

    @property
    def rise_length(self) -> float:
        return self.tan_rise * self.perimeter / self.npf

    @property
    def shear_angle(self) -> float:
        return math.degrees(90 - self.skew_tilt_angle + self.rise_angle)


def given(s) -> TypeGuard[Any]:
    return s is not None


def solve_cylinder(
    spacing=None,
    skew_tilt_angle=None,
    skew_angle=None,
    rise_angle=None,
    rise_length=None,
    radius=None,
    npf=None,
    start=None,
    *,
    allow_duplicate=False,
    rise_sign=1,
) -> CylindricParameters:
    if given(skew_angle) and given(skew_tilt_angle) and not allow_duplicate:
        raise ValueError("Cannot specify both skew_angle and skew_tilt.")
    if given(rise_angle) and given(rise_length) and not allow_duplicate:
        raise ValueError("Cannot specify both rise_angle and rise_length.")

    _skew_is_known = given(skew_angle) or given(skew_tilt_angle)
    _rise_is_known = given(rise_angle) or given(rise_length)
    _start_is_known = given(start)
    _npf_is_known = given(npf)
    _spacing_is_known = given(spacing)
    _radius_is_known = given(radius)

    if not (_spacing_is_known and _radius_is_known and _npf_is_known):
        raise ValueError("spacing, radius and npf must be provided.")

    s0 = sum([int(_skew_is_known), int(_rise_is_known), int(_start_is_known)])
    if s0 == 3:
        if allow_duplicate:
            _rise_is_known = False
        else:
            raise ValueError("Nothing to solve.")
    elif s0 < 2:
        raise ValueError("Not enough information to solve.")

    perimeter = 2 * math.pi * radius
    npf = roundint(npf)

    if _skew_is_known and given(skew_angle):
        skew_tilt_angle = math.degrees(
            math.atan(math.radians(skew_angle) * radius / 2 / spacing)
        )

    if _rise_is_known and given(rise_length):
        tan_rise = rise_length / (perimeter / npf)
        rise_angle = math.degrees(math.atan(tan_rise))
        zero_start = rise_angle == 0
    else:
        zero_start = start == 0

    if zero_start and not _skew_is_known:
        raise ValueError("Cannot solve skew_tilt_angle when start is zero.")

    # now, two of skew_tilt_angle, rise_angle and start are known.
    if not _skew_is_known:
        tan_rise = math.tan(math.radians(rise_angle)) * rise_sign
        tan_skew_tilt = (perimeter / start / spacing * tan_rise - 1) / tan_rise
        skew_tilt_angle = math.degrees(math.atan(tan_skew_tilt))
    elif not _rise_is_known:
        tan_rise = (
            rise_sign
            * start
            * spacing
            / (
                perimeter
                - rise_sign * math.tan(math.radians(skew_tilt_angle)) * start * spacing
            )
        )
        rise_angle = math.degrees(math.atan(tan_rise)) * rise_sign
    else:
        tan_skew_tilt = math.tan(math.radians(skew_tilt_angle))
        tan_rise = math.tan(math.radians(rise_angle)) * rise_sign
        start_f = perimeter / spacing / (tan_skew_tilt * tan_rise + 1) * tan_rise
        start = roundint(start_f)
        tan_rise = start * spacing / (perimeter - tan_skew_tilt * start * spacing)
        rise_angle = math.degrees(math.atan(tan_rise)) * rise_sign

    return CylindricParameters(skew_tilt_angle, rise_angle, spacing, radius, npf)
