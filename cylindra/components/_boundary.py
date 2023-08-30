from __future__ import annotations

from dataclasses import dataclass
import math as m
from typing import Any, Literal
from typing_extensions import TypeGuard

from cylindra.utils import roundint


@dataclass(frozen=True)
class CylindricParameters:
    skew_tilt_angle: float
    rise_angle: float
    pitch: float
    radius: float
    npf: int

    @property
    def spacing(self) -> float:
        _r = self.rise_angle_rad
        _s = self.skew_tilt_angle_rad
        return self.pitch * m.cos(_r) / m.cos(_r - _s)

    @property
    def spacing_proj(self) -> float:
        return self.spacing * m.cos(self.skew_tilt_angle_rad)

    @property
    def lat_spacing(self) -> float:
        if self.tan_rise != 0:
            return self.rise_length / m.sin(self.rise_angle_rad)
        return self.perimeter / self.npf

    @property
    def lat_spacing_proj(self) -> float:
        if self.tan_rise != 0:
            return self.rise_length / self.tan_rise
        return self.perimeter / self.npf

    @property
    def perimeter(self) -> float:
        return 2 * m.pi * self.radius

    @property
    def tan_skew_tilt(self) -> float:
        return m.tan(m.radians(self.skew_tilt_angle))

    @property
    def tan_rise(self) -> float:
        return m.tan(self.rise_angle_rad)

    @property
    def start(self) -> int:
        return roundint(self.perimeter * self.tan_rise / self.pitch)

    @property
    def skew_angle(self) -> float:
        return m.degrees(self.skew_angle_rad)

    @property
    def skew_tilt_angle_rad(self) -> float:
        return m.radians(self.skew_tilt_angle)

    @property
    def skew_angle_rad(self) -> float:
        # == m.sin(self.skew_tilt_angle_rad) * 2 * self.spacing / self.radius
        if self.start != 0:
            tt = self.tan_rise * self.tan_skew_tilt
            return 4 * m.pi / self.start * tt / (1 + tt)
        return m.tan(self.skew_tilt_angle_rad) * 2 * self.pitch / self.radius

    @property
    def rise_angle_rad(self) -> float:
        return m.radians(self.rise_angle)

    @property
    def rise_length(self) -> float:
        return (
            self.perimeter
            / self.npf
            * self.tan_rise
            / (1 + self.tan_rise * self.tan_skew_tilt)
        )


def given(s) -> TypeGuard[Any]:
    return s is not None


def solve_cylinder(
    spacing: float | None = None,
    pitch: float | None = None,
    skew_tilt_angle: float | None = None,
    skew_angle: float | None = None,
    rise_angle: float | None = None,
    rise_length: float | None = None,
    radius: float | None = None,
    npf: int | None = None,
    start: int | None = None,
    *,
    allow_duplicate: bool = False,
    rise_sign: Literal[1, -1] = 1,  # TODO: hard-coded for MTs
) -> CylindricParameters:
    """Normalize the inputs and return the parameters of the cylinder."""
    if given(skew_angle) and given(skew_tilt_angle) and not allow_duplicate:
        raise ValueError("Cannot specify both skew_angle and skew_tilt.")
    if given(rise_angle) and given(rise_length) and not allow_duplicate:
        raise ValueError("Cannot specify both rise_angle and rise_length.")
    if given(spacing) and given(pitch) and not allow_duplicate:
        raise ValueError("Cannot specify both spacing and pitch.")

    _skew_is_known = given(skew_angle) or given(skew_tilt_angle)
    _rise_is_known = given(rise_angle) or given(rise_length)
    _spacing_is_known = given(spacing) or given(pitch)

    if not all([_skew_is_known, given(radius), given(npf), _spacing_is_known]):
        raise ValueError("spacing, radius and npf must be provided.")

    perimeter = 2 * m.pi * radius
    npf = roundint(npf)

    if given(pitch):
        if given(start):
            if _rise_is_known and not allow_duplicate:
                raise ValueError("Cannot specify both start and rise.")
            tan_rise = start * spacing / perimeter
            if given(skew_angle):
                skew_tilt_angle = _tilt_to_skew(start, tan_rise, skew_angle)
        elif given(rise_angle):
            start = roundint(perimeter * m.tan(m.radians(rise_angle)) / pitch)
            if given(skew_angle):
                skew_tilt_angle = _tilt_to_skew(start, tan_rise, skew_angle)
        elif given(rise_length):
            raise NotImplementedError
        else:
            raise ValueError("Not enough information to solve.")

    elif given(spacing):
        if given(skew_angle):
            skew_tilt_rad = m.asin(m.radians(skew_angle) * radius / 2 / spacing)
        skew_tilt_angle = m.radians(skew_tilt_rad)
        if given(start):
            if _rise_is_known and not allow_duplicate:
                raise ValueError("Cannot specify both start and rise.")
            if start == 0:
                tan_rise = 0
            else:
                tan_rise = m.cos(skew_tilt_rad) / (
                    perimeter / start / spacing - m.sin(skew_tilt_rad)
                )
            rise_angle = m.degrees(m.atan(tan_rise))
        elif given(rise_angle):
            start = _rise_to_start(rise_angle, skew_tilt_rad, spacing, perimeter)
        elif given(rise_length):
            nl = npf * rise_length / perimeter
            tan_rise = nl / (1 - nl * m.tan(skew_tilt_rad))
            rise_angle = m.degrees(m.atan(tan_rise))
            start = _rise_to_start(rise_angle, skew_tilt_rad, spacing, perimeter)
        else:
            raise ValueError("Not enough information to solve.")

    return CylindricParameters(skew_tilt_angle, rise_angle, spacing, radius, npf)


def _tilt_to_skew(start: int, tan_rise: float, skew_angle: float) -> float:
    _s_sk = start * m.radians(skew_angle)
    tan_skew_tilt = _s_sk / tan_rise / (4 * m.pi - _s_sk)
    return m.degrees(m.atan(tan_skew_tilt))


def _rise_to_start(rise_angle, skew_tilt_rad, spacing, perimeter):
    tan_rise = m.tan(m.radians(rise_angle))
    return roundint(
        perimeter / spacing / (m.cos(skew_tilt_rad) / tan_rise - m.sin(skew_tilt_rad))
    )
