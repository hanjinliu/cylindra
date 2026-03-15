from __future__ import annotations

import math as m
from dataclasses import dataclass
from typing import Literal, TypeGuard

from cylindra.utils import roundint


@dataclass(frozen=True)
class CylinderParameters:
    """Data structure that represents the parameters of a cylinder.

    There are several parameters that are mutually dependent. This class
    provides a way to normalize the parameters and calculate parameters
    easily.
    """

    radius: float
    pitch: float
    moire_period: float  # maybe negative!!
    npf: int
    start_raw: int
    rise_sign: Literal[1, -1] = -1

    @property
    def spacing(self) -> float:
        """Longitudinal spacing in nm."""
        _r = m.radians(self.rise_angle_raw)
        _s = self.skew_rad
        return self.pitch * m.cos(_r - _s) / m.cos(_r)

    @property
    def spacing_proj(self) -> float:
        """The y-projection of the spacing."""
        return self.spacing * m.cos(self.skew_rad)

    @property
    def rise_angle_raw(self) -> float:
        """Rise angle in degrees before applying rise_sign."""
        return m.degrees(m.atan(self.tan_rise_raw))

    @property
    def lat_spacing(self) -> float:
        """Lateral spacing in nm."""
        if self.rise_angle_raw != 0:
            return self.rise_length / m.sin(m.radians(self.rise_angle))
        return self.perimeter / self.npf

    @property
    def lat_spacing_proj(self) -> float:
        """The θ-projection of the lateral spacing."""
        if self.tan_rise_raw != 0:
            return self.rise_length / self.tan_rise_raw * self.rise_sign
        return self.perimeter / self.npf

    @property
    def perimeter(self) -> float:
        """Perimeter of the cylinder in nm."""
        return 2 * m.pi * self.radius

    @property
    def tan_skew(self) -> float:
        """Tangent of the skew tilt angle."""
        return self.dx / self.moire_period

    @property
    def rise_angle(self) -> float:
        """Rise angle in degrees."""
        return self.rise_angle_raw * self.rise_sign

    @property
    def tan_rise_raw(self) -> float:
        """Tangent of the rise angle."""
        return self.pitch * self.start_raw / (self.dx * self.npf)

    @property
    def dx(self) -> float:
        """Distance between PFs in the y-projection"""
        return 2 * m.pi * self.radius / self.npf

    @property
    def start(self) -> int:
        """The start number."""
        return self.start_raw * self.rise_sign

    @property
    def twist(self) -> float:
        """Twist angle in degrees."""
        return m.degrees(self.twist_rad)

    @property
    def skew_rad(self) -> float:
        """Skew tilt angle in radians."""
        return m.atan(self.tan_skew)

    @property
    def skew(self) -> float:
        """Skew tilt angle in degrees."""
        return m.degrees(self.skew_rad)

    @property
    def twist_rad(self) -> float:
        """Twist angle in radians."""
        # When the sampling point moved moire_period forward, cylinder will be twisted
        # by 2*pi / npf.
        return 2 * m.pi / self.npf / self.moire_period * self.pitch

    @property
    def rise_angle_rad(self) -> float:
        """Rise angle in radians."""
        return m.radians(self.rise_angle_raw)

    @property
    def rise_length(self) -> float:
        """Rise length in nm."""
        # rise_length * npf == pitch * start
        return self.pitch * self.start / self.npf

    @classmethod
    def solve(
        cls,
        spacing: float | None = None,
        pitch: float | None = None,
        skew: float | None = None,
        twist: float | None = None,
        rise_angle: float | None = None,
        rise_length: float | None = None,
        radius: float | None = None,
        npf: int | None = None,
        start: int | None = None,
        moire_period: float | None = None,
        *,
        allow_duplicate: bool = False,
        rise_sign: Literal[1, -1] = -1,
    ):
        """Normalize the inputs and return the parameters of the cylinder."""
        if given(twist) and given(skew) and not allow_duplicate:
            raise ValueError("Cannot specify both twist and skew_tilt.")
        if given(rise_angle) and given(rise_length) and not allow_duplicate:
            raise ValueError("Cannot specify both rise_angle and rise_length.")
        if given(spacing) and given(pitch) and not allow_duplicate:
            raise ValueError("Cannot specify both spacing and pitch.")
        if rise_sign not in (1, -1):
            raise ValueError("rise_sign must be either 1 or -1.")

        _skew_is_known = given(twist) or given(skew)
        _rise_is_known = given(rise_angle) or given(rise_length)
        _spacing_is_known = given(spacing) or given(pitch)

        if not all([_skew_is_known, given(radius), given(npf), _spacing_is_known]):
            raise ValueError("spacing, radius and npf must be provided.")

        radius_: float = radius
        npf = roundint(npf)
        perimeter = 2 * m.pi * radius_

        # NOTE: the `roundint` part will be the reason of unmatch between input twist
        # and the output.
        if given(pitch):
            if given(rise_angle):
                rise_angle *= rise_sign
                start_ = roundint(perimeter * m.tan(m.radians(rise_angle)) / pitch)
                tan_rise = m.tan(m.radians(rise_angle))
                if given(twist):
                    if start_ != 0:
                        skew = _twist_to_skew(start_, tan_rise, twist)
                    else:
                        skew = _twist_to_skew_no_rise(pitch, radius_, twist)
            elif given(start):
                start_ = rise_sign * start
                if _rise_is_known and not allow_duplicate:
                    raise ValueError("Cannot specify both start and rise.")
                tan_rise = start_ * pitch / perimeter
                if given(twist):
                    if start_ != 0:
                        skew = _twist_to_skew(start_, tan_rise, twist)
                    else:
                        skew = _twist_to_skew_no_rise(pitch, radius_, twist)
            elif given(rise_length):
                raise NotImplementedError
            else:
                raise ValueError("Not enough information to solve.")
            if given(moire_period):
                moire_period_ = moire_period
            else:
                moire_period_ = _skew_to_moire_period(perimeter, npf, skew)

        elif given(spacing):
            if given(twist):
                skew_rad = m.asin(m.radians(twist) * radius_ / spacing)
                skew = m.degrees(skew_rad)
            elif given(skew):
                skew_rad = m.radians(skew)
            else:
                raise ValueError("Not enough information to solve.")
            if given(rise_angle):
                rise_angle *= rise_sign
                start_ = _rise_to_start(rise_angle, skew_rad, spacing, perimeter)
            elif given(start):
                start_ = start * rise_sign
                if _rise_is_known and not allow_duplicate:
                    raise ValueError("Cannot specify both start and rise.")
                if start_ == 0:
                    tan_rise = 0.0
                else:
                    tan_rise = m.cos(skew_rad) / (
                        perimeter / start_ / spacing - m.sin(skew_rad)
                    )
                rise_angle = m.degrees(m.atan(tan_rise))
            elif given(rise_length):
                nl = npf * rise_length / perimeter * rise_sign
                tan_rise = nl / (1.0 - nl * m.tan(skew_rad))
                rise_angle = m.degrees(m.atan(tan_rise))
            else:
                raise ValueError("Not enough information to solve.")
            pitch = (
                spacing
                * m.cos(m.radians(rise_angle))
                / m.cos(m.radians(rise_angle - skew))
            )
            if given(moire_period):
                moire_period_ = moire_period
            else:
                moire_period_ = _skew_to_moire_period(perimeter, npf, skew)
        else:
            raise ValueError("Not enough information to solve.")

        return CylinderParameters(radius_, pitch, moire_period_, npf, start_, rise_sign)


def given(s) -> TypeGuard[float]:
    return s is not None


def _twist_to_skew(start: int, tan_rise: float, twist: float) -> float:
    _s_sk = start * m.radians(twist)
    tan_skew = _s_sk / tan_rise / (2 * m.pi + _s_sk)
    return m.degrees(m.atan(tan_skew))


def _twist_to_skew_no_rise(pitch: float, radius: float, twist: float) -> float:
    #    o   o -+- pitch
    #           |
    # twist \|  |
    #        o -+
    rad = m.tan(m.radians(twist)) * pitch / radius
    return m.degrees(rad)


def _rise_to_start(rise_angle, skew_rad, spacing, perimeter):
    tan_rise = m.tan(m.radians(rise_angle))
    return roundint(
        perimeter / spacing / (m.cos(skew_rad) / tan_rise + m.sin(skew_rad))
    )


def _skew_to_moire_period(perimeter, npf, skew):
    if skew == 0:
        return float("inf")
    return perimeter / npf / m.tan(m.radians(skew))
