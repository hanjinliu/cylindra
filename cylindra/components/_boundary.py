from __future__ import annotations

from dataclasses import dataclass
import math as m
from typing import Any, Literal
from typing_extensions import TypeGuard

from cylindra.utils import roundint


@dataclass(frozen=True)
class CylindricParameters:
    skew: float
    rise_angle: float
    pitch: float
    radius: float
    npf: int

    @property
    def spacing(self) -> float:
        """Longitudinal spacing in nm."""
        _r = self.rise_angle_rad
        _s = self.skew_rad
        return self.pitch * m.cos(_r) / m.cos(_r - _s)

    @property
    def spacing_proj(self) -> float:
        """The y-projection of the spacing."""
        return self.spacing * m.cos(self.skew_rad)

    @property
    def lat_spacing(self) -> float:
        """Lateral spacing in nm."""
        if self.tan_rise != 0:
            return self.rise_length / m.sin(self.rise_angle_rad)
        return self.perimeter / self.npf

    @property
    def lat_spacing_proj(self) -> float:
        """The θ-projection of the lateral spacing."""
        if self.tan_rise != 0:
            return self.rise_length / self.tan_rise
        return self.perimeter / self.npf

    @property
    def perimeter(self) -> float:
        """Perimeter of the cylinder in nm."""
        return 2 * m.pi * self.radius

    @property
    def tan_skew(self) -> float:
        """Tangent of the skew tilt angle."""
        return m.tan(self.skew_rad)

    @property
    def tan_rise(self) -> float:
        """Tangent of the rise angle."""
        return m.tan(self.rise_angle_rad)

    @property
    def start(self) -> int:
        """The start number."""
        return roundint(self.perimeter * self.tan_rise / self.pitch)

    @property
    def dimer_twist(self) -> float:
        """Skew angle in degrees."""
        return m.degrees(self.dimer_twist_rad)

    @property
    def skew_rad(self) -> float:
        """Skew tilt angle in radians."""
        return m.radians(self.skew)

    @property
    def dimer_twist_rad(self) -> float:
        """Skew angle in radians."""
        # == m.sin(self.skew_tilt_angle_rad) * 2 * self.spacing / self.radius
        if self.start != 0:
            tt = self.tan_rise * self.tan_skew
            return 4 * m.pi / self.start * tt / (1 + tt)
        return m.tan(self.skew_rad) * 2 * self.pitch / self.radius

    @property
    def rise_angle_rad(self) -> float:
        """Rise angle in radians."""
        return m.radians(self.rise_angle)

    @property
    def rise_length(self) -> float:
        """Rise length in nm."""
        return (
            self.perimeter
            / self.npf
            * self.tan_rise
            / (1 + self.tan_rise * self.tan_skew)
        )

    @classmethod
    def solve(
        cls,
        spacing: float | None = None,
        pitch: float | None = None,
        skew: float | None = None,
        dimer_twist: float | None = None,
        rise_angle: float | None = None,
        rise_length: float | None = None,
        radius: float | None = None,
        npf: int | None = None,
        start: int | None = None,
        *,
        allow_duplicate: bool = False,
        rise_sign: Literal[1, -1] = 1,  # TODO: hard-coded for MTs
    ):
        """Normalize the inputs and return the parameters of the cylinder."""
        if given(dimer_twist) and given(skew) and not allow_duplicate:
            raise ValueError("Cannot specify both dimer_twist and skew_tilt.")
        if given(rise_angle) and given(rise_length) and not allow_duplicate:
            raise ValueError("Cannot specify both rise_angle and rise_length.")
        if given(spacing) and given(pitch) and not allow_duplicate:
            raise ValueError("Cannot specify both spacing and pitch.")

        _skew_is_known = given(dimer_twist) or given(skew)
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
                tan_rise = start * pitch / perimeter
                if given(dimer_twist):
                    skew = _twist_to_skew(start, tan_rise, dimer_twist)
            elif given(rise_angle):
                start = roundint(perimeter * m.tan(m.radians(rise_angle)) / pitch)
                if given(dimer_twist):
                    skew = _twist_to_skew(start, tan_rise, dimer_twist)
            elif given(rise_length):
                raise NotImplementedError
            else:
                raise ValueError("Not enough information to solve.")

        elif given(spacing):
            if given(dimer_twist):
                skew_rad = m.asin(m.radians(dimer_twist) * radius / 2 / spacing)
            skew = m.degrees(skew_rad)
            if given(start):
                if _rise_is_known and not allow_duplicate:
                    raise ValueError("Cannot specify both start and rise.")
                if start == 0:
                    tan_rise = 0
                else:
                    tan_rise = m.cos(skew_rad) / (
                        perimeter / start / spacing - m.sin(skew_rad)
                    )
                rise_angle = m.degrees(m.atan(tan_rise))
            elif given(rise_angle):
                start = _rise_to_start(rise_angle, skew_rad, spacing, perimeter)
            elif given(rise_length):
                nl = npf * rise_length / perimeter
                tan_rise = nl / (1 - nl * m.tan(skew_rad))
                rise_angle = m.degrees(m.atan(tan_rise))
                start = _rise_to_start(rise_angle, skew_rad, spacing, perimeter)
            else:
                raise ValueError("Not enough information to solve.")
            pitch = (
                spacing
                * m.cos(m.radians(rise_angle - skew))
                / m.cos(m.radians(rise_angle))
            )

        return CylindricParameters(skew, rise_angle, pitch, radius, npf)


def given(s) -> TypeGuard[Any]:
    return s is not None


def _twist_to_skew(start: int, tan_rise: float, dimer_twist: float) -> float:
    _s_sk = start * m.radians(dimer_twist)
    tan_skew = _s_sk / tan_rise / (4 * m.pi - _s_sk)
    return m.degrees(m.atan(tan_skew))


def _rise_to_start(rise_angle, skew_rad, spacing, perimeter):
    tan_rise = m.tan(m.radians(rise_angle))
    return roundint(
        perimeter / spacing / (m.cos(skew_rad) / tan_rise - m.sin(skew_rad))
    )
