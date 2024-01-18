from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import impy as ip
import numpy as np
from numpy.typing import NDArray

from cylindra import utils
from cylindra.const import nm

if TYPE_CHECKING:
    from cylindra.components.spline import SplineConfig


class PickerIterator:
    def __init__(
        self,
        picker: Picker,
        img: ip.ImgArray,
        prev: NDArray[np.float32],
        prevprev: NDArray[np.float32],
    ):
        self._picker = picker
        self._img = weakref.ref(img)
        self._prev = prev
        self._prevprev = prevprev

    def __iter__(self):
        return self

    def __next__(self):
        img = self._img()
        if img is None:
            raise ValueError("Image has been deleted.")
        out = self._picker.pick_next(self._prev, self._prevprev, self._img())
        if isinstance(out, Exception):
            raise out
        self._prevprev = self._prev
        self._prev = out
        return out

    def next(self) -> NDArray[np.float32]:
        return next(self)


class Picker(ABC):
    """The abstract picker class."""

    @abstractmethod
    def pick_next(
        self,
        prev: NDArray[np.float32],
        prevprev: NDArray[np.float32],
        img: ip.ImgArray,
    ) -> Exception | NDArray[np.float32]:
        """Pick a new point using two existing points and the image."""

    def _check_path(
        self, img: ip.ImgArray, point: NDArray[np.float32]
    ) -> Exception | None:
        imgshape_nm = np.array(img.shape) * img.scale.x

        if not all(0 <= p < s for p, s in zip(point, imgshape_nm, strict=True)):
            # outside image
            return StopIteration("Outside boundary.")
        return None

    def iter_pick(
        self,
        img: ip.ImgArray,
        prev: NDArray[np.float32],
        prevprev: NDArray[np.float32],
    ) -> PickerIterator:
        return PickerIterator(self, img, prev, prevprev)


class AutoCorrelationPicker(Picker):
    """Picker using auto-correlation."""

    def __init__(
        self,
        interval: nm,
        angle_step: float,
        max_angle: float,
        max_shifts: nm,
        config: SplineConfig,
    ):
        self._interval = interval
        self._angle_step = angle_step
        self._max_angle = max_angle
        self._max_shifts = max_shifts
        self._config = config

    def pick_next(
        self,
        prev: NDArray[np.float32],
        prevprev: NDArray[np.float32],
        img: ip.ImgArray,
    ) -> Exception | NDArray[np.float32]:
        interv_nm = self._interval
        angle_pre = self._angle_step
        angle_dev = self._max_angle
        scale = img.scale.x
        max_shifts = self._max_shifts / scale

        # orientation is point0 -> point1
        point0 = prevprev / scale  # unit: pixel
        point1 = prev / scale

        length_px = utils.roundint(self._config.fit_depth / scale)
        width_px = utils.roundint(self._config.fit_width / scale)

        shape = (width_px,) + (utils.roundint((width_px + length_px) / 1.41),) * 2

        orientation = point1[1:] - point0[1:]
        subimg = utils.crop_tomogram(img, point1, shape)
        center = np.rad2deg(np.arctan2(*orientation)) % 180 - 90
        angle_deg = utils.angle_corr(
            subimg,
            ang_center=center,
            drot=angle_dev,
            nrots=utils.ceilint(angle_dev / angle_pre),
        )
        angle_rad = np.deg2rad(angle_deg)
        dr = (
            np.array(
                [0.0, interv_nm * np.cos(angle_rad), -interv_nm * np.sin(angle_rad)]
            )
            / scale
        )
        if np.dot(orientation, dr[1:]) > np.dot(orientation, -dr[1:]):
            point2 = point1 + dr
        else:
            point2 = point1 - dr

        img_next = utils.crop_tomogram(img, point2, shape)
        new_point = (
            centering(img_next, point2, angle_deg, max_shifts=max_shifts) * scale
        )

        return self._check_path(img, new_point) or new_point


def centering(
    img: ip.ImgArray,
    point: np.ndarray,
    angle: float,
    drot: float = 5,
    nrots: int = 7,
    max_shifts: float | None = None,
):
    """
    Find the center of cylinder using self-correlation.

    Parameters
    ----------
    img : ip.ImgArray
        Target image.
    point : np.ndarray
        Current center of cylinder.
    angle : float
        The central angle of the cylinder.
    drot : float, default 5
        Deviation of the rotation angle.
    nrots : int, default 7
        Number of rotations to try.
    max_shifts : float, optional
        Maximum shift in pixel.

    """
    angle_deg2 = utils.angle_corr(img, ang_center=angle, drot=drot, nrots=nrots)

    img_next_rot = img.rotate(-angle_deg2, cval=np.mean(img))
    proj = img_next_rot.mean(axis="y")
    shift = utils.mirror_zncc(proj, max_shifts=max_shifts)

    shiftz, shiftx = shift / 2
    shift = np.array([shiftz, 0, shiftx])
    rad = -np.deg2rad(angle_deg2)
    cos = np.cos(rad)
    sin = np.sin(rad)
    shift = shift @ [[1.0, 0.0, 0.0], [0.0, cos, sin], [0.0, -sin, cos]]
    return point + shift
