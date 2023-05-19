from __future__ import annotations

from abc import ABC, abstractmethod
import weakref
import numpy as np
from numpy.typing import NDArray
import impy as ip

from cylindra import utils
from cylindra.const import nm, GlobalVariables as GVar


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
        box_size = (GVar.fitWidth,) + ((GVar.fitWidth + GVar.fitLength) / 1.41,) * 2

        if not all(
            r / 4 <= p < s - r / 4 for p, s, r in zip(point, imgshape_nm, box_size)
        ):
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
    def __init__(
        self,
        interval: nm,
        angle_step: float,
        max_angle: float,
        max_shifts: nm,
    ):
        self._interval = interval
        self._angle_step = angle_step
        self._max_angle = max_angle
        self._max_shifts = max_shifts

    def pick_next(
        self,
        prev: NDArray[np.float32],
        prevprev: NDArray[np.float32],
        img: ip.ImgArray,
    ) -> Exception | NDArray[np.float32]:
        interv_nm = self._interval
        angle_pre = self._angle_step
        angle_dev = self._max_angle
        max_shifts = self._max_shifts
        scale = img.scale.x

        # orientation is point0 -> point1
        point0: np.ndarray = prevprev / scale  # unit: pixel
        point1: np.ndarray = prev / scale

        length_px = utils.roundint(GVar.fitLength / scale)
        width_px = utils.roundint(GVar.fitWidth / scale)

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

        utils.centering(
            img_next, point2, angle_deg, drot=5.0, max_shifts=max_shifts / scale
        )

        new_point = point2 * scale
        return self._check_path(img, new_point) or new_point
