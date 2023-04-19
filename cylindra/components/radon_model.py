from pydantic import BaseModel, Field, validator
from typing import NamedTuple
import numpy as np
import impy as ip

from cylindra.const import nm


class TiltRange(NamedTuple):
    """Tilt range in degrees."""

    min: float
    max: float
    num: int

    @classmethod
    def zeros(cls):
        return cls(0, 0, 1)

    def asarray(self):
        return np.linspace(self.min, self.max, self.num)


class RadonModel(BaseModel):
    """A model for radon transformation."""

    range: TiltRange = Field(default_factory=TiltRange.zeros)
    height: int = 0
    order: int = 1

    @validator("range")
    def _validate_range(cls, v):
        rng = TiltRange(*v)
        if rng.num < 1:
            raise ValueError("Number of tilt angles must be positive.")
        if rng.min > rng.max:
            raise ValueError("Minimum tilt angle must be smaller than maximum.")
        return rng

    def transform(self, img: ip.ImgArray) -> ip.ImgArray:
        degs = self.range.asarray()
        return img.radon(degs, central_axis="y", order=self.order)

    def inverse_transform(self, tilt_series: ip.ImgArray) -> ip.ImgArray:
        degs = self.range.asarray()
        rec = tilt_series.iradon(
            degs, central_axis="y", order=self.order, height=self.height
        )
        rec.set_scale(xyz=rec.scale.x)
        return rec
