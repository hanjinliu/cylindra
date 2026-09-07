from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from acryo import Molecules

from cylindra.const import MoleculesHeader as Mole
from cylindra.const import PropertyNames as H

if TYPE_CHECKING:
    from cylindra.components import CylSpline


def extend_longitudinally(
    source_spl: CylSpline,
    mole: Molecules,
    n_extend: dict[int, tuple[int, int]],
    n_to_fit: int = 5,
) -> Molecules:
    from cylindra.components import CylSpline

    extended_pf_moles: list[Molecules] = []
    spacing = source_spl.props.get_glob(H.spacing)
    for pf_id, pf_mole in mole.group_by(Mole.pf):
        n0, n1 = n_extend.get(pf_id, (0, 0))
        if n0 + n1 > 0 and pf_mole.count() >= n_to_fit:
            pf_mole = pf_mole.sort(by=Mole.nth)
            if n0 > 0:
                spl = CylSpline(order=1).fit(pf_mole.pos[:n_to_fit])
                _u = -spacing / spl.length() * np.arange(-n0, 0)
                _nth_min = pf_mole.features[Mole.nth][0]
                coords = spl.map(_u)
                mole_ext = Molecules.from_quat(
                    coords,
                    np.repeat([pf_mole.quaternion()[0]], n0, axis=0),
                    features={
                        Mole.nth: np.arange(-n0 - _nth_min, -_nth_min, dtype=np.int32),
                        Mole.pf: np.full(n0, pf_id, dtype=np.int32),
                    },
                )
                pf_mole = mole_ext.concat_with(pf_mole)
            if n1 > 0:
                spl = CylSpline(order=1).fit(pf_mole.pos[-n_to_fit:])
                _u = 1 + spacing / spl.length() * np.arange(1, n1 + 1)
                _nth_max = pf_mole.features[Mole.nth][-1]
                coords = spl.map(_u)
                mole_ext = Molecules.from_quat(
                    coords,
                    np.repeat([pf_mole.quaternion()[-1]], n1, axis=0),
                    features={
                        Mole.nth: np.arange(
                            _nth_max + 1, _nth_max + n1 + 1, dtype=np.int32
                        ),
                        Mole.pf: np.full(n1, pf_id, dtype=np.int32),
                    },
                )
                pf_mole = pf_mole.concat_with(mole_ext)
        extended_pf_moles.append(pf_mole)

    return Molecules.concat(extended_pf_moles)


def extend_laterally(
    mole: Molecules,
    pre_rotation: float,
    translation: tuple[float, float, float],
    n_extend_left: int = 1,
    n_extend_right: int = 1,
):
    if abs(pre_rotation) >= 1e-4:
        mole = mole.rotate_by_rotvec_internal([0, np.deg2rad(pre_rotation), 0])

    dz, dy, dx = translation
    _trans_l = np.array([dz, -dy, -dx])
    _trans_r = np.array([dz, dy, dx])
    _rot_r_rad = 2 * np.arctan2(dz, dx)
    _rot_l_rad = -_rot_r_rad

    mole_l_list = _prep_mole_list(
        mole_start=mole.filter(pl.col(Mole.pf) == pl.col(Mole.pf).min()),
        n_extend=n_extend_left,
        translation=_trans_l,
        rot_rad=_rot_l_rad,
        pf_id_increment=-1,
    )
    mole_r_list = _prep_mole_list(
        mole_start=mole.filter(pl.col(Mole.pf) == pl.col(Mole.pf).max()),
        n_extend=n_extend_right,
        translation=_trans_r,
        rot_rad=_rot_r_rad,
        pf_id_increment=1,
    )
    mole_all = Molecules.concat(mole_l_list + [mole] + mole_r_list)
    return mole_all.with_features(pl.col(Mole.pf) - mole_all.features[Mole.pf].min())


def _prep_mole_list(
    mole_start: Molecules,
    n_extend: int,
    translation: tuple[float, float, float],
    rot_rad: float,
    pf_id_increment: int,
) -> list[Molecules]:
    cur_mole = mole_start
    mole_out_list = []
    for _ in range(n_extend):
        pf_id_unique = cur_mole.features[Mole.pf].unique()
        if len(pf_id_unique) != 1:
            raise ValueError("Expected exactly one unique pf_id in the molecule.")

        cur_mole = (
            cur_mole.translate_internal(translation)
            .rotate_by_rotvec_internal([0, rot_rad, 0])
            .with_features(pl.lit(pf_id_unique[0] + pf_id_increment).alias(Mole.pf))
        )
        mole_out_list.append(cur_mole)
    return mole_out_list
