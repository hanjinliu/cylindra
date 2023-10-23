import tempfile

import napari

from cylindra import start  # NOTE: Set ApplicationAttributes

from magicclass.types import Path
from magicclass.ext.polars import DataFrameView
import numpy as np
from acryo import Molecules
from cylindra.widgets import CylindraMainWidget
from cylindra.const import MoleculesHeader as Mole
from cylindra.cylstructure import calc_lateral_interval
from cylindra.types import MoleculesLayer

import polars as pl

from .user_consts import TEMPLATE_A, TEMPLATE_B, TEMPLATE_X, WOBBLE_TEMPLATES


def create_microtubule(ui: CylindraMainWidget):
    ui.simulator.create_empty_image(size=(60.0, 180.0, 60.0), scale=0.25)
    initialize_molecules(ui)
    layer = ui.mole_layers.last()
    ui.simulator.displace(
        layer, twist=pl.when(pl.col("isotype-id")).then(-0.02).otherwise(0.02)
    )
    ui.simulator.expand(layer=layer, by=0.1, yrange=(6, 16), arange=(0, 6), allev=True)
    ui.simulator.expand(
        layer=layer, by=0.1, yrange=(22, 32), arange=(7, 13), allev=True
    )
    ui.calculate_lattice_structure(layer=layer, props=["spacing"])
    return layer.molecules


def initialize_molecules(ui: CylindraMainWidget):
    ui.simulator.create_straight_line((30.0, 15.0, 30.0), (30.0, 165.0, 30.0))
    ui.simulator.generate_molecules(
        spacing=4.08, twist=0.04, start=3, npf=13, radius=11.0, offsets=(0.0, 0.0)
    )
    ui.sta.seam_search_manually(ui.mole_layers.last(), location=0)
