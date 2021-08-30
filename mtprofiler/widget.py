from __future__ import annotations
import pandas as pd
import napari
from napari._qt.widgets.qt_viewer_dock_widget import QtViewerDockWidget
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
import numpy as np
from napari.utils.colormaps.colormap import Colormap
from napari.qt import progress
from qtpy.QtWidgets import (QWidget, QMainWindow, QAction, QLabel, QPushButton, QDockWidget, QVBoxLayout)

from .core import MTPath

BlueToRed = Colormap([[0,0,1,1], [1,0,0,1]], name="BlueToRed")

def start(viewer:"napari.Viewer"):
    scale = [r[2] for r in viewer.dims.range]
    layer_prof = viewer.add_points(ndim=3, n_dimensional=True, scale=scale, size=4/np.mean(scale), name="MT Profiles")
    layer_work = viewer.add_points(ndim=3, n_dimensional=True, scale=scale, size=4/np.mean(scale), name="Working Layer",
                                   face_color="yellow")
    layer_prof.editable = False
    for layer in viewer.layers:
        if isinstance(layer, napari.layers.Image) and layer.visible:
            mtprof = MTProfiler(layer.data, layer_work, layer_prof)
            break
    else:
        raise ValueError("No visible image layer was found")
    
    dock = viewer.window.add_dock_widget(mtprof, area="right", name="MT Profiler")
    dock.setFloating(True)
    return mtprof
    

class MTProfiler(QMainWindow):
    def __init__(self, image, layer_work, layer_prof):
        super().__init__()
        self.image = image
        self.layer_work = layer_work
        self.layer_prof = layer_prof
        self.fig = None
        self.ax = None
        self.figure_widget = None
        self.paths = []
        self.dataframe = None
        
        self._add_central_widget()
        self.setUnifiedTitleAndToolBarOnMac(True)
        self.setWindowTitle("MT Profiler")
    
    def register_path(self):
        self.layer_prof.add(self.layer_work.data)
        self.paths.append(self.layer_work.data)
        self.layer_work.data = []
        return None
        
    def run_for_all_path(self):
        if self.dataframe is not None:
            raise ValueError("Data Frame list is not empty")
        
        df_list = []
        for i, path in enumerate(progress(self.paths)):
            mtp = MTPath(self.image.scale.x)
            mtp.run_all(self.image, path)
            df = mtp.to_dataframe(i)
            df_list.append(df)
            
        self.dataframe = pd.concat(df_list, axis=0)
        self.layer_prof.data = [] # clear data
        
        # for df in self.dataframes:
        #     for k, v in df.items():
        #         n = df.shape[0]
        #         if k not in self.layer_prof.properties:
        #             self.layer_prof.properties[k] = np.array([0]*n, dtype=v.dtype)
        #         self.layer_prof.properties[k][-n:] = np.asarray(v)
        self.layer_prof.data = self.dataframe[["z", "y", "x"]].values/self.image.scale
        self.layer_prof.properties.update(self.dataframe.to_dict())
        self.layer_prof.face_color = "pitch"
        self.layer_prof.face_contrast_limits = [4.08, 4.36]
        self.layer_prof.face_colormap = BlueToRed
        
        self.layer_prof.size = mtp.radius[1]/mtp.scale
    
    def get_selected_point(self):
        # TODO: load selected point only.
        selected = self.layer_prof.selected_data
        if len(selected) != 1:
            raise ValueError("Select one data.")
        i = list(selected)[0]
        label = self.dataframe.loc[i]["label"]
        mtp = MTPath(self.image.scale.x)
        mtp._even_interval_points = self.dataframe[self.dataframe["label"]==label][["z","y","x"]].values
        mtp.load_images(self.image)
        mtp.grad_path()
        mtp.rotate3d()
        mtp.determine_radius()
        mtp.calc_pitch_length()
        mtp.calc_pf_number()
        mtp.rotational_averages()
        self.mtp_cache = mtp
        return None
        
    def _add_central_widget(self):
        central_widget = QWidget(self)
        central_widget.setLayout(QVBoxLayout())
        
        self.register_button = QPushButton("Register Path", central_widget)
        self.register_button.setToolTip("Register current points in 'Working Layer' as a MT path.")
        self.register_button.clicked.connect(self.register_path)
        
        self.run_button = QPushButton("Run", central_widget)
        self.run_button.setToolTip("Run profiler for all the paths.")
        self.run_button.clicked.connect(self.run_for_all_path)
        
        self.load_button = QPushButton("Load", central_widget)
        self.load_button.setToolTip("Load path from the selected point.")
        self.load_button.clicked.connect(self.get_selected_point)
        
        central_widget.layout().addWidget(self.register_button)
        central_widget.layout().addWidget(self.run_button)
        central_widget.layout().addWidget(self.load_button)
        
        self.setCentralWidget(central_widget)
        
        return None
        
    def _add_figuire(self):
        if self.fig is None:
            self.fig = plt.figure()
            canvas = FigureCanvas(self.fig)
            self.figure_widget = QtViewerDockWidget(self, canvas, name="Figure",
                                                    area="bottom", allowed_areas=["right", "bottom"])
            self.figure_widget.setMinimumHeight(120)
            self.addDockWidget(self.figure_widget.qt_area, self.figure_widget)
        else:
            self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        
        return None
    
    