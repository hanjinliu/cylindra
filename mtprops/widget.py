from __future__ import annotations
import pandas as pd
import napari
from napari._qt.widgets.qt_viewer_dock_widget import QtViewerDockWidget
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
import numpy as np
from napari.utils.colormaps.colormap import Colormap
from napari.qt import progress
from qtpy.QtWidgets import (QWidget, QMainWindow, QPushButton, QFrame, QVBoxLayout, QHBoxLayout, QSlider)
from qtpy.QtCore import Qt

from .core import MTPath

BlueToRed = Colormap([[0,0,1,1], [1,0,0,1]], name="BlueToRed")

def start(viewer:"napari.Viewer"):
    scale = [r[2] for r in viewer.dims.range]
    common_properties = dict(ndim=3, n_dimensional=True, scale=scale, size=4/np.mean(scale))
    layer_prof = viewer.add_points(**common_properties,
                                   name="MT Profiles",
                                   opacity=0.4, 
                                   edge_color="black",
                                   face_color="black",
                                   text={"text": "{label}-{number}", "color":"black", "size":4, "visible": False},
                                   )
    layer_work = viewer.add_points(**common_properties,
                                   name="Working Layer",
                                   face_color="yellow"
                                   )
    
    layer_prof.editable = False
    layer_prof.properties = {"pitch": np.array([], dtype=np.float64)}
    layer_prof.current_properties = {"pitch": np.array([0.0], dtype=np.float64)}
    
    for layer in viewer.layers:
        if isinstance(layer, napari.layers.Image) and layer.visible:
            mtprof = MTProfiler(layer.data, layer_work, layer_prof, parent=viewer.window._qt_window)
            break
    else:
        raise ValueError("No visible image layer was found")
    
    dock = viewer.window.add_dock_widget(mtprof, area="right", name="MT Profiler")
    dock.setFloating(True)
    # TODO: viewer.window._status_bar._toggle_activity_dock(True)
    layer_work.mode = "add"
    return mtprof
    

class MTProfiler(QMainWindow):
    def __init__(self, image, layer_work, layer_prof, parent=None):
        super().__init__(parent=parent)
        self.image = image
        self.layer_work = layer_work
        self.layer_prof = layer_prof
        self.figure_widget = None
        self.mt_paths = []
        self.dataframe = None
        
        self._add_central_widget()
        self.setUnifiedTitleAndToolBarOnMac(True)
        self.setWindowTitle("MT Profiler")
    
    def register_path(self):
        self.layer_prof.add(self.layer_work.data)
        self.mt_paths.append(self.layer_work.data)
        self.layer_work.data = []
        return None
        
    def run_for_all_path(self):
        if self.dataframe is not None:
            raise ValueError("Data Frame list is not empty")
        
        df_list = []
        with progress(self.mt_paths) as pbr:
            for i, path in enumerate(pbr):
                subpbr = progress(total=10, nest_under=pbr)
                mtp = MTPath(self.image.scale.x)
                subpbr.set_description("Loading images")
                mtp.set_path(path)
                mtp.load_images(self.image)
                subpbr.update(1)
                subpbr.set_description("Calculating MT path")
                mtp.grad_path()
                mtp.smooth_path()
                subpbr.update(1)
                subpbr.set_description("XY-rotation")
                mtp.rot_correction()
                subpbr.update(1)
                subpbr.set_description("X/Z-shift")
                mtp.xshift_correction()
                mtp.zshift_correction()
                subpbr.update(1)
                subpbr.set_description("Updating path edges")
                mtp.calc_center_shift()
                mtp.update_points()
                subpbr.update(1)
                subpbr.set_description("Reloading images")
                mtp.load_images(self.image)
                mtp.grad_path()
                subpbr.update(1)
                subpbr.set_description("XYZ-rotation")
                mtp.rotate3d()
                subpbr.update(1)
                subpbr.set_description("Determining MT radius")
                mtp.determine_radius()
                subpbr.update(1)
                subpbr.set_description("Calculating pitch lengths")
                mtp.calc_pitch_length()
                subpbr.update(1)
                subpbr.set_description("Calculating PF numbers")
                mtp.calc_pf_number()
                
                df = mtp.to_dataframe(i)
                df_list.append(df)
                
        self.dataframe = pd.concat(df_list, axis=0)
        
        self.layer_prof.data = self.dataframe[["z", "y", "x"]].values
        self.layer_prof.properties = self.dataframe
        # self.layer_prof.current_properties = self.dataframe.iloc[-1,:].to_dict()
        self.layer_prof.face_color = "pitch"
        self.layer_prof.face_contrast_limits = [4.08, 4.36]
        self.layer_prof.face_colormap = BlueToRed
        self.layer_prof.text.visible = True
        self.layer_prof.size = mtp.radius[1]/mtp.scale
        
        self.layer_work.mode = "pan_zoom"
        self.layer_prof.mode = "select"
        
        return None
    
    def get_selected_points(self):
        selected = list(self.layer_prof.selected_data)
        label = np.unique([self.dataframe.iloc[i]["label"] for i in selected])
        if len(label) != 1:
            raise ValueError(f"{len(label)} MTs were selected.")
        label = label[0]
        df = self.dataframe[self.dataframe["label"]==label]
        mtp = MTPath(self.image.scale.x)
        mtp._even_interval_points = df[["z","y","x"]].values
        mtp.grad_path()
        indices = self.dataframe["number"].values[selected]
        mtp._even_interval_points = mtp._even_interval_points[indices]
        mtp.grad_angles_yx = mtp.grad_angles_yx[indices]
        mtp.grad_angles_zy = mtp.grad_angles_zy[indices]
        
        mtp.load_images(self.image)
        mtp.rotate3d()
        mtp.radius_peak = float(self.dataframe["MTradius"].values[0])
        mtp.pitch_lengths = self.dataframe["pitch"].values[selected]
        mtp._pf_numbers = self.dataframe["nPF"].values[selected]
        self.mtp_cache = mtp
        self._add_figure()
        self.canvas.imshow_yx_raw()
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
        self.load_button.clicked.connect(self.get_selected_points)
        
        central_widget.layout().addWidget(self.register_button)
        central_widget.layout().addWidget(self.run_button)
        central_widget.layout().addWidget(self.load_button)
        
        self.setCentralWidget(central_widget)
        
        return None
        
    def _add_figure(self):
        if self.figure_widget is not None:
            self.removeDockWidget(self.figure_widget)
        self.canvas = SlidableFigureCanvas(self, self.mtp_cache)
    
        dock = QtViewerDockWidget(self, self.canvas, name="Figure",
                                  area="bottom", allowed_areas=["right", "bottom"])
        dock.setMinimumHeight(200)
        self.resize(self.width(), max(self.height(), 250))
        self.addDockWidget(dock.qt_area, dock)
        self.figure_widget = dock
        return None
    

class SlidableFigureCanvas(QWidget):
    def __init__(self, parent=None, mtpath:MTPath=None):
        super().__init__(parent)
        self.mtpath = mtpath
        self.setLayout(QVBoxLayout())
        
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(0, mtpath.npoints-1)
        self.slider.valueChanged.connect(self.call)
        
        self.fig = self.fig = plt.figure()
        canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        
        buttons = QFrame(self)
        buttons.setLayout(QHBoxLayout())
        
        
        imshow0 = QPushButton("XY raw", buttons)
        imshow0.setToolTip("Call imshow_yx_raw")
        imshow0.clicked.connect(self.imshow_yx_raw)
        buttons.layout().addWidget(imshow0)
        
        imshow1 = QPushButton("YZ raw", buttons)
        imshow1.setToolTip("Call imshow_zy_raw")
        imshow1.clicked.connect(self.imshow_zy_raw)
        buttons.layout().addWidget(imshow1)
        
        imshow2 = QPushButton("YZ ave", buttons)
        imshow2.setToolTip("Call imshow_zy_ave")
        imshow2.clicked.connect(self.imshow_zy_ave)
        buttons.layout().addWidget(imshow2)
        
        self.layout().addWidget(canvas)
        self.layout().addWidget(buttons)
        self.layout().addWidget(self.slider)
        
        self.last_called = self.imshow_zy_raw
    
    def imshow_yx_raw(self):
        self.ax.cla()
        i = self.slider.value()
        self.mtpath.imshow_yx_raw(i, ax=self.ax)
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.last_called = self.imshow_yx_raw
        return None
    
    def imshow_zy_raw(self):
        self.ax.cla()
        i = self.slider.value()
        self.mtpath.imshow_zy_raw(i, ax=self.ax)
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.last_called = self.imshow_zy_raw
        return None
    
    def imshow_zy_ave(self):
        self.ax.cla()
        i = self.slider.value()
        self.mtpath.imshow_zy_ave(i, ax=self.ax)
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.last_called = self.imshow_zy_ave
        return None
    
    def call(self):
        return self.last_called()