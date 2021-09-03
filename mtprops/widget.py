from __future__ import annotations
import pandas as pd
import traceback
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import OrderedDict
from matplotlib.backends.backend_qt5agg import FigureCanvas
import numpy as np
import napari
from napari.utils.colormaps.colormap import Colormap
from napari.qt import progress, thread_worker
from qtpy.QtWidgets import (QWidget, QPushButton, QFrame, QVBoxLayout, QHBoxLayout, QSlider, QFileDialog,
                            QSpinBox, QLabel, QLineEdit, QMessageBox)
from qtpy.QtCore import Qt

from ._impy import impy as ip
from .mtpath import MTPath, calc_total_length

BlueToRed = Colormap([[0,0,1,1], [1,0,0,1]], name="BlueToRed")

class CacheMap:
    def __init__(self, maxgb:float=2.0):
        self.maxgb = maxgb
        self.cache = OrderedDict()
        self.gb = 0.0
    
    def __getitem__(self, key):
        real_key, identifier = key
        idf, value = self.cache[real_key]
        if idf == identifier:
            return value
        else:
            raise KeyError("Wrong identifier")
    
    def __setitem__(self, key, value:list[np.ndarray]):
        real_key, identifier = key
        self.cache[real_key] = (identifier, value)
        size = sum(a.nbytes for a in value)/1e9
        self.gb += size
        while self.gb > self.maxgb:
            self.pop()
    
    def pop(self):
        _, item = self.cache.popitem(last=False)
        self.gb -= sum(a.nbytes for a in item)/1e9
        return None

    def keys(self):
        return self.cache.keys()
    
    def clear(self):
        self.cache.clear()
        self.gb = 0
        return None

cachemap = CacheMap(maxgb=ip.Const["MAX_GB"])

def cached_rotate(mtp:MTPath, image):
    try:
        mtp._sub_images = cachemap[f"{image.name}-{mtp.label}", 
                                   hash(str(mtp._even_interval_points))]
    except KeyError:
        mtp.load_images(image)
        mtp.grad_path()
        mtp.rotate3d()
        cachemap[f"{image.name}-{mtp.label}",
                 hash(str(mtp._even_interval_points))] = mtp._sub_images
    return None

def raise_error_message(parent, msg:str):
    return QMessageBox.critical(parent, "Error", msg,QMessageBox.Ok)

@thread_worker(progress={'total': 0, 'desc':'Reading Image'})
def imread(img, binsize):
    with ip.SetConst("SHOW_PROGRESS", False):
        imgb = img.binning(binsize, check_edges=False).data
    
    return imgb

class MTProfiler(QWidget):
    def __init__(self, viewer:"napari.Viewer", interval_nm=24, light_background:bool=True):
        if interval_nm <= 0:
            raise ValueError("interval_nm must be a positive float.")

        self.interval = interval_nm
        self.light_background = light_background
        
        super().__init__(parent=viewer.window._qt_window)
        self.viewer = viewer
        self.image = None
        self.layer_image = None
        self.layer_prof = None
        self.layer_work = None
        
        self._add_widgets()
        self.setWindowTitle("MT Profiler")
        
    def load_image(self, img=None, binsize=4):
        self.viewer.window._status_bar._toggle_activity_dock(True)
        worker = imread(img, binsize)
        @worker.returned.connect
        def _(imgb):
            self.canvas._init_widget_params()
            tr = (binsize-1)/2*img.scale.x
            if self.layer_image not in self.viewer.layers:
                self.layer_image = self.viewer.add_image(
                    imgb, 
                    scale=imgb.scale, 
                    name=imgb.name, 
                    translate=[tr, tr, tr],
                    rendering="minip" if self.light_background else "mip"
                    )
            else:
                self.layer_image.data = imgb
                self.layer_image.scale = imgb.scale
                self.layer_image.name = imgb.name
                self.layer_image.translate = [tr, tr, tr]
                self.layer_image.rendering = "minip" if self.light_background else "mip"
                
            self.viewer.scale_bar.unit = img.scale_unit
            self.viewer.dims.axis_labels = ("z", "y", "x")

            self.image = img

            self.clear()
            self.viewer.window._status_bar._toggle_activity_dock(False)
            return None
        worker.start()
        return None
    
    def open_image_file(self):
        dlg = QFileDialog()
        hist = napari.utils.history.get_open_history()
        dlg.setHistory(hist)
        filenames, _ = dlg.getOpenFileNames(
            parent=self.viewer.window.qt_viewer,
            caption="Select image ...",
            directory=hist[0],
        )
        if filenames != [] and filenames is not None:
            path = filenames[0]
            napari.utils.history.update_open_history(filenames[0])
        else:
            return None
            
        img = ip.lazy_imread(path, chunks=(64, 1024, 1024))

        self.load_image(img)
        return None
    
    def _init_layers(self):
        viewer = self.viewer
        img = self.image
        
        common_properties = dict(ndim=3, n_dimensional=True, scale=img.scale, size=4/img.scale.x)
        if self.layer_prof in self.viewer.layers:
            self.layer_prof.name = "MT Profiles-old"
    
        self.layer_prof = viewer.add_points(**common_properties,
                                    name="MT Profiles",
                                    opacity=0.4, 
                                    edge_color="black",
                                    face_color="black",
                                    properties = {"pitch": np.array([0.0], dtype=np.float64)},
                                    text={"text": "{label}-{number}", 
                                            "color":"black", 
                                            "size": ip.Const["FONT_SIZE_FACTOR"]*4, 
                                            "visible": False},
                                    )
        self.layer_prof.editable = False
            
        if self.layer_work in self.viewer.layers:
            self.layer_work.name = "Working Layer-old"
        
        self.layer_work = viewer.add_points(**common_properties,
                                    name="Working Layer",
                                    face_color="yellow"
                                    )
    
        self.layer_work.mode = "add"
        
        if "MT Profiles-old" in self.viewer.layers:
            self.viewer.layers.remove("MT Profiles-old")
        if "Working Layer-old" in self.viewer.layers:
            self.viewer.layers.remove("Working Layer-old")
            
        self.mt_paths = []
        self.dataframe = None

        return None
    
    def register_path(self):
        # check length
        total_length = calc_total_length(self.layer_work.data)*self.image.scale.x
        if total_length < self.interval*3:
            raise_error_message(self,
                                f"Path is too short: {total_length:.2f} nm\n"
                                f"Must be longer than {self.interval*3} nm.")
            return None
        
        self.layer_prof.add(self.layer_work.data)
        self.mt_paths.append(self.layer_work.data)
        self.canvas.ax.plot(self.layer_work.data[:,2], self.layer_work.data[:,1], color="gray", lw=2.5)
        self.canvas.ax.set_xlim(0, self.image.sizeof("x"))
        self.canvas.ax.set_ylim(self.image.sizeof("y"), 0)
        self.canvas.ax.tick_params(labelbottom=False,labelleft=False, labelright=False, labeltop=False)
        self.canvas.fig.tight_layout()
        self.canvas.fig.canvas.draw()
        self.layer_work.data = []
        return None
            
    def run_for_all_path(self):
        if self.dataframe is not None:
            raise ValueError("Data Frame list is not empty")
        
        df_list = []
        first_mtp = None
        self.viewer.window._status_bar._toggle_activity_dock(True)
        with progress(self.mt_paths) as pbr:
            for i, path in enumerate(pbr):
                subpbr = progress(total=10, nest_under=pbr)
                mtp = MTPath(self.image.scale.x, 
                             label=i, 
                             interval_nm=self.interval, 
                             light_background=self.light_background
                             )
                             
                try:
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
                    mtp.zxshift_correction()
                    subpbr.update(1)
                    subpbr.set_description("Updating path edges")
                    mtp.calc_center_shift()
                    mtp.update_points()
                    subpbr.update(1)
                    subpbr.set_description("Reloading images")
                    mtp.load_images(self.image)
                    subpbr.update(1)
                    subpbr.set_description("XYZ-rotation")
                    mtp.grad_path()
                    mtp.rotate3d()
                    cachemap[f"{self.image.name}-{mtp.label}",
                            hash(str(mtp._even_interval_points))] = mtp._sub_images
                    subpbr.update(1)
                    subpbr.set_description("Determining MT radius")
                    mtp.determine_radius()
                    subpbr.update(1)
                    subpbr.set_description("Calculating pitch lengths")
                    mtp.calc_pitch_length()
                    subpbr.update(1)
                    subpbr.set_description("Calculating PF numbers")
                    mtp.calc_pf_number()
                    subpbr.update(1)
                    
                    df = mtp.to_dataframe()
                    df_list.append(df)
                    
                    subpbr.close()
                    
                    if i == 0:
                        first_mtp = mtp
                        
                except Exception as e:
                    # TODO: show full traceback. Before that add QScrollBar into QMessageBox
                    # raise_error_message(self, f"Error in iteration {i}.\n\n{traceback.format_exc()}")
                    raise_error_message(self, f"Error in iteration {i}.\n\n{e.__class__.__name__}: {e}")
                    break
            
        self.viewer.window._status_bar._toggle_activity_dock(False)
        self.from_dataframe(pd.concat(df_list, axis=0), first_mtp)
        return None
    
    def from_path(self):
        """
        Open a file dialog, choose a csv file and load it.
        """        
        dlg = QFileDialog()
        hist = napari.utils.history.get_open_history()
        dlg.setHistory(hist)
        filenames, _ = dlg.getOpenFileNames(
            parent=self.viewer.window.qt_viewer,
            caption="Select image ...",
            directory=hist[0],
        )
        if filenames != [] and filenames is not None:
            path = filenames[0]
            napari.utils.history.update_open_history(filenames[0])
        else:
            return None
        
        self.viewer.window._status_bar._toggle_activity_dock(True)
        with progress(total=0) as pbr:
            pbr.set_description("Connecting csv to image")
            df = pd.read_csv(path)
            self.from_dataframe(df)
        self.viewer.window._status_bar._toggle_activity_dock(False)
        return None
    
    def from_dataframe(self, df:pd.DataFrame, mtp:MTPath=None):
        """
        Convert data frame information into points layer and update widgets. If the first MTPath object
        is available, use mtp argument.
        """        
        self._init_layers()
        
        self.dataframe = df
        
        if mtp is None:
            mtp = self.get_one_mt(0)
            mtp._even_interval_points = df[df["label"]==0][["z", "y", "x"]].values
        
        df["Note"] = np.array([""]*df.shape[0], dtype="<U32")
        
        self.layer_prof.data = df[["z", "y", "x"]].values
        self.layer_prof.properties = df
        self.layer_prof.face_color = "pitch"
        self.layer_prof.face_contrast_limits = [4.08, 4.36]
        self.layer_prof.face_colormap = BlueToRed
        self.layer_prof.text.visible = True
        self.layer_prof.size = mtp.radius[1]/mtp.scale
        
        self.layer_work.mode = "pan_zoom"
        
        self.viewer.layers.selection = {self.layer_prof}
        self.canvas._init_widget_params()
        self.canvas.label_choice.setMaximum(len(df["label"].unique())-1)
        self.canvas.slider.setRange(0, mtp.npoints-1)
        self.canvas.call()
        self.canvas.add_note_edit()
        
        self.viewer.dims.current_step = (int(df["z"].mean()), 0, 0)
        return None
    
    def get_one_mt(self, label:int=0):
        """
        Prepare current MTPath object from data frame.
        """        
        df = self.dataframe[self.dataframe["label"]==label]
        mtp = MTPath(self.image.scale.x, 
                     label=label, 
                     interval_nm=self.interval,
                     light_background=self.light_background
                     )
        mtp._even_interval_points = df[["z","y","x"]].values
        cached_rotate(mtp, self.image)
        mtp.radius_peak = float(df["MTradius"].values[0])
        mtp.pitch_lengths = df["pitch"].values
        mtp._pf_numbers = df["nPF"].values
        return mtp
    
    def save_results(self, path:str=None):
        # open file dialog if path is not specified.
        if not isinstance(path, str):
            dlg = QFileDialog()
            hist = napari.utils.history.get_save_history()
            dlg.setHistory(hist)
            filename, _ = dlg.getSaveFileName(
                parent=self,
                caption="Save results ...",
                directory=hist[0],
            )
            if filename:
                path = filename
                napari.utils.history.update_save_history(filename)
            else:
                return None
        if not path.endswith(".csv"):
            path += ".csv"
        self.dataframe.to_csv(path)
        return None
    
    
    def paint_mt(self):
        # TODO: paint using labels layer
        lbl = np.zeros(self.layer_image.data.shape, dtype=np.uint8)
        color = dict()
        for i, row in self.dataframe.iterrows():
            crds = row[["z","y","x"]]
            color[i] = BlueToRed.map((row["pitch"] - 4.08)/(4.36 - 4.08))
            # update lbl
            
        self.viewer.add_labels(lbl, color=color, scale=self.layer_image.scale,
                               translate=self.layer_image.translate)
    
    def clear(self):
        self._init_layers()
        if hasattr(self, "canvas"):
            self.canvas.ax.cla()
            self.canvas.fig.canvas.draw()
        cachemap.clear()
        return None
        
    def _add_widgets(self):
        self.setLayout(QVBoxLayout())
        
        main_buttons = QFrame(self)
        main_buttons.setLayout(QHBoxLayout())
        
        register_button = QPushButton("Mark 📝", main_buttons)
        register_button.setToolTip("Register current points in 'Working Layer' as a MT path.")
        register_button.clicked.connect(self.register_path)
        
        run_button = QPushButton("Run 👉", main_buttons)
        run_button.setToolTip("Run profiler for all the paths.")
        run_button.clicked.connect(self.run_for_all_path)
        
        main_buttons.layout().addWidget(register_button)
        main_buttons.layout().addWidget(run_button)

        other_buttons = QFrame(self)
        other_buttons.setLayout(QHBoxLayout())

        load_img_button = QPushButton("Open image 🔬", other_buttons)
        load_img_button.setToolTip("Open an image and start analysis.")
        load_img_button.clicked.connect(self.open_image_file)
                
        load_csv_button = QPushButton("Load csv 📂", other_buttons)
        load_csv_button.setToolTip("Load results from csv.")
        load_csv_button.clicked.connect(self.from_path)
        
        save_button = QPushButton("Save 💾", other_buttons)
        save_button.setToolTip("Save results.")
        save_button.clicked.connect(self.save_results)
        
        clear_button = QPushButton("Clear ✘", other_buttons)
        clear_button.setToolTip("Clear all.")
        clear_button.clicked.connect(self.clear)
        

        other_buttons.layout().addWidget(load_img_button)
        other_buttons.layout().addWidget(load_csv_button)
        other_buttons.layout().addWidget(save_button)
        other_buttons.layout().addWidget(clear_button)
        
        self.layout().addWidget(main_buttons)
        self.layout().addWidget(other_buttons)
        
        self.canvas = SlidableFigureCanvas(self)
        
        self.layout().addWidget(self.canvas)
        return None
    

class SlidableFigureCanvas(QWidget):
    def __init__(self, mtprofiler:MTProfiler=None):
        super().__init__(mtprofiler)
        self.mtprofiler = mtprofiler
        self._mtpath = None
        self.setLayout(QVBoxLayout())
        self.last_called = self.imshow_zx_raw
        
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setMinimumWidth(50)
        self.slider.setToolTip("Slide along a MT")
        self.slider.valueChanged.connect(self.call)
        
        self.fig = self.fig = plt.figure()
        canvas = FigureCanvas(self.fig)        
        self._ax = None
        
        figindex = QFrame(self)
        figindex.setLayout(QHBoxLayout())
        
        label = QLabel()
        label.setText("MT No.")
        figindex.layout().addWidget(label)
        
        self.label_choice = QSpinBox(self)
        self.label_choice.setMinimumWidth(25)
        self.label_choice.setToolTip("MT label")
        @self.label_choice.valueChanged.connect
        def _(*args):
            # We have to block value changed event here, otherwise event will be emitted twice.
            self.label_choice.setEnabled(False)
            self.update_mtpath()
            self.update_note()
            self.label_choice.setEnabled(True)
        
        figindex.layout().addWidget(self.label_choice)
        figindex.layout().addWidget(self.slider)
        
        frame1 = QFrame(self)
        frame1.setLayout(QHBoxLayout())
        
        self.imshow_buttons:list[QPushButton] = []
        
        imshow0 = QPushButton("XY raw 📈", frame1)
        imshow0.setCheckable(True)
        imshow0.setToolTip("Call imshow_yx_raw")
        imshow0.clicked.connect(self.imshow_yx_raw)
        frame1.layout().addWidget(imshow0)
        self.imshow_buttons.append(imshow0)
        
        imshow1 = QPushButton("XZ raw 📈", frame1)
        imshow1.setCheckable(True)
        imshow1.setToolTip("Call imshow_zx_raw")
        imshow1.clicked.connect(self.imshow_zx_raw)
        frame1.layout().addWidget(imshow1)
        self.imshow_buttons.append(imshow1)
        
        imshow2 = QPushButton("XZ avg 📈", frame1)
        imshow2.setCheckable(True)
        imshow2.setToolTip("Call imshow_zx_ave")
        imshow2.clicked.connect(self.imshow_zx_ave)
        frame1.layout().addWidget(imshow2)
        self.imshow_buttons.append(imshow2)
        
        frame2 = QFrame()
        frame2.setLayout(QHBoxLayout())
        
        send = QPushButton("View 👁", frame2)
        send.setToolTip("Send current MT fragment to viewer.")
        send.clicked.connect(self.send_to_napari)
        frame2.layout().addWidget(send)
        
        self.info = QLabel()
        frame2.layout().addWidget(self.info)
        
        self.layout().addWidget(figindex)
        self.layout().addWidget(canvas)
        self.layout().addWidget(frame1)
        self.layout().addWidget(frame2)
        
        self._init_widget_params()
        
        return None
    
    @property
    def mtpath(self):
        if self._mtpath is None:
            self._mtpath = self.mtprofiler.get_one_mt(0)
        return self._mtpath
    
    @property
    def ax(self):
        if self._ax is None:
            self._ax = self.fig.add_subplot(111)
            self._ax.set_aspect("equal")
        return self._ax
    
    def _init_widget_params(self):
        self.label_choice.setValue(0)
        self.label_choice.setRange(0, 0)
        self.slider.setValue(0)
        self.slider.setRange(0, 0)
        self.info.setText("X.XX nm / XX pf")
        return None
        
    
    def add_note_edit(self):
        if hasattr(self, "line_edit"):
            return None
        note_frame = QFrame(self)
        note_frame.setLayout(QHBoxLayout())
        
        label = QLabel()
        label.setText("Note:")
        note_frame.layout().addWidget(label)
        
        self.line_edit = QLineEdit()
        self.line_edit.setToolTip("Add note to the current MT.")
        self.line_edit.editingFinished.connect(self.update_note)
            
        note_frame.layout().addWidget(self.line_edit)
        
        self.layout().addWidget(note_frame)
        return None
    
    def update_note(self):
        df = self.mtprofiler.dataframe
        l = self.label_choice.value()
        df.loc[df["label"]==l, "Note"] = self.line_edit.text()
        return None
        
    def update_mtpath(self):
        label = self.label_choice.value()
        self._mtpath = self.mtprofiler.get_one_mt(label)
        self.slider.setRange(0, self._mtpath.npoints-1)
        sl = self.mtprofiler.dataframe["label"] == label
        note = self.mtprofiler.dataframe[sl]["Note"].values[0]
        self.line_edit.setText(note)
        return self.call()
    
    def send_to_napari(self):
        if self.mtprofiler.dataframe is None:
            return None
        i = self.slider.value()
        img = self._mtpath._sub_images[i]
        self.mtprofiler.viewer.add_image(img, scale=img.scale, name=img.name,
                                         rendering="minip" if self.light_background else "mip")
        return None
    
    
    def imshow_yx_raw(self):
        if self.mtprofiler.dataframe is None:
            return None
        self.ax.cla()
        i = self.slider.value()
        self.mtpath.imshow_yx_raw(i, ax=self.ax)
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.last_called = self.imshow_yx_raw
        for b in self.imshow_buttons:
            b.setDown(False)
        self.imshow_buttons[0].setDown(True)
        return None
    
    def imshow_zx_raw(self):
        if self.mtprofiler.dataframe is None:
            return None
        self.ax.cla()
        i = self.slider.value()
        self.mtpath.imshow_zx_raw(i, ax=self.ax)
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.last_called = self.imshow_zx_raw
        for b in self.imshow_buttons:
            b.setDown(False)
        self.imshow_buttons[1].setDown(True)
        return None
    
    def imshow_zx_ave(self):
        if self.mtprofiler.dataframe is None:
            return None
        self.ax.cla()
        i = self.slider.value()
        self.mtpath.imshow_zx_ave(i, ax=self.ax)
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.last_called = self.imshow_zx_ave
        for b in self.imshow_buttons:
            b.setDown(False)
        self.imshow_buttons[2].setDown(True)
        return None
    
    def call(self):
        i = self.slider.value()
        pitch = self.mtpath.pitch_lengths[i]
        npf = self.mtpath.pf_numbers[i]
        self.info.setText(f"{pitch:.2f} nm / {npf} pf")
        return self.last_called()