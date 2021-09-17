import pandas as pd
from typing import TYPE_CHECKING
from collections import OrderedDict
import numpy as np
from napari.utils.colormaps.colormap import Colormap
from napari.qt import progress, thread_worker
from qtpy.QtWidgets import QMessageBox
from pathlib import Path

from ._dependencies import impy as ip
from ._dependencies import magicclass, field, button_design, click, set_options, Figure
from .mtpath import MTPath, calc_total_length

if TYPE_CHECKING:
    from napari.layers import Image, Points

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

@magicclass
class MTProfiler:
    
    ### Index ###########################################################################
    
    @magicclass(layout="horizontal", labels=False)
    class operation:
        def register_path(self): ...
        def run_for_all_path(self): ...
        def settings(self): ...
        def clear_all(self): ...
    
    @magicclass(layout="horizontal", labels=False)
    class io:
        def open_image_file(self, path: Path): ...
        def from_csv_file(self, path: Path): ...
        def save_results(self, path: Path): ...
    
    @magicclass(layout="horizontal")
    class mt:
        mtlabel = field(int, options={"max": 0}, name="MTLabel")
        pos = field(int, widget_type="Slider", options={"max":0}, name="Pos")
    
    canvas = field(Figure, name="Figure")
    
    @magicclass(layout="horizontal", labels=False)
    class imshow:
        def imshow_yx_raw(self): ...
        def imshow_zx_raw(self): ...
        def imshow_zx_ave(self): ...
    
    @magicclass(layout="horizontal", labels=False)
    class viewer_op:
        def send_to_napari(self): ...
        def focus_on(self): ...
        txt = field("X.XX nm / XX pf", options={"enabled": False}, name="result")
    
    line_edit = field(str, name="Note: ")
    
    #####################################################################################
    
    def __init__(self, interval_nm=24, light_background:bool=True):
        if interval_nm <= 0:
            raise ValueError("interval_nm must be a positive float.")

        self.interval = interval_nm
        self.light_background = light_background
        
        self._mtpath = None
        self.last_called = self.imshow_zx_raw
        self.image = None
        self.layer_image: Image = None
        self.layer_prof: Points = None
        self.layer_work: Points = None
    
    def __post_init__(self):
        self.mt.pos.min_width = 70
        
    @property
    def current_mt(self):
        if self._mtpath is None:
            self._mtpath = self._get_one_mt(0)
        return self._mtpath
    
    @operation.wraps
    @click(enabled=False, enables="operation.run_for_all_path")
    @button_design(text="Mark 📝")
    def register_path(self):
        """
        Register current selected points as a MT path.
        """        
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
        self.canvas.ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        self.canvas.figure.tight_layout()
        self.canvas.figure.canvas.draw()
        self.layer_work.data = []
        return None
    
    @operation.wraps
    @click(enabled=False, enables=["imshow.imshow_yx_raw", "imshow.imshow_zx_raw", "imshow.imshow_zx_ave", 
                                   "io.save_results", "viewer_op.send_to_napari", "viewer_op.focus_on"])
    @button_design(text="Run 👉")
    def run_for_all_path(self):
        """
        Run MTProps.
        """        
        if self.dataframe is not None:
            raise ValueError("Data Frame list is not empty")
        
        df_list = []
        first_mtp = None
        self.parent_viewer.window._status_bar._toggle_activity_dock(True)
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
                    raise_error_message(self.native, f"Error in iteration {i}.\n\n{e.__class__.__name__}: {e}")
                    break
            
            else:
                self.parent_viewer.window._status_bar._toggle_activity_dock(False)
                self._from_dataframe(pd.concat(df_list, axis=0), first_mtp)
                
        return None
    
    @operation.wraps
    @button_design(text="⚙", width=50)
    def settings(self, interval_nm:float=33.4, light_background:bool=True):
        """
        Change MTProps setting.

        Parameters
        ----------
        interval_nm : float, optional
            Interval between points to analyze.
        light_background : bool, optional
            Light background or not
        """        
        self.interval = interval_nm
        self.light_background = light_background
    
    @operation.wraps
    @button_design(text="Clear ✘")
    def clear_all(self):
        self._init_layers()
        if hasattr(self, "canvas"):
            self.canvas.ax.cla()
            self.canvas.draw()
        cachemap.clear()
        return None
    
    @io.wraps
    @set_options(scale={"max": 4, "step": 0.1, "label": "scale (nm)"})
    @click(enables=["io.from_csv_file", "operation.register_path"])
    @button_design(text="Open image 🔬")
    def open_image_file(self, path: Path, scale:float=0):
        img = ip.lazy_imread(path, chunks=(64, 1024, 1024))
        if scale > 0:
            img.scale.x = img.scale.y = img.scale.z = scale
        self._load_image(img)
        return None
    
    @io.wraps
    @click(enabled=False)
    @button_design(text="Load csv 📂")
    def from_csv_file(self, path: Path):
        """
        Choose a csv file and load it.
        """        
        self.parent_viewer.window._status_bar._toggle_activity_dock(True)
        with progress(total=0) as pbr:
            pbr.set_description("Connecting csv to image")
            df = pd.read_csv(path)
            self._from_dataframe(df)
        self.parent_viewer.window._status_bar._toggle_activity_dock(False)
        return None
    
    @io.wraps
    @button_design(text="Save 💾")
    @click(enabled=False)
    @set_options(path={"mode": "w"})
    def save_results(self, path:Path):
        """
        Save the results as csv.
        """        
        self.dataframe.to_csv(path)
        return None
            
    @imshow.wraps
    @click(enabled=False)
    @button_design(text="XY raw 📈")
    def imshow_yx_raw(self):
        if self.dataframe is None:
            return None
        self.canvas.ax.cla()
        i = self.mt.pos.value
        self.current_mt.imshow_yx_raw(i, ax=self.canvas.ax)
        self.canvas.figure.tight_layout()
        self.canvas.draw()
        self.last_called = self.imshow_yx_raw
        self.imshow[0].font_color = (240, 241, 242, 255)
        self.imshow[1].font_color = "gray"
        self.imshow[2].font_color = "gray"
    
    @imshow.wraps
    @click(enabled=False)
    @button_design(text="XZ raw 📈")
    def imshow_zx_raw(self):
        if self.dataframe is None:
            return None
        self.canvas.ax.cla()
        i = self.mt.pos.value
        self.current_mt.imshow_zx_raw(i, ax=self.canvas.ax)
        self.canvas.figure.tight_layout()
        self.canvas.draw()
        self.last_called = self.imshow_zx_raw
        self.imshow[0].font_color = "gray"
        self.imshow[1].font_color = (240, 241, 242, 255)
        self.imshow[2].font_color = "gray"
    
    @imshow.wraps
    @click(enabled=False)
    @button_design(text="XZ avg 📈")
    def imshow_zx_ave(self):
        if self.dataframe is None:
            return None
        self.canvas.ax.cla()
        i = self.mt.pos.value
        self.current_mt.imshow_zx_ave(i, ax=self.canvas.ax)
        self.canvas.figure.tight_layout()
        self.canvas.draw()
        self.last_called = self.imshow_zx_ave
        self.imshow[0].font_color = "gray"
        self.imshow[1].font_color = "gray"
        self.imshow[2].font_color = (240, 241, 242, 255)
    
    @viewer_op.wraps
    @click(enabled=False)
    @button_design(text="View 👁")    
    def send_to_napari(self):
        """
        Send the current MT fragment 3D image (not binned) to napari viewer.
        """        
        if self.dataframe is None:
            return None
        i = self.mt.pos.value
        img = self.current_mt._sub_images[i]
        self.viewer.add_image(img, scale=img.scale, name=img.name,
                                        rendering="minip" if self.light_background else "mip")
        return None
    
    @viewer_op.wraps
    @click(enabled=False)
    @button_design(text="Focus 📷")
    def focus_on(self):
        """
        Change camera focus to the position of current MT fragment.
        """        
        viewer = self.parent_viewer
        i = self.mt.pos.value
        scale = viewer.layers["MT Profiles"].scale
        next_coords = self.current_mt._even_interval_points[i]
        next_center = next_coords * scale
        viewer.dims.current_step = list(next_coords.astype(np.int64))
        
        viewer.camera.center = next_center
        zoom = viewer.camera.zoom
        viewer.camera.events.zoom() # Here events are emitted and zoom changes automatically.
        viewer.camera.zoom = zoom
        return None
    
    def _get_one_mt(self, label:int=0):
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
    
    def _load_image(self, img=None, binsize=4):
        self.parent_viewer.window._status_bar._toggle_activity_dock(True)
        worker = imread(img, binsize)
        @worker.returned.connect
        def _(imgb):
            self._init_widget_params()
            tr = (binsize-1)/2*img.scale.x
            if self.layer_image not in self.parent_viewer.layers:
                self.layer_image = self.parent_viewer.add_image(
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
                
            self.parent_viewer.scale_bar.unit = img.scale_unit
            self.parent_viewer.dims.axis_labels = ("z", "y", "x")

            self.image = img

            self.clear_all()
            self.parent_viewer.window._status_bar._toggle_activity_dock(False)
            return None
        worker.start()
        return None
    
    def _from_dataframe(self, df:pd.DataFrame, mtp:MTPath=None):
        """
        Convert data frame information into points layer and update widgets. If the first MTPath object
        is available, use mtp argument.
        """        
        self._init_layers()
        
        self.dataframe = df
        
        if mtp is None:
            mtp = self._get_one_mt(0)
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
        
        self.parent_viewer.layers.selection = {self.layer_prof}
        self._init_widget_params()
        self.mt.mtlabel.max = len(df["label"].unique())-1
        self.mt.pos.max = mtp.npoints-1
        self._mtpath = mtp
        self._call()
        
        self.parent_viewer.dims.current_step = (int(df["z"].mean()), 0, 0)
        return None
    
    def _init_widget_params(self):
        self.mt.mtlabel.value = 0
        self.mt.mtlabel.min = 0
        self.mt.mtlabel.max = 0
        self.mt.pos.value = 0
        self.mt.pos.min = 0
        self.mt.pos.max = 0
        self.viewer_op.txt.value = "X.XX nm / XX pf"
        return None
    
    def _init_layers(self):
        viewer = self.parent_viewer
        img = self.image
        
        common_properties = dict(ndim=3, n_dimensional=True, scale=img.scale, size=4/img.scale.x)
        if self.layer_prof in self.parent_viewer.layers:
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
            
        if self.layer_work in self.parent_viewer.layers:
            self.layer_work.name = "Working Layer-old"
        
        self.layer_work = viewer.add_points(**common_properties,
                                    name="Working Layer",
                                    face_color="yellow"
                                    )
    
        self.layer_work.mode = "add"
        
        if "MT Profiles-old" in self.parent_viewer.layers:
            self.parent_viewer.layers.remove("MT Profiles-old")
        if "Working Layer-old" in self.parent_viewer.layers:
            self.parent_viewer.layers.remove("Working Layer-old")
            
        self.mt_paths = []
        self.dataframe = None

        return None
    
    def _paint_mt(self):
        # TODO: paint using labels layer
        lbl = np.zeros(self.layer_image.data.shape, dtype=np.uint8)
        color = dict()
        for i, row in self.dataframe.iterrows():
            crds = row[["z","y","x"]]
            color[i] = BlueToRed.map((row["pitch"] - 4.08)/(4.36 - 4.08))
            # update lbl
            
        self.parent_viewer.add_labels(lbl, color=color, scale=self.layer_image.scale,
                               translate=self.layer_image.translate)
    
    @line_edit.connect
    def _update_note(self, event=None):
        # TODO: chekc if correctly updated
        df = self.dataframe
        df.loc[df["label"]==self.mt.mtlabel, "Note"] = self.line_edit.value
        return None
    
    @mt.mtlabel.connect
    def _update_mtpath(self, event=None):
        self._mtpath = self._get_one_mt(self.mt.mtlabel.value)
        self.mt.pos.max = self._mtpath.npoints-1
        sl = self.dataframe["label"] == self.mt.mtlabel.value
        note = self.dataframe[sl]["Note"].values[0]
        self.line_edit.value = note
        return self._call()
    
    @mt.pos.connect
    def _call(self, event=None):
        i = self.mt.pos.value
        pitch = self.current_mt.pitch_lengths[i]
        npf = self.current_mt.pf_numbers[i]
        self.viewer_op.txt.value = f"{pitch:.2f} nm / {npf} pf"
        return self.last_called()