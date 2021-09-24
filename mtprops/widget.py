import pandas as pd
from typing import TYPE_CHECKING, TypeVar
from collections import OrderedDict
import numpy as np
from scipy import ndimage as ndi
from napari.utils.colormaps.colormap import Colormap
from napari.qt import progress, thread_worker
from qtpy.QtWidgets import QMessageBox
from qtpy.QtGui import QFont
from pathlib import Path
from magicgui.widgets import Table, TextEdit
import matplotlib.pyplot as plt

from ._dependencies import impy as ip
from ._dependencies import mcls, magicclass, field, button_design, click, set_options, Figure, TupleEdit, CheckButton
from .mtpath import MTPath, calc_total_length

if TYPE_CHECKING:
    from napari.layers import Image, Points, Labels

_V = TypeVar("_V")

class CacheMap:
    def __init__(self, maxgb:float=2.0):
        self.maxgb = maxgb
        self.cache: OrderedDict[int, _V] = OrderedDict()
        self.gb = 0.0
    
    def __getitem__(self, key: tuple[str, int]) -> _V:
        real_key, identifier = key
        idf, value = self.cache[real_key]
        if idf == identifier:
            return value
        else:
            raise KeyError("Wrong identifier")
    
    def __setitem__(self, key: tuple[str, int], value:_V):
        real_key, identifier = key
        self.cache[real_key] = (identifier, value)
        size = sum(a.nbytes for a in value)/1e9
        self.gb += size
        while self.gb > self.maxgb:
            self.pop()
    
    def pop(self) -> None:
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
                                   hash(str(mtp.points))]
    except KeyError:
        mtp.load_images(image)
        mtp.grad_path()
        mtp.rotate3d()
        cachemap[f"{image.name}-{mtp.label}",
                 hash(str(mtp.points))] = mtp._sub_images
    return None

def raise_error_message(parent, msg:str):
    return QMessageBox.critical(parent, "Error", msg,QMessageBox.Ok)

@thread_worker(progress={"total": 0, "desc": "Reading Image"})
def imread(img, binsize):
    with ip.SetConst("SHOW_PROGRESS", False):
        imgb = img.binning(binsize, check_edges=False).data
    
    return imgb

@magicclass
class ImageLoader:
    path = field(Path)
    scale = field(str, options={"label": "scale (nm)"})
    
    @button_design(text="OK")
    def call_button(self):
        try:
            scale = float(self.scale.value)
        except Exception as e:
            raise type(e)(f"Invalid input: {self.scale.value}")
        
        self.img.scale.x = self.img.scale.y = self.img.scale.z = scale
        return self.img
    
    @path.connect
    def _read_scale(self, event):
        path = event.value
        self.img = ip.lazy_imread(path, chunks=(64, 1024, 1024))
        self.scale.value = f"{self.img.scale.x:.3f}"

@magicclass
class MTProfiler:
    
    ### Index ###########################################################################
    
    _loader = field(ImageLoader)
    
    @magicclass(layout="horizontal", labels=False)
    class operation:
        def register_path(self): ...
        def run_for_all_path(self): ...
        def settings(self): ...
        def clear_current(self): ...
        def clear_all(self): ...
        def info(self): ...
    
    @magicclass(layout="horizontal", labels=False)
    class io:
        def open_image_file(self, path: Path): ...
        def from_csv_file(self, path: Path): ...
        def save_results(self, path: Path): ...
    
    @magicclass(layout="horizontal")
    class mt:
        mtlabel = field(int, options={"max": 0}, name="MTLabel")
        pos = field(int, widget_type="Slider", options={"max":0}, name="Pos")
    
    canvas = field(Figure, name="Figure", options={"figsize":(4.2, 1.8), "tooltip": "Projections"})
        
    @magicclass(layout="horizontal", labels=False)
    class viewer_op:
        def send_to_napari(self): ...
        focus = field(CheckButton, options={"text": "🔍"})
        def show_3d_path(self): ...
        def show_table(self): ...
        txt = field("X.XX nm / XX pf", options={"enabled": False}, name="result")
    
    line_edit = field(str, name="Note: ")
    
    plot = field(Figure, name="Plot", options={"figsize":(4.2, 1.8), "tooltip": "Plot of pitch lengths"})
    
    #####################################################################################
        
    POST_PROCESSING = [io.save_results, viewer_op.send_to_napari,
                   viewer_op.show_table, viewer_op.show_3d_path]

    POST_IMREAD = [io.from_csv_file, operation.register_path]
    
    @set_options(start={"widget_type": TupleEdit, "options": {"step": 0.1}}, 
                 end={"widget_type": TupleEdit, "options": {"step": 0.1}},
                 limit={"widget_type": TupleEdit, "options": {"step": 0.02}, "label": "limit (nm)"})
    def set_colormap(self, start=(0.0, 0.0, 1.0), end=(1.0, 0.0, 0.0), limit=(4.10, 4.36)):
        """
        Set the color-map for painting microtubules.
        Parameters
        ----------
        start : tuple, default is (0.0, 0.0, 1.0)
            RGB color that corresponds to the most compacted microtubule.
        end : tuple, default is (1.0, 0.0, 0.0)
            RGB color that corresponds to the most expanded microtubule.
        limit : tuple, default is (4.10, 4.36)
            Color limit (nm).
        """        
        self.label_colormap = Colormap([start+(1,), end+(1,)], name="PitchLength")
        self.label_colorlimit = limit
        self._update_colormap()
        return None
    
    def subtomogram_averaging(self):
        """
        Average subtomograms along a MT.
        """        
        df = self.current_mt.average_tomograms()
        avg_img = self.current_mt.rot_average_tomogram()
        table = Table(value=df)
        self.parent_viewer.window.add_dock_widget(table)
        
        if self.light_background:
            avg_img = -avg_img
            
        self.parent_viewer.add_image(avg_img, scale=avg_img.scale, name="AVG",
                                     rendering="iso", iso_threshold=0.8)
        return None

    def _update_colormap(self):
        if self.layer_paint is None:
            return None
        color = self.layer_paint.color
        lim0, lim1 = self.label_colorlimit
        for i, (_, row) in enumerate(self.dataframe.iterrows()):
            color[i+1] = self.label_colormap.map((row["pitch"] - lim0)/(lim1 - lim0))
        self.layer_paint.color = color
        return None
        
    def __post_init__(self):
        self._mtpath = None
        self.image = None
        self.layer_image: Image = None
        self.layer_prof: Points = None
        self.layer_work: Points = None
        self.layer_paint: Labels = None
        
        self.settings()
        self.set_colormap()
        self.mt.pos.min_width = 70
        self.canvas.ax.set_aspect("equal")
        self.canvas.ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        call_button = self._loader["call_button"]
        call_button.changed.connect(self._load_image)
                
    @property
    def current_mt(self):
        if self._mtpath is None:
            self._mtpath = self._get_one_mt(0)
        return self._mtpath
    
    @operation.wraps
    @click(enabled=False, enables=operation.run_for_all_path)
    @button_design(text="📝")
    def register_path(self):
        """
        Register current selected points as a MT path.
        """        
        # check length
        total_length = calc_total_length(self.layer_work.data)*self.image.scale.x
        if total_length < self.interval*3:
            raise_error_message(self.native,
                                f"Path is too short: {total_length:.2f} nm\n"
                                f"Must be longer than {self.interval*3:.2f} nm.")
            return None
        
        self.layer_prof.add(self.layer_work.data)
        self.mt_paths.append(self.layer_work.data)
        self.canvas.ax.plot(self.layer_work.data[:,2], self.layer_work.data[:,1], color="gray", lw=2.5)
        self.canvas.ax.set_xlim(0, self.image.sizeof("x"))
        self.canvas.ax.set_ylim(self.image.sizeof("y"), 0)
        self.canvas.ax.set_aspect("equal")
        self.canvas.ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        self.canvas.figure.tight_layout()
        self.canvas.figure.canvas.draw()
        self.layer_work.data = []
        return None
    
    @operation.wraps
    @click(enabled=False, enables=POST_PROCESSING)
    @button_design(text="👉")
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
                             light_background=self.light_background,
                             radius_pre_nm=self.radius_pre_nm,
                             radius_nm=self.radius_nm
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
                             hash(str(mtp.points))] = mtp._sub_images
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
    @set_options(radius_pre_nm={"widget_type": TupleEdit, "label": "z/y/x-radius-pre (nm)"}, 
                 radius_nm={"widget_type": TupleEdit, "label": "z/y/x-radius (nm)"},
                 bin_size={"min": 1, "max": 8})
    @button_design(text="⚙")
    def settings(self, interval_nm:float=24, light_background:bool=True,
                 radius_pre_nm=(22.0, 28.0, 28.0), radius_nm=(16.7, 16.7, 16.7),
                 bin_size:int=4):
        """
        Change MTProps setting.
        Parameters
        ----------
        interval_nm : float, optional
            Interval between points to analyze.
        light_background : bool, optional
            Light background or not
        radius_pre_nm : tuple[float, float, float]
            Images in this range will be considered to determine MT tilt and shift.
        radius_nm : tuple[float, float, float]
            Images in this range will be considered to determine MT pitch length and PF number.
        bin_size : int
            Binning (pixel) that will be applied to the image in the viewer. This parameter does
            not affect the results of analysis.
        """        
        self.interval = interval_nm
        self.light_background = light_background
        self.radius_pre_nm = radius_pre_nm
        self.radius_nm = radius_nm
        self.binsize = bin_size
        if self.layer_image is not None:
            self.layer_image.rendering = "minip" if self.light_background else "mip"
    
    @operation.wraps
    @button_design(text="❌")
    def clear_current(self):
        """
        Clear current selection.
        """        
        self.layer_work.data = []
        return None
    
    @operation.wraps
    @set_options(_={"widget_type":"Label"})
    @button_design(text="💥")
    def clear_all(self, _="Are you sure to clear all?"):
        """
        Clear all.
        """        
        self._init_layers()
        if hasattr(self, "canvas"):
            self.canvas.figure.clf()
            self.canvas.figure.add_subplot(111)
            self.canvas.draw()
        cachemap.clear()
        return None
    
    @operation.wraps
    @button_design(text="❓")
    def info(self):
        """
        Show information of dependencies.
        """        
        import napari
        import magicgui
        from .__init__ import __version__
        import dask
        
        value = f"MTProps: {__version__}"\
                f"impy: {ip.__version__}"\
                f"magicgui: {magicgui.__version__}"\
                f"magicclass: {mcls.__version__}"\
                f"napari: {napari.__version__}"\
                f"dask: {dask.__version__}"
        
        txt = TextEdit(value=value)
        self.read_only = True
        txt.native.setFont(QFont("Consolas"))
        txt.show()
    
    @io.wraps
    @button_design(text="Open image 🔬")
    def open_image_file(self):
        """
        Open an image and add to viewer.
        """
        self._loader.show()
        return None
        
    @io.wraps
    @click(enabled=False, enables=POST_PROCESSING)
    @button_design(text="Load csv 📂")
    def from_csv_file(self, path: Path):
        """
        Choose a csv file and load it.
        """        
        self.parent_viewer.window._status_bar._toggle_activity_dock(True)
        with progress(total=0) as pbr:
            pbr.set_description("Loading csv")
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
    
    @viewer_op.wraps
    @click(enabled=False)
    @button_design(text="👁️")    
    def send_to_napari(self):
        """
        Send the current MT fragment 3D image (not binned) to napari viewer.
        """        
        if self.dataframe is None:
            return None
        i = self.mt.pos.value
        img = self.current_mt._sub_images[i]
        self.parent_viewer.add_image(img, scale=img.scale, name=img.name,
                                     rendering="minip" if self.light_background else "mip")
        return None
    
    @viewer_op.focus.connect
    def _set_down(self, event=None):
        # ensure button is down in napari
        focused = self.viewer_op.focus.value
        if focused:
            self._focus_on()
        elif self.layer_paint is not None:
            self.layer_paint.show_selected_label = False
        self.viewer_op.focus.native.setDown(focused)
        
    @mt.mtlabel.connect
    @mt.pos.connect
    def _focus_on(self):
        """
        Change camera focus to the position of current MT fragment.
        """        
        if not self.viewer_op.focus.value or self.layer_paint is None:
            return None
        
        viewer = self.parent_viewer
        i = self.mt.pos.value
        scale = viewer.layers["MT Profiles"].scale
        next_coords = self.current_mt.points[i]
        next_center = next_coords * scale
        viewer.dims.current_step = list(next_coords.astype(np.int64))
        
        viewer.camera.center = next_center
        zoom = viewer.camera.zoom
        viewer.camera.events.zoom() # Here events are emitted and zoom changes automatically.
        viewer.camera.zoom = zoom
        
        self.layer_paint.show_selected_label = True
        for k, (_, row) in enumerate(self.dataframe.iterrows()):
            if row["label"] == self.current_mt.label and row["number"] == i:
                self.layer_paint.selected_label = k+1
                break
        return None
    
    @viewer_op.wraps
    @click(enabled=False)
    @button_design(text="📜")
    def show_table(self):
        """
        Show result table.
        """        
        table = Table(value=self.dataframe)
        self.parent_viewer.window.add_dock_widget(table)
        return None
    
    @viewer_op.wraps
    @click(enabled=False)
    @button_design(text="📈")
    def show_3d_path(self):
        """
        Show 3D paths of microtubule center axes.
        """        
        paths = []
        for _, df in self.dataframe.groupby("label"):
            paths.append(df[["z", "y", "x"]].values/self.binsize)
        self.parent_viewer.add_shapes(paths, shape_type="path", edge_color="lime", edge_width=1,
                                      scale=self.layer_image.scale, translate=self.layer_image.translate)
        return None
    
    @click(visible=False)
    def iter_mt(self):
        i = 0
        while True:
            try:
                yield self._get_one_mt(i)
            except IndexError:
                break
            finally:
                i += 1

    def _paint_mt(self):
        """
        Paint microtubule fragments by its pitch length.
        
        1. Prepare small boxes and make masks inside them.
        2. Map the masks to the reference image.
        3. Erase masks using reference image, based on intensity.
        """        
        from .mtpath import rot3dinv, da, make_slice_and_pad
        
        lbl = ip.zeros(self.layer_image.data.shape, dtype=np.uint8)
        color: dict[int, list[float]] = {0: [0, 0, 0, 0]}
        bin_scale = self.layer_image.scale[0] # scale of binned reference image
        lz, ly, lx = [int(r/bin_scale*1.4)*2 + 1 for r in self.radius_nm]

        with ip.SetConst("SHOW_PROGRESS", False):
            tasks = []
            for i, (_, row) in enumerate(self.dataframe.iterrows()):                
                z, y, x = np.indices((lz, ly, lx))
                r0 = self.current_mt.radius_peak/self.current_mt.scale*0.9/self.binsize
                r1 = self.current_mt.radius_peak/self.current_mt.scale*1.1/self.binsize
                _sq = (z-lz//2)**2 + (x-lx//2)**2
                domain = (r0**2 < _sq) & (_sq < r1**2)
                domain = domain.astype(np.float32)
                ry = int(self.interval/bin_scale/2)
                domain[:, :ly//2-ry] = 0
                domain[:, ly//2+ry+1:] = 0
                domain = ip.array(domain, axes="zyx")
                ang_zy = row["angle_zy"]
                ang_yx = row["angle_yx"]
                tasks.append(da.from_delayed(rot3dinv(domain, ang_yx, ang_zy), shape=domain.shape, 
                                             dtype=np.float32) > 0.3)
            
            out: list[np.ndarray] = da.compute(tasks)[0]

        # paint roughly
        for i, (_, row) in enumerate(self.dataframe.iterrows()):
            center = row[["z","y","x"]].astype(np.int16)//self.binsize
            sl = []
            outsl = []
            # We should deal with the borders of image.
            for c, l, size in zip(center, [lz, ly, lx], lbl.shape):
                _sl, _pad = make_slice_and_pad(c, l//2, size)
                sl.append(_sl)
                if _pad[0] > 0:
                    outsl.append(slice(_pad[0], None))
                elif _pad[1] > 0:
                    outsl.append(slice(None, -_pad[1]))
                else:
                    outsl.append(slice(None, None))

            sl = tuple(sl)
            outsl = tuple(outsl)
            lbl.value[sl][out[i][outsl]] = i + 1
        
        # paint finely
        ref_filt = ndi.gaussian_filter(self.layer_image.data, sigma=2)
        
        if self.light_background:
            thr = np.percentile(ref_filt[lbl>0], 95)
            lbl[ref_filt>thr] = 0
        else:
            thr = np.percentile(ref_filt[lbl>0], 5)
            lbl[ref_filt<thr] = 0
        
        # Labels layer properties
        props = pd.DataFrame([])
        props["ID"] = self.dataframe.apply(lambda x: "{}-{}".format(x["label"], x["number"]), axis=1)
        props["pitch"] = self.dataframe["pitch"].map("{:.2f}".format)
        props["nPF"] = self.dataframe["nPF"]
        back = pd.DataFrame({"ID": [np.nan], "pitch": [np.nan], "nPF": [np.nan]})
        props = pd.concat([back, props])
        
        # Add labels layer
        self.layer_paint = self.parent_viewer.add_labels(
            lbl.value, color=color, scale=self.layer_image.scale,
            translate=self.layer_image.translate, opacity=0.33, name="Label",
            properties=props
            )
        
        self._update_colormap()
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
        mtp.points = df[["z","y","x"]].values
        cached_rotate(mtp, self.image)
        mtp.radius_peak = float(df["MTradius"].values[0])
        mtp.pitch_lengths = df["pitch"].values
        mtp._pf_numbers = df["nPF"].values
        mtp.grad_angles_yx = df["angle_yx"]
        mtp.grad_angles_zy = df["angle_zy"]
        return mtp
    
    def _plot_pitch(self):
        x = np.arange(self.current_mt.npoints)*self.interval
        with plt.style.context("dark_background"):
            self.plot.ax.cla()
            self.plot.ax.plot(x, self.current_mt.pitch_lengths, color="white")
            self.plot.ax.set_xlabel("position (nm)")
            self.plot.ax.set_ylabel("pitch (nm)")
            self.plot.ax.set_ylim(*self.label_colorlimit)
            self.plot.figure.tight_layout()
            self.plot.draw()
        
        return None
    
    @click(disables=POST_PROCESSING, enables=POST_IMREAD)
    def _load_image(self, event=None):
        self.parent_viewer.window._status_bar._toggle_activity_dock(True)
        img = self._loader.img
        worker = imread(img, self.binsize)
        @worker.returned.connect
        def _(imgb):
            self._init_widget_params()
            tr = (self.binsize-1)/2*img.scale.x
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
        self._loader.close()
        return None
    
    def _from_dataframe(self, df:pd.DataFrame, mtp:MTPath=None):
        """
        Convert data frame information into points layer and update widgets. If the first MTPath object
        is available, use mtp argument.
        """        
        self._init_layers()
        
        self.dataframe = df
        
        # Set current MT to the first MT in the DataFrame
        if mtp is None:
            mtp = self._get_one_mt(0)
            mtp.points = df[df["label"]==0][["z", "y", "x"]].values
        
        # Add a column for note
        df["Note"] = np.array([""]*df.shape[0], dtype="<U32")
        
        # Show text
        self.layer_prof.data = df[["z", "y", "x"]].values
        self.layer_prof.properties = df
        self.layer_prof.text.visible = False
        self.layer_prof.face_color = [0, 0, 0, 0]
        self.layer_prof.edge_color = [0, 0, 0, 0]
        self.layer_prof.selected_data = {}
        
        # Paint MTs by its pitch length
        self._paint_mt()
        self._plot_pitch()
        
        self.layer_work.mode = "pan_zoom"
        self.parent_viewer.layers.selection = {self.layer_paint}
        
        # initialize GUI
        self._init_widget_params()
        self.mt.mtlabel.max = len(df["label"].unique())-1
        self.mt.pos.max = mtp.npoints-1
        self._mtpath = mtp
        
        self.canvas.figure.clf()
        self.canvas.figure.add_subplot(131)
        self.canvas.figure.add_subplot(132)
        self.canvas.figure.add_subplot(133)
        self._imshow_all()
        
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
        # TODO: simpler implementation after napari==0.4.12
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
                                    properties={"pitch": np.array([0.0], dtype=np.float64)},
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
        
        if self.layer_paint is not None:
            self.parent_viewer.layers.remove(self.layer_paint.name)
            self.layer_paint = None
            
        self.mt_paths = []
        self.dataframe = None

        return None
    
    @mt.pos.connect
    def _imshow_all(self):
        if self.dataframe is None:
            return None
        i = self.mt.pos.value
        pitch = self.current_mt.pitch_lengths[i]
        npf = self.current_mt.pf_numbers[i]
        self.viewer_op.txt.value = f"{pitch:.2f} nm / {npf} pf"
        
        for j in range(3):
            self.canvas.axes[j].cla()
            self.canvas.axes[j].tick_params(labelbottom=False,labelleft=False, labelright=False, labeltop=False)
            
        subimg = self.current_mt._sub_images[i]
        lz, ly, lx = subimg.shape
        with ip.SetConst("SHOW_PROGRESS", False):
            self.canvas.axes[0].imshow(subimg.proj("z"), cmap="gray")
            self.canvas.axes[0].set_xlabel("x")
            self.canvas.axes[0].set_ylabel("y")
            self.canvas.axes[1].imshow(subimg.proj("y"), cmap="gray")
            self.canvas.axes[1].set_xlabel("x")
            self.canvas.axes[1].set_ylabel("z")
            self.canvas.axes[2].imshow(self.current_mt.average_images[i], cmap="gray")
            self.canvas.axes[2].set_xlabel("x")
            self.canvas.axes[2].set_ylabel("z")
        
        ylen = int(self.current_mt.radius[1]/self.current_mt.scale)
        ymin, ymax = ly/2 - ylen, ly/2 + ylen
        r = self.current_mt.radius_peak/self.current_mt.scale*self.current_mt.__class__.outer
        xmin, xmax = -r + lx/2, r + lx/2
        self.canvas.axes[0].plot([xmin, xmin, xmax, xmax, xmin], [ymin, ymax, ymax, ymin, ymin], color="lime")
        self.canvas.axes[0].text(1, 1, f"{self.mt.mtlabel.value}-{i}", 
                                 color="lime", font="Consolas", size=15, va="top")
    
        theta = np.deg2rad(np.arange(360))
        r = self.current_mt.radius_peak/self.current_mt.scale*self.current_mt.__class__.inner
        self.canvas.axes[1].plot(r*np.cos(theta) + lx/2, r*np.sin(theta) + lz/2, color="lime")
        r = self.current_mt.radius_peak/self.current_mt.scale*self.current_mt.__class__.outer
        self.canvas.axes[1].plot(r*np.cos(theta) + lx/2, r*np.sin(theta) + lz/2, color="lime")
                
        self.canvas.figure.tight_layout()
        self.canvas.draw()
    
    @line_edit.connect
    def _update_note(self, event=None):
        df = self.dataframe
        df.loc[df["label"]==self.mt.mtlabel.value, "Note"] = self.line_edit.value
        return None
    
    @mt.mtlabel.connect
    def _update_mtpath(self, event=None):
        self._mtpath = self._get_one_mt(self.mt.mtlabel.value)
        self.mt.pos.max = self._mtpath.npoints-1
        sl = self.dataframe["label"] == self.mt.mtlabel.value
        note = self.dataframe[sl]["Note"].values[0]
        self.line_edit.value = note
        self._plot_pitch()
        self._imshow_all()
        return None
    