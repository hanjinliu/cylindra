import pandas as pd
from typing import TYPE_CHECKING, Iterable
from collections import OrderedDict
import numpy as np
from dask import array as da
from scipy import ndimage as ndi
import napari
from napari.utils.colormaps.colormap import Colormap
from napari.qt import thread_worker, create_worker
from qtpy.QtGui import QFont
from pathlib import Path
from magicgui.widgets import Table, TextEdit
import matplotlib.pyplot as plt

from ._dependencies import impy as ip
from ._dependencies import (mcls, magicclass, field, button_design, click, set_options, 
                            Figure, TupleEdit, CheckButton, Separator, ListWidget)
from .tomogram import MtTomogram, cachemap, angle_corr, dask_affine
from .utils import load_a_subtomogram, make_slice_and_pad
from .const import nm, H, INNER, OUTER

if TYPE_CHECKING:
    from napari.layers import Image, Points, Labels
    from napari._qt.qthreading import GeneratorWorker
    from matplotlib.axes import Axes


@thread_worker(progress={"total": 0, "desc": "Reading Image"})
def bin_image_worker(img, binsize):
    with ip.SetConst("SHOW_PROGRESS", False):
        imgb = img.binning(binsize, check_edges=False).data
    
    return imgb
        

@magicclass
class ImageLoader:
    path = field(Path)
    scale = field(str, options={"label": "scale (nm)"})
    bin_size = field(4, options={"label": "bin size"})
    light_background = field(True, options={"label": "light background"})
    
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
        self._imread(path)
    
    def _imread(self, path:str):
        self.img = ip.lazy_imread(path, chunks=(64, 1024, 1024))
        self.scale.value = f"{self.img.scale.x:.3f}"



@magicclass(layout="horizontal", labels=False)
class WorkerControl:
    info = field(str)
    
    def __post_init__(self):
        self.paused = False
        self.worker: GeneratorWorker = None
        self._last_info = ""
        self.metadata: dict[str] = {}
        self.info.enabled = False
    
    def _set_worker(self, worker):
        self.worker = worker
        
    def Pause(self):
        """
        Pause/Resume thread.
        """        
        if self.paused:
            self.worker.resume()
            self["Pause"].text = "Pause"
            self.info.value = self._last_info
        else:
            self.worker.pause()
            self["Pause"].text = "Resume"
            self._last_info = self.info.value
            self.info.value = "Pausing"
        self.paused = not self.paused
        
    def Interrupt(self):
        """
        Interrupt thread.
        """        
        self.worker.quit()
    
@magicclass
class MTProfiler:
    
    _loader = field(ImageLoader)
    _worker_control = field(WorkerControl)
    
    @magicclass(layout="horizontal", labels=False)
    class operation:
        def register_path(self): ...
        def run_for_all_path(self): ...
        def clear_current(self): ...
        def clear_all(self): ...
        def info(self): ...
    
    @magicclass(layout="horizontal", labels=False)
    class io:
        def open_image_file(self, path: Path): ...
        def from_json(self, path: Path): ...
        def save_results(self, path: Path): ...
    
    tomograms = field(ListWidget, options={"name": "Tomograms"})
    
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
        txt = field(str, options={"enabled": False}, name="result")
        
    line_edit = field(str, name="Note: ")
    
    plot = field(Figure, name="Plot", options={"figsize":(4.2, 1.8), "tooltip": "Plot of pitch lengths"})
        
    POST_PROCESSING = [io.save_results, viewer_op.send_to_napari,
                       viewer_op.show_table, viewer_op.show_3d_path]

    POST_IMREAD = [io.from_json, operation.register_path]
    
    sep1 = field(Separator)
    
    @magicclass(layout="horizontal")
    class auto_picker:
        stride = field(50.0, widget_type="FloatSlider", options={"min": 10, "max": 100}, name="stride (nm)")
        def pick_next(self): ...
        def auto_center(self): ...
        
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
        
    @button_design(text="Create Macro")
    def create_macro_(self):
        self.create_macro(show=True)
        return None
    
    @property
    def active_binsize(self):
        bin_scale = self.layer_image.scale[0] # scale of binned reference image
        tomo = self.active_tomogram
        binsize = int(bin_scale/tomo.scale)
        return binsize

    def _update_colormap(self, prop: str = H.yPitch):
        # TODO: color by other properties
        if self.layer_paint is None:
            return None
        color = self.layer_paint.color
        lim0, lim1 = self.label_colorlimit
        df = self.active_tomogram.collect_localprops()[prop]
        for i, value in enumerate(df):
            color[i+1] = self.label_colormap.map((value - lim0)/(lim1 - lim0))
        self.layer_paint.color = color
        return None
        
    def __post_init__(self):
        self._mtpath = None
        self.active_tomogram: MtTomogram = None
        self.layer_image: Image = None
        self.layer_prof: Points = None
        self.layer_work: Points = None
        self.layer_paint: Labels = None
        
        self.set_colormap()
        self.mt.pos.min_width = 70
        call_button = self._loader["call_button"]
        call_button.changed.connect(self.load_image)
        
        @self.tomograms.register_callback(MtTomogram)
        def open_tomogram(tomo: MtTomogram, i: int):
            if tomo is self.active_tomogram:
                return None
            self.active_tomogram = tomo
            
            self._bin_image(tomo.image, tomo.metadata["binsize"], 
                            tomo.light_background, new=False)
            if tomo.paths:
                self._load_tomogram_results()
            else:
                self._init_layers()
                self._init_widget_params()
        
        @self.tomograms.register_contextmenu(MtTomogram)
        def Load_tomogram(tomo: MtTomogram, i: int):
            open_tomogram(tomo, i)
        
        @self.tomograms.register_contextmenu(MtTomogram)
        def Remove_tomogram_from_list(tomo: MtTomogram, i: int):
            self.tomograms.pop_item(i)
            
        self.tomograms.height = 120
        self.tomograms.max_height = 120
             
    @operation.wraps
    @click(enabled=False, enables=operation.run_for_all_path)
    @button_design(text="📝")
    def register_path(self):
        """
        Register current selected points as a MT path.
        """        
        coords = self.layer_work.data
        if coords.size == 0:
            return None
        tomo = self.active_tomogram
        self.active_tomogram.add_path(coords)
        spl = self.active_tomogram.paths[-1]
        
        # check/draw path
        interval = 30
        length = spl.length()
        if length < 50.0:
            raise ValueError(f"Path is too short: {length:.2f} nm")
        
        n = int(length/interval) + 1
        fit = spl(np.linspace(0, 1, n))
        self.layer_prof.add(fit)
        self.canvas.ax.plot(fit[:,2], fit[:,1], color="gray", lw=2.5)
        self.canvas.ax.set_xlim(0, tomo.image.shape.x*tomo.image.scale.x)
        self.canvas.ax.set_ylim(tomo.image.shape.y*tomo.image.scale.y, 0)
        self.canvas.ax.set_aspect("equal")
        self.canvas.ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        self.canvas.figure.tight_layout()
        self.canvas.figure.canvas.draw()
        self.layer_work.data = []
        return None
    
    @operation.wraps
    @click(enabled=False, enables=POST_PROCESSING)
    @set_options(interval={"min":1.0, "max": 100.0, "label": "Interval (nm)"},
                 box_radius_pre={"widget_type": TupleEdit, "label": "Initial box radius (nm)"}, 
                 box_radius={"widget_type": TupleEdit, "label": "Final box radius (nm)"},
                 upsample_factor={"min":12, "max":50, "label": "Up-sampling factor"})
    @button_design(text="👉")
    def run_for_all_path(self, 
                         interval: nm = 24.0,
                         box_radius_pre: tuple[nm, nm, nm] = (22.0, 28.0, 28.0),
                         box_radius: tuple[nm, nm, nm] = (16.7, 16.7, 16.7),
                         upsample_factor: int = 20):
        """
        Run MTProps.

        Parameters
        ----------
        interval : nm, default is 24.0
            Interval of sampling points of microtubule fragments.
        box_radius_pre : tuple[nm, nm, nm], default is (22.0, 28.0, 28.0)
            Box size of microtubule fragments used for angle correction and centering.
        box_radius : tuple[nm, nm, nm], default is (16.7, 16.7, 16.7)
            Box size of MT fragments used for final analysis.
        upsample_factor : int, default is 20
            Up-sampling factor of Fourier transformation.
        """        
        if self.layer_work.data.size > 0:
            self.register_path()
        
        worker = create_worker(self._run_all, 
                               interval=interval,
                               box_radius_pre=box_radius_pre,
                               box_radius=box_radius,
                               upsample_factor=upsample_factor,
                               _progress={"total": self.active_tomogram.n_paths*3 + 1, 
                                          "desc": "Running MTProps"}
                               )
        
        self._connect_worker(worker)
        @worker.yielded.connect
        def _on_yield(out):
            if isinstance(out, str):
                self._worker_control.info.value = out
            
        @worker.returned.connect
        def _on_return(out: MtTomogram):
            self._load_tomogram_results()
        
        self._worker_control.info.value = f"Spline fitting (0/{self.active_tomogram.n_paths})"
        worker.start()
        return None
    
    def _run_all(self, 
                 interval: nm = 24.0,
                 box_radius_pre: tuple[nm, nm, nm] = (22.0, 28.0, 28.0),
                 box_radius: tuple[nm, nm, nm] = (16.7, 16.7, 16.7),
                 upsample_factor: int = 20):
        tomo = self.active_tomogram
        tomo.box_radius_pre = box_radius_pre
        tomo.box_radius = box_radius
        for i in range(tomo.n_paths):
            tomo.fit(i)
            tomo.make_anchors(interval=interval)
            
            yield f"Reloading subtomograms  ({i}/{tomo.n_paths})"
            tomo.get_subtomograms(i)
            
            yield f"MT analysis ({i}/{tomo.n_paths}) "
            tomo.measure_radius(i)
            tomo.calc_ft_params(i, upsample_factor=upsample_factor)
            
            yield f"Spline fitting ({i+1}/{tomo.n_paths})"
        
        return tomo
    
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
        Clear all the paths and heatmaps.
        """        
        self._init_layers()
        self.canvas.figure.clf()
        self.canvas.figure.add_subplot(111)
        self.canvas.draw()
        self.canvas.ax.set_aspect("equal")
        self.canvas.ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        self.plot.figure.clf()
        self.plot.figure.add_subplot(111)
        self.plot.draw()
        
        cachemap.clear()
        self.active_tomogram._paths.clear()
        
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
        return None
    
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
    @set_options(path={"filter": "*.json;*.txt"})
    @button_design(text="Load json 📂")
    def from_json(self, path: Path):
        """
        Choose a json file and load it.
        """        
        tomo = self.active_tomogram
        tomo.load(path)
        tomo.get_subtomograms()
        self._load_tomogram_results()
        return None
    
    @io.wraps
    @button_design(text="Save 💾")
    @click(enabled=False)
    @set_options(path={"mode": "w"})
    def save_results(self, file_path: Path, contain_results: bool = True):
        """
        Save the results as json.
        
        Parameters
        ----------
        file_path: Path
            File path to save splines.
        contain_results: bool
            Check and local MT properties will also saved in the same json file.
        """        
        self.active_tomogram.save(file_path, contain_results=contain_results)
        return None            
    
    @viewer_op.wraps
    @click(enabled=False)
    @button_design(text="👁️")    
    def send_to_napari(self):
        """
        Send the current MT fragment 3D image (not binned) to napari viewer.
        """        
        i = self.mt.pos.value
        tomo = self.active_tomogram
        img = tomo._sample_subtomograms(i)
        self.parent_viewer.add_image(img, scale=img.scale, name=img.name,
                                     rendering="minip" if tomo.light_background else "mip")
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
        
        viewer: napari.Viewer = self.parent_viewer
        i = self.mt.mtlabel.value
        j = self.mt.pos.value
        
        tomo = self.active_tomogram
        spl = tomo.paths[i]
        pos = spl.anchors[j]
        next_center = spl(pos)
        viewer.dims.current_step = list(next_center.astype(np.int64))
        
        viewer.camera.center = next_center
        zoom = viewer.camera.zoom
        viewer.camera.events.zoom() # Here events are emitted and zoom changes automatically.
        viewer.camera.zoom = zoom
        
        self.layer_paint.show_selected_label = True
        
        j_offset = sum(spl.anchors.size for spl in tomo.path[:i])
        self.layer_paint.selected_label = j_offset + j + 1
        return None
    
    @viewer_op.wraps
    @click(enabled=False)
    @button_design(text="📜")
    def show_table(self):
        """
        Show result table.
        """        
        table = Table(value=self.active_tomogram.collect_localprops())
        self.parent_viewer.window.add_dock_widget(table)
        return None
    
    @viewer_op.wraps
    @click(enabled=False)
    @button_design(text="📈")
    def show_3d_path(self):
        """
        Show 3D paths of microtubule center axes.
        """        
        paths = [r.partition(100) for r in self.active_tomogram.paths]
        
        self.parent_viewer.add_shapes(paths, shape_type="path", edge_color="lime", edge_width=1,
                                      translate=self.layer_image.translate)
        return None
    
    def _paint_mt(self):
        """
        Paint microtubule fragments by its pitch length.
        
        1. Prepare small boxes and make masks inside them.
        2. Map the masks to the reference image.
        3. Erase masks using reference image, based on intensity.
        """        
        lbl = ip.zeros(self.layer_image.data.shape, dtype=np.uint8)
        color: dict[int, list[float]] = {0: [0, 0, 0, 0]}
        bin_scale = self.layer_image.scale[0] # scale of binned reference image
        tomo = self.active_tomogram
        
        lz, ly, lx = [int(r/bin_scale*1.4)*2 + 1 for r in tomo.box_radius]
        binsize = self.active_binsize
        with ip.SetConst("SHOW_PROGRESS", False):
            center = np.array([lz, ly, lx])/2 + 0.5
            z, y, x = np.indices((lz, ly, lx))
            cylinders = []
            matrices = []
            for i, spl in enumerate(tomo.paths):
                # Prepare template hollow image
                r0 = spl.radius/tomo.scale*0.9/binsize
                r1 = spl.radius/tomo.scale*1.1/binsize
                _sq = (z - lz//2)**2 + (x - lx//2)**2
                domains = []
                dist = [-np.inf] + list(spl.distances()) + [np.inf]
                for j in range(spl.anchors.size):
                    domain = (r0**2 < _sq) & (_sq < r1**2)
                    ry = min((dist[j+1] - dist[j]) / 2, 
                             (dist[j+2] - dist[j+1]) / 2, 
                              tomo.box_radius[1]) / bin_scale + 0.5 
                        
                    ry = max(int(np.ceil(ry)), 1)
                    domain[:, :ly//2-ry] = 0
                    domain[:, ly//2+ry+1:] = 0
                    domain = domain.astype(np.float32)
                    domains.append(domain)
                    
                cylinders.append(domains)
                matrices.append(spl.rotation_matrix(center=center))
            
            cylinders = np.concatenate(cylinders, axis=0)
            matrices = np.concatenate(matrices, axis=0)
            out = dask_affine(cylinders, matrices) > 0.3
            
        # paint roughly
        for i, crd in enumerate(tomo.collect_anchor_coords()):
            center = tomo.nm2pixel(crd)//binsize
            sl = []
            outsl = []
            # We should deal with the borders of image.
            for c, l, size in zip(center, [lz, ly, lx], lbl.shape):
                _sl, _pad = make_slice_and_pad(c, l//2, size)
                sl.append(_sl)
                outsl.append(
                    slice(_pad[0] if _pad[0] > 0 else None,
                         -_pad[1] if _pad[1] > 0 else None)
                )

            sl = tuple(sl)
            outsl = tuple(outsl)
            lbl.value[sl][out[i][outsl]] = i + 1
        
        # paint finely
        ref = self.layer_image.data
        
        if tomo.light_background:
            thr = np.percentile(ref[lbl>0], 95)
            lbl[ref>thr] = 0
        else:
            thr = np.percentile(ref[lbl>0], 5)
            lbl[ref<thr] = 0
        
        # Labels layer properties
        _id = "ID"
        _type = "type"
        columns = [_id, H.riseAngle, H.yPitch, H.skewAngle, _type]
        df = tomo.collect_localprops()[[H.riseAngle, H.yPitch, H.skewAngle, H.nPF, H.start]]
        df_reset = df.reset_index()
        df_reset[_id] = df_reset.apply(lambda x: "{}-{}".format(int(x["level_0"]), int(x["level_1"])), axis=1)
        df_reset[_type] = df_reset.apply(lambda x: "{}_{}".format(int(x[H.nPF]), int(x[H.start])), axis=1)
        
        back = pd.DataFrame({c: [np.nan] for c in columns})
        props = pd.concat([back, df_reset[columns]])
        
        # Add labels layer
        if self.layer_paint is None:
            self.layer_paint = self.parent_viewer.add_labels(
                lbl.value, color=color, scale=self.layer_image.scale,
                translate=self.layer_image.translate, opacity=0.33, name="Label",
                properties=props
                )
        else:
            self.layer_paint.data = lbl.value
            self.layer_paint.properties = props
        self._update_colormap()
        return None
        
            
    def _plot_pitch(self):
        i = self.mt.mtlabel.value
        props = self.active_tomogram.paths[i].localprops
        x = props[H.splDistance]
        with plt.style.context("dark_background"):
            self.plot.ax.cla()
            self.plot.ax.plot(x, props[H.yPitch], color="white")
            self.plot.ax.set_xlabel("position (nm)")
            self.plot.ax.set_ylabel("pitch (nm)")
            self.plot.ax.set_ylim(*self.label_colorlimit)
            self.plot.figure.tight_layout()
            self.plot.draw()
        
        return None
    
    @click(disables=POST_PROCESSING, enables=POST_IMREAD, visible=False)
    def load_image(self, event=None):
        img = self._loader.img
        light_bg = self._loader.light_background.value
        binsize = self._loader.bin_size.value
        
        self._bin_image(img, binsize, light_bg)
        self._loader.close()
        return None
    
    def _bin_image(self, img, binsize:int, light_bg:bool, new:bool=True):
        viewer: napari.Viewer = self.parent_viewer
        worker = bin_image_worker(img, binsize)
        self._connect_worker(worker)
        self._worker_control.info.value = \
            f"original image: {img.shape}, " \
            f"binned image: {tuple(s//binsize for s in img.shape)}"
        
        @worker.returned.connect
        def _on_return(imgb):
            tr = (binsize - 1)/2*img.scale.x
            rendering = "minip" if light_bg else "mip"
            if self.layer_image not in viewer.layers:
                self.layer_image = viewer.add_image(
                    imgb, 
                    scale=imgb.scale, 
                    name=imgb.name, 
                    translate=[tr, tr, tr],
                    contrast_limits=[np.min(imgb), np.max(imgb)],
                    rendering=rendering
                    )
            else:
                self.layer_image.data = imgb
                self.layer_image.scale = imgb.scale
                self.layer_image.name = imgb.name
                self.layer_image.translate = [tr, tr, tr]
                self.layer_image.contrast_limits = [np.min(imgb), np.max(imgb)]
                self.layer_image.rendering = rendering
                
            viewer.scale_bar.unit = img.scale_unit
            viewer.dims.axis_labels = ("z", "y", "x")
            
            if new:
                self._init_widget_params()
                
                tomo = MtTomogram(light_background=light_bg, name=img.name)
                
                tomo.metadata["source"] = str(self._loader.path.value)
                tomo.metadata["binsize"] = binsize
                
                self.active_tomogram = tomo
                tomo.image = img
                self.tomograms.add_item(tomo)
                
                self.clear_all()
            
            return None
        
        worker.start()
        return None
    
    def _load_tomogram_results(self):
        tomo = self.active_tomogram
        self._active_rot_ave = tomo.rotational_average()
        self._init_layers()
                
        # Paint MTs by its pitch length
        self._paint_mt()
        
        self.layer_work.mode = "pan_zoom"
        self.parent_viewer.layers.selection = {self.layer_paint}
        
        # initialize GUI
        self._init_widget_params()
        self.mt.mtlabel.max = tomo.n_paths - 1
        self.mt.pos.max = len(tomo.paths[0].localprops[H.splDistance]) - 1
        
        self.canvas.figure.clf()
        self.canvas.figure.add_subplot(131)
        self.canvas.figure.add_subplot(132)
        self.canvas.figure.add_subplot(133)
        self._imshow_all()
        
        self._plot_pitch()
        
        return None
    
    def _init_widget_params(self):
        self.mt.mtlabel.value = 0
        self.mt.mtlabel.min = 0
        self.mt.mtlabel.max = 0
        self.mt.pos.value = 0
        self.mt.pos.min = 0
        self.mt.pos.max = 0
        self.viewer_op.txt.value = ""
        return None
    
    @auto_picker.wraps
    def pick_next(self):
        """
        Automatically pick MT center using previous two points.
        """        
        stride_nm = self.auto_picker.stride.value
        imgb = self.layer_image.data
        try:
            # orientation is point0 -> point1
            point0: np.ndarray = self.layer_work.data[-2]/imgb.scale.x # unit: pixel
            point1: np.ndarray = self.layer_work.data[-1]/imgb.scale.x
        except IndexError:
            raise IndexError("Auto pick needs at least two points in the working layer.")
        
        tomo = self.active_tomogram
        binsize = int(self.layer_image.scale[0]/tomo.scale) # scale of binned reference image
        
        radius = tomo.nm2pixel(np.array(tomo.box_radius_pre)/binsize)
        with ip.SetConst("SHOW_PROGRESS", False):
            orientation = point1[1:] - point0[1:]
            img = load_a_subtomogram(imgb, point1.astype(np.uint16), radius, dask=False)
            center = np.rad2deg(np.arctan2(*orientation)) % 180 - 90
            angle_deg = angle_corr(img, ang_center=center, drot=25, nrots=25)
            angle_rad = np.deg2rad(angle_deg)
            dr = np.array([0.0, stride_nm*np.cos(angle_rad), -stride_nm*np.sin(angle_rad)])
            if np.dot(orientation, dr[1:]) > np.dot(orientation, -dr[1:]):
                point2 = point1 + dr
            else:
                point2 = point1 - dr
            
            centering(imgb, point2, angle_deg, radius)
            
        next_data = point2 * imgb.scale.x
        msg = self._check_path()
        if msg:
            self.layer_work.data = self.layer_work.data[:-1]
            raise ValueError(msg)
        self.layer_work.add(next_data)
        change_viewer_focus(self.parent_viewer, point2, next_data)
        return None
    
    @auto_picker.wraps
    def auto_center(self):
        """
        Auto centering of selected points.
        """        
        imgb = self.layer_image.data
        tomo = self.active_tomogram
        binsize = int(self.layer_image.scale[0]/tomo.scale) # scale of binned reference image
        selected = self.layer_work.selected_data
        radius = tomo.nm2pixel(np.array(tomo.box_radius_pre)/binsize)
        
        points = self.layer_work.data / imgb.scale.x
        last_i = -1
        with ip.SetConst("SHOW_PROGRESS", False):
            for i, point in enumerate(points):
                if i not in selected:
                    continue
                angle_deg = angle_corr(imgb, ang_center=0, drot=90, nrots=17)
                centering(imgb, point, angle_deg, radius)
                last_i = i
        
        self.layer_work.data = points * imgb.scale.x
        if len(selected) == 1:
            change_viewer_focus(self.parent_viewer, points[last_i], self.layer_work.data[last_i])
        return None
    
    def _check_path(self) -> str:
        tomo = self.active_tomogram
        imgshape_nm = np.array(tomo.image.shape) * tomo.image.scale.x
        if self.layer_work.data.shape[0] == 0:
            return ""
        else:
            point0 = self.layer_work.data[-1]
            if not np.all([r*0.5 <= p < s - r*0.5
                           for p, s, r in zip(point0, imgshape_nm, tomo.box_radius_pre)]):
                # outside image
                return "Outside boundary."
            elif self.layer_work.data.shape[0] >= 3:
                point2, point1, point0 = self.layer_work.data[-3:]
                vec2 = point2 - point1
                vec0 = point0 - point1
                len0 = np.sqrt(vec0.dot(vec0))
                len2 = np.sqrt(vec2.dot(vec2))
                cos1 = vec0.dot(vec2)/(len0*len2)
                curvature = 2 * np.sqrt((1 - cos1**2) / sum((point2 - point0)**2))
                if curvature > 0.02:
                    # curvature is too large
                    return f"Curvature {curvature} is too large for a MT."
        
        return ""
    
    def _init_layers(self):
        # TODO: simpler implementation after napari==0.4.12
        viewer: napari.Viewer = self.parent_viewer
        
        common_properties = dict(ndim=3, n_dimensional=True, size=8)
        if self.layer_prof in self.parent_viewer.layers:
            self.layer_prof.name = "MT Profiles-old"
    
        self.layer_prof = viewer.add_points(**common_properties,
                                    name="MT Profiles",
                                    opacity=0.4, 
                                    edge_color="black",
                                    face_color="black",
                                    )
        self.layer_prof.editable = False
            
        if self.layer_work in viewer.layers:
            self.layer_work.name = "Working Layer-old"
        
        self.layer_work = viewer.add_points(**common_properties,
                                    name="Working Layer",
                                    face_color="yellow"
                                    )
    
        self.layer_work.mode = "add"
        
        if "MT Profiles-old" in viewer.layers:
            viewer.layers.remove("MT Profiles-old")
        if "Working Layer-old" in viewer.layers:
            viewer.layers.remove("Working Layer-old")
        
        if self.layer_paint is not None:
            self.layer_paint.data = np.zeros_like(self.layer_paint.data)
            
        return None
    
    @mt.pos.connect
    def _imshow_all(self, event=None):
        tomo = self.active_tomogram
        i = self.mt.mtlabel.value
        j = self.mt.pos.value
        results = tomo.paths[i]
        pitch, skew, npf, start = results.localprops[[H.yPitch, H.skewAngle, H.nPF, H.start]].iloc[j]
        self.viewer_op.txt.value = f"{pitch:.2f} nm / {skew:.2f}°/ {int(npf)}_{int(start)}"
        
        axes: Iterable[Axes] = self.canvas.axes
        for k in range(3):
            axes[k].cla()
            axes[k].tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
            
        subtomo = tomo._sample_subtomograms(i)[j]
        lz, ly, lx = subtomo.shape
        with ip.SetConst("SHOW_PROGRESS", False):
            axes[0].imshow(subtomo.proj("z"), cmap="gray")
            axes[0].set_xlabel("x")
            axes[0].set_ylabel("y")
            axes[1].imshow(subtomo.proj("y"), cmap="gray")
            axes[1].set_xlabel("x")
            axes[1].set_ylabel("z")
            axes[2].imshow(self._active_rot_ave[i][j], cmap="gray")
            axes[2].set_xlabel("x")
            axes[2].set_ylabel("z")
        
        ylen = tomo.nm2pixel(tomo.box_radius[1])
        ymin, ymax = ly/2 - ylen, ly/2 + ylen
        r = tomo.nm2pixel(results.radius)*OUTER
        xmin, xmax = -r + lx/2, r + lx/2
        axes[0].plot([xmin, xmin, xmax, xmax, xmin], [ymin, ymax, ymax, ymin, ymin], color="lime")
        axes[0].text(1, 1, f"{i}-{j}", 
                                 color="lime", font="Consolas", size=15, va="top")
    
        theta = np.linspace(0, 2*np.pi, 360)
        r = tomo.nm2pixel(results.radius) * INNER
        axes[1].plot(r*np.cos(theta) + lx/2, r*np.sin(theta) + lz/2, color="lime")
        r = tomo.nm2pixel(results.radius) * OUTER
        axes[1].plot(r*np.cos(theta) + lx/2, r*np.sin(theta) + lz/2, color="lime")
                
        self.canvas.figure.tight_layout()
        self.canvas.draw()
    
    @line_edit.connect
    def _update_note(self, event=None):
        i = self.mt.mtlabel.value
        self.active_tomogram.paths[i].orientation = self.line_edit.value
        return None
    
    @mt.mtlabel.connect
    def _update_mtpath(self, event=None):
        i = self.mt.mtlabel.value
        tomo = self.active_tomogram
        
        self.mt.pos.max = len(tomo.paths[i].localprops) - 1
        note = tomo.paths[i].orientation
        self.line_edit.value = note
        self._plot_pitch()
        self._imshow_all()
        return None
    
    def _connect_worker(self, worker):
        self._worker_control._set_worker(worker)
        viewer: napari.Viewer = self.parent_viewer
        viewer.window._status_bar._toggle_activity_dock(True)
        dialog = viewer.window._qt_window._activity_dialog
        
        @worker.finished.connect
        def _on_finish():
            viewer.window._status_bar._toggle_activity_dock(False)
            dialog.layout().removeWidget(self._worker_control.native)
            
        dialog.layout().addWidget(self._worker_control.native)
        return None

def centering(imgb, point, angle, radius):
    img_next = load_a_subtomogram(imgb, point.astype(np.uint16), radius, dask=False)
            
    angle_deg2 = angle_corr(img_next, ang_center=angle, drot=5, nrots=7)
    
    img_next_rot = img_next.rotate(-angle_deg2, cval=np.median(img_next))
    proj = img_next_rot.proj("y")
    proj_mirror = proj["z=::-1;x=::-1"]
    shift = ip.pcc_maximum(proj, proj_mirror)
    
    shiftz, shiftx = shift/2
    shift = np.array([shiftz, 0, shiftx])
    rad = -np.deg2rad(angle_deg2)
    cos = np.cos(rad)
    sin = np.sin(rad)
    shift = shift @ [[1.,   0.,  0.],
                     [0.,  cos, sin],
                     [0., -sin, cos]]
    point += shift

def change_viewer_focus(viewer, next_center, next_coord):
    viewer.camera.center = next_center
    zoom = viewer.camera.zoom
    viewer.camera.events.zoom()
    viewer.camera.zoom = zoom
    viewer.dims.current_step = list(next_coord.astype(np.int64))