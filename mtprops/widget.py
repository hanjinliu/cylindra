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
                            Figure, TupleEdit, CheckButton, Separator)
# from .mtpath import (MTPath, angle_corr, calc_total_length, load_a_subtomogram, vector_to_grad, 
#                      rot3d, rot3dinv, make_slice_and_pad)
from .tomogram import MtTomogram, cachemap, angle_corr
from .utils import load_a_subtomogram, make_slice_and_pad
from .const import nm, H, INNER, OUTER

# TODO: QListView

if TYPE_CHECKING:
    from napari.layers import Image, Points, Labels
    from napari._qt.qthreading import GeneratorWorker
    from matplotlib.axes import Axes


@thread_worker(progress={"total": 0, "desc": "Reading Image"})
def imread(img, binsize):
    with ip.SetConst("SHOW_PROGRESS", False):
        imgb = img.binning(binsize, check_edges=False).data
    
    return imgb
        

@magicclass
class ImageLoader:
    path = field(Path)
    scale = field(str, options={"label": "scale (nm)"})
    light_background = field(True)
    
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
        def settings(self): ...
        def clear_current(self): ...
        def clear_all(self): ...
        def info(self): ...
    
    @magicclass(layout="horizontal", labels=False)
    class io:
        def open_image_file(self, path: Path): ...
        def from_csv_file(self, path: Path): ...
        def save_results(self, path: Path): ...
    
    sep0 = field(Separator)
    
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
        
    POST_PROCESSING = [io.save_results, viewer_op.send_to_napari,
                       viewer_op.show_table, viewer_op.show_3d_path]

    POST_IMREAD = [io.from_csv_file, operation.register_path]
    
    sep1 = field(Separator)
    
    @magicclass(layout="horizontal")
    class auto_picker:
        stride = field(50.0, options={"min": 10, "max": 100}, name="stride (nm)")
        def pick(self): ...
        
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
    
    # @set_options(niter={"min": 1, "max": 5, "label": "Number of iterations"},
    #              nshifts={"min": 3, "max": 31, "label": "Number of shifts"},
    #              nrots={"min": 3, "max": 25, "label": "Number of rotations"})
    # def subtomogram_averaging(self, niter:int=3, nshifts:int=7, nrots:int=5):
    #     """
    #     Average subtomograms along a MT.

    #     Parameters
    #     ----------
    #     niter : int, default is 2
    #         Number of iterations. MT template will be updated using the previous iteration.
    #     nshifts : int, default is 19
    #         Number of shifts along Y-axis to apply.
    #     nrots : int, default is 5
    #         Number of rotation angles in around Y-axis to apply.
    #     """
    #     viewer: napari.Viewer = self.parent_viewer
        
    #     worker = create_worker(self.current_mt.average_subtomograms, 
    #                            niter, nshifts, nrots,
    #                            _progress={"total": niter, 
    #                                       "desc": "Subtomogram averaging"}
    #                            )
        
    #     self._connect_worker(worker)
    #     self._worker_control.metadata["iter"] = 1
    #     self._worker_control.info.value = "Iteration 1"
        
    #     @worker.yielded.connect
    #     def _on_yield(out):
    #         df, avg_img = out
            
    #         if self.light_background:
    #             avg_img = -avg_img
            
    #         avg_image_name = "AVG"
    #         try:
    #             viewer.layers[avg_image_name].data = avg_img
    #         except KeyError:
    #             layer = viewer.add_image(avg_img, scale=avg_img.scale, name=avg_image_name,
    #                                      rendering="iso", iso_threshold = 0.6)
            
    #         it = self._worker_control.metadata["iter"] + 1
    #         self._worker_control.info.value = f"Iteration {it}"
    #         self._worker_control.metadata["iter"] = it
    #         viewer.camera.center = np.array(avg_img.shape)/2*avg_img.scale
        
    #     @worker.returned.connect
    #     def _on_return(out):
    #         df, avg_img = out
    #         table = Table(value=df)
    #         viewer.window.add_dock_widget(table, name="Results")
                
    #     worker.start()
    #     return None
    
    # @set_options(binsize={"min":1, "max":8, "label": "Binning for peak detection"},
    #              niter={"min":1, "max":5, "label": "Number of PCC iteration"},
    #              remap={"label": "Remap coordinates"})
    # def tubulin_averaging(self, binsize:int=2, niter:int=3, remap:bool=True):
    #     """
    #     Run tubulin averagin algorithm along current MT path.

    #     Parameters
    #     ----------
    #     binsize : int, default is 2
    #         Bin size applied to averaged subtomogram to speed up local peak detection.
    #     niter : int, default is 3
    #         Number of iteration of refining tubulin template.
    #     remap : bool, default is True
    #         If true, tubulin coordinates will remapped to world coordinate system.
    #     """        
    #     mtp = self.current_mt
    #     viewer: napari.Viewer = self.parent_viewer

    #     coords = mtp.find_tubulin(binsize)
    #     template = mtp.crop_out_tubulin(coords, niter=niter)
    #     if self.light_background:
    #         template = -template
    #     viewer.add_points(coords, face_color="gold", size=2, edge_color="white", 
    #                       name="Tubulin on template")
    #     viewer.add_image(template, scale=template.scale, name="Tubulin template",
    #                      rendering="iso", iso_threshold=0.6)
        
    #     if remap:
    #         coords_remapped = np.concatenate(
    #             list(mtp.transform_coordinates()),
    #             axis=0
    #             )
    #         viewer.add_points(coords_remapped, size=2, face_color="gold", edge_color="white", 
    #                           name="Remapped tubulins")
    #     return None
    
    # @set_options(position={"min":0.0, "max":1.0, "step":0.05})
    # def pick_subtomogram(self, position:float=0.5):
    #     scale = self.image.scale.x
    #     with ip.SetConst("SHOW_PROGRESS", False):
    #         mtp = self.current_mt
    #         tomo = load_a_subtomogram(self.image, mtp.spl(position)/scale, self.box_radius_pre)
    #         dr = mtp.spl(position, der=1).reshape(1, -1)
    #         zy, yx = vector_to_grad(dr)
    #         tomo = rot3d(tomo, yx[0], zy[0])
    #         self.parent_viewer.add_image(tomo, scale=scale)
    
    @button_design(text="Create Macro")
    def create_macro_(self):
        self.create_macro(show=True)
        return None

    def _update_colormap(self): # TODO
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
        self.tomograms: list[MtTomogram] = []
        self.active_tomogram: MtTomogram = None
        self.layer_image: Image = None
        self.layer_prof: Points = None
        self.layer_work: Points = None
        self.layer_paint: Labels = None
        
        self.settings()
        self.set_colormap()
        self.mt.pos.min_width = 70
        call_button = self._loader["call_button"]
        call_button.changed.connect(self.load_image)
             
    @operation.wraps
    @click(enabled=False, enables=operation.run_for_all_path)
    @button_design(text="📝")
    def register_path(self):
        """
        Register current selected points as a MT path.
        """        
        coords = self.layer_work.data
        tomo = self.active_tomogram
        self.active_tomogram.add_path(coords)
        spl = self.active_tomogram.paths[-1]
        if spl.length() < 72.0:
            raise ValueError(f"Path is too short: {spl.length():.2f} nm")
        
        fit = spl.partition(100)
        self.layer_prof.add(fit)
        self.canvas.ax.plot(fit[:,2], fit[:,1], color="gray", lw=2.5)
        self.canvas.ax.set_xlim(0, tomo.image.sizeof("x")*tomo.image.scale.x)
        self.canvas.ax.set_ylim(tomo.image.sizeof("y")*tomo.image.scale.y, 0)
        self.canvas.ax.set_aspect("equal")
        self.canvas.ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        self.canvas.figure.tight_layout()
        self.canvas.figure.canvas.draw()
        self.layer_work.data = []
        return None
    
    @operation.wraps
    @click(enabled=False, enables=POST_PROCESSING)
    @button_design(text="👉")
    def run_for_all_path(self): # TODO
        """
        Run MTProps.
        """        
        if self.layer_work.data.size > 0:
            self.register_path()
        
        worker = create_worker(self._run_all, 
                               _progress={"total": len(self.mt_paths)*13, 
                                          "desc": "Running MTProps"}
                               )
        
        self._connect_worker(worker)
        @worker.yielded.connect
        def _on_yield(out):
            if isinstance(out, str):
                self._worker_control.info.value = out
            
        @worker.returned.connect
        def _on_return(out: MtTomogram):
            self._from_dataframe(pd.concat(out, axis=0))
        
        worker.start()
        return None
    
    def _run_all(self): # TODO
        tomo = self.active_tomogram
        for i in range(tomo.n_paths):
            prog = f"{i}/{tomo.n_paths}"
            yield f"{prog} Spline fitting ..."
            tomo.fit(i)
            length = tomo.paths[i].length()
            nseg = int(length//self.interval)
            position = tuple(np.linspace(0, nseg/length, nseg+1))
            yield f"{prog} Reloading subtomograms ..."
            tomo.get_subtomograms(i, position)
            yield f"{prog} MT analysis ..."
            tomo.calc_ft_params(i, position)
            
            yield
        
        return tomo
    
    @operation.wraps
    @set_options(box_radius_pre={"widget_type": TupleEdit, "label": "z/y/x-radius-pre (nm)"}, 
                 radius_nm={"widget_type": TupleEdit, "label": "z/y/x-radius (nm)"},
                 upsample_factor={"min":12, "max":50},
                 bin_size={"min": 1, "max": 8})
    @button_design(text="⚙")
    def settings(self,
                 interval_nm: float = 24.0, 
                 box_radius_pre: tuple[float, float, float] = (22.0, 28.0, 28.0), 
                 box_radius: tuple[float, float, float] = (16.7, 16.7, 16.7),
                 upsample_factor: int = 20,
                 bin_size: int = 4):
        """
        Change MTProps setting.
        Parameters
        ----------
        interval_nm : float, optional
            Interval between points to analyze.
        light_background : bool, optional
            Light background or not
        box_radius_pre : tuple[float, float, float]
            Images in this range will be considered to determine MT tilt and shift.
        radius_nm : tuple[float, float, float]
            Images in this range will be considered to determine MT pitch length and PF number.
        upsample_factor : int
            Up-sampling factor for pitch length calculation.
        bin_size : int
            Binning (pixel) that will be applied to the image in the viewer. This parameter does
            not affect the results of analysis.
        """        
        self.interval = interval_nm
        self.box_radius = box_radius
        self.box_radius_pre = box_radius_pre
        self.upsample_factor = upsample_factor
        self.binsize = bin_size
    
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
        self.canvas.figure.clf()
        self.canvas.figure.add_subplot(111)
        self.canvas.draw()
        self.canvas.ax.set_aspect("equal")
        self.canvas.ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        self.plot.figure.clf()
        self.plot.figure.add_subplot(111)
        self.plot.draw()
        
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
        return None
    
    @io.wraps
    @button_design(text="Open image 🔬")
    def open_image_file(self):
        """
        Open an image and add to viewer.
        """
        self._loader.show()
        return None
        
    # @io.wraps
    # @click(enabled=False, enables=POST_PROCESSING)
    # @set_options(path={"filter": "*.tif;*.mrc;*.rec"})
    # @button_design(text="Load csv 📂")
    # def from_csv_file(self, path: Path):
    #     """
    #     Choose a csv file and load it.
    #     """        
    #     self.parent_viewer.window._status_bar._toggle_activity_dock(True)
    #     with progress(total=0) as pbr:
    #         pbr.set_description("Loading csv")
    #         df = pd.read_csv(path)
    #         self._from_dataframe(df)
    #     self.parent_viewer.window._status_bar._toggle_activity_dock(False)
    #     return None
    
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
    def send_to_napari(self): # TODO
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
        
        viewer = self.parent_viewer
        i = self.mt.pos.value
        scale = viewer.layers["MT Profiles"].scale
        next_center = self.current_mt.points[i]
        viewer.dims.current_step = list(next_center.astype(np.int64))
        
        viewer.camera.center = next_center
        zoom = viewer.camera.zoom
        viewer.camera.events.zoom() # Here events are emitted and zoom changes automatically.
        viewer.camera.zoom = zoom
        
        self.layer_paint.show_selected_label = True
        for k, (_, row) in enumerate(self.dataframe.iterrows()):
            if row[Header.label] == self.current_mt.label and row[Header.number] == i:
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
        paths = [r.partition(100) for r in self.active_tomogram.paths]
        
        self.parent_viewer.add_shapes(paths, shape_type="path", edge_color="lime", edge_width=1,
                                      translate=self.layer_image.translate)
        return None
    
    def _paint_mt(self): # TODO
        """
        Paint microtubule fragments by its pitch length.
        
        1. Prepare small boxes and make masks inside them.
        2. Map the masks to the reference image.
        3. Erase masks using reference image, based on intensity.
        """        
        lbl = ip.zeros(self.layer_image.data.shape, dtype=np.uint8)
        color: dict[int, list[float]] = {0: [0, 0, 0, 0]}
        bin_scale = self.layer_image.scale[0] # scale of binned reference image
        lz, ly, lx = [int(r/bin_scale*1.4)*2 + 1 for r in self.box_radius_pre]
        tomo = self.active_tomogram
        with ip.SetConst("SHOW_PROGRESS", False):
            z, y, x = np.indices((lz, ly, lx))
            domains = np.zeros((len(tomo.paths), lz, ly, lx), dtype=np.float32)
            for i, r in enumerate(tomo.paths):
                # Prepare template hollow image
                r0 = r.radius/tomo.scale*0.9/self.binsize
                r1 = r.radius/tomo.scale*1.1/self.binsize
                _sq = (z - lz//2)**2 + (x - lx//2)**2
                domains[i][(r0**2 < _sq) & (_sq < r1**2)] = 1.0
                ry = max(int(self.interval/bin_scale/2 + 0.5), 1)
                domains[i, :, :ly//2-ry] = 0
                domains[i, :, ly//2+ry+1:] = 0
                
            domains = ip.asarray(domains, axes="pzyx")
            
            tasks = []
            for i, (_, row) in enumerate(self.dataframe.iterrows()):                
                z, y, x = np.indices((lz, ly, lx))
                r0 = self.current_mt.radius_peak/self.current_mt.scale*0.9/self.binsize
                r1 = self.current_mt.radius_peak/self.current_mt.scale*1.1/self.binsize
                _sq = (z-lz//2)**2 + (x-lx//2)**2
                domain = (r0**2 < _sq) & (_sq < r1**2)
                domain = domain.astype(np.float32)
                ry = max(int(self.interval/bin_scale/2 + 0.5), 1)
                domain[:, :ly//2-ry] = 0
                domain[:, ly//2+ry+1:] = 0
                domain = ip.array(domain, axes="zyx")
                ang_zy = row[Header.angle_zy]
                ang_yx = row[Header.angle_yx]
                tasks.append(da.from_delayed(rot3dinv(domain, ang_yx, ang_zy), shape=domain.shape, 
                                             dtype=np.float32) > 0.3)
            
            out: list[np.ndarray] = da.compute(tasks)[0]

        # paint roughly
        for i, (_, row) in enumerate(self.dataframe.iterrows()):
            center = (row[Header.zyx()]/self.image.scale.x).astype(np.int16)//self.binsize
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
        props["ID"] = self.dataframe.apply(lambda x: "{}-{}".format(x[Header.label], x[Header.number]), axis=1)
        props["pitch"] = self.dataframe[Header.pitch].map("{:.2f} nm".format)
        props["nPF"] = self.dataframe[Header.nPF]
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
        viewer: napari.Viewer = self.parent_viewer
        worker = imread(img, self.binsize)
        self._connect_worker(worker)
        self._worker_control.info.value = \
            f"original image: {img.shape}, " \
            f"binned image: {tuple(s//self.binsize for s in img.shape)}"
        
        @worker.returned.connect
        def _on_return(imgb):
            self._init_widget_params()
            tr = (self.binsize-1)/2*img.scale.x
            if self.layer_image not in viewer.layers:
                self.layer_image = viewer.add_image(
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
                
            viewer.scale_bar.unit = img.scale_unit
            viewer.dims.axis_labels = ("z", "y", "x")
            
            tomo = MtTomogram(self.box_radius_pre, 
                              self.box_radius,
                              self.light_background,
                              name=img.name)
            
            self.tomograms.append(tomo)

            self.clear_all()
            return None
        
        worker.start()
        self._loader.close()
        return None
    
    def _load_tomogram_results(self):
        tomo = self.active_tomogram
        self._init_layers()
                
        # Paint MTs by its pitch length
        self._paint_mt()
        
        self.layer_work.mode = "pan_zoom"
        self.parent_viewer.layers.selection = {self.layer_paint}
        
        # initialize GUI
        self._init_widget_params()
        self.mt.mtlabel.max = tomo.n_paths
        self.mt.pos.max = len(tomo.paths[0].localprops[H.splDistance])
        
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
        self.viewer_op.txt.value = "XX ° / X.XX nm / XX pf"
        return None
    
    @auto_picker.wraps
    def pick(self):
        """
        Automatically pick MT center using previous two points.
        """        
        point = self._pick()
        self._add_point(point)
        return None
    
    def _pick(self):
        stride_nm = self.auto_picker.stride.value
        imgb = self.layer_image.data
        try:
            # orientation is point0 -> point1
            point0: np.ndarray = self.layer_work.data[-2]/imgb.scale.x # unit: pixel
            point1: np.ndarray = self.layer_work.data[-1]/imgb.scale.x
        except IndexError:
            raise IndexError("Auto pick needs at least two points in the working layer.")
        
        radius = [int(v/imgb.scale.x) for v in self.box_radius_pre]
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
            
            img_next = load_a_subtomogram(imgb, point2.astype(np.uint16), radius, dask=False)
            
            angle_deg2 = angle_corr(img_next, ang_center=angle_deg, drot=5, nrots=7)
            
            if np.deg2rad(angle_deg - angle_deg2)/stride_nm > 0.02:
                raise ValueError("Angle changed too much.")
            img_next_rot = img_next.rotate(-angle_deg2, cval=np.median(img_next))
            proj = img_next_rot.proj("y")
            proj_mirror = proj["z=::-1;x=::-1"]
            shift = ip.pcc_maximum(proj, proj_mirror)
            dz, dx = shift/2
            
        point2[0] += dz
        point2[2] += dx
        return point2
        
    def _add_point(self, point):
        imgb = self.layer_image.data
        next_data = point * imgb.scale.x
        self.layer_work.add(next_data)
        msg = self._check_path()
        if msg:
            self.layer_work.data = self.layer_work.data[:-1]
            raise ValueError(msg)
        viewer = self.parent_viewer
        viewer.camera.center = point
        zoom = viewer.camera.zoom
        viewer.camera.events.zoom()
        viewer.camera.zoom = zoom
        viewer.dims.current_step = list(next_data.astype(np.int64))
        return None
                    
    def _check_path(self) -> str:
        imgshape_nm = np.array(self.image.shape) * self.image.scale.x
        if self.layer_work.data.shape[0] == 0:
            return ""
        else:
            point0 = self.layer_work.data[-1]
            if not np.all([r*0.7 <= p < s - r*0.7
                           for p, s, r in zip(point0, imgshape_nm, self.box_radius_pre)]):
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
        img = self.active_tomogram.image
        
        common_properties = dict(ndim=3, n_dimensional=True, size=16*img.scale.x)
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
            viewer.layers.remove(self.layer_paint.name)
            self.layer_paint = None

        return None
    
    @mt.pos.connect
    def _imshow_all(self, event=None):
        tomo = self.active_tomogram
        i = self.mt.mtlabel.value
        j = self.mt.pos.value
        results = tomo.paths[i]
        positions, skew, pitch, npf = results.localprops[[H.splDistance, H.skew, H.yPitch, H.nPF]].iloc[j]
        positions = tuple(positions)
        self.viewer_op.txt.value = f"{skew}° / {pitch:.2f} nm / {npf} pf"
        
        axes: Iterable[Axes] = self.canvas.axes
        for k in range(3):
            axes[k].cla()
            axes[k].tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
            
        subtomo = tomo._sample_subtomograms(i, positions)
        lz, ly, lx = subtomo.shape
        with ip.SetConst("SHOW_PROGRESS", False):
            axes[0].imshow(subtomo.proj("z"), cmap="gray")
            axes[0].set_xlabel("x")
            axes[0].set_ylabel("y")
            axes[1].imshow(subtomo.proj("y"), cmap="gray")
            axes[1].set_xlabel("x")
            axes[1].set_ylabel("z")
            axes[2].imshow(self.current_mt.average_images[i], cmap="gray")
            axes[2].set_xlabel("x")
            axes[2].set_ylabel("z")
        
        ylen = tomo.nm2pixel(tomo.box_radius[1])
        ymin, ymax = ly/2 - ylen, ly/2 + ylen
        r = tomo.nm2pixel(results.radius)*OUTER
        xmin, xmax = -r + lx/2, r + lx/2
        axes[0].plot([xmin, xmin, xmax, xmax, xmin], [ymin, ymax, ymax, ymin, ymin], color="lime")
        axes[0].text(1, 1, f"{i}-{j}", 
                                 color="lime", font="Consolas", size=15, va="top")
    
        theta = np.linspace(0, np.pi, 360)
        r = tomo.nm2pixel(results.radius)*INNER
        axes[1].plot(r*np.cos(theta) + lx/2, r*np.sin(theta) + lz/2, color="lime")
        r = tomo.nm2pixel(results.radius)*OUTER
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
            # BUG: not called after loading image
            viewer.window._status_bar._toggle_activity_dock(False)
            dialog.layout().removeWidget(self._worker_control.native)
            
        dialog.layout().addWidget(self._worker_control.native)
        return None
        