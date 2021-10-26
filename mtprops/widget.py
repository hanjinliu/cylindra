import pandas as pd
from typing import TYPE_CHECKING, Iterable
import numpy as np
import napari
from napari.utils.colormaps.colormap import Colormap
from napari.qt import thread_worker, create_worker
from qtpy.QtGui import QFont
from pathlib import Path
from magicgui.widgets import Table, TextEdit
import matplotlib.pyplot as plt

from ._dependencies import impy as ip
from ._dependencies import (mcls, magicclass, magicmenu, field, set_design, click, set_options, 
                            Figure, TupleEdit, Separator, ListWidget)
from .tomogram import MtTomogram, cachemap, angle_corr, dask_affine
from .utils import load_a_subtomogram, make_slice_and_pad, map_coordinates, roundint, ceilint
from .const import nm, H, Ori, GVar

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
    path = field(Path, options={"filter": "*.tif;*.tiff;*.mrc;*.rec"})
    scale = field(str, options={"label": "scale (nm)"})
    bin_size = field(4, options={"label": "bin size", "min": 1, "max": 8})
    light_background = field(True, options={"label": "light background"})
    
    @set_design(text="OK")
    def call_button(self):
        try:
            scale = float(self.scale.value)
        except Exception as e:
            raise type(e)(f"Invalid input: {self.scale.value}")
        
        self.img.scale.x = self.img.scale.y = self.img.scale.z = scale
        return self.img
    
    @path.connect
    def _read_scale(self):
        self._imread(self.path.value)
    
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
    # Main GUI class.
    
    _loader = ImageLoader()
    _worker_control = WorkerControl()
    
    @magicmenu
    class File:
        def Open_image(self): ...
        def Load_json(self, path: Path): ...
        def Save_results_as_json(self, path: Path): ...
    
    @magicmenu
    class View:
        def View_current_MT_fragment(self): ...
        def View_straightened_image(self): ...
        sep0 = Separator()
        def show_current_ft(self): ...
        def show_global_ft(self): ...
        def show_r_proj(self): ...
        def show_global_r_proj(self): ...
        sep1 = Separator()
        def Show_splines(self): ...
        def Show_results_in_a_table_widget(self): ...
        def Paint_MT(self): ...
        def Set_colormap(self): ...
        focus = field(False, options={"text": "Focus"})
    
    @magicmenu
    class Analysis:
        def Fit_splines(self): ...
        def Refine_splines(self): ...
        sep0 = Separator()
        def Local_FT_analysis(self): ...
        def Global_FT_analysis(self): ...
        sep1 = Separator()
        def Reconstruct_MT(self): ...
        def cylindric_reconstruction(self): ...
        def Map_tubulin(self): ...
    
    @magicmenu
    class Others:
        def Create_macro(self): ...
        def Global_variables(self): ...
        def MTProps_info(self): ...
        
    @magicclass(layout="horizontal", labels=False)
    class operation:
        def register_path(self): ...
        def run_for_all_path(self): ...
        def clear_current(self): ...
        def clear_all(self): ...
    
    @magicclass(layout="horizontal")
    class auto_picker:
        stride = field(50.0, widget_type="FloatSlider", options={"min": 10, "max": 100}, name="stride (nm)")
        def pick_next(self): ...
        def auto_center(self): ...
        
    tomograms = field(ListWidget, options={"name": "Tomograms"})
    
    @magicclass(layout="horizontal")
    class mt:
        mtlabel = field(int, options={"max": 0}, name="MTLabel")
        pos = field(int, widget_type="Slider", options={"max":0}, name="Pos")
    
    canvas = field(Figure, name="Figure", options={"figsize":(4.2, 1.8), "tooltip": "Projections"})
        
    txt = field(str, options={"enabled": False}, name="result")
        
    orientation_choice = field(Ori.none, name="Orientation: ")
    
    plot = field(Figure, name="Plot", options={"figsize":(4.2, 1.8), "tooltip": "Plot of local properties"})

    @View.wraps
    @set_options(start={"widget_type": TupleEdit, "options": {"step": 0.1}}, 
                 end={"widget_type": TupleEdit, "options": {"step": 0.1}},
                 limit={"widget_type": TupleEdit, "options": {"step": 0.02}, "label": "limit (nm)"})
    def Set_colormap(self, start=(0.0, 0.0, 1.0), end=(1.0, 0.0, 0.0), limit=(4.10, 4.36)):
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
    
    @Others.wraps
    def Create_macro(self):
        """
        Create Python executable script.
        """        
        self.create_macro(show=True)
        return None
    
    def _update_colormap(self, prop: str = H.yPitch):
        # TODO: color by other properties
        if self.layer_paint is None:
            return None
        color = {0: np.array([0., 0., 0., 0.], dtype=np.float32),
                 None: np.array([0., 0., 0., 1.], dtype=np.float32)}
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
        
        self.Set_colormap()
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
    @set_design(text="📝")
    def register_path(self):
        """
        Register current selected points as a MT path.
        """        
        coords = self.layer_work.data
        if coords.size == 0:
            return None
        tomo = self.active_tomogram
        self.active_tomogram.add_spline(coords)
        spl = self.active_tomogram.paths[-1]
        
        # check/draw path
        interval = 30
        length = spl.length()
        
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
    @set_options(interval={"min":1.0, "max": 100.0, "label": "Interval (nm)"},
                 box_size={"widget_type": TupleEdit, "label": "Initial box size (nm)"}, 
                 ft_size={"label": "Local DFT window size (nm)"})
    @set_design(text="👉")
    def run_for_all_path(self, 
                         interval: nm = 24.0,
                         box_size: tuple[nm, nm, nm] = (44.0, 56.0, 56.0),
                         ft_size: nm = 33.4):
        """
        Run MTProps.

        Parameters
        ----------
        interval : nm, default is 24.0
            Interval of sampling points of microtubule fragments.
        box_size : tuple[nm, nm, nm], default is (44.0, 56.0, 56.0)
            Box size of microtubule fragments used for angle correction and centering.
        ft_size : nm, default is 33.4
            Longitudinal length of local discrete Fourier transformation used for 
            structural analysis.
        """        
        if self.layer_work.data.size > 0:
            self.register_path()
        
        worker = create_worker(self._run_all, 
                               interval=interval,
                               box_size=box_size,
                               ft_size=ft_size,
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
                 interval: nm,
                 box_size,
                 ft_size):
        tomo = self.active_tomogram
        tomo.box_size = box_size
        tomo.ft_size = ft_size
        for i in range(tomo.n_paths):
            tomo.fit(i)
            tomo.make_anchors(interval=interval)
            
            yield f"Reloading subtomograms  ({i}/{tomo.n_paths})"
            tomo.get_subtomograms(i)
            
            yield f"Local Fourier transform ({i}/{tomo.n_paths}) "
            tomo.measure_radius(i)
            tomo.ft_params(i)
            if i+1 < tomo.n_paths:
                yield f"Spline fitting ({i+1}/{tomo.n_paths})"
        yield "Finishing ..."
        return tomo
    
    @operation.wraps
    @set_design(text="❌")
    def clear_current(self):
        """
        Clear current selection.
        """        
        self.layer_work.data = []
        return None
    
    @operation.wraps
    @set_options(_={"widget_type":"Label"})
    @set_design(text="💥")
    def clear_all(self, _="Are you sure to clear all?"):
        """
        Clear all the paths and heatmaps.
        """        
        self._init_widget_params()
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
    
    @Others.wraps
    def Global_variables(self, 
                         nPFmin: int = 11,
                         nPFmax: int = 17,
                         splOrder: int = 3,
                         yPitchAvg: nm = 4.16,
                         splError: nm = 0.8,
                         inner: float = 0.7,
                         outer: float = 1.6):
        """
        Set global variables.

        Parameters
        ----------
        nPFmin : int, default is 11
            Minimum protofilament numbers. 
        nPFmax : int, default is 17
            Maximum protofilament numbers.
        splOrder : int, default is 3
            Maximum order of spline curve.
        yPitchAvg : nm, default is 4.16
            Average pitch length estimation.
        splError : nm, default is 0.8
            Average error of spline fitting.
        inner : float, default is 0.7
            Radius x inner will be the inner surface of MT.
        outer : float, default is 1.6
            Radius x outer will be the outer surface of MT.
        """        
        GVar.set_value(**locals())
        
    
    @Others.wraps
    def MTProps_info(self):
        """
        Show information of dependencies.
        """        
        import napari
        import magicgui
        from .__init__ import __version__
        import dask
        
        value = f"MTProps: {__version__}\n"\
                f"impy: {ip.__version__}\n"\
                f"magicgui: {magicgui.__version__}\n"\
                f"magicclass: {mcls.__version__}\n"\
                f"napari: {napari.__version__}\n"\
                f"dask: {dask.__version__}\n"
        
        txt = TextEdit(value=value)
        txt.native.setParent(self.native, txt.native.windowFlags())
        self.read_only = True
        txt.native.setFont(QFont("Consolas"))
        txt.show()
        return None
    
    @File.wraps
    def Open_image(self):
        """
        Open an image and add to viewer.
        """
        self._loader.show()
        return None
        
    @File.wraps
    @set_options(path={"filter": "*.json;*.txt"})
    def Load_json(self, path: Path):
        """
        Choose a json file and load it.
        """        
        tomo = self.active_tomogram
        tomo.load(path)
        tomo.get_subtomograms()
        self._load_tomogram_results()
        return None
    
    @File.wraps
    @set_design(text="Save results as json")
    @set_options(file_path={"mode": "w", "filter": "*.json;*.txt"})
    def Save_results_as_json(self, file_path: Path):
        """
        Save the results as json.
        
        Parameters
        ----------
        file_path: Path
        """        
        self.active_tomogram.save(file_path)
        return None            
    
    @View.wraps
    def View_current_MT_fragment(self):
        """
        Send the current MT fragment 3D image (not binned) to napari viewer.
        """        
        i = self.mt.mtlabel.value
        j = self.mt.pos.value
        tomo = self.active_tomogram
        img = tomo._sample_subtomograms(i)[j]
        self.parent_viewer.add_image(img, scale=img.scale, name=img.name,
                                     rendering="minip" if tomo.light_background else "mip")
        return None
            
    @mt.mtlabel.connect
    @mt.pos.connect
    def _focus_on(self):
        """
        Change camera focus to the position of current MT fragment.
        """        
        if not self.View.focus.value or self.layer_paint is None:
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
        
        j_offset = sum(spl.anchors.size for spl in tomo.paths[:i])
        self.layer_paint.selected_label = j_offset + j + 1
        return None
    
    @View.wraps
    @set_design(text="Show results in a table widget")
    def Show_results_in_a_table_widget(self):
        """
        Show result table.
        """        
        table = Table(value=self.active_tomogram.collect_localprops())
        self.parent_viewer.window.add_dock_widget(table)
        return None
    
    @View.wraps
    @set_design(text="View straightened image")
    def View_straightened_image(self):
        """
        Send straightened image of the current MT to viewer.
        """        
        i = self.mt.mtlabel.value
        tomo = self.active_tomogram
        
        worker = create_worker(tomo.straighten, 
                               i=i, 
                               _progress={"total": 0, 
                                          "desc": "Running"}
                               )
        
        @worker.returned.connect
        def _on_return(out: ip.arrays.ImgArray):
            self.parent_viewer.add_image(out, scale=out.scale)
        
        self._worker_control.info.value = f"Straightening spline No. {i}"
        self._connect_worker(worker)
        worker.start()
        return None
    
    @View.wraps
    @set_design(text="R-projection")
    def show_r_proj(self):
        """
        Show radial projection of cylindrical image around the current MT fragment.
        """        
        polar = self._current_cylindrical_img().proj("r")
        self.parent_viewer.add_image(polar, scale=polar.scale, name="R-projection")
        return None
    
    @View.wraps
    @set_design(text="R-projection (Global)")
    def show_global_r_proj(self):
        """
        Show radial projection of cylindrical image along current MT.
        """        
        i = self.mt.mtlabel.value
        polar = self.active_tomogram.straighten(i, cylindrical=True).proj("r")
        self.parent_viewer.add_image(polar, scale=polar.scale, name="R-projection (Global)")
        return None
    
    @View.wraps
    @set_design(text="2D-FT")
    def show_current_ft(self):
        """
        View Fourier space of local cylindrical coordinate system at current position.
        """        
        with ip.SetConst("SHOW_PROGRESS", False):
            polar = self._current_cylindrical_img()
            pw = polar.power_spectra(zero_norm=True, dims="rya").proj("r")
        self.parent_viewer.add_image(pw, scale=pw.scale, colormap="inferno", name="FT")
        return None
    
    @View.wraps
    @set_design(text="2D-FT (Global)")
    def show_global_ft(self):
        """
        View Fourier space along current MT.
        """  
        i = self.mt.mtlabel.value
        with ip.SetConst("SHOW_PROGRESS", False):
            polar = self.active_tomogram.straighten(i, cylindrical=True)
            pw = polar.power_spectra(zero_norm=True, dims="rya").proj("r")
        self.parent_viewer.add_image(pw, scale=pw.scale, colormap="inferno", name="FT (Global)")
        return None
    
    @View.wraps
    def Show_splines(self):
        """
        Show 3D spline paths of microtubule center axes as a layer.
        """        
        paths = [r.partition(100) for r in self.active_tomogram.paths]
        
        self.parent_viewer.add_shapes(paths, shape_type="path", edge_color="lime", edge_width=1,
                                      translate=self.layer_image.translate)
        return None
    
    @Analysis.wraps
    @set_options(box_size={"widget_type": TupleEdit, "label": "Initial box size (nm)"})
    def Fit_splines(self, 
                   box_size: tuple[nm, nm, nm] = (44.0, 56.0, 56.0)):
        """
        Fit MT with spline curve, using manually selected points.

        Parameters
        ----------
        box_size : tuple[nm, nm, nm], default is (44.0, 56.0, 56.0)
            Box size that will be cropped from the original image when calculate MT center.
        """        
        tomo = self.active_tomogram
        tomo.box_size = box_size
        for i in range(tomo.n_paths):
            tomo.fit(i)
            tomo.make_anchors(n=3)
            tomo.measure_radius(i)
        
        self._init_layers()
        self.Show_splines()
        return None
    
    @Analysis.wraps
    def Refine_splines(self):
        """
        Refine splines using the global MT structural parameters.
        """        
        tomo = self.active_tomogram
        
        def _run():
            tomo.refine()
            tomo.make_anchors()
            tomo.measure_radius()
        
        worker = create_worker(_run,
                               _progress={"total": 0, 
                                          "desc": "Running"})
        
        self._connect_worker(worker)
        self._worker_control.info.value = f"Refining splines ..."
        worker.start()
                
        self._init_widget_params()
        self.canvas.figure.clf()
        self.canvas.draw()
        self.plot.figure.clf()
        self.plot.figure.add_subplot(111)
        self.plot.draw()
        return None
    
    @Analysis.wraps
    def Local_FT_analysis(self):
        tomo = self.active_tomogram
        worker = create_worker(tomo.ft_params,
                               _progress={"total": 0, 
                                          "desc": "Running"})
        @worker.returned.connect
        def _on_return(df):
            self._load_tomogram_results()
        
        self._connect_worker(worker)
        self._worker_control.info.value = f"Local Fourier transform ..."
        worker.start()
        return None
        
    @Analysis.wraps
    def Global_FT_analysis(self):
        tomo = self.active_tomogram
        worker = create_worker(tomo.global_ft_params,
                               _progress={"total": 0, 
                                          "desc": "Running"})
        @worker.returned.connect
        def _on_return(out):
            df = pd.DataFrame({f"MT-{k}": v for k, v in enumerate(out)})
            table = Table(value=df)
            self.parent_viewer.window.add_dock_widget(table, name="Global structures")
        
        self._connect_worker(worker)
        self._worker_control.info.value = f"Global Fourier transform ..."
        worker.start()
        
        return None
        
    @Analysis.wraps
    @set_options(rot_ave={"label": "Rotational averaging"},
                 y_length={"label": "Longitudinal length (nm)"})
    def Reconstruct_MT(self, rot_ave=False, y_length=50.0):
        """
        Coarse reconstruction of MT.

        Parameters
        ----------
        rot_ave : bool, default is False
            Check to run rotational averaging after reconstruction.
        y_length : nm, default is 50.0
            Longitudinal length (nm) of reconstructed image.
        """        
        tomo = self.active_tomogram
        i = self.mt.mtlabel.value
        
        worker = create_worker(tomo.reconstruct, 
                               i=i,
                               rot_ave=rot_ave, 
                               y_length=y_length,
                               _progress={"total": 0, 
                                          "desc": "Running"}
                               )
        
        @worker.returned.connect
        def _on_return(out: ip.arrays.ImgArray):
            if tomo.light_background:
                out = -out
            self.parent_viewer.add_image(out, scale=out.scale, 
                                         name=f"Reconstruction of MT-{i}")
        
        self._connect_worker(worker)
        self._worker_control.info.value = f"Reconstruction ..."
        worker.start()
        return None
    
    @Analysis.wraps
    @set_options(rot_ave={"label": "Rotational averaging"},
                 y_length={"label": "Longitudinal length (nm)"})
    @set_design(text="Reconstruct MT (cylindric)")
    def cylindric_reconstruction(self, rot_ave=False, y_length=50.0):
        """
        Coarse reconstruction of MT.

        Parameters
        ----------
        rot_ave : bool, default is False
            Check to run rotational averaging after reconstruction.
        y_length : nm, default is 50.0
            Longitudinal length (nm) of reconstructed image.
        """        
        tomo = self.active_tomogram
        i = self.mt.mtlabel.value
        
        worker = create_worker(tomo.cylindric_reconstruct, 
                               i=i,
                               rot_ave=rot_ave, 
                               y_length=y_length,
                               _progress={"total": 0, 
                                          "desc": "Running"}
                               )
        
        @worker.returned.connect
        def _on_return(out: ip.arrays.ImgArray):
            if tomo.light_background:
                out = -out
            self.parent_viewer.add_image(out, scale=out.scale)
        
        self._connect_worker(worker)
        self._worker_control.info.value = f"Cylindric reconstruction ..."
        worker.start()
        return None
    
    @Analysis.wraps
    @set_design(text="Map tubulin")
    def Map_tubulin(self):
        tomo = self.active_tomogram
        i = self.mt.mtlabel.value
        
        worker = create_worker(tomo.map_monomer, 
                               i=i,
                               _progress={"total": 0, 
                                          "desc": "Running"}
                               )
        
        @worker.returned.connect
        def _on_return(out):
            self.parent_viewer.add_points(out.world, size=3, face_color="lime",
                                          n_dimensional=True,
                                          name="tubulin monomers")
        
        self._connect_worker(worker)
        self._worker_control.info.value = f"Tubulin mapping ..."
        worker.start()
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
        binsize = roundint(self.layer_image.scale[0]/tomo.scale) # scale of binned reference image
        
        shape = tomo.nm2pixel(np.array(tomo.box_size)/binsize)
        with ip.SetConst("SHOW_PROGRESS", False):
            orientation = point1[1:] - point0[1:]
            img = load_a_subtomogram(imgb, point1, shape, dask=False)
            center = np.rad2deg(np.arctan2(*orientation)) % 180 - 90
            angle_deg = angle_corr(img, ang_center=center, drot=25, nrots=25)
            angle_rad = np.deg2rad(angle_deg)
            dr = np.array([0.0, stride_nm*np.cos(angle_rad), -stride_nm*np.sin(angle_rad)])
            if np.dot(orientation, dr[1:]) > np.dot(orientation, -dr[1:]):
                point2 = point1 + dr
            else:
                point2 = point1 - dr
            img_next = load_a_subtomogram(imgb, point2, shape, dask=False)
            centering(img_next, point2, angle_deg)
            
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
        binsize = roundint(self.layer_image.scale[0]/tomo.scale) # scale of binned reference image
        selected = self.layer_work.selected_data
        shape = tomo.nm2pixel(np.array(tomo.box_size)/binsize)
        
        points = self.layer_work.data / imgb.scale.x
        last_i = -1
        with ip.SetConst("SHOW_PROGRESS", False):
            for i, point in enumerate(points):
                if i not in selected:
                    continue
                img_input = load_a_subtomogram(imgb, point, shape, dask=False)
                angle_deg = angle_corr(img_input, ang_center=0, drot=89.5, nrots=19)
                centering(img_input, point, angle_deg, drot=5, nrots=7)
                last_i = i
        
        self.layer_work.data = points * imgb.scale.x
        if len(selected) == 1:
            change_viewer_focus(self.parent_viewer, points[last_i], self.layer_work.data[last_i])
        return None
    
    @View.wraps
    def Paint_MT(self):
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
        
        lz, ly, lx = [int(r/bin_scale*1.4)*2 + 1 for r in [15, tomo.ft_size/2, 15]]
        bin_scale = self.layer_image.scale[0] # scale of binned reference image
        binsize = roundint(bin_scale/tomo.scale)
        with ip.SetConst("SHOW_PROGRESS", False):
            center = np.array([lz, ly, lx])/2 + 0.5
            z, y, x = np.indices((lz, ly, lx))
            cylinders = []
            matrices = []
            for i, spl in enumerate(tomo.paths):
                # Prepare template hollow image
                r0 = spl.radius/tomo.scale*0.9/binsize
                r1 = spl.radius/tomo.scale*1.1/binsize
                _sq = (z - lz/2 - 0.5)**2 + (x - lx/2 - 0.5)**2
                domains = []
                dist = [-np.inf] + list(spl.distances()) + [np.inf]
                for j in range(spl.anchors.size):
                    domain = (r0**2 < _sq) & (_sq < r1**2)
                    ry = min((dist[j+1] - dist[j]) / 2, 
                             (dist[j+2] - dist[j+1]) / 2, 
                              tomo.ft_size/2) / bin_scale + 0.5 
                        
                    ry = max(ceilint(ry), 1)
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
                _sl, _pad = make_slice_and_pad(c - l//2, c + l//2 + 1, size)
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
        
            
    def _plot_properties(self):
        i = self.mt.mtlabel.value
        props = self.active_tomogram.paths[i].localprops
        x = props[H.splDistance]
        pitch_color = "lime"
        skew_color = "gold"
        self.plot.ax.cla()
        if hasattr(self.plot, "ax2"):
            self.plot.ax2.cla()
        
        self.plot.ax.plot(x, props[H.yPitch], color=pitch_color)
        self.plot.ax.set_xlabel("position (nm)")
        self.plot.ax.set_ylabel("pitch (nm)")
        self.plot.ax.set_ylim(*self.label_colorlimit)
        
        self.plot.ax2 = self.plot.ax.twinx()
        self.plot.ax2.plot(x, props[H.skewAngle], color=skew_color)
        self.plot.ax2.set_ylabel("skew (deg)")
        self.plot.ax2.set_ylim(-2.0, 2.0)
        
        self.plot.ax2.spines["left"].set_color(pitch_color)
        self.plot.ax2.spines["right"].set_color(skew_color)
                    
        self.plot.figure.tight_layout()
        self.plot.draw()
    
        return None
    
    @click(visible=False)
    def load_image(self):
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
            f"Loading with {binsize}x{binsize} binned size: {tuple(s//binsize for s in img.shape)}"
        
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
                tomo = MtTomogram(light_background=light_bg)
                
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
        # initialize GUI
        self._init_widget_params()
        self.mt.mtlabel.max = tomo.n_paths - 1
        self.mt.pos.max = len(tomo.paths[0].localprops[H.splDistance]) - 1
        
        self._active_rot_ave = tomo.rotational_average()
        self._init_layers()
                        
        self.layer_work.mode = "pan_zoom"
        
        self._imshow_all()
        
        self._plot_properties()
        
        return None
    
    def _init_widget_params(self):
        self.mt.mtlabel.value = 0
        self.mt.mtlabel.min = 0
        self.mt.mtlabel.max = 0
        self.mt.pos.value = 0
        self.mt.pos.min = 0
        self.mt.pos.max = 0
        self.txt.value = ""
        return None
    
    def _check_path(self) -> str:
        tomo = self.active_tomogram
        imgshape_nm = np.array(tomo.image.shape) * tomo.image.scale.x
        if self.layer_work.data.shape[0] == 0:
            return ""
        else:
            point0 = self.layer_work.data[-1]
            if not np.all([r/4 <= p < s - r/4
                           for p, s, r in zip(point0, imgshape_nm, tomo.box_size)]):
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
    
    
    def _current_cylindrical_img(self):
        """
        Return cylindric-transformed image at the current position
        """        
        i = self.mt.mtlabel.value
        j = self.mt.pos.value
        tomo = self.active_tomogram
        ylen = tomo.nm2pixel(tomo.ft_size)
        spl = tomo._paths[i]
        
        rmin = tomo.nm2pixel(spl.radius*GVar.inner)
        rmax = tomo.nm2pixel(spl.radius*GVar.outer)
        
        coords = spl.local_cylindrical((rmin, rmax), ylen, spl.anchors[j])
        coords = np.moveaxis(coords, -1, 0)
        img = tomo.image
        polar = map_coordinates(img, coords, prefilter=False, order=1)
        polar = ip.asarray(polar, axes="rya") # radius, y, angle
        polar.set_scale(r=img.scale.x, y=img.scale.x, a=img.scale.x)
        polar.scale_unit = img.scale_unit
        return polar
    
    def _init_layers(self):
        viewer: napari.Viewer = self.parent_viewer
        
        common_properties = dict(ndim=3, n_dimensional=True, size=8)
        if self.layer_prof in self.parent_viewer.layers:
            viewer.layers.remove(self.layer_prof)
    
        self.layer_prof = viewer.add_points(**common_properties,
                                    name="MT Profiles",
                                    opacity=0.4, 
                                    edge_color="black",
                                    face_color="black",
                                    )
        self.layer_prof.editable = False
            
        if self.layer_work in viewer.layers:
            viewer.layers.remove(self.layer_work)
        
        self.layer_work = viewer.add_points(**common_properties,
                                    name="Working Layer",
                                    face_color="yellow"
                                    )
    
        self.layer_work.mode = "add"
        
        if self.layer_paint is not None:
            self.layer_paint.data = np.zeros_like(self.layer_paint.data)
        self.orientation_choice.value = Ori.none
        return None
    
    @mt.pos.connect
    def _imshow_all(self):
        tomo = self.active_tomogram
        i = self.mt.mtlabel.value
        j = self.mt.pos.value
        try:
            results = tomo.paths[i]
        except IndexError:
            # sometimes i takes wrong value due to event emission
            i = 0
            results = tomo.paths[i]
        pitch, skew, npf, start = results.localprops[[H.yPitch, H.skewAngle, H.nPF, H.start]].iloc[j]
        self.txt.value = f"{pitch:.2f} nm / {skew:.2f}°/ {int(npf)}_{int(start)}"
        
        if len(self.canvas.axes) < 3:
            self.canvas.figure.clf()
            self.canvas.figure.add_subplot(131)
            self.canvas.figure.add_subplot(132)
            self.canvas.figure.add_subplot(133)
        
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
        
        ylen = tomo.nm2pixel(tomo.ft_size/2)
        ymin, ymax = ly/2 - ylen, ly/2 + ylen
        r = tomo.nm2pixel(results.radius)*GVar.outer
        xmin, xmax = -r + lx/2, r + lx/2
        axes[0].plot([xmin, xmin, xmax, xmax, xmin], [ymin, ymax, ymax, ymin, ymin], color="lime")
        axes[0].text(1, 1, f"{i}-{j}", color="lime", font="Consolas", size=15, va="top")
    
        theta = np.linspace(0, 2*np.pi, 360)
        r = tomo.nm2pixel(results.radius) * GVar.inner
        axes[1].plot(r*np.cos(theta) + lx/2, r*np.sin(theta) + lz/2, color="lime")
        r = tomo.nm2pixel(results.radius) * GVar.outer
        axes[1].plot(r*np.cos(theta) + lx/2, r*np.sin(theta) + lz/2, color="lime")
                
        self.canvas.figure.tight_layout()
        self.canvas.draw()
    
    @orientation_choice.connect
    def _update_note(self):
        i = self.mt.mtlabel.value
        self.active_tomogram.paths[i].orientation = self.orientation_choice.value
        return None
    
    @mt.mtlabel.connect
    def _update_mtpath(self):
        self.mt.mtlabel.enabled = False
        i = self.mt.mtlabel.value
        tomo = self.active_tomogram
        
        self.mt.pos.max = len(tomo.paths[i].localprops) - 1
        note = tomo.paths[i].orientation
        self.orientation_choice.value = Ori(note)
        self._plot_properties()
        self._imshow_all()
        self.mt.mtlabel.enabled = True
        return None
    
    def _connect_worker(self, worker):
        self._worker_control._set_worker(worker)
        viewer: napari.Viewer = self.parent_viewer
        viewer.window._status_bar._toggle_activity_dock(True)
        dialog = viewer.window._qt_window._activity_dialog
        
        @worker.finished.connect
        def _on_finish(*args):
            viewer.window._status_bar._toggle_activity_dock(False)
            dialog.layout().removeWidget(self._worker_control.native)

        dialog.layout().addWidget(self._worker_control.native)
        return None

def centering(imgb, point, angle, drot=5, nrots=7):
            
    angle_deg2 = angle_corr(imgb, ang_center=angle, drot=drot, nrots=nrots)
    
    img_next_rot = imgb.rotate(-angle_deg2, cval=np.median(imgb))
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