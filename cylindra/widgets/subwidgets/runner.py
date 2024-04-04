from typing import Annotated, Sequence

from magicclass import (
    do_not_record,
    magicclass,
    set_design,
    vfield,
)
from magicclass.types import Optional
from magicclass.utils import thread_worker

from cylindra.const import nm
from cylindra.widgets._widget_ext import CheckBoxes
from cylindra.widgets.subwidgets._child_widget import ChildWidget


@magicclass(widget_type="groupbox", name="Fitting parameters", record=False)
class runner_params1:
    """
    Parameters used in spline fitting.

    Attributes
    ----------
    edge_sigma : nm
        Sharpness of dense-mode mask at the edges. Useful if cylindric structures are
        densely packed. Initial spline position must be 'almost' fitted in dense-mode.
    max_shift : nm
        Maximum shift in nm of manually selected spline to the true center.
    """

    edge_sigma = vfield(Optional[float], label="Edge sigma").with_options(
        value=2.0,
        options={"step": 0.1, "min": 0.0, "max": 50.0},
        text="Don't use dense mode",
    )
    max_shift = vfield(5.0, label="Max shift (nm)").with_options(max=50.0, step=0.5)


@magicclass(widget_type="groupbox", name="Local-CFT parameters", record=False)
class runner_params2:
    """
    Parameters used in calculation of local properties.

    Attributes
    ----------
    interval : nm
        Interval of sampling points of cylinder fragments.
    depth: nm
        Longitudinal length of local discrete Fourier transformation used
        for structural analysis.
    """

    interval = vfield(50.0, label="Interval (nm)").with_options(min=1.0, max=200.0)
    depth = vfield(50.0, label="FT window size (nm)").with_options(min=1.0, max=200.0)


@magicclass(name="_Run cylindrical fitting", record=False)
class Runner(ChildWidget):
    """
    Attributes
    ----------
    splines : list of int
        Splines that will be analyzed
    bin_size : int
        Set to >1 to use binned image for fitting.
    n_refine : int
        Iteration number of spline refinement.
    local_props : bool
        Check if calculate local properties.
    global_props : bool
        Check if calculate global properties.
    infer_polarity : bool
        Check if infer spline polarity after run.
    """

    def _get_splines(self, _=None) -> list[tuple[str, int]]:
        """Get list of spline objects for categorical widgets."""
        try:
            tomo = self._get_main().tomogram
        except Exception:
            return []
        if tomo is None:
            return []
        return [(f"({i}) {spl}", i) for i, spl in enumerate(tomo.splines)]

    def _get_available_binsize(self, _=None) -> list[int]:
        try:
            parent = self._get_main()
        except Exception:
            return [1]
        out = [x[0] for x in parent.tomogram.multiscaled]
        if 1 not in out:
            out = [1] + out
        return sorted(out)

    splines = vfield(widget_type=CheckBoxes).with_choices(_get_splines)
    bin_size = vfield(int).with_choices(choices=_get_available_binsize)

    fit = vfield(True, label="Fit splines")
    params1 = runner_params1
    n_refine = vfield(1, label="Refinement iteration").with_options(max=10)
    local_props = vfield(True, label="Calculate local properties")
    params2 = runner_params2
    global_props = vfield(True, label="Calculate global properties")
    infer_polarity = vfield(True, label="Infer polarity")
    map_monomers = vfield(False, label="Map monomers")

    @fit.connect
    def _toggle_fit_params(self, val: bool):
        self.params1.enabled = val

    @local_props.connect
    def _toggle_localprops_params(self, val: bool):
        self.params2.enabled = val

    def _get_max_shift(self, w=None):
        if self.fit:
            return self.params1.max_shift
        else:
            return -1.0

    @set_design(text="Fit and Measure")
    @do_not_record(recursive=False)
    @thread_worker.with_progress(
        desc="Running fit/measure workflow",
        total="max_shift>0 + n_refine + local_props + global_props + infer_polarity + map_monomers",
    )
    def run(
        self,
        splines: Annotated[Sequence[int], {"bind": splines}] = (),
        bin_size: Annotated[int, {"bind": bin_size}] = 1,
        max_shift: Annotated[nm, {"bind": _get_max_shift}] = 5.0,
        edge_sigma: Annotated[nm, {"bind": params1.edge_sigma}] = 2.0,
        n_refine: Annotated[int, {"bind": n_refine}] = 1,
        local_props: Annotated[bool, {"bind": local_props}] = True,
        interval: Annotated[nm, {"bind": params2.interval}] = 50.0,
        depth: Annotated[nm, {"bind": params2.depth}] = 50.0,
        global_props: Annotated[bool, {"bind": global_props}] = True,
        infer_polarity: Annotated[bool, {"bind": infer_polarity}] = True,
        map_monomers: Annotated[bool, {"bind": map_monomers}] = False,
    ):
        """Run workflow."""
        main = self._get_main()
        if main._reserved_layers.work.data.size > 0:
            raise ValueError("The last spline is not registered yet.")
        if len(main.tomogram.splines) == 0:
            raise ValueError("No spline is added to the viewer canvas.")
        elif len(splines) == 0:
            splines = list(range(len(main.tomogram.splines)))
        yield thread_worker.callback(main._runner.close)

        if max_shift > 0.0:
            yield from main.fit_splines.arun(
                splines=splines,
                bin_size=bin_size,
                edge_sigma=edge_sigma,
                max_shift=max_shift,
            )
            yield
        for _ in range(n_refine):
            yield from main.refine_splines.arun(
                splines=splines,
                max_interval=max(interval, 30),
                bin_size=bin_size,
            )
            yield
        yield from main.measure_radius.arun(splines=splines, bin_size=bin_size)
        yield
        if local_props:
            yield from main.local_cft_analysis.arun(
                splines=splines, interval=interval, depth=depth, bin_size=bin_size
            )
            yield
        if infer_polarity:
            yield from main.infer_polarity.arun(bin_size=bin_size)
            yield
        if global_props:
            yield from main.global_cft_analysis.arun(splines=splines, bin_size=bin_size)
            yield
        if map_monomers:
            yield from main.map_monomers.arun(splines, orientation="MinusToPlus")
            yield
        return None
