from typing import Annotated, Sequence

from magicclass import (
    do_not_record,
    magicclass,
    vfield,
    MagicTemplate,
    set_design,
)
from magicclass.types import Optional

from cylindra.widgets._widget_ext import CheckBoxes

from cylindra.const import nm


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
        options=dict(step=0.1, min=0.0, max=50.0),
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
    ft_size: nm
        Longitudinal length of local discrete Fourier transformation used
        for structural analysis.
    paint : bool
        Check if paint the tomogram with the local properties.
    """

    interval = vfield(32.64, label="Interval (nm)").with_options(min=1.0, max=200.0)
    ft_size = vfield(32.64, label="FT window size (nm)").with_options(
        min=1.0, max=200.0
    )
    paint = vfield(False)


@magicclass(name="_Run cylindrical fitting", record=False)
class Runner(MagicTemplate):
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

    def _get_parent(self):
        from cylindra.widgets.main import CylindraMainWidget

        return self.find_ancestor(CylindraMainWidget)

    def _get_splines(self, _=None) -> list[tuple[str, int]]:
        """Get list of spline objects for categorical widgets."""
        try:
            tomo = self._get_parent().tomogram
        except Exception:
            return []
        if tomo is None:
            return []
        return [(f"({i}) {spl}", i) for i, spl in enumerate(tomo.splines)]

    def _get_available_binsize(self, _=None) -> list[int]:
        try:
            parent = self._get_parent()
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
    def run(
        self,
        splines: Annotated[Sequence[int], {"bind": splines}] = (),
        bin_size: Annotated[int, {"bind": bin_size}] = 1,
        max_shift: Annotated[nm, {"bind": _get_max_shift}] = 5.0,
        edge_sigma: Annotated[nm, {"bind": params1.edge_sigma}] = 2.0,
        n_refine: Annotated[int, {"bind": n_refine}] = 1,
        local_props: Annotated[bool, {"bind": local_props}] = True,
        interval: Annotated[nm, {"bind": params2.interval}] = 32.64,
        ft_size: Annotated[nm, {"bind": params2.ft_size}] = 32.64,
        global_props: Annotated[bool, {"bind": global_props}] = True,
        paint: Annotated[bool, {"bind": params2.paint}] = False,
        infer_polarity: Annotated[bool, {"bind": infer_polarity}] = True,
        map_monomers: Annotated[bool, {"bind": map_monomers}] = False,
    ):
        """Run workflow."""
        parent = self._get_parent()
        if parent._reserved_layers.work.data.size > 0:
            raise ValueError("The last spline is not registered yet.")
        if len(parent.tomogram.splines) == 0:
            raise ValueError("No spline is added to the viewer canvas.")
        elif len(splines) == 0:
            splines = list(range(len(parent.tomogram.splines)))
        parent._runner.close()

        if max_shift > 0.0:
            parent.fit_splines(
                splines=splines,
                bin_size=bin_size,
                edge_sigma=edge_sigma,
                max_shift=max_shift,
            )
        for _ in range(n_refine):
            parent.refine_splines(
                splines=splines,
                max_interval=max(interval, 30),
                bin_size=bin_size,
            )
        parent.measure_radius(splines=splines, bin_size=bin_size)
        if local_props:
            parent.local_ft_analysis(
                splines=splines, interval=interval, depth=ft_size, bin_size=bin_size
            )
        if infer_polarity:
            parent.auto_align_to_polarity(bin_size=bin_size)
        if global_props:
            parent.global_ft_analysis(splines=splines, bin_size=bin_size)
        if local_props and paint:
            cfg = parent.tomogram.splines[splines[0]].config
            parent.paint_cylinders(limits=cfg.spacing_range.astuple())
        if map_monomers:
            _plus_idx = list[int]()
            _minus_idx = list[int]()
            for idx in splines:
                if parent.tomogram.splines[idx].config.clockwise == "PlusToMinus":
                    _plus_idx.append(idx)
                else:
                    _minus_idx.append(idx)
            if _plus_idx:
                parent.map_monomers(_plus_idx, orientation="PlusToMinus")
            if _minus_idx:
                parent.map_monomers(_minus_idx, orientation="PlusToMinus")
        return None
