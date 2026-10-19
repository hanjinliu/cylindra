import json
import logging
import tempfile
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator, Iterable

import polars as pl
from pydantic import ConfigDict, Field

from cylindra import _config
from cylindra.const import ImageFilter, cast_dataframe, get_versions
from cylindra.const import PropertyNames as H
from cylindra.project._base import (
    BaseProject,
    MissingWedge,
    PathLike,
    TiltSeriesInfo,
    resolve_path,
)
from cylindra.project._json import project_json_encoder
from cylindra.project._layer_info import InteractionInfo, LandscapeInfo, MoleculesInfo
from cylindra.project._utils import (
    as_main_function,
    as_relative,
    extract,
    resolve_relative_paths,
)

if TYPE_CHECKING:
    import tarfile

    import macrokit
    from acryo import BackProjectionLoader, Molecules

    from cylindra.components import CylSpline, CylTomogram
    from cylindra.widgets.main import CylindraMainWidget

LOGGER = logging.getLogger("cylindra")


class CylindraProject(BaseProject):
    """A project of cylindra."""

    # this allows extra fields in the json file, for backward compatibility
    model_config = ConfigDict(extra="allow")

    datetime: str
    version: str
    dependency_versions: dict[str, str]
    image: PathLike | None
    """Path to the image file."""
    image_relative: PathLike | None = None
    """Relative path to the image file, used as a fallback."""
    image_halves: tuple[PathLike, PathLike] | None = None
    """Paths to the two half tomograms."""
    cache_image: bool = False
    """Whether to cache the image in SSD."""
    scale: float
    """Scale of the image, in nm/pixel."""
    image_reference: PathLike | None = None
    """Path to the reference image."""
    invert: bool = False
    """Whether to invert the image when loaded."""
    invert_reference: bool = False
    """Whether to invert the reference image when loaded from `image_reference`."""
    multiscales: list[int]
    """List of bin factors for multiscale tomogram."""
    tilt_series: TiltSeriesInfo | None = None
    """The tilt series, if the tomogram is opened from a tilt series."""
    molecules_info: list[MoleculesInfo] = Field(default_factory=list)
    landscape_info: list[LandscapeInfo] = Field(default_factory=list)
    interaction_info: list[InteractionInfo] = Field(default_factory=list)
    missing_wedge: MissingWedge = Field(default_factory=MissingWedge.default)
    """Missing wedge model, used for masking subtomograms."""
    project_path: Path | None = None
    project_description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def resolve_path(self, file_dir: PathLike):
        """Resolve the path of the project."""
        file_dir = Path(file_dir)
        self.image = resolve_path(self.image, file_dir)
        if self.image_halves is not None:
            self.image_halves = (
                resolve_path(self.image_halves[0], file_dir),
                resolve_path(self.image_halves[1], file_dir),
            )
        # NOTE: relative paths of the tilt series are resolved when opened, as they
        # may be in a tar or zip file (`file_dir` is a temporary directory for zip).
        return self

    @classmethod
    def new(
        cls,
        image: PathLike | None,
        scale: float | None,
        image_reference: PathLike | None = None,
        multiscales: list[int] = [1],
        missing_wedge: Any | None = None,
        invert: bool = False,
        project_path: Path | None = None,
        invert_reference: bool = False,
        tilt_series: TiltSeriesInfo | None = None,
    ):
        """Create a new project.

        If `tilt_series` is given, the tomogram is opened from the tilt series, and
        `image` can be None.
        """
        from cylindra import _io

        _versions = get_versions()
        if image is None and tilt_series is None:
            raise ValueError("image must not be None.")
        if scale is None:
            if image is None:
                raise ValueError("scale must be given for a tilt series.")
            header = _io.read_header(image)
            scale = header.scale["x"]
        return CylindraProject(
            datetime=datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            version=_versions.pop("cylindra", "unknown"),
            dependency_versions=_versions,
            image=image,
            image_reference=image_reference,
            scale=scale,
            multiscales=list(multiscales),
            tilt_series=tilt_series,
            missing_wedge=MissingWedge.parse(missing_wedge),
            invert=invert,
            invert_reference=invert_reference,
            project_path=project_path,
        )

    def save(
        self,
        project_dir: Path,
        splines: "list[CylSpline]" = [],
        molecules: "dict[str, Molecules]" = {},
        molecule_sources: "dict[str, int | None]" = {},
    ) -> None:
        """Save this project."""
        from macrokit import parse

        from cylindra.components.tomogram._spline_list import SplineList

        with _prep_save_dir(project_dir) as results_dir:
            expr_open = parse(self._open_image_script(), squeeze=False)
            # TODO: consider half tomos
            expr = as_main_function(expr_open)
            self._script_py_path(results_dir).write_text(expr)
            self_copy = self.model_copy()
            spl_list = SplineList(splines)
            if (df_loc := spl_list.collect_localprops()) is not None:
                df_loc.write_csv(self._localprops_path(results_dir))
            if (df_glob := spl_list.collect_globalprops()) is not None:
                df_glob.write_csv(self._globalprops_path(results_dir))
            for ith, spl in spl_list.enumerate():
                spl.to_json(results_dir / f"spline-{ith}.json")
            for name, mole in molecules.items():
                save_path = results_dir / name
                if save_path.suffix == "":
                    save_path = save_path.with_suffix(".csv")
                source = molecule_sources.get(name, None)
                self_copy.molecules_info.append(
                    MoleculesInfo(name=save_path.name, source=source)
                )
                mole.to_file(save_path)
            self_copy.to_json(self._project_json_path(results_dir))

    def _open_image_script(self) -> str:
        """The script that opens the tomogram of this new project."""
        scale = self.scale
        if (ts := self.tilt_series) is None:
            path = Path(self.image).as_posix()
            bin_size = self.multiscales
            tilt_range = self.missing_wedge.as_param()
            return f"ui.open_image({path=}, {scale=:.4f}, {bin_size=}, {tilt_range=})"
        path = Path(ts.path).as_posix()
        tomo_hand = ts.tomo_hand
        bin_size = ts.binsizes
        if ts.reference is not None:
            reference_path = Path(ts.reference).as_posix()
            invert_reference = self.invert_reference
            return (
                f"ui.open_tilt_series_with_reference({path=}, {scale=:.4f}, "
                f"{reference_path=}, {tomo_hand=}, {bin_size=}, {invert_reference=})"
            )
        # same bin size from the pixel sizes rounded in the script
        reference_scale = round(round(scale, 4) * ts.reference_binsize, 6)
        if ts.reference_shape:
            tomo_height = ts.reference_shape[0] * ts.reference_binsize * scale
        else:
            tomo_height = 300.0
        return (
            f"ui.open_tilt_series({path=}, {scale=:.4f}, {reference_scale=}, "
            f"{tomo_hand=}, {tomo_height=:.2f}, {bin_size=}, "
            "save_coarse_tomogram=False)"
        )

    @classmethod
    def from_gui(
        cls,
        gui: "CylindraMainWidget",
        project_dir: Path,
        mole_ext: str = ".csv",
        save_landscape: bool = False,
        save_coarse_tomogram: bool = True,
    ) -> "CylindraProject":
        """Construct a project from a widget state."""
        from cylindra._napari import InteractionVector, LandscapeSurface

        _versions = get_versions()
        tomo = gui.tomogram

        # Save path of molecules
        mole_infos = list[MoleculesInfo]()
        for layer in gui.mole_layers:
            mole_infos.append(MoleculesInfo.from_layer(gui, layer, mole_ext))

        # Save paths of landscape
        landscape_infos = list[LandscapeInfo]()
        if save_landscape:
            for layer in gui.parent_viewer.layers:
                if isinstance(layer, LandscapeSurface):
                    landscape_infos.append(LandscapeInfo.from_layer(gui, layer))

        # Save paths of interaction
        interaction_infos = list[InteractionInfo]()
        for layer in gui.parent_viewer.layers:
            if isinstance(layer, InteractionVector):
                interaction_infos.append(InteractionInfo.from_layer(gui, layer))

        orig_path = tomo.metadata.get("orig_path", None)
        if _image_halves := tomo.image_halves:
            path_halves = tuple(i.source for i in _image_halves)
        else:
            path_halves = None
        img_ref_path = gui._reserved_layers.image_data.source
        if tomo.metadata.get("source", None) == img_ref_path:
            img_ref_path = None
        tilt_series_info = _tilt_series_info(tomo, save_coarse_tomogram)
        if (
            tilt_series_info is not None
            and img_ref_path is not None
            and tilt_series_info.reference is not None
            and Path(img_ref_path) == Path(tilt_series_info.reference)
        ):
            img_ref_path = None  # the reference is loaded with the tilt series
        return cls(
            datetime=datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            version=_versions.pop("cylindra", "unknown"),
            dependency_versions=_versions,
            image=orig_path,
            image_relative=as_relative(orig_path, project_dir),
            image_halves=path_halves,
            cache_image=tomo.metadata.get("cache_image", False),
            scale=tomo.scale,
            image_reference=img_ref_path,
            invert=tomo.is_inverted,
            invert_reference=gui._reserved_layers.ref_inverted,
            multiscales=[x[0] for x in tomo.multiscaled],
            tilt_series=tilt_series_info,
            molecules_info=mole_infos,
            landscape_info=landscape_infos,
            interaction_info=interaction_infos,
            missing_wedge=MissingWedge.parse(tomo.tilt),
            project_path=project_dir,
        )

    @classmethod
    def save_gui(
        cls: "type[CylindraProject]",
        gui: "CylindraMainWidget",
        project_dir: Path,
        mole_ext: str = ".csv",
        save_landscape: bool = False,
        save_coarse_tomogram: bool = True,
    ) -> None:
        """Serialize the GUI state to a json file.

        Parameters
        ----------
        gui : CylindraMainWidget
            The main widget from which project model will be constructed.
        project_dir : Path
            The path to the project json file.
        save_coarse_tomogram : bool, default True
            If the tomogram is opened from a tilt series, save its coarse tomogram that
            was reconstructed from the tilt series (not read from a file) in the
            project. Otherwise, it is reconstructed again when the project is opened.
        """
        self = cls.from_gui(
            gui, project_dir, mole_ext, save_landscape, save_coarse_tomogram
        )

        tomo = gui.tomogram
        localprops = tomo.splines.collect_localprops(allow_none=True)
        globalprops = tomo.splines.collect_globalprops(allow_none=True)

        with _prep_save_dir(project_dir) as results_dir:
            if localprops is not None:
                localprops.write_csv(self._localprops_path(results_dir))
            if globalprops is not None:
                globalprops.write_csv(self._globalprops_path(results_dir))
            for i, spl in enumerate(gui.tomogram.splines):
                spl.to_json(results_dir / f"spline-{i}.json")
            for info in (
                self.molecules_info + self.landscape_info + self.interaction_info
            ):
                info.save_layer(gui, results_dir)
            if self.tilt_series is not None:
                _save_tilt_series(tomo, results_dir, save_coarse_tomogram)

            _cfg_json = gui.default_config.json_dumps()
            self._default_spline_config_path(results_dir).write_text(_cfg_json)

            # save macro
            expr = as_main_function(
                gui._format_macro(gui.macro[gui._macro_offset :]),
                imports=[plg.import_statement() for plg in gui._plugins_called],
            )
            self._script_py_path(results_dir).write_text(expr)

            self.project_description = gui.GeneralInfo.project_desc.value

            # dry run metadata serialization
            try:
                json.dumps(
                    gui._project_metadata,
                    default=project_json_encoder,
                )
            except Exception:  # pragma: no cover
                warnings.warn(
                    "Project metadata is not serializable. Skipping.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                self.metadata = gui._project_metadata
            self.to_json(self._project_json_path(results_dir))

    def _to_gui(
        self,
        gui: "CylindraMainWidget | None" = None,
        filter: "ImageFilter | None" = True,
        read_image: bool = True,
        read_reference: bool = True,
        update_config: bool = True,
    ):
        """Update CylindraMainWidget state based on the project model."""
        import impy as ip
        from magicclass.utils import thread_worker

        from cylindra.components import SplineConfig
        from cylindra.widget_utils import fix_reference_scale

        gui = _get_instance(gui)
        if (
            read_reference
            and self.image_reference
            and Path(self.image_reference).exists()
        ):
            path_ref = Path(self.image_reference)  # user-supplied reference image
        else:
            path_ref = None
        with self.open_project() as project_dir:
            _need_compute = read_image and path_ref is None
            tomogram = self.load_tomogram(project_dir, compute=_need_compute)
            macro_expr = extract(self._script_py_path(project_dir).read_text()).args
            cfg_path = project_dir / "default_spline_config.json"
            if cfg_path.exists() and update_config:
                default_config = SplineConfig.from_file(cfg_path)
            else:
                default_config = None

            cb = gui._send_tomogram_to_viewer.with_args(
                tomogram, filt=filter, invert=self.invert
            )
            yield cb
            cb.await_call()
            gui._init_macro_state()

            # we need to make sure that the `ref_inverted` flag will be set to True
            # after the inversion
            gui._reserved_layers.ref_inverted = False
            if path_ref:
                try:
                    img_ref = ip.imread(path_ref)

                    if _need_fix_reference_scale(macro_expr):
                        LOGGER.info("Fixing reference image scale based on the macro.")
                        img_ref = fix_reference_scale(img_ref, tomogram)
                except Exception as e:
                    LOGGER.warning(
                        f"Cannot read reference image at {path_ref.as_posix()}: {e}"
                    )
                    img_ref = None
                else:
                    cb = thread_worker.callback(gui._update_reference_image).with_args(
                        img_ref
                    )
                    yield cb
                    cb.await_call()
                    # filter and invert are needed to be called on the newly opened
                    # user-supplied reference image
                    if filter is not None:
                        yield from gui.filter_reference_image.arun(filter)
                    if self.invert_reference:
                        yield from gui.invert_image.arun(reference_only=True)
            elif tomogram.tilt_series is not None and self.invert_reference:
                # the reference of the tilt series is already shown and filtered
                yield from gui.invert_image.arun(reference_only=True)

            @thread_worker.callback
            def _update_widget():
                if len(tomogram.splines) > 0:
                    gui._update_splines_in_images()
                    with gui.macro.blocked():
                        gui.sample_subtomograms()
                if default_config is not None:
                    gui.default_config = default_config

                gui.macro.extend(macro_expr)

                # load subtomogram analyzer state
                gui.reset_choices()
                gui._need_save = False

            yield _update_widget
            _update_widget.await_call()

            # load molecules and landscapes
            _add_layer = thread_worker.callback(gui.parent_viewer.add_layer)
            with gui._pend_reset_choices():
                for info in (
                    self.molecules_info + self.landscape_info + self.interaction_info
                ):
                    if layer := info.to_layer(gui, project_dir):
                        cb = _add_layer.with_args(layer)
                        yield cb
                        cb.await_call(timeout=10)

            gui._project_metadata = self.metadata

        @thread_worker.callback
        def out():
            # update project description widget
            gui.GeneralInfo.project_desc.value = self.project_description
            gui.reset_choices()

        return out

    def load_spline(
        self,
        idx: int,
        dir: Path | None = None,
        props: bool = True,
    ) -> "CylSpline":
        """Load the spline of the given index.

        >>> spl = project.load_spline(0)  # load the 0-th spline instance

        Parameters
        ----------
        idx : int
            The index of the spline.
        dir : Path, optional
            The project directory. Can be given if the project is already opened.
        props : bool, default True
            Whether to load the properties of the spline.

        Returns
        -------
        CylSpline
            The spline instance.
        """
        from cylindra.components import CylSpline

        if dir is None:
            with self.open_project() as dir:
                return self.load_spline(idx, dir, props)
        spl = CylSpline.from_json(dir / f"spline-{idx}.json")
        if not props:
            return spl
        localprops_path = self._localprops_path(dir)
        globalprops_path = self._globalprops_path(dir)
        if localprops_path.exists():
            _loc = pl.read_csv(localprops_path).filter(pl.col(H.spline_id) == idx)
            _loc = _drop_null_columns(_loc)
        else:
            _loc = pl.DataFrame([])
        if globalprops_path.exists():
            _glob = pl.read_csv(globalprops_path)[idx]
            _glob = _drop_null_columns(_glob)
        else:
            _glob = pl.DataFrame([])

        if H.spl_dist in _loc.columns:
            _loc = _loc.drop(H.spl_dist)
        if H.spl_pos in _loc.columns:
            spl._anchors = _loc[H.spl_pos].to_numpy()
            _loc = _loc.drop(H.spl_pos)
        for c in [H.spline_id, H.pos_id]:
            if c in _loc.columns:
                _loc = _loc.drop(c)
        spl.props.loc = cast_dataframe(_loc)
        spl.props.glob = cast_dataframe(_glob)

        return spl

    def iter_spline_paths(
        self, dir: Path | None = None
    ) -> "Iterable[tuple[int, Path]]":
        """Iterate over the paths of splines and their indices."""
        if dir is None:
            with self.open_project() as dir:
                paths = list(dir.glob("spline-*.json"))
        else:
            paths = list(dir.glob("spline-*.json"))
        # sort by index
        idx_paths = [(int(p.stem.split("-")[1]), p) for p in paths]
        idx_paths.sort(key=lambda x: x[0])
        yield from idx_paths

    def iter_load_splines(
        self,
        dir: Path | None = None,
        drop_columns: bool = True,
    ) -> "Iterable[CylSpline]":
        """Load all splines including its properties iteratively."""
        from cylindra.components import CylSpline

        if dir is None:
            with self.open_project() as dir:
                yield from self.iter_load_splines(dir, drop_columns)
            return

        localprops_path = self._localprops_path(dir)
        globalprops_path = self._globalprops_path(dir)
        if localprops_path.exists():
            _localprops = pl.read_csv(localprops_path)
        else:
            _localprops = None
        if globalprops_path.exists():
            _globalprops = pl.read_csv(globalprops_path)
        else:
            _globalprops = None
        for idx, spl_path in self.iter_spline_paths(dir):
            spl = CylSpline.from_json(spl_path)
            if _localprops is not None:
                _loc = _localprops.filter(pl.col(H.spline_id) == idx)
                _loc = _drop_null_columns(_loc)
                if len(_loc) == 0:
                    _loc = pl.DataFrame([])
            else:
                _loc = pl.DataFrame([])
            if _globalprops is not None:
                _glob = _globalprops.filter(pl.col(H.spline_id) == idx)
                _glob = _drop_null_columns(_glob)
                if len(_glob) == 0:
                    _glob = pl.DataFrame([])
            else:
                _glob = pl.DataFrame([])

            if H.spl_dist in _loc.columns and drop_columns:
                _loc = _loc.drop(H.spl_dist)
            if H.spl_pos in _loc.columns:
                spl._anchors = _loc[H.spl_pos].to_numpy()
                if drop_columns:
                    _loc = _loc.drop(H.spl_pos)
            for c in [H.spline_id, H.pos_id]:
                if c in _loc.columns and drop_columns:
                    _loc = _loc.drop(c)
            spl.props.loc = cast_dataframe(_loc)
            spl.props.glob = cast_dataframe(_glob)
            yield spl

    def iter_load_molecules(
        self, dir: Path | None = None
    ) -> "Iterable[tuple[MoleculesInfo, Molecules]]":
        """Load all molecules iteratively."""
        from acryo import Molecules

        if dir is None:
            with self.open_project() as dir:
                yield from self.iter_load_molecules(dir)
            return

        for info in self.molecules_info:
            path = dir / info.name
            if not path.exists():
                LOGGER.warning(
                    f"Cannot find molecule file: {path.as_posix()}. "
                    "Probably it was moved?"
                )
                continue
            mole = Molecules.from_file(path)
            yield info, mole

    def load_molecules(self, name: str, dir: Path | None = None) -> "Molecules":
        """Load the molecule with the given name.

        >>> mole = project.load_molecules("Mole-0")  # load one with name "Mole-0"

        Parameters
        ----------
        name : str
            Name of the molecules layer.
        dir : Path, optional
            The project directory. Can be given if the project is already opened.

        Returns
        -------
        Molecules
            The molecules instance that matches the given name.
        """
        from acryo import Molecules

        if dir is None:
            with self.open_project() as dir:
                return self.load_molecules(name, dir)
        for info in self.molecules_info:
            if info.name == name or info.stem == name:
                path = dir / info.name
                if not path.exists():
                    raise ValueError(
                        f"Cannot find molecule file: {path.as_posix()}. Probably the "
                        "`dir` parameter is not correct."
                    )
                return Molecules.from_file(path)
        raise ValueError(f"Cannot find molecule with name: {name}.")

    def load_tomogram(
        self,
        dir: Path | None = None,
        compute: bool = True,
    ) -> "CylTomogram":
        """Load the tomogram object of the project.

        Parameters
        ----------
        dir : Path, optional
            Can be given if the project is already opened.
        compute : bool, optional
            Whether to compute the binned tomograms.
        """
        import impy as ip

        from cylindra.components import CylTomogram
        from cylindra.components.tomogram import _tilt_series as _ts

        kwargs = {
            "scale": self.scale,
            "tilt": self.missing_wedge.as_param(),
            # The reference of a tilt series is not a multi-scale image, so this may be
            # empty. Tomograms (and dummies) need a multi-scale image to be shown.
            "binsize": self.multiscales or [1],
        }
        if (ts_info := self.tilt_series) is not None:
            _paths = [p for p in (ts_info.path, ts_info.reference) if p is not None]
            if dir is None and not all(Path(p).is_absolute() for p in _paths):
                # files saved in the project, which may be a tar or zip file
                with self.open_project() as _dir:
                    return self.load_tomogram(_dir, compute=compute)
            ts_path = _resolve_in(ts_info.path, dir)
            if ts_info.reference is None:
                ref_path = None  # reconstructed from the tilt series
            else:
                ref_path = _resolve_in(ts_info.reference, dir)
            if ts_path.exists() and (ref_path is None or ref_path.exists()):
                if ref_path is None:
                    reference = _ts.reconstruct_coarse_tomogram(
                        _ts.read_tilt_series(ts_path, self.scale),
                        binsize=ts_info.reference_binsize,
                        shape=ts_info.reference_shape,
                        tomo_hand=ts_info.tomo_hand,
                    )
                else:
                    reference = ip.imread(ref_path)
                tomo = CylTomogram.from_tilt_series(
                    ts_path,
                    reference,
                    scale=self.scale,
                    reference_binsize=ts_info.reference_binsize,
                    tomo_hand=ts_info.tomo_hand,
                    binsize=ts_info.binsizes,
                    tomogram_center=ts_info.tomogram_center,
                )
                # The files in the project may be temporary (tar or zip). They are
                # saved again when the project is saved.
                tomo.tilt_series.mark_unsaved(
                    tilt_series=_is_in_dir(ts_path, dir),
                    reference=ref_path is None or _is_in_dir(ref_path, dir),
                )
                # files in a tar or zip project are shown as the files in the project
                for key, path in [("source", ts_path), ("reference", ref_path)]:
                    if (_disp := self._archive_member_name(path, dir)) is not None:
                        tomo.metadata[f"display_{key}"] = _disp
                if compute:  # multi-scale images reconstructed from the tilt series
                    for b in self.multiscales:
                        if b not in (x[0] for x in tomo.multiscaled):
                            tomo.add_multiscale(b)
            else:
                missing = ts_path if not ts_path.exists() else ref_path
                LOGGER.warning(
                    f"Cannot find {Path(missing).as_posix()} of the tilt series. Load "
                    "other components only.",
                )
                tomo = CylTomogram.dummy(**kwargs, name="<Tilt series not found>")
        elif self.image is not None:
            if self.cache_image:
                read_path = _config.cache_tomogram(self.image)
            else:
                read_path = Path(self.image)
            if read_path.exists():
                tomo = CylTomogram.imread(
                    path=read_path,
                    **kwargs,
                    compute=compute,
                    image_halves=self.image_halves,
                ).with_cache_info(orig_path=Path(self.image), cached=self.cache_image)
            elif _rpath := self._try_resolve_image_relative():
                tomo = CylTomogram.imread(
                    path=_rpath,
                    **kwargs,
                    compute=compute,
                    image_halves=self.image_halves,
                ).with_cache_info(orig_path=Path(self.image), cached=self.cache_image)
            else:
                LOGGER.warning(
                    f"Cannot find image file: {read_path.as_posix()}. "
                    "Load other components only.",
                )
                tomo = CylTomogram.dummy(**kwargs, name="<Image not found>")
        elif self.image_halves is not None:
            path1, path2 = self.image_halves
            if Path(path1).exists() and Path(path2).exists():
                tomo = CylTomogram.imread_halves(
                    path1,
                    path2,
                    **kwargs,
                    compute=compute,
                )
            else:
                LOGGER.warning(
                    f"Cannot find half tomogram files: {path1}, {path2}. "
                    "Load other components only.",
                )
                tomo = CylTomogram.dummy(
                    scale=self.scale,
                    tilt=self.missing_wedge.as_param(),
                    binsize=kwargs["binsize"],
                    name="<Half tomograms not found>",
                )
        else:
            tomo = CylTomogram.dummy(**kwargs, name="<No image>")
        tomo.splines.extend(self.iter_load_splines(dir))
        return tomo

    def _try_resolve_image_relative(self) -> Path | None:
        if self.image_relative is None or self.project_path is None:
            return None
        return resolve_relative_paths(self.image_relative, self.project_path.parent)

    def _batch_image_path(self) -> Path | None:
        """Path that identifies the tomogram of this project in batch analysis.

        For a tomogram opened from a tilt series, it is the path of the tilt series,
        which may be a file in the project (or in the tar or zip file of the project).
        """
        if (ts_info := self.tilt_series) is None:
            return None if self.image is None else Path(self.image)
        path = Path(ts_info.path)
        if not path.is_absolute() and self.project_path is not None:
            path = Path(self.project_path) / path
        return path

    def load_tilt_series_loader(self) -> "BackProjectionLoader":
        """Create the loader that reconstructs subtomograms from the tilt series."""
        from cylindra.components.tomogram import _tilt_series as _ts

        if (ts_info := self.tilt_series) is None:
            raise ValueError("The project is not opened from a tilt series.")
        if Path(ts_info.path).is_absolute():
            tilt_series = _ts.read_tilt_series(ts_info.path, self.scale)
        else:  # in the project, which may be a tar or zip file
            with self.open_project() as dir:
                path = _resolve_in(ts_info.path, dir)
                tilt_series = _ts.read_tilt_series(path, self.scale)
        return _ts.make_loader(tilt_series, ts_info.tomogram_center, ts_info.tomo_hand)

    def _archive_member_name(self, path: Path | None, dir: Path | None) -> str | None:
        """Name of a file in a tar or zip project, such as "path/to/project.zip:x.json".

        The files in the archive are read from a temporary directory `dir`, so their
        paths are not meaningful.
        """
        if (
            path is None
            or dir is None
            or self.project_path is None
            or Path(self.project_path).suffix not in (".tar", ".zip")
            or not _is_in_dir(path, dir)
        ):
            return None
        member = Path(path).resolve().relative_to(Path(dir).resolve()).as_posix()
        return f"{Path(self.project_path).as_posix()}:{member}"

    def make_project_viewer(self):
        """Build a project viewer widget from this project."""
        from cylindra.project._widgets import ProjectViewer

        pviewer = ProjectViewer()
        pviewer._from_project(self)
        return pviewer

    def make_component_viewer(self):
        """Build a molecules viewer widget from this project."""
        from cylindra.project._widgets import ComponentsViewer

        mviewer = ComponentsViewer()
        mviewer._from_project(self)
        return mviewer

    @contextmanager
    def open_project(self) -> Generator[Path, None, None]:
        """Open the project within this context."""
        if self.project_path is None:
            raise ValueError("Project path is not set.")
        ext = self.project_path.suffix
        if ext == "":
            yield self.project_path

        elif ext in (".tar",):
            import tarfile

            with tempfile.TemporaryDirectory() as tmpdir:
                with tarfile.open(self.project_path) as tar:
                    _tar_extract_all(tar, tmpdir)
                yield Path(tmpdir)

        elif ext in (".zip",):
            import zipfile

            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(self.project_path) as _zip:
                    _zip.extractall(tmpdir)
                yield Path(tmpdir)

        else:
            raise ValueError(f"Unsupported extension {ext}.")

        return None

    def rewrite(self, dir: Path):
        """Rewrite tar/zip file using given temporary directory.

        This method is only used after some mutable operation on the
        project directory.
        """
        if self.project_path is None:
            raise ValueError("Project path is not set.")

        ext = self.project_path.suffix
        if ext == "":
            return

        elif ext in (".tar",):
            import tarfile

            self.project_path.unlink()

            with tarfile.open(self.project_path, mode="w") as tar:
                for file in Path(dir).glob("*"):
                    tar.add(file, arcname=file.name)

        elif ext in (".zip",):
            import zipfile

            self.project_path.unlink()
            with zipfile.ZipFile(self.project_path, mode="w") as zip:
                for file in Path(dir).glob("*"):
                    zip.write(file, arcname=file.name)

        else:
            raise ValueError(f"Unsupported extension {ext}.")

        return None

    def _localprops_path(self, dir: Path) -> Path:
        """Path to the spline local properties file."""
        return dir / "localprops.csv"

    def _globalprops_path(self, dir: Path) -> Path:
        """Path to the spline global properties file."""
        return dir / "globalprops.csv"

    def _default_spline_config_path(self, dir: Path) -> Path:
        """Path to the default spline config file."""
        return dir / "default_spline_config.json"

    def _script_py_path(self, dir: Path) -> Path:
        """Path to the script.py file."""
        return dir / "script.py"

    def _project_json_path(self, dir: Path) -> Path:
        """Path to the project.json file."""
        return dir / "project.json"


def _get_instance(gui: "CylindraMainWidget | None" = None):
    if gui is not None:
        return gui
    from cylindra import instance

    ui = instance()
    if ui is None:
        raise RuntimeError("No CylindraMainWidget GUI found.")
    return ui


TILT_SERIES_FILE = "tilt_series.json"
COARSE_TOMOGRAM_FILE = "coarse_tomogram.mrc"


def _tilt_series_info(
    tomo: "CylTomogram", save_coarse_tomogram: bool = True
) -> TiltSeriesInfo | None:
    if (ts := tomo.tilt_series) is None:
        return None
    # the tilt series and the reference that are not saved in any file (such as the
    # refined ones) are saved in the project directory
    if ts.has_unsaved_tilt_series:
        path = TILT_SERIES_FILE
    else:
        path = tomo.metadata.get("source", None)
    if not ts.has_unsaved_reference:
        reference = tomo.metadata.get("reference", None)
    elif save_coarse_tomogram:
        reference = COARSE_TOMOGRAM_FILE
    else:
        reference = None  # reconstructed again when the project is opened
    if path is None or ts.reference is None:
        warnings.warn(
            "Tilt series or its reference tomogram is not available. The tomogram "
            "cannot be saved in the project.",
            UserWarning,
            stacklevel=2,
        )
        return None
    return TiltSeriesInfo(
        path=path,
        reference=reference,
        reference_binsize=ts.reference_binsize,
        reference_shape=ts.reference.shape,
        tomogram_center=tuple(float(c) for c in ts.tomogram_center),
        tomo_hand=ts.loader.tomo_hand,
        binsizes=ts.binsizes,
    )


def _save_tilt_series(
    tomo: "CylTomogram", results_dir: Path, save_coarse_tomogram: bool = True
) -> None:
    """Save the tilt series and the reference that are not saved in any file."""
    if (ts := tomo.tilt_series) is None:
        return None
    if ts.has_unsaved_tilt_series:
        tilt_series = ts.tilt_series
        if (image_path := tilt_series.image_path) is not None:
            # paths of the tilt images are absolute, as the JSON is in the project
            if isinstance(image_path, tuple):
                image_path = tuple(Path(p).resolve() for p in image_path)
            else:
                image_path = Path(image_path).resolve()
            tilt_series = tilt_series.replace(image_path=image_path)
        (results_dir / TILT_SERIES_FILE).write_text(tilt_series.dump_json())
    if save_coarse_tomogram and ts.has_unsaved_reference and ts.reference is not None:
        ts.reference.imsave(results_dir / COARSE_TOMOGRAM_FILE)
    return None


def _resolve_in(path: "str | Path", dir: Path | None) -> Path:
    """Resolve a path relative to the project directory."""
    path = Path(path)
    if path.is_absolute() or dir is None:
        return path
    return Path(dir) / path


def _is_in_dir(path: Path, dir: Path | None) -> bool:
    """Whether the path is in the directory."""
    if dir is None:
        return False
    return Path(path).resolve().is_relative_to(Path(dir).resolve())


def _drop_null_columns(df: pl.DataFrame) -> pl.DataFrame:
    nrows = df.shape[0]
    to_drop = list[str]()
    for idx, count in enumerate(df.null_count().row(0)):
        if count == nrows:
            to_drop.append(df.columns[idx])
    if to_drop:
        return df.drop(to_drop)
    return df


@contextmanager
def _prep_save_dir(project_path: Path) -> Generator[Path, None, None]:
    ext = project_path.suffix
    if ext == "":
        if not project_path.exists():
            project_path.mkdir()
        yield project_path

    elif ext in (".tar",):
        import tarfile

        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
            with tarfile.open(project_path, mode="w") as tar:
                for file in Path(tmpdir).glob("*"):
                    tar.add(file, arcname=file.name)

    elif ext in (".zip",):
        import zipfile

        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
            with zipfile.ZipFile(project_path, mode="w") as zip:
                for file in Path(tmpdir).glob("*"):
                    zip.write(file, arcname=file.name)

    else:
        raise ValueError(f"Unsupported extension {ext}.")

    return None


def _tar_extract_all(tar: "tarfile.TarFile", path: Path) -> None:
    tar.extractall(path, filter="fully_trusted")


def _need_fix_reference_scale(macro_expr: "list[macrokit.Expr]") -> bool:
    from macrokit import Expr, Head

    for expr in macro_expr:
        if expr.head is not Head.call:
            continue
        _fn, _args, _kwargs = expr.split_call()
        if (
            isinstance(_fn, Expr)
            and _fn.head == Head.getattr
            and _fn.args[1].name == "open_image_with_reference"
        ):
            return _kwargs["fix_reference_scale"].eval()
    return False
