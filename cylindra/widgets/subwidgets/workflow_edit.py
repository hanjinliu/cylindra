import inspect
from functools import partial
from typing import TYPE_CHECKING, Annotated

from macrokit import Head, Symbol, parse
from macrokit.utils import check_attributes, check_call_args
from magicclass import (
    abstractapi,
    bind_key,
    confirm,
    do_not_record,
    field,
    get_button,
    magicclass,
    nogui,
    set_design,
)
from magicclass.logging import getLogger
from magicclass.types import Path
from magicclass.widgets import CodeEdit

from cylindra import _config
from cylindra.widget_utils import capitalize, get_code_theme
from cylindra.widgets.subwidgets._child_widget import ChildWidget

if TYPE_CHECKING:
    from cylindra.widgets import CylindraMainWidget

_Logger = getLogger("cylindra")


@magicclass(labels=False)
class WorkflowEdit(ChildWidget):
    def _get_workflow_names(self, *_) -> list[str]:
        return [file.stem for file in _config.get_config().list_workflow_paths()]

    filename = field(str).with_choices(_get_workflow_names)
    filename_new = field(str, label="filename")
    code = field(CodeEdit)

    @magicclass(layout="horizontal", widget_type="frame")
    class Buttons(ChildWidget):
        new = abstractapi()
        edit = abstractapi()
        save = abstractapi()
        define_workflow = abstractapi()
        cancel_edit = abstractapi()
        run = abstractapi()

    def __post_init__(self):
        self.code.syntax_highlight("python", theme=get_code_theme(self))
        self.code.read_only = True
        self.filename_new.visible = False
        self.filename_new.native.setPlaceholderText("Enter new workflow filename here")
        self._set_edit_state(False)
        self._set_new_state(False)

    def _init(self):
        self._on_name_change(self.filename.value)
        self.width = 600
        self.height = 400

    @property
    def _workflow_menu(self):
        main = self._get_main()
        return main.OthersMenu.Workflows

    @filename.connect
    def _on_name_change(self, filename: str | None):
        if filename is None:
            return
        self.code.value = _config.workflow_path(filename).read_text()

    @set_design(text=capitalize, location=Buttons)
    def run(self):
        filename = self.filename.value
        fname = self._make_method_name(filename)
        self._workflow_menu[fname].changed()  # run workflow by clicking the menu

    def _set_new_state(self, vis: bool):
        get_button(self.new).visible = not vis
        get_button(self.edit).visible = not vis
        get_button(self.run).enabled = not vis
        get_button(self.delete).enabled = not vis
        get_button(self.define_workflow).visible = vis
        get_button(self.cancel_edit).visible = vis
        self.filename_new.visible = vis
        self.filename.visible = not vis
        if vis:
            self.filename_new.value = ""

    def _set_edit_state(self, vis: bool):
        get_button(self.new).visible = not vis
        get_button(self.edit).visible = not vis
        get_button(self.run).enabled = not vis
        get_button(self.delete).enabled = not vis
        get_button(self.save).visible = vis
        get_button(self.cancel_edit).visible = vis
        self.filename.enabled = not vis

    @set_design(text=capitalize, location=Buttons)
    @bind_key("Ctrl+N")
    def new(self):
        """Create a new workflow script."""
        self.filename.enabled = self.code.read_only = False
        self._set_new_state(True)
        self.code.value = _config.WORKFLOW_TEMPLATE.format("# Write your workflow here")
        self.filename_new.native.setFocus()

    @set_design(text=capitalize, location=Buttons)
    @bind_key("Ctrl+E")
    def edit(self):
        """Edit this workflow script."""
        self.filename.enabled = self.code.read_only = False
        self._set_edit_state(True)

    @set_design(text="Save", location=Buttons)
    def save(self):
        """Save current workflow script."""
        self.define_workflow(filename=self.filename.value, code=self.code.value)

    @set_design(text="Save", location=Buttons)
    def define_workflow(
        self,
        filename: Annotated[str, {"bind": filename_new}],
        code: Annotated[str, {"bind": code}],
    ):
        """Save current workflow script."""
        if filename == "":
            raise ValueError("Filename must be specified.")
        code = normalize_workflow(code, self._get_main())
        path = _config.workflow_path(filename)
        if path.exists():
            old_text: str | None = path.read_text()
        else:
            old_text = None
        path.write_text(code, encoding="utf-8")
        try:
            self.append_workflow(path)
        except Exception as e:
            if old_text:
                path.write_text(old_text, encoding="utf-8")
            else:
                path.unlink(missing_ok=True)
            raise e
        _Logger.print("Workflow saved: " + path.as_posix())
        self.cancel_edit()
        self.filename.value = filename

    @set_design(text="Cancel", location=Buttons)
    def cancel_edit(self):
        """Cancel edit."""
        self.filename.enabled = self.code.read_only = True
        self.reset_choices()
        self._on_name_change(self.filename.value)
        self._set_new_state(False)
        self._set_edit_state(False)

    def _make_method_name(self, path: Path) -> str:
        abs_path = _config.workflow_path(path)
        return f"Run_{hex(hash(abs_path))}"

    @nogui
    def append_workflow(self, path: Path):
        """Append workflow as a widget to the menu."""
        main = self._get_main()
        main_func = _config.get_main_function(path)
        partial_func = partial(main_func, main)
        prms = list(inspect.signature(main_func).parameters.values())[1:]
        partial_func.__signature__ = inspect.Signature(prms)

        fn = set_design(text=f"Run `{path.stem}`")(do_not_record(partial_func))
        fn.__name__ = self._make_method_name(path)
        # Old menu should be removed
        try:
            del self._workflow_menu[fn.__name__]
        except (IndexError, KeyError):
            pass
        return self._workflow_menu.append(fn)

    @set_design(text=capitalize, location=Buttons)
    @confirm(text="Are you sure you want to delete this workflow?")
    def delete(self, name: Annotated[str, {"bind": filename}]):
        """Delete an existing workflow file."""
        path = _config.workflow_path(name)
        if path.exists():
            assert path.suffix == ".py"
            path.unlink()
        else:
            raise FileNotFoundError(f"Workflow file not found: {path.as_posix()}")
        name = self._make_method_name(path)
        for i, action in enumerate(self._workflow_menu):
            if action.name == name:
                del self._workflow_menu[i]
                break
        self.reset_choices()


def normalize_workflow(workflow: str, ui: "CylindraMainWidget") -> str:
    """Normalize the workflow script."""
    workflow = workflow.replace("\t", "    ")
    expr = parse(workflow)
    if errors := check_call_args(expr, {"ui": ui}):
        msg = "".join(f"\n - {s}" for s in errors)
        raise ValueError(f"Method errors found in workflow script: {msg}")
    if errors := check_attributes(expr, {"ui": ui}):
        msg = "".join(f"\n - {s}" for s in errors)
        raise ValueError(f"Attribute errors found in workflow script: {msg}")
    _main_function_found = False
    for line in expr.args:
        if isinstance(line, Symbol):
            continue
        if line.head is Head.function and line.args[0].args[0].name == "main":
            _main_function_found = True
            break

    if not _main_function_found:
        raise ValueError("No main function found in workflow script.")
    return workflow
