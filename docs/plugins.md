# The Plugin System

With [workflows](workflows.md), you can automate your tasks using combination of
methods defined in `cylindra`. However, you may need some additional methods (e.g.
particle detection based on neural network or custom subtomogram alignment methods) that
you want to use during sessions.

The `cylindra.plugin` submodules provide interfaces to register plugins. The registered
plugins will be available in the "Plugin" menu and will be recorded in the macro, thus
works as other methods in `cylindra`.

## Your First Plugin

Plugins must be defined in a python package, as they must be importable by the
`from XXX import YYY` statement.

### Step 1. Make a Package

The file structure of a package should be as follows:

```
my-first-plugin/
├── LICENSE
├── pyproject.toml
├── README.md
├── my_first_plugin/
│   ├── __init__.py
:   :
│
└── tests/
```

The easiest way to create a package is to use the `cylindra plugin` command. It uses
[cookiecutter](https://github.com/cookiecutter/cookiecutter) to automatically create the
necessary files and directories for you.

```bash
pip install cookiecutter -U
cylindra plugin new .
```

??? note "Other ways?"
    Alternatively, you can also use [`hatch`](https://hatch.pypa.io/latest/).

    ```bash
    pip install hatch -U
    hatch new my-first-plugin
    ```

!!! note
    If your plugin needs special build steps, such as compiling C or Rust code, you will
    need to rewrite the `pyproject.toml` file.

### Step 2. Define Plugin Functions

```
my-first-plugin/
:
├── my_first_plugin/
│   ├── __init__.py
:   └── core.py
```

When `cylindra` looks for the plugins, it searches for the variables defined under your
modules.

- If your plugins are to be defined under the namespace `my_first_plugin`, you should
  expose all the plugin functions in the `__init__.py` file.
- Plugin functions should be decorated with `cylindra.plugin.register_function`.

Let's start with a simple plugin function that prints a message and a random array.

``` python title="my_first_plugin/__init__.py"
from .core import my_plugin_function
```

``` python title="my_first_plugin/core.py"
import numpy as np
from cylindra.plugin import register_function

@register_function
def my_plugin_function(ui):
    ui.logger.print("My first plugin!")
    ui.logger.print(np.random.rand(5))
```

### Step 3. Metadata

The "pyproject.toml" file describes the metadata of the package. The mandatory fields
are alreadly filled by the `cylindra plugin new` command, but there are still some
fields that are needed to be updated.

If your plugin depends on other packages, you should list them in the `dependencies`.
Because `cylindra` plugins always depend on `cylindra`, `"cylindra"` is already included
by default. If there are others, add them like below:

```toml
[project]
dependencies = [
    "cylindra",
    "numpy>=2.1.0",
]
```

### Step 4. Install the Plugin

Now, your package is ready to be installed by Python package manager.

```bash
pip install -e my_first_plugin
```

??? note "The `-e` option"
    The `-e` option installs the package in the editable mode, so you can modify the
    plugin functions without reinstalling the package. This is very useful during the
    development. However, if you modified the pyproject.toml itself, you'll have to
    reinstall the package.
