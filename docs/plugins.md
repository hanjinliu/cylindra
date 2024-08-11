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
my_first_plugin/
├── LICENSE
├── pyproject.toml
├── README.md
├── src/
│   └── my_first_plugin/
│       ├── __init__.py
:       :
│
└── tests/
```

One of the easiest ways to create a package is to use the [`hatch`](https://hatch.pypa.io/latest/) command. It automatically creates the necessary files and directories for you.

```bash
pip install hatch -U
hatch new my_first_plugin
```

### Step 2. Define Plugin Functions

```
my_first_plugin/
:
├── src/
│   └── my_first_plugin/
│       ├── __init__.py
:       └── main.py
```

When `cylindra` looks for the plugins, it searches for the variables defined under your
modules.

- If your plugins are to be defined under the namespace `my_first_plugin`, you should
  expose all the plugin functions in the `__init__.py` file.
- Plugin functions should be decorated with `cylindra.plugin.register_function`.

Let's start with a simple plugin function that prints a message and a random array.

``` python title="my_first_plugin/__init__.py"
from .main import my_plugin_function
```

``` python title="my_first_plugin/main.py"
import numpy as np
from cylindra.plugin import register_function

@register_function
def my_plugin_function(ui):
    ui.logger.print("My first plugin!")
    ui.logger.print(np.random.rand(5))
```

### Step 3. Metadata

The "pyproject.toml" file describes the metadata of the package. The mandatory fields
are

```toml
[project]
dependencies = ...

[project.entry-points."cylindra.plugin"]
...
```

The `dependencies` field under the `project` section should include the dependencies
required for your plugin. All the packages imported in your package should be listed.

Under the `project.entry-points."cylindra.plugin"` section, you should define the
display name and the location of the plugin functions. Fields listed here will be used
to search for the plugin functions.

In this example, this section will be as follows:

```toml
[project]
dependencies = [
    "cylindra",
    "numpy",
]

[project.entry-points."cylindra.plugin"]
"Run my first plugin" = "my_first_plugin"
```

### Step 4. Install the Plugin

Now, your package is ready to be installed by Python package manager.

```bash
pip install -e my_first_plugin
```

.. note::
    The `-e` option installs the package in the editable mode, so you can modify the
    plugin functions without reinstalling the package. This is very useful during the
    development. However, if you modified the pyproject.toml itself, you'll have to
    reinstall the package.

```
