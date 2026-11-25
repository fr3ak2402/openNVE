# Plugins

Plugins are separate packages that add model runtimes (backends), commands and system
information to openNVE. openNVE finds installed plugins automatically; a catalog lists the plugins
that can be installed.

| Plugin | What it adds | Platforms | Repository |
|---|---|---|---|
| `hailo` | Hailo accelerators (Hailo-8, 8L, 10H, 15; Raspberry Pi AI Kit, AI HAT+, AI HAT+ 2): agents with compiled `.hef` models, `opennve hailo`, Hailo in `opennve system` | Linux | [openNVE-hailo](https://github.com/fr3ak2402/openNVE-hailo) |
| `vitis` | DPU of AMD Kria boards (KV260, KR260, K26 / K24 SOMs): agents with compiled Vitis AI `.xmodel` models, `opennve vitis`, the DPU in `opennve system` | Linux (Kria, aarch64) | [openNVE-vitis](https://github.com/fr3ak2402/openNVE-vitis) |

## Use

```sh
opennve plugin list                 # installed and installable plugins
opennve plugin info hailo           # what it adds, how it is installed
opennve plugin install hailo        # install (into the Python of openNVE)
opennve plugin install hailo -- --yes --reboot     # options for the plugin's installer after --
opennve plugin remove hailo         # uninstall
```

```text
Installed plugins:
  hailo        0.1.0      Hailo accelerators (Hailo-8, 8L, 10H, 15): agents with compiled .hef models
Installable plugins (opennve plugin install <name>):
  (all installed)
```

`opennve plugin list --json` gives the same as JSON; `opennve version` lists the installed
plugins with their versions. A plugin that is installed but cannot be loaded (for instance a
missing library) is listed with the error; openNVE keeps working without it.

Install methods (from the catalog):

- **pip**: `pip install <package>` into the Python of openNVE.
- **script**: downloads the plugin's repository (`--ref` branch or tag, `--source DIR` a local
  copy) and runs its install script with that Python (`PYTHON=...`). For plugins that also set
  up the system, like `hailo`: on a Raspberry Pi it enables PCIe Gen 3, installs driver,
  firmware and HailoRT with apt (asks first) and the ready-made `opennve_hailo` package, see
  [Hailo](accelerators.md#hailo); `vitis` installs XRT and the DPU design on a Kria board, see
  [AMD Kria DPU](accelerators.md#amd-kria-dpu).

`remove` uninstalls the plugin's package; system parts that its installer added (drivers, apt
packages) stay.

Agents use the backends of plugins like the built-in ones: `"backend": "auto"` picks them by
the model file (`.hef` → `hailo`, `.xmodel` → `vitis`). A model file of a plugin that is not installed gives a hint:
`.hef models need the plugin hailo: opennve plugin install hailo`.

## Writing a plugin

A plugin is a Python package with an entry point in the group `opennve.plugins`:

```toml
# pyproject.toml of the plugin
[project.entry-points."opennve.plugins"]
onnx = "opennve_onnx.plugin:plugin"
```

The entry point is a `Plugin`, or a function that returns one (then the plugin imports openNVE
only when openNVE loads it, and stays usable without openNVE):

```python
# opennve_onnx/plugin.py
def plugin():
    from opennve.plugins import Plugin

    return Plugin(
        name="onnx",
        description="ONNX models with ONNX Runtime",
        register=register,          # registers backends / tasks (once, when openNVE needs them)
        commands=commands,          # adds `opennve <command>`
        system_info=system_info,    # section "onnx" of opennve.system_info()["plugins"]
        system_lines=system_lines,  # lines of `opennve system`
    )


def register():
    from opennve.agents import register_backend

    register_backend("onnx", create_backend, [".onnx"], threads="planned")


def create_backend(config):         # config: opennve.AgentConfig
    from opennve.agents.backends import Backend, ModelInput
    ...                             # a Backend: input (ModelInput), predict(batch) -> list of arrays


def commands(subparsers):           # argparse subparsers of the opennve command
    command = subparsers.add_parser("onnx", help="show ONNX Runtime")
    command.set_defaults(function=lambda args: print("...") or 0)
```

`register_backend(name, factory, suffixes, threads=..., accelerator=...)`:

| Option | Meaning |
|---|---|
| `suffixes` | model files that `"backend": "auto"` runs with this backend |
| `threads="planned"` | workflows run one model per thread and split the cores between the threads (like TensorFlow Lite); `"self"` (default): the runtime uses the cores itself, one workflow thread |
| `accelerator=True` | the backend always runs on an accelerator: workflows give it 2 threads |

A `Backend` has `input` (`ModelInput(shape, dtype)`), `predict(batch) -> list[np.ndarray]`,
optionally `layout` (`"nhwc"` / `"nchw"`), `device` (`"cpu"`, `"gpu"`, `"npu"`), `device_label`
(shown by `Agent.device_description()`), `batch_size` and `close()`.

To make a plugin installable with `opennve plugin install`, add it to the catalog
(`opennve/plugins/catalog.py`): name, description, package, repository, install method
(`pip` or `script`), platforms, and the backends, model suffixes and commands it adds.
