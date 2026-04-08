# Installation

## Requirements

- **Python** >= 3.11, < 3.14
- **Rust** toolchain (for building the native tiling engine from source)
- **uv** (recommended) or pip

## 1. Install Rust

muDM includes a Rust extension for high-performance tiling. You need a Rust toolchain to build from source.

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

Verify with `rustc --version` (1.70+ required).

## 2. Install uv (recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager that handles virtual environments and dependency resolution.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 3. Create a virtual environment

```bash
uv venv --python=3.13
source .venv/bin/activate
```

## 4. Install muDM

### From GitHub (development branch)

```bash
uv pip install "mudm @ git+https://github.com/bengtl/microjson.git@feature/3d"
```

This clones the repo, compiles the Rust extension via maturin, and installs the package with all dependencies. The first install takes a few minutes for the Rust compilation.

### From source (editable)

If you want to modify the code:

```bash
git clone https://github.com/bengtl/microjson.git
cd microjson
git checkout feature/3d
uv venv --python=3.13
source .venv/bin/activate
uv pip install -e ".[draco]"
```

The Rust extension is built automatically by maturin during the editable install. After modifying Rust code, rebuild with:

```bash
maturin develop --uv
```

### Optional extras

```bash
# Draco mesh compression support
uv pip install "mudm[draco] @ git+https://github.com/bengtl/microjson.git@feature/3d"
```

## 5. Verify the installation

```bash
python -c "
import json
from mudm.model import MuDM
from mudm._rs import StreamingTileGenerator, StreamingTileGenerator2D

data = {
    'type': 'FeatureCollection',
    'features': [{
        'type': 'Feature',
        'geometry': {'type': 'Point', 'coordinates': [10, 20]},
        'properties': {'label': 'test'},
    }],
}
obj = MuDM.model_validate(data)
gen2d = StreamingTileGenerator2D(min_zoom=0, max_zoom=2)
gen2d.add_geojson(json.dumps(data), (0.0, 0.0, 100.0, 100.0))
gen3d = StreamingTileGenerator(min_zoom=0, max_zoom=2)
print('All checks passed.')
"
```

## Troubleshooting

| Problem | Solution |
| --- | --- |
| `FileNotFoundError: maturin` | Rust is not installed or not on PATH. Run `source "$HOME/.cargo/env"` or reinstall Rust. |
| `PyO3's maximum supported version (3.13)` | Use Python 3.13, not 3.14. PyO3 doesn't support 3.14 yet. |
| `error: can't find Rust compiler` | Run `rustup default stable` to set a default toolchain. |
| Slow first install | Normal — Rust compilation takes 2-4 minutes on first build. Subsequent installs are cached. |
