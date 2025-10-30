# TensorFlow CPU environment (macOS Intel) — Reproducible setup

This document captures the exact working setup used to run the labs on an Intel mac (x86_64) without GPU. It uses micromamba with conda-forge packages and pins NumPy < 2 to avoid ABI issues.

If you follow these steps, you should end up with a Jupyter kernel called "Python (tf-cpu)" that can run the labs including Conv2D and plotting.

## Target system
- macOS (Intel/x86_64). Verify with: `uname -m` -> `x86_64`
- Python 3.10

Notes:
- On Intel macs, do NOT use `tensorflow-macos` (that is for Apple Silicon arm64). Use conda-forge CPU builds instead.
- TensorFlow 2.15 works with Python 3.10. Older TF (e.g., 2.4) won’t install on 3.10.
- Pin `numpy < 2` to avoid C-ABI breakages with TF 2.15.

## 1) Install micromamba (if not already installed)
You can skip if `micromamba --version` works.

```zsh
# install micromamba to ~/micromamba (from official docs)
/bin/bash -c "$(curl -L https://micro.mamba.pm/install.sh)"
```

## 2) Initialize micromamba for zsh
Add this minimal block to `~/.zshrc` (replace any previous broken init):

```zsh
# micromamba init (minimal, zsh)
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "$($HOME/micromamba/micromamba shell hook -s zsh)"
# fallback for direct binary on PATH
export PATH="$HOME/micromamba:$PATH"
```

Open a new terminal or `source ~/.zshrc`, then verify:

```zsh
micromamba --version
```

## 3) Create the tf-cpu environment
Create an environment with TensorFlow CPU from conda-forge and NumPy < 2.

```zsh
micromamba create -n tf-cpu -c conda-forge \
  python=3.10 tensorflow=2.15 "numpy<2" typing_extensions ipykernel -y
```

Then activate it:

```zsh
micromamba activate tf-cpu
```

Install plotting and imaging deps used in the labs:

```zsh
micromamba install -n tf-cpu -c conda-forge matplotlib scikit-image -y
```

Verified versions from the working state:
- Python 3.10.19
- TensorFlow 2.15.0
- NumPy 1.26.4
- matplotlib 3.10.7
- scikit-image 0.25.2

## 4) Register the Jupyter kernel
Register the environment as a Jupyter kernel so notebooks can use it explicitly:

```zsh
python -m ipykernel install --user --name tf-cpu --display-name "Python (tf-cpu)"
```

In VS Code notebooks, select the kernel "Python (tf-cpu)" from the top-right kernel picker.

## 5) Verify TensorFlow and friends
Sanity check imports and CPU visibility:

```zsh
python - <<'PY'
import tensorflow as tf, numpy as np
import matplotlib
print("TF", tf.__version__, "NP", np.__version__, "MPL", matplotlib.__version__)
print("Devices:", tf.config.list_physical_devices())
from tensorflow.keras.layers import Conv2D
print("Conv2D import OK")
PY
```

Expected output (similar):
- TF 2.15.0, NP 1.26.4, MPL 3.10.7
- Devices: [PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU')]
- "Conv2D import OK"

## 6) VS Code interpreter + Pylance (optional but recommended)
Point VS Code’s Python interpreter to the env so Pylance resolves imports:
- Command Palette → "Python: Select Interpreter"
- Pick: `~/micromamba/envs/tf-cpu/bin/python`

Notebooks can still use the "Python (tf-cpu)" kernel even if the workspace interpreter differs, but aligning both reduces false "missing import" diagnostics.

## 7) Troubleshooting / known pitfalls
- "tensorflow==2.4" on Python 3.10 → won’t install. Use TF 2.15 on Python 3.10.
- "tensorflow-macos" on Intel x86_64 → wrong arch; it’s for arm64.
- NumPy 2.x → may cause ABI/runtime issues with TF 2.15; pin `<2`.
- Missing plotting libs → install inside env: `micromamba install -n tf-cpu -c conda-forge matplotlib scikit-image`.
- Kernel not visible → re-run `python -m ipykernel install --user --name tf-cpu --display-name "Python (tf-cpu)"`.
- Shell not recognizing `micromamba` → ensure the `~/.zshrc` block above and open a new shell.

## 8) Use without activation (handy for CI)
You can run tools in the env without activating it:

```zsh
micromamba run -n tf-cpu python -c "import tensorflow as tf; print(tf.__version__)"
micromamba run -n tf-cpu jupyter kernelspec list | grep tf-cpu
```

## 9) Export the environment (optional)
For portability or CI, export a lock-ish file:

```zsh
micromamba env export -n tf-cpu --no-builds > tf-cpu-env.yaml
```

You can then recreate it on another machine:

```zsh
micromamba env create -n tf-cpu -f tf-cpu-env.yaml
```

---

Completion summary
- Environment: `tf-cpu` (Python 3.10), TensorFlow 2.15 (CPU), NumPy 1.26.x, matplotlib, scikit-image.
- Jupyter kernel: installed as "Python (tf-cpu)" and selectable in VS Code.
- Shell init: minimal micromamba zsh hook in `~/.zshrc`.

This mirrors the first fully working state that ran the labs successfully on macOS Intel.
