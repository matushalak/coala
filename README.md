# coala
Self-contained repository for the former `cc/datasets` and `cc/ml` code.

## Layout
- `coala/datasets`: dataset registry and dataset loaders
- `coala/architecture`, `coala/heads`, `coala/pretraining`, `coala/logs`: model code and experiment assets
- `coala/utils`: shared utility package
- `tests`: repository-level tests targeting the `coala` package

## Setup
Create the environment and install the package in editable mode:

```bash
conda env create -f environment.yml
conda activate coala
python -m pip install -e .
```

After that, imports such as `import coala`, `from coala.datasets import get_dataloaders`, and `from coala.pretraining import MAE` should resolve from this repository alone.
