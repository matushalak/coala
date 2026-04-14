from __future__ import annotations

import importlib

_EXPORTS = {
    "MAE": "coala.pretraining.MAEmodel:MAE",
    "JEPA": "coala.pretraining.JEPAmodel:JEPA",
    "LeJEPA": "coala.pretraining.LeJEPAmodel:LeJEPA",
    "COALA": "coala.pretraining.COALA:COALA",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, attr_name = _EXPORTS[name].split(":")
    module = importlib.import_module(module_path)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
