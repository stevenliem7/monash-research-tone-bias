"""Load :mod:`config` from any script depth without a pre-set ``PYTHONPATH``.

Typical use at the top of a script (before ``from config import ...``)::

    import importlib.util
    from pathlib import Path

    for _p in Path(__file__).resolve().parents:
        _loader = _p / "load_config.py"
        if _loader.is_file():
            _spec = importlib.util.spec_from_file_location("load_config", _loader)
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            _mod.bootstrap(__file__)
            break

With ``uv run``, setting ``PYTHONPATH=src`` in ``.env`` (see ``.env.example``) is
enough and you can import ``config`` directly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def bootstrap(caller: str | Path) -> ModuleType:
    """Insert ``src/`` on ``sys.path`` and return the ``config`` module.

    Args:
        caller: Typically ``__file__`` of the calling script.

    Returns:
        The loaded ``config`` module.

    Raises:
        RuntimeError: If ``config.py`` cannot be located above the caller.
    """
    for parent in Path(caller).resolve().parents:
        config_file = parent / "config.py"
        if not config_file.is_file():
            continue
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        if "config" in sys.modules:
            return sys.modules["config"]
        spec = importlib.util.spec_from_file_location("config", config_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {config_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["config"] = module
        spec.loader.exec_module(module)
        return module
    raise RuntimeError(f"Could not locate config.py above {caller}")
