from __future__ import annotations

import sys


def install_numpy_pickle_compat() -> None:
    """Install minimal NumPy module aliases for cross-version pickle loading.

    Some saved SB3 model pickles may reference NumPy 2-style internal module
    paths such as ``numpy._core.numeric``. When loading those models under
    NumPy 1.x, the module does not exist even though the underlying objects are
    still available under ``numpy.core``.
    """

    try:
        import numpy as np
        import numpy.core as np_core
        import numpy.core.numeric as np_numeric
    except Exception:
        return

    sys.modules.setdefault("numpy._core", np_core)
    sys.modules.setdefault("numpy._core.numeric", np_numeric)

    try:
        import numpy.core.multiarray as np_multiarray

        sys.modules.setdefault("numpy._core.multiarray", np_multiarray)
    except Exception:
        pass


def build_sb3_custom_objects(config: dict, algorithm: str, env) -> dict:
    """Build custom_objects for robust SB3 model loading across env/version drift."""

    algo = str(algorithm).lower()
    custom_objects = {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "_last_obs": None,
        "_last_original_obs": None,
        "_last_episode_starts": None,
        "ep_info_buffer": None,
        "ep_success_buffer": None,
    }

    if algo == "ppo":
        ppo_cfg = dict(config.get("ppo", {}) or {})
        lr = float(ppo_cfg.get("learning_rate", 3e-4))
        clip = float(ppo_cfg.get("clip_range", 0.2))
        custom_objects["lr_schedule"] = lambda _: lr
        custom_objects["clip_range"] = lambda _: clip
    elif algo == "dqn":
        dqn_cfg = dict(config.get("dqn", {}) or {})
        lr = float(dqn_cfg.get("learning_rate", 1e-4))
        custom_objects["lr_schedule"] = lambda _: lr

    return custom_objects

    try:
        import numpy.core._multiarray_umath as np_multiarray_umath

        sys.modules.setdefault("numpy._core._multiarray_umath", np_multiarray_umath)
    except Exception:
        pass
