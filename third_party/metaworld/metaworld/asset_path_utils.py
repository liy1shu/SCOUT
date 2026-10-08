"""Set of utilities for retrieving asset paths for the environments."""

from __future__ import annotations

from pathlib import Path

_CURRENT_FILE_DIR = Path(__file__).parent.absolute()

ENV_ASSET_DIR_V3 = _CURRENT_FILE_DIR / "assets"


def full_V3_path_for(file_name: str) -> str:
    """Retrieves the full, absolute path for a given V3 asset

    Args:
        file_name: Name of the asset file. Can include subdirectories.

    Returns:
        The full path to the asset file.
    """
    return str(ENV_ASSET_DIR_V3 / file_name)


ENV_ASSET_DIR_V1 = _CURRENT_FILE_DIR / "assets_v1"
ENV_ASSET_DIR_V2 = _CURRENT_FILE_DIR / "assets_v2"


def full_v1_path_for(file_name):
    return str(ENV_ASSET_DIR_V1 / file_name)


def full_v2_path_for(file_name):
    return str(ENV_ASSET_DIR_V2 / file_name)


def write_generated_xml(env, xml: str, directory: str, prefix: str) -> str:
    """Write an env's generated MJCF to a file private to that env instance.

    Envs that build their XML at construction time (random centre-of-mass bar
    etc.) used to write it to one fixed path, so two envs created at the same
    time (parallel processes, or pick-bar and push-bar side by side) could load
    each other's file.  The file stays in ``directory`` because the XML uses
    relative ``<include>`` paths, and is removed when the process exits.
    """
    import atexit
    import os
    import tempfile

    path = getattr(env, "_generated_xml_path", None)
    if path is None:
        fd, path = tempfile.mkstemp(prefix=f".{prefix}_", suffix=".xml", dir=directory)
        os.close(fd)
        atexit.register(lambda p=path: os.path.exists(p) and os.remove(p))
        env._generated_xml_path = path
    with open(path, "w") as f:
        f.write(xml)
    return path
