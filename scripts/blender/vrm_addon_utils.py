"""Shared VRM add-on install/enable for Blender helper scripts (4.0 classic + 4.2+)."""
from __future__ import annotations

import os
import traceback


def _vrm_ops_ready() -> bool:
    import bpy

    try:
        export_ops = dir(bpy.ops.export_scene)
    except Exception:
        return False
    return "vrm" in export_ops


def _try_enable_module(module: str) -> bool:
    import addon_utils
    import bpy

    def _on_error(ex: BaseException) -> None:
        print(f"VRM addon enable error ({module}): {ex}")
        traceback.print_exception(type(ex), ex, ex.__traceback__)

    try:
        # Prefer addon_utils.enable — bpy.ops.preferences.addon_enable can
        # return FINISHED even when the module failed to register operators.
        addon_utils.enable(
            module,
            default_set=True,
            persistent=True,
            handle_error=_on_error,
        )
    except Exception as exc:
        print(f"VRM addon enable failed ({module}): {exc}")
        return False

    loaded, loaded_state = addon_utils.check(module)
    if not (loaded and loaded_state):
        print(f"VRM addon '{module}' not loaded (check={addon_utils.check(module)})")
        return False
    if not _vrm_ops_ready():
        print(f"VRM addon '{module}' loaded but export_scene.vrm missing")
        return False
    try:
        bpy.ops.wm.save_userpref()
    except Exception:
        pass
    return True


def try_enable_vrm_addon(zip_path: str) -> bool:
    """
    Install (optional) and enable a VRM import/export add-on.

    Blender 4.0 needs a classic ``bl_info`` zip (e.g. VRM_Addon_for_Blender-4_0_0).
    The UniRig ``add-on-vrm-v2.20.77_modified.zip`` requires Blender >= 4.2 and
    will not load on 4.0 (extension-only / no ``bl_info``).
    """
    import addon_utils
    import bpy

    if _vrm_ops_ready():
        return True

    if zip_path and os.path.isfile(zip_path):
        try:
            bpy.ops.preferences.addon_install(filepath=zip_path, overwrite=True)
        except Exception as exc:
            print(f"VRM addon install skipped: {exc}")

    preferred = (
        "VRM_Addon_for_Blender-release",  # official 4.0.x classic package
        "vrm",  # Blender 4.2+ extension package name
        "io_scene_vrm",  # legacy folder name
        "vrn_importer",
    )
    for mod in preferred:
        if _try_enable_module(mod):
            return True

    for mod in addon_utils.modules():
        name = getattr(mod, "__name__", "") or ""
        if "vrm" not in name.lower():
            continue
        if name in preferred:
            continue
        if _try_enable_module(name):
            return True

    return _vrm_ops_ready()
