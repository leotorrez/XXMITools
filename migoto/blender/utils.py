import time
from collections import OrderedDict
from typing import TypedDict

import bpy
import numpy as np
from bpy.types import Context, Key, Mesh, Modifier, Object, ShapeKey
from numpy.typing import NDArray


class Shapekey_Properties(TypedDict):
    slider_min: float
    slider_max: float
    value: float
    mute: bool
    vertex_group: str
    relative_key: str | None


def apply_modifiers_to_shapekey_objects(
    context: Context,
    base_obj: Object,
    modifiers_to_apply: list[str],
    shapekeys_to_skip: list[str] | None = None,
) -> None:
    """Applies modifiers to objects with shapekeys.

    Shapekeys named in ``shapekeys_to_skip`` (e.g. the DEFORM/CUSTOM keys that
    the exporter writes as deltas) are left untouched: modifiers are not baked
    into them and their original data is restored instead, so they keep the
    coordinates they are exported from.
    """
    skip: set[str] = set(shapekeys_to_skip or ())
    start_time = time.time()
    assert base_obj.type == "MESH" and isinstance(base_obj.data, Mesh), (
        "Invalid mesh object."
    )

    shape_keys: Key | None = base_obj.data.shape_keys
    if shape_keys is None or not modifiers_to_apply:
        return base_obj

    # Backup shape key settings & modifier visibility
    sk_settings: OrderedDict[str, Shapekey_Properties] = backup_shape_key_settings(
        shape_keys
    )

    modifier_visibility: dict[str, bool] = {}
    for m in base_obj.modifiers:
        modifier_visibility[m.name] = m.show_viewport
        m.show_viewport = m.name in modifiers_to_apply

    # We setup basis_obj to return
    key_blocks = shape_keys.key_blocks
    for b in key_blocks:
        b.value = 0.0

    depsgraph = context.evaluated_depsgraph_get()
    assert depsgraph is not None, "Failed to get evaluated depsgraph."
    eval_obj = base_obj.evaluated_get(depsgraph)
    result_obj = base_obj.copy()
    assert result_obj is not None, "Failed to create new mesh data."
    result_obj.data = bpy.data.meshes.new_from_object(eval_obj)
    assert result_obj.data is not None, "Failed to create new mesh data."

    result_obj.modifiers.clear()
    _ = result_obj.shape_key_add(name="Basis", from_mix=False)

    # Apply modifiers to virtual meshes for efficiency
    vertex_count = len(result_obj.data.vertices)
    for i, block in enumerate(key_blocks):
        if i == 0:
            continue  # Skip Basis shape key

        if block.name in skip:
            # Export-target shapekeys are written as deltas from the original
            # mesh, so baking modifiers into them would corrupt the export.
            # Re-add them untouched to keep a complete shapekey set.
            result_block = result_obj.shape_key_add(
                name=block.name, from_mix=False
            )
            if len(result_block.data) == len(block.data):
                coords: NDArray = np.empty(len(block.data) * 3, dtype=np.float32)
                block.data.foreach_get("co", coords)
                result_block.data.foreach_set("co", coords)
            continue

        block.value = 1.0
        depsgraph.update()
        mesh: Mesh = base_obj.evaluated_get(depsgraph).to_mesh()
        block.value = 0.0

        assert result_obj.data.shape_keys is not None, (
            "Result object has no shape keys."
        )
        result_obj.shape_key_add(name=mesh.name, from_mix=False)

        coords: NDArray = np.zeros((len(mesh.vertices), 3), dtype=np.float32)
        mesh.vertices.foreach_get("co", coords.ravel())
        if vertex_count != len(mesh.vertices):
            raise ValueError(
                "During the process of applying modifiers to shape keys, the vertex count changed. "
                f"Context: {vertex_count} vs {len(mesh.vertices)} in shape key '{block.name}'"
            )
        result_obj.data.shape_keys.key_blocks[mesh.name].data.foreach_set(
            "co", coords.ravel()
        )
        base_obj.to_mesh_clear()

    # Restore shape key settings for original and new object
    restore_shape_key_settings(base_obj.data.shape_keys, sk_settings)
    restore_shape_key_settings(result_obj.data.shape_keys, sk_settings)

    # Restore modifier visibility
    for mod_name, visible in modifier_visibility.items():
        if mod_name in modifiers_to_apply:
            mod: Modifier = base_obj.modifiers[mod_name]
            base_obj.modifiers.remove(mod)
            continue
        base_obj.modifiers[mod_name].show_viewport = visible

    base_obj.data = result_obj.data
    base_obj = result_obj
    total_time = time.time() - start_time
    print(f"Applied modifiers in {total_time:.4f} seconds")


def backup_shape_key_settings(shape_keys) -> OrderedDict[str, Shapekey_Properties]:
    """Backup shape key settings into an ordered dictionary"""
    sk_settings: OrderedDict[str, Shapekey_Properties] = OrderedDict()
    key_blocks = shape_keys.key_blocks
    for block in key_blocks:
        sk_settings[block.name] = {
            "slider_min": block.slider_min,
            "slider_max": block.slider_max,
            "value": block.value,
            "mute": block.mute,
            "vertex_group": block.vertex_group,
            "relative_key": block.relative_key.name if block.relative_key else None,
        }
    return sk_settings


def restore_shape_key_settings(
    shape_keys: Key, sk_settings: OrderedDict[str, Shapekey_Properties]
) -> None:
    """Restore shape key settings from an ordered dictionary"""
    key_blocks = shape_keys.key_blocks
    for i, sk_name in enumerate(sk_settings.keys()):
        settings: Shapekey_Properties = sk_settings[sk_name]
        key_block: ShapeKey = key_blocks[i]
        key_block.name = sk_name
        key_block.slider_min = settings["slider_min"]
        key_block.slider_max = settings["slider_max"]
        key_block.value = settings["value"]
        key_block.mute = settings["mute"]
        key_block.vertex_group = settings["vertex_group"]
        if settings["relative_key"]:
            key_block.relative_key = key_blocks[settings["relative_key"]]
