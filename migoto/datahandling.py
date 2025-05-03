import json
import os
import re
from pathlib import Path


import bmesh
import bpy
from bpy.types import Context, Mesh, Object, Operator
from bpy_extras.io_utils import axis_conversion
from mathutils import Vector

from .datastructures import (
    ConstantBuffer,
    FALogFile,
    Fatal,
    IndexBuffer,
    VBSOMapEntry,
    VertexBufferGroup,
    keys_to_ints,
    keys_to_strings,
)


def new_custom_attribute_int(mesh: Mesh, layer_name: str):
    # vertex_layers were dropped in 4.0. Looks like attributes were added in
    # 3.0 (to confirm), so we could probably start using them or add a
    # migration function on older versions as well
    if bpy.app.version >= (4, 0):
        mesh.attributes.new(name=layer_name, type="INT", domain="POINT")
        return mesh.attributes[layer_name]
    else:
        mesh.vertex_layers_int.new(name=layer_name)
        return mesh.vertex_layers_int[layer_name]


def new_custom_attribute_float(mesh: Mesh, layer_name: str):
    if bpy.app.version >= (4, 0):
        # TODO: float2 and float3 could be stored directly as 'FLOAT2' /
        # 'FLOAT_VECTOR' types (in fact, UV layers in 4.0 show up in attributes
        # using FLOAT2) instead of saving each component as a separate layer.
        # float4 is missing though. For now just get it working equivelently to
        # the old vertex layers.
        mesh.attributes.new(name=layer_name, type="FLOAT", domain="POINT")
        return mesh.attributes[layer_name]
    else:
        mesh.vertex_layers_float.new(name=layer_name)
        return mesh.vertex_layers_float[layer_name]


# TODO: Refactor to prefer attributes over vertex layers even on 3.x if they exist
def custom_attributes_int(mesh: Mesh):
    if bpy.app.version >= (4, 0):
        return {
            k: v
            for k, v in mesh.attributes.items()
            if (v.data_type, v.domain) == ("INT", "POINT")
        }
    else:
        return mesh.vertex_layers_int


def custom_attributes_float(mesh: Mesh):
    if bpy.app.version >= (4, 0):
        return {
            k: v
            for k, v in mesh.attributes.items()
            if (v.data_type, v.domain) == ("FLOAT", "POINT")
        }
    else:
        return mesh.vertex_layers_float


def assert_pointlist_ib_is_pointless(ib: IndexBuffer, vb: VertexBufferGroup):
    # Index Buffers are kind of pointless with point list topologies, because
    # the advantages they offer for triangle list topologies don't really
    # apply and there is little point in them being used at all... But, there
    # is nothing technically stopping an engine from using them regardless, and
    # we do see this in One Piece Burning Blood. For now, just verify that the
    # index buffers are the trivial case that lists every vertex in order, and
    # just ignore them since we already loaded the vertex buffer in that order.
    assert len(vb) == len(ib)  # FIXME: Properly implement point list index buffers
    assert all(
        [(i,) == j for i, j in enumerate(ib.faces)]
    )  # FIXME: Properly implement point list index buffers


# from export_obj:
def mesh_triangulate(me: Mesh):
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()


def find_stream_output_vertex_buffers(log):
    vb_so_map = {}
    for so_draw_call, bindings in log.slot_class["so"].items():
        for so_slot, so in bindings.items():
            # print(so_draw_call, so_slot, so.resource_address)
            # print(list(sorted(log.find_resource_uses(so.resource_address, 'vb'))))
            for vb_draw_call, slot_type, vb_slot in log.find_resource_uses(
                so.resource_address, "vb"
            ):
                # NOTE: Recording the stream output slot here, but that won't
                # directly help determine which VB inputs we need from this
                # draw call (all of them, or just some?), but we might want
                # this slot if we write out an ini file for reinjection
                vb_so_map[VBSOMapEntry(vb_draw_call, vb_slot)] = VBSOMapEntry(
                    so_draw_call, so_slot
                )
    # print(sorted(vb_so_map.items()))
    return vb_so_map


def open_frame_analysis_log_file(dirname: Path) -> FALogFile:
    basename = os.path.basename(dirname)
    if basename.lower().startswith("ctx-0x"):
        context = basename[6:]
        path = os.path.join(dirname, "..", f"log-0x{context}.txt")
    else:
        path = os.path.join(dirname, "log.txt")
    return FALogFile(open(path, "r"))


# Parsing the headers for vb0 txt files
# This has been constructed by the collect script, so its headers are much more accurate than the originals
def parse_buffer_headers(headers, filters):
    results = []
    # https://docs.microsoft.com/en-us/windows/win32/api/dxgiformat/ne-dxgiformat-dxgi_format
    for element in headers.split("]:")[1:]:
        lines = element.strip().splitlines()
        name = lines[0].split(": ")[1]
        index = lines[1].split(": ")[1]
        data_format = lines[2].split(": ")[1]
        bytewidth = (
            sum(
                [
                    int(x)
                    for x in re.findall(
                        "([0-9]*)[^0-9]", data_format.split("_")[0] + "_"
                    )
                    if x
                ]
            )
            // 8
        )

        # A bit annoying, but names can be very similar so need to match filter format exactly
        element_name = name
        if index != "0":
            element_name += index
        if element_name + ":" not in filters:
            continue

        results.append(
            {
                "semantic_name": name,
                "element_name": element_name,
                "index": index,
                "format": data_format,
                "bytewidth": bytewidth,
            }
        )

    return results


def apply_vgmap(
    operator: Operator,
    context: Context,
    targets=None,
    filepath: Path = "",
    commit: bool = False,
    reverse: bool = False,
    suffix="",
    rename: bool = False,
    cleanup: bool = False,
):
    if not targets:
        targets = context.selected_objects

    if not targets:
        raise Fatal("No object selected")

    vgmap = json.load(open(filepath, "r"))

    if reverse:
        vgmap = {int(v): int(k) for k, v in vgmap.items()}
    else:
        vgmap = {k: int(v) for k, v in vgmap.items()}

    for obj in targets:
        if commit:
            raise Fatal("commit not yet implemented")

        prop_name = "3DMigoto:VGMap:" + suffix
        obj[prop_name] = keys_to_strings(vgmap)

        if rename:
            for k, v in vgmap.items():
                if str(k) in obj.vertex_groups.keys():
                    continue
                if str(v) in obj.vertex_groups.keys():
                    obj.vertex_groups[str(v)].name = k
                else:
                    obj.vertex_groups.new(name=str(k))
        if cleanup:
            for vg in obj.vertex_groups:
                if vg.name not in vgmap:
                    obj.vertex_groups.remove(vg)

        if "3DMigoto:VBLayout" not in obj:
            operator.report(
                {"WARNING"},
                "%s is not a 3DMigoto mesh. Vertex Group Map custom property applied anyway"
                % obj.name,
            )
        else:
            operator.report({"INFO"}, "Applied vgmap to %s" % obj.name)


def update_vgmap(operator: Operator, context: Context, vg_step: int = 1):
    if not context.selected_objects:
        raise Fatal("No object selected")

    for obj in context.selected_objects:
        vgmaps = {
            k: keys_to_ints(v)
            for k, v in obj.items()
            if k.startswith("3DMigoto:VGMap:")
        }
        if not vgmaps:
            raise Fatal("Selected object has no 3DMigoto vertex group maps")
        for suffix, vgmap in vgmaps.items():
            highest = max(vgmap.values())
            for vg in obj.vertex_groups.keys():
                if vg.isdecimal():
                    continue
                if vg in vgmap:
                    continue
                highest += vg_step
                vgmap[vg] = highest
                operator.report(
                    {"INFO"}, "Assigned named vertex group %s = %i" % (vg, vgmap[vg])
                )
            obj[suffix] = vgmap


def import_pose(
    operator: Operator,
    context: Context,
    filepath: Path = None,
    limit_bones_to_vertex_groups: bool = True,
    axis_forward="-Z",
    axis_up="Y",
    pose_cb_off: list[int] = [0, 0],
    pose_cb_step: int = 1,
):
    pose_buffer = ConstantBuffer(open(filepath, "r"), *pose_cb_off)

    matrices = pose_buffer.as_3x4_matrices()

    obj = context.object
    if not context.selected_objects:
        obj = None

    if limit_bones_to_vertex_groups and obj:
        matrices = matrices[: len(obj.vertex_groups)]

    name = os.path.basename(filepath)
    arm_data = bpy.data.armatures.new(name)
    arm = bpy.data.objects.new(name, object_data=arm_data)

    conversion_matrix = axis_conversion(
        from_forward=axis_forward, from_up=axis_up
    ).to_4x4()

    context.scene.collection.objects.link(arm)

    # Construct bones (FIXME: Position these better)
    # Must be in edit mode to add new bones
    arm.select_set(True)
    context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    for i, matrix in enumerate(matrices):
        bone = arm_data.edit_bones.new(str(i * pose_cb_step))
        bone.tail = Vector((0.0, 0.10, 0.0))
    bpy.ops.object.mode_set(mode="OBJECT")

    # Set pose:
    for i, matrix in enumerate(matrices):
        bone = arm.pose.bones[str(i * pose_cb_step)]
        matrix.resize_4x4()
        bone.matrix_basis = operator.matmul(
            operator.matmul(conversion_matrix, matrix), conversion_matrix.inverted()
        )

    # Apply pose to selected object, if any:
    if obj is not None:
        mod = obj.modifiers.new(arm.name, "ARMATURE")
        mod.object = arm
        obj.parent = arm
        # Hide pose object if it was applied to another object:
        arm.hide_set(True)


def find_armature(obj: Object) -> Object:
    if obj is None:
        return None
    if obj.type == "ARMATURE":
        return obj
    return obj.find_armature()


def copy_bone_to_target_skeleton(
    context: Context, target_arm: Object, new_name: str, src_bone
):
    is_hidden = target_arm.hide_get()
    is_selected = target_arm.select_get()
    prev_active = context.view_layer.objects.active
    target_arm.hide_set(False)
    target_arm.select_set(True)
    context.view_layer.objects.active = target_arm

    bpy.ops.object.mode_set(mode="EDIT")
    bone = target_arm.data.edit_bones.new(new_name)
    bone.tail = Vector((0.0, 0.10, 0.0))
    bpy.ops.object.mode_set(mode="OBJECT")

    bone = target_arm.pose.bones[new_name]
    bone.matrix_basis = src_bone.matrix_basis

    context.view_layer.objects.active = prev_active
    target_arm.select_set(is_selected)
    target_arm.hide_set(is_hidden)


def merge_armatures(operator: Operator, context: Context):
    target_arm = find_armature(context.object)
    if target_arm is None:
        raise Fatal("No active target armature")
    # print('target:', target_arm)

    for src_obj in context.selected_objects:
        src_arm = find_armature(src_obj)
        if src_arm is None or src_arm == target_arm:
            continue
        # print('src:', src_arm)

        # Create mapping between common bones:
        bone_map = {}
        for src_bone in src_arm.pose.bones:
            for dst_bone in target_arm.pose.bones:
                # Seems important to use matrix_basis - if using 'matrix'
                # and merging multiple objects together, the last inserted bone
                # still has the identity matrix when merging the next pose in
                if src_bone.matrix_basis == dst_bone.matrix_basis:
                    if src_bone.name in bone_map:
                        operator.report(
                            {"WARNING"},
                            "Source bone %s.%s matched multiple bones in the destination: %s, %s"
                            % (
                                src_arm.name,
                                src_bone.name,
                                bone_map[src_bone.name],
                                dst_bone.name,
                            ),
                        )
                    else:
                        bone_map[src_bone.name] = dst_bone.name

        # Can't have a duplicate name, even temporarily, so rename all the
        # vertex groups first, and rename the source pose bones to match:
        orig_names = {}
        for vg in src_obj.vertex_groups:
            orig_name = vg.name
            vg.name = "%s.%s" % (src_arm.name, vg.name)
            orig_names[vg.name] = orig_name

        # Reassign vertex groups to matching bones in target armature:
        for vg in src_obj.vertex_groups:
            orig_name = orig_names[vg.name]
            if orig_name in bone_map:
                print("%s.%s -> %s" % (src_arm.name, orig_name, bone_map[orig_name]))
                vg.name = bone_map[orig_name]
            elif orig_name in src_arm.pose.bones:
                # FIXME: Make optional
                print("%s.%s -> new %s" % (src_arm.name, orig_name, vg.name))
                copy_bone_to_target_skeleton(
                    context, target_arm, vg.name, src_arm.pose.bones[orig_name]
                )
            else:
                print(
                    "Vertex group %s missing corresponding bone in %s"
                    % (orig_name, src_arm.name)
                )

        # Change existing armature modifier to target:
        for modifier in src_obj.modifiers:
            if modifier.type == "ARMATURE" and modifier.object == src_arm:
                modifier.object = target_arm
        src_obj.parent = target_arm
        context.scene.collection.objects.unlink(src_arm)
