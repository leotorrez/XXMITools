import itertools
import os
import struct

import re
import numpy
from pathlib import Path
from typing import Callable
from glob import glob, escape as glob_escape

import bpy
from bpy.props import BoolProperty, CollectionProperty, StringProperty
from bpy.types import (
    Operator,
    OperatorFileListElement,
    PropertyGroup,
    Context,
    Object,
    Mesh,
)
from bpy_extras.io_utils import (
    ImportHelper,
    orientation_helper,
    unpack_list,
    axis_conversion,
)

from .datahandling import (
    find_stream_output_vertex_buffers,
    open_frame_analysis_log_file,
    apply_vgmap,
    new_custom_attribute_float,
    new_custom_attribute_int,
    assert_pointlist_ib_is_pointless,
    import_pose,
)
from .datastructures import (
    Fatal,
    ImportPaths,
    IOOBJOrientationHelper,
    VBSOMapEntry,
    VertexBufferGroup,
    IndexBuffer,
    vertex_color_layer_channels,
)
from .export_ops import XXMIProperties


def load_3dmigoto_mesh_bin(operator: Operator, vb_paths, ib_paths, pose_path):
    if len(vb_paths) != 1 or len(ib_paths) > 1:
        raise Fatal("Cannot merge meshes loaded from binary files")

    # Loading from binary files, but still need to use the .txt files as a
    # reference for the format:
    ib_bin_path, ib_txt_path = ib_paths[0]

    use_drawcall_range = False
    if hasattr(operator, "load_buf_limit_range"):  # Frame analysis import only
        use_drawcall_range = operator.load_buf_limit_range

    vb = VertexBufferGroup()
    vb.parse_vb_bin(vb_paths[0], use_drawcall_range)

    ib = None
    if ib_bin_path:
        ib = IndexBuffer(open(ib_txt_path, "r"), load_indices=False)
        if ib.used_in_drawcall is False:
            operator.report(
                {"WARNING"},
                "{}: Discarding index buffer not used in draw call".format(
                    os.path.basename(ib_bin_path)
                ),
            )
            ib = None
        else:
            ib.parse_ib_bin(open(ib_bin_path, "rb"), use_drawcall_range)

    return vb, ib, os.path.basename(vb_paths[0][0][0]), pose_path


def load_3dmigoto_mesh(operator: Operator, paths: ImportPaths):
    vb_paths, ib_paths, use_bin, pose_path = zip(*paths)
    pose_path = pose_path[0]

    if use_bin[0]:
        return load_3dmigoto_mesh_bin(operator, vb_paths, ib_paths, pose_path)

    vb = VertexBufferGroup(vb_paths[0])
    # Merge additional vertex buffers for meshes split over multiple draw calls:
    for vb_path in vb_paths[1:]:
        tmp = VertexBufferGroup(vb_path)
        vb.merge(tmp)

    # For quickly testing how importent any unsupported semantics may be:
    # vb.wipe_semantic_for_testing('POSITION.w', 1.0)
    # vb.wipe_semantic_for_testing('TEXCOORD.w', 0.0)
    # vb.wipe_semantic_for_testing('TEXCOORD5', 0)
    # vb.wipe_semantic_for_testing('BINORMAL')
    # vb.wipe_semantic_for_testing('TANGENT')
    # vb.write(open(os.path.join(os.path.dirname(vb_paths[0]), 'TEST.vb'), 'wb'), operator=operator)

    ib = None
    if ib_paths and ib_paths != (None,):
        ib = IndexBuffer(open(ib_paths[0], "r"))
        # Merge additional vertex buffers for meshes split over multiple draw calls:
        for ib_path in ib_paths[1:]:
            tmp = IndexBuffer(open(ib_path, "r"))
            ib.merge(tmp)
        if ib.used_in_drawcall is False:
            operator.report(
                {"WARNING"},
                "{}: Discarding index buffer not used in draw call".format(
                    os.path.basename(ib_paths[0])
                ),
            )
            ib = None

    return vb, ib, os.path.basename(vb_paths[0][0]), pose_path


def normal_import_translation(elem, flip):
    unorm = elem.Format.endswith("_UNORM")
    if unorm:
        # Scale UNORM range 0:+1 to normal range -1:+1
        if flip:
            return lambda x: -(x * 2.0 - 1.0)
        else:
            return lambda x: x * 2.0 - 1.0
    if flip:
        return lambda x: -x
    else:
        return lambda x: x


def import_normals_step1(
    mesh: Mesh,
    data: list,
    vertex_layers,
    operator: Operator,
    translate_normal: Callable,
    flip_mesh: bool,
):
    # Ensure normals are 3-dimensional:
    # XXX: Assertion triggers in DOA6
    if len(data[0]) == 4:
        if [x[3] for x in data] != [0.0] * len(data):
            # raise Fatal('Normals are 4D')
            operator.report(
                {"WARNING"},
                "Normals are 4D, storing W coordinate in NORMAL.w vertex layer. Beware that some types of edits on this mesh may be problematic.",
            )
            vertex_layers["NORMAL.w"] = [[x[3]] for x in data]
    normals = [tuple(map(translate_normal, (x[0], x[1], x[2]))) for x in data]
    normals = [(-(2 * flip_mesh - 1) * x[0], x[1], x[2]) for x in normals]
    # To make sure the normals don't get lost by Blender's edit mode,
    # or mesh.update() we need to set custom normals in the loops, not
    # vertices.
    #
    # For testing, to make sure our normals are preserved let's use
    # garbage ones:
    # import random
    # normals = [(random.random() * 2 - 1,random.random() * 2 - 1,random.random() * 2 - 1) for x in normals]
    #
    # Comment from other import scripts:
    # Note: we store 'temp' normals in loops, since validate() may alter final mesh,
    #       we can only set custom lnors *after* calling it.
    if bpy.app.version >= (4, 1):
        return normals
    mesh.create_normals_split()
    for loop in mesh.loops:
        loop.normal[:] = normals[loop.vertex_index]
    return []


def import_normals_step2(mesh: Mesh, flip_mesh: bool):
    clnors = numpy.zeros(len(mesh.loops) * 3, dtype=numpy.float32)
    mesh.loops.foreach_get("normal", clnors)
    clnors = clnors.reshape((-1, 3))
    clnors[:, 0] *= -(2 * flip_mesh - 1)
    # Not sure this is still required with use_auto_smooth, but the other
    # importers do it, and at the very least it shouldn't hurt...
    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    mesh.normals_split_custom_set(clnors.tolist())
    mesh.use_auto_smooth = (
        True  # This has a double meaning, one of which is to use the custom normals
    )
    # XXX CHECKME: show_edge_sharp moved in 2.80, but I can't actually
    # recall what it does and have a feeling it was unimportant:
    # mesh.show_edge_sharp = True


def import_vertex_groups(mesh: Mesh, obj: Object, blend_indices, blend_weights):
    # assert len(blend_indices) == len(blend_weights), (
    #     "Mismatched blend indices and weights"
    # )
    if blend_indices:
        if len(blend_weights) == 0:
            # If no blend weights are provided, assume uniform weights
            blend_weights = {
                sem_idx: [[1] * len(verts[0])] * len(verts)
                for sem_idx, verts in blend_indices.items()
            }
        # We will need to make sure we re-export the same blend indices later -
        # that they haven't been renumbered. Not positive whether it is better
        # to use the vertex group index, vertex group name or attach some extra
        # data. Make sure the indices and names match:
        num_vertex_groups = (
            max(itertools.chain(*itertools.chain(*blend_indices.values()))) + 1
        )
        for i in range(num_vertex_groups):
            obj.vertex_groups.new(name=str(i))
        for vertex in mesh.vertices:
            for semantic_index in sorted(blend_indices.keys()):
                for i, w in zip(
                    blend_indices[semantic_index][vertex.index],
                    blend_weights[semantic_index][vertex.index],
                ):
                    if w == 0.0:
                        continue
                    obj.vertex_groups[i].add((vertex.index,), w, "REPLACE")


def import_uv_layers(mesh: Mesh, obj: Object, texcoords, flip_texcoord_v: bool):
    for texcoord, data in sorted(texcoords.items()):
        # TEXCOORDS can have up to four components, but UVs can only have two
        # dimensions. Not positive of the best way to handle this in general,
        # but for now I'm thinking that splitting the TEXCOORD into two sets of
        # UV coordinates might work:
        dim = len(data[0])
        if dim == 4:
            components_list = ("xy", "zw")
        elif dim == 3:
            components_list = ("xy", "z")
        elif dim == 2:
            components_list = ("xy",)
        elif dim == 1:
            components_list = ("x",)
        else:
            raise Fatal("Unhandled TEXCOORD%s dimension: %i" % (texcoord, dim))
        cmap = {"x": 0, "y": 1, "z": 2, "w": 3}

        for components in components_list:
            uv_name = "TEXCOORD%s.%s" % (texcoord and texcoord or "", components)
            if hasattr(mesh, "uv_textures"):  # 2.79
                mesh.uv_textures.new(uv_name)
            else:  # 2.80
                mesh.uv_layers.new(name=uv_name)
            blender_uvs = mesh.uv_layers[uv_name]

            # This will assign a texture to the UV layer, which works fine but
            # working out which texture maps to which UV layer is guesswork
            # before the import and the artist may as well just assign it
            # themselves in the UV editor pane when they can see the unwrapped
            # mesh to compare it with the dumped textures:
            #
            # path = textures.get(uv_layer, None)
            # if path is not None:
            #    image = load_image(path)
            #    for i in range(len(mesh.polygons)):
            #        mesh.uv_textures[uv_layer].data[i].image = image

            # Can't find an easy way to flip the display of V in Blender, so
            # add an option to flip it on import & export:
            if len(components) % 2 == 1:
                # 1D or 3D TEXCOORD, save in a UV layer with V=0
                translate_uv = lambda u: (u[0], 0)
            elif flip_texcoord_v:
                translate_uv = lambda uv: (uv[0], 1.0 - uv[1])
                # Record that V was flipped so we know to undo it when exporting:
                obj["3DMigoto:" + uv_name] = {"flip_v": True}
            else:
                translate_uv = lambda uv: uv

            uvs = [[d[cmap[c]] for c in components] for d in data]
            for loop in mesh.loops:
                blender_uvs.data[loop.index].uv = translate_uv(uvs[loop.vertex_index])


# This loads unknown data from the vertex buffers as vertex layers
def import_vertex_layers(mesh: Mesh, obj: Object, vertex_layers):
    for element_name, data in sorted(vertex_layers.items()):
        dim = len(data[0])
        cmap = {0: "x", 1: "y", 2: "z", 3: "w"}
        for component in range(dim):
            if dim != 1 or element_name.find(".") == -1:
                layer_name = "%s.%s" % (element_name, cmap[component])
            else:
                layer_name = element_name

            if type(data[0][0]) is int:
                layer = new_custom_attribute_int(mesh, layer_name)
                for v in mesh.vertices:
                    val = data[v.index][component]
                    # Blender integer layers are 32bit signed and will throw an
                    # exception if we are assigning an unsigned value that
                    # can't fit in that range. Reinterpret as signed if necessary:
                    if val < 0x80000000:
                        layer.data[v.index].value = val
                    else:
                        layer.data[v.index].value = struct.unpack(
                            "i", struct.pack("I", val)
                        )[0]
            elif type(data[0][0]) is float:
                layer = new_custom_attribute_float(mesh, layer_name)
                for v in mesh.vertices:
                    layer.data[v.index].value = data[v.index][component]
            else:
                raise Fatal("BUG: Bad layer type %s" % type(data[0][0]))


def import_faces_from_ib(mesh: Mesh, ib: IndexBuffer, flip_winding: bool):
    mesh.loops.add(len(ib.faces) * 3)
    mesh.polygons.add(len(ib.faces))
    if flip_winding:
        mesh.loops.foreach_set("vertex_index", unpack_list(map(reversed, ib.faces)))
    else:
        mesh.loops.foreach_set("vertex_index", unpack_list(ib.faces))
    mesh.polygons.foreach_set("loop_start", [x * 3 for x in range(len(ib.faces))])
    mesh.polygons.foreach_set("loop_total", [3] * len(ib.faces))


def import_faces_from_vb_trianglelist(
    mesh: Mesh, vb: VertexBufferGroup, flip_winding: bool
):
    # Only lightly tested
    num_faces = len(vb.vertices) // 3
    mesh.loops.add(num_faces * 3)
    mesh.polygons.add(num_faces)
    if flip_winding:
        raise Fatal(
            "Flipping winding order untested without index buffer"
        )  # export in particular needs support
        mesh.loops.foreach_set(
            "vertex_index", [x for x in reversed(range(num_faces * 3))]
        )
    else:
        mesh.loops.foreach_set("vertex_index", [x for x in range(num_faces * 3)])
    mesh.polygons.foreach_set("loop_start", [x * 3 for x in range(num_faces)])
    mesh.polygons.foreach_set("loop_total", [3] * num_faces)


def import_faces_from_vb_trianglestrip(
    mesh: Mesh, vb: VertexBufferGroup, flip_winding: bool
):
    # Only lightly tested
    if flip_winding:
        raise Fatal(
            "Flipping winding order with triangle strip topology is not implemented"
        )
    num_faces = len(vb.vertices) - 2
    if num_faces <= 0:
        raise Fatal("Insufficient vertices in trianglestrip")
    mesh.loops.add(num_faces * 3)
    mesh.polygons.add(num_faces)

    # Every 2nd face has the vertices out of order to keep all faces in the same orientation:
    # https://learn.microsoft.com/en-us/windows/win32/direct3d9/triangle-strips
    tristripindex = [
        (
            i,
            i % 2 and i + 2 or i + 1,
            i % 2 and i + 1 or i + 2,
        )
        for i in range(num_faces)
    ]

    mesh.loops.foreach_set("vertex_index", unpack_list(tristripindex))
    mesh.polygons.foreach_set("loop_start", [x * 3 for x in range(num_faces)])
    mesh.polygons.foreach_set("loop_total", [3] * num_faces)


def import_vertices(
    mesh: Mesh,
    obj: Object,
    vb: VertexBufferGroup,
    operator: Operator,
    semantic_translations={},
    flip_normal: bool = False,
    flip_mesh: bool = False,
):
    mesh.vertices.add(len(vb.vertices))

    blend_indices = {}
    blend_weights = {}
    texcoords = {}
    vertex_layers = {}
    use_normals = False
    normals = []

    for elem in vb.layout:
        if elem.InputSlotClass != "per-vertex" or elem.reused_offset:
            continue

        if elem.InputSlot not in vb.slots:
            # UE4 known to proclaim it has attributes in all the slots in the
            # layout description, but only ends up using two (and one of those
            # is per-instance data)
            print(
                "NOTICE: Vertex semantic %s unavailable due to missing vb%i"
                % (elem.name, elem.InputSlot)
            )
            continue

        translated_elem_name, translated_elem_index = semantic_translations.get(
            elem.name, (elem.name, elem.SemanticIndex)
        )

        # Some games don't follow the official DirectX UPPERCASE semantic naming convention:
        translated_elem_name = translated_elem_name.upper()

        data = tuple(x[elem.name] for x in vb.vertices)
        if translated_elem_name == "POSITION":
            # Ensure positions are 3-dimensional:
            if len(data[0]) == 4:
                if [x[3] for x in data] != [1.0] * len(data):
                    # XXX: There is a 4th dimension in the position, which may
                    # be some artibrary custom data, or maybe something weird
                    # is going on like using Homogeneous coordinates in a
                    # vertex buffer. The meshes this triggers on in DOA6
                    # (skirts) lie about almost every semantic and we cannot
                    # import them with this version of the script regardless.
                    # But perhaps in some cases it might still be useful to be
                    # able to import as much as we can and just preserve this
                    # unknown 4th dimension to export it later or have a game
                    # specific script perform some operations on it - so we
                    # store it in a vertex layer and warn the modder.
                    operator.report(
                        {"WARNING"},
                        "Positions are 4D, storing W coordinate in POSITION.w vertex layer. Beware that some types of edits on this mesh may be problematic.",
                    )
                    vertex_layers["POSITION.w"] = [[x[3]] for x in data]
            positions = [(-(2 * flip_mesh - 1) * x[0], x[1], x[2]) for x in data]
            mesh.vertices.foreach_set("co", unpack_list(positions))
        elif translated_elem_name.startswith("COLOR"):
            if len(data[0]) <= 3 or vertex_color_layer_channels == 4:
                # Either a monochrome/RGB layer, or Blender 2.80 which uses 4
                # channel layers
                mesh.vertex_colors.new(name=elem.name)
                color_layer = mesh.vertex_colors[elem.name].data
                c = vertex_color_layer_channels
                for loop in mesh.loops:
                    color_layer[loop.index].color = list(data[loop.vertex_index]) + [
                        0
                    ] * (c - len(data[loop.vertex_index]))
            else:
                mesh.vertex_colors.new(name=elem.name + ".RGB")
                mesh.vertex_colors.new(name=elem.name + ".A")
                color_layer = mesh.vertex_colors[elem.name + ".RGB"].data
                alpha_layer = mesh.vertex_colors[elem.name + ".A"].data
                for loop in mesh.loops:
                    color_layer[loop.index].color = data[loop.vertex_index][:3]
                    alpha_layer[loop.index].color = [data[loop.vertex_index][3], 0, 0]
        elif translated_elem_name == "NORMAL":
            use_normals = True
            translate_normal = normal_import_translation(elem, flip_normal)
            normals = import_normals_step1(
                mesh, data, vertex_layers, operator, translate_normal, flip_mesh
            )
        elif translated_elem_name in ("TANGENT", "BINORMAL"):
            #    # XXX: loops.tangent is read only. Not positive how to handle
            #    # this, or if we should just calculate it when re-exporting.
            #    for l in mesh.loops:
            #        FIXME: rescale range if elem.Format.endswith('_UNORM')
            #        assert data[l.vertex_index][3] in (1.0, -1.0)
            #        l.tangent[:] = data[l.vertex_index][0:3]
            operator.report(
                {"INFO"},
                "Skipping import of %s in favour of recalculating on export"
                % elem.name,
            )
        elif translated_elem_name.startswith("BLENDINDICES"):
            blend_indices[translated_elem_index] = data
        elif translated_elem_name.startswith("BLENDWEIGHT"):
            blend_weights[translated_elem_index] = data
        elif translated_elem_name.startswith("TEXCOORD") and elem.is_float():
            texcoords[translated_elem_index] = data
        else:
            operator.report(
                {"INFO"},
                "Storing unhandled semantic %s %s as vertex layer"
                % (elem.name, elem.Format),
            )
            vertex_layers[elem.name] = data

    return (
        blend_indices,
        blend_weights,
        texcoords,
        vertex_layers,
        use_normals,
        normals,
    )


def import_3dmigoto_vb_ib(
    operator: Operator,
    context: Context,
    paths: ImportPaths,
    flip_texcoord_v: bool = True,
    flip_winding: bool = False,
    flip_mesh: bool = False,
    flip_normal: bool = False,
    axis_forward="-Z",
    axis_up="Y",
    pose_cb_off=[0, 0],
    pose_cb_step=1,
    merge_verts: bool = False,
    tris_to_quads: bool = False,
    clean_loose: bool = False,
):
    vb, ib, name, pose_path = load_3dmigoto_mesh(operator, paths)

    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(mesh.name, mesh)

    global_matrix = axis_conversion(from_forward=axis_forward, from_up=axis_up).to_4x4()
    obj.matrix_world = global_matrix

    if hasattr(operator.properties, "semantic_remap"):
        semantic_translations = vb.layout.apply_semantic_remap(operator)
    else:
        semantic_translations = vb.layout.get_semantic_remap()

    # Attach the vertex buffer layout to the object for later exporting. Can't
    # seem to retrieve this if attached to the mesh - to_mesh() doesn't copy it:
    obj["3DMigoto:VBLayout"] = vb.layout.serialise()
    obj["3DMigoto:Topology"] = vb.topology
    for raw_vb in vb.vbs:
        obj["3DMigoto:VB%iStride" % raw_vb.idx] = raw_vb.stride
    obj["3DMigoto:FirstVertex"] = vb.first
    # Record these import options so the exporter can set them to match by
    # default. Might also consider adding them to the .fmt file so reimporting
    # a previously exported file can also set them by default?
    obj["3DMigoto:FlipWinding"] = flip_winding
    obj["3DMigoto:FlipNormal"] = flip_normal
    obj["3DMigoto:FlipMesh"] = flip_mesh
    if flip_mesh:
        flip_winding = not flip_winding

    if ib is not None:
        if ib.topology in ("trianglelist", "trianglestrip"):
            import_faces_from_ib(mesh, ib, flip_winding)
        elif ib.topology == "pointlist":
            assert_pointlist_ib_is_pointless(ib, vb)
        else:
            raise Fatal("Unsupported topology (IB): {}".format(ib.topology))
        # Attach the index buffer layout to the object for later exporting.
        obj["3DMigoto:IBFormat"] = ib.format
        obj["3DMigoto:FirstIndex"] = ib.first
    elif vb.topology == "trianglelist":
        import_faces_from_vb_trianglelist(mesh, vb, flip_winding)
    elif vb.topology == "trianglestrip":
        import_faces_from_vb_trianglestrip(mesh, vb, flip_winding)
    elif vb.topology != "pointlist":
        raise Fatal("Unsupported topology (VB): {}".format(vb.topology))
    if vb.topology == "pointlist":
        operator.report(
            {"WARNING"},
            "{}: uses point list topology, which is highly experimental and may have issues with normals/tangents/lighting. This may not be the mesh you are looking for.".format(
                mesh.name
            ),
        )

    (blend_indices, blend_weights, texcoords, vertex_layers, use_normals, normals) = (
        import_vertices(
            mesh, obj, vb, operator, semantic_translations, flip_normal, flip_mesh
        )
    )

    import_uv_layers(mesh, obj, texcoords, flip_texcoord_v)
    if not texcoords:
        operator.report(
            {"WARNING"},
            "{}: No TEXCOORDs / UV layers imported. This may cause issues with normals/tangents/lighting on export.".format(
                mesh.name
            ),
        )

    import_vertex_layers(mesh, obj, vertex_layers)

    import_vertex_groups(mesh, obj, blend_indices, blend_weights)

    # Validate closes the loops so they don't disappear after edit mode and probably other important things:
    mesh.validate(
        verbose=False, clean_customdata=False
    )  # *Very* important to not remove lnors here!
    # Not actually sure update is necessary. It seems to update the vertex normals, not sure what else:
    mesh.update()

    # Must be done after validate step:
    if use_normals:
        if bpy.app.version >= (4, 1):
            mesh.normals_split_custom_set_from_vertices(normals)
        else:
            import_normals_step2(mesh, flip_mesh)
    elif hasattr(mesh, "calc_normals"):  # Dropped in Blender 4.0
        mesh.calc_normals()

    context.scene.collection.objects.link(obj)
    obj.select_set(True)
    context.view_layer.objects.active = obj

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    if merge_verts:
        bpy.ops.mesh.remove_doubles(use_sharp_edge_from_normals=True)
    if tris_to_quads:
        bpy.ops.mesh.tris_convert_to_quads(
            uvs=True, vcols=True, seam=True, sharp=True, materials=True
        )
    if clean_loose:
        bpy.ops.mesh.delete_loose()
    bpy.ops.object.mode_set(mode="OBJECT")
    if pose_path is not None:
        import_pose(
            operator,
            context,
            pose_path,
            limit_bones_to_vertex_groups=True,
            axis_forward=axis_forward,
            axis_up=axis_up,
            pose_cb_off=pose_cb_off,
            pose_cb_step=pose_cb_step,
        )
        context.view_layer.objects.active = obj

    return obj


def import_3dmigoto(
    operator: Operator,
    context: Context,
    paths: ImportPaths,
    merge_meshes: bool = True,
    **kwargs,
):
    if merge_meshes:
        return import_3dmigoto_vb_ib(operator, context, paths, **kwargs)
    else:
        obj = []
        for p in paths:
            try:
                obj.append(import_3dmigoto_vb_ib(operator, context, [p], **kwargs))
            except Fatal as e:
                operator.report({"ERROR"}, str(e) + ": " + str(p[:2]))
        # FIXME: Group objects together
        return obj


def import_3dmigoto_raw_buffers(
    operator: Operator,
    context: Context,
    vb_fmt_path: Path,
    ib_fmt_path: Path,
    vb_path: Path = None,
    ib_path: Path = None,
    vgmap_path: Path = None,
    **kwargs,
):
    paths = (
        ImportPaths(
            vb_paths=list(zip(vb_path, [vb_fmt_path] * len(vb_path))),
            ib_paths=(ib_path, ib_fmt_path),
            use_bin=True,
            pose_path=None,
        ),
    )
    obj = import_3dmigoto(operator, context, paths, merge_meshes=False, **kwargs)
    if obj and vgmap_path:
        apply_vgmap(
            operator,
            context,
            targets=obj,
            filepath=vgmap_path,
            rename=True,
            cleanup=True,
        )


semantic_remap_enum = [
    (
        "None",
        "No change",
        "Do not remap this semantic. If the semantic name is recognised the script will try to interpret it, otherwise it will preserve the existing data in a vertex layer",
    ),
    (
        "POSITION",
        "POSITION",
        "This data will be used as the vertex positions. There should generally be exactly one POSITION semantic for hopefully obvious reasons",
    ),
    (
        "NORMAL",
        "NORMAL",
        "This data will be used as split (custom) normals in Blender.",
    ),
    (
        "TANGENT",
        "TANGENT (CAUTION: Discards data!)",
        "Data in the TANGENT semantics are discarded on import, and recalculated on export",
    ),
    # ('BINORMAL', 'BINORMAL', "Don't encourage anyone to choose this since the data will be entirely discarded"),
    (
        "BLENDINDICES",
        "BLENDINDICES",
        "This semantic holds the vertex group indices, and should be paired with a BLENDWEIGHT semantic that has the corresponding weights for these groups",
    ),
    (
        "BLENDWEIGHT",
        "BLENDWEIGHT",
        "This semantic holds the vertex group weights, and should be paired with a BLENDINDICES semantic that has the corresponding vertex group indices that these weights apply to",
    ),
    (
        "TEXCOORD",
        "TEXCOORD",
        "Typically holds UV coordinates, though can also be custom data. Choosing this will import the data as a UV layer (or two) in Blender",
    ),
    (
        "COLOR",
        "COLOR",
        "Typically used for vertex colors, though can also be custom data. Choosing this option will import the data as a vertex color layer in Blender",
    ),
    (
        "Preserve",
        "Unknown / Preserve",
        "Don't try to interpret the data. Choosing this option will simply store the data in a vertex layer in Blender so that it can later be exported unmodified",
    ),
]


class SemanticRemapItem(PropertyGroup):
    semantic_from: bpy.props.StringProperty(name="From", default="ATTRIBUTE")
    semantic_to: bpy.props.EnumProperty(
        items=semantic_remap_enum, name="Change semantic interpretation"
    )
    # Extra information when this is filled out automatically that might help guess the correct semantic:
    Format: bpy.props.StringProperty(name="DXGI Format")
    InputSlot: bpy.props.IntProperty(name="Vertex Buffer")
    InputSlotClass: bpy.props.StringProperty(name="Input Slot Class")
    AlignedByteOffset: bpy.props.IntProperty(name="Aligned Byte Offset")
    valid: bpy.props.BoolProperty(default=True)
    tooltip: bpy.props.StringProperty(
        default="This is a manually added entry. It's recommended to pre-fill semantics from selected files via the menu to the right to avoid typos"
    )

    def update_tooltip(self):
        if not self.Format:
            return
        self.tooltip = "vb{}+{} {}".format(
            self.InputSlot, self.AlignedByteOffset, self.Format
        )
        if self.InputSlotClass == "per-instance":
            self.tooltip = ". This semantic holds per-instance data (such as per-object transformation matrices) which will not be used by the script"
        elif self.valid is False:
            self.tooltip += ". This semantic is invalid - it may share the same location as another semantic or the vertex buffer it belongs to may be missing / too small"


class ClearSemanticRemapList(Operator):
    """Clear the semantic remap list"""

    bl_idname = "import_mesh.migoto_semantic_remap_clear"
    bl_label = "Clear list"

    def execute(self, context):
        import_operator = context.space_data.active_operator
        import_operator.properties.semantic_remap.clear()
        return {"FINISHED"}


class PrefillSemanticRemapList(Operator):
    """Add semantics from the selected files to the semantic remap list"""

    bl_idname = "import_mesh.migoto_semantic_remap_prefill"
    bl_label = "Prefill from selected files"

    def execute(self, context):
        import_operator = context.space_data.active_operator
        semantic_remap_list = import_operator.properties.semantic_remap
        semantics_in_list = {x.semantic_from for x in semantic_remap_list}

        paths = import_operator.get_vb_ib_paths(load_related=False)

        for p in paths:
            vb, ib, name, pose_path = load_3dmigoto_mesh(import_operator, [p])
            valid_semantics = vb.get_valid_semantics()
            for semantic in vb.layout:
                if semantic.name not in semantics_in_list:
                    remap = semantic_remap_list.add()
                    remap.semantic_from = semantic.name
                    # Store some extra information that can be helpful to guess the likely semantic:
                    remap.Format = semantic.Format
                    remap.InputSlot = semantic.InputSlot
                    remap.InputSlotClass = semantic.InputSlotClass
                    remap.AlignedByteOffset = semantic.AlignedByteOffset
                    remap.valid = semantic.name in valid_semantics
                    remap.update_tooltip()
                    semantics_in_list.add(semantic.name)

        return {"FINISHED"}


@orientation_helper(axis_forward="-Z", axis_up="Y")
class Import3DMigotoFrameAnalysis(Operator, ImportHelper, IOOBJOrientationHelper):
    """Import a mesh dumped with 3DMigoto's frame analysis"""

    bl_idname = "import_mesh.migoto_frame_analysis"
    bl_label = "Import 3DMigoto Frame Analysis Dump"
    bl_options = {"PRESET", "UNDO"}

    filename_ext = ".txt"
    filter_glob: StringProperty(
        default="*.txt",
        options={"HIDDEN"},
    )

    files: CollectionProperty(
        name="File Path",
        type=OperatorFileListElement,
    )

    flip_texcoord_v: BoolProperty(
        name="Flip TEXCOORD V",
        description="Flip TEXCOORD V asix during importing",
        default=True,
    )

    flip_winding: BoolProperty(
        name="Flip Winding Order",
        description="Flip winding order (face orientation) during importing. Try if the model doesn't seem to be shading as expected in Blender and enabling the 'Face Orientation' overlay shows **RED** (if it shows BLUE, try 'Flip Normal' instead). Not quite the same as flipping normals within Blender as this only reverses the winding order without flipping the normals. Recommended for Unreal Engine",
        default=False,
    )

    flip_mesh: BoolProperty(
        name="Flip Mesh",
        description="Mirrors mesh over the X Axis on import, and invert the winding order.",
        default=False,
    )

    flip_normal: BoolProperty(
        name="Flip Normal",
        description="Flip Normals during importing. Try if the model doesn't seem to be shading as expected in Blender and enabling 'Face Orientation' overlay shows **BLUE** (if it shows RED, try 'Flip Winding Order' instead). Not quite the same as flipping normals within Blender as this won't reverse the winding order",
        default=False,
    )

    load_related: BoolProperty(
        name="Auto-load related meshes",
        description="Automatically load related meshes found in the frame analysis dump",
        default=True,
    )

    load_related_so_vb: BoolProperty(
        name="Load pre-SO buffers (EXPERIMENTAL)",
        description="Scans the frame analysis log file to find GPU pre-skinning Stream Output techniques in prior draw calls, and loads the unposed vertex buffers from those calls that are suitable for editing. Recommended for Unity games to load neutral poses",
        default=False,
    )

    load_buf: BoolProperty(
        name="Load .buf files instead",
        description="Load the mesh from the binary .buf dumps instead of the .txt files\nThis will load the entire mesh as a single object instead of separate objects from each draw call",
        default=False,
    )

    load_buf_limit_range: BoolProperty(
        name="Limit to draw range",
        description="Load just the vertices/indices used in the draw call (equivalent to loading the .txt files) instead of the complete buffer",
        default=False,
    )

    merge_meshes: BoolProperty(
        name="Merge meshes together",
        description="Merge all selected meshes together into one object. Meshes must be related",
        default=False,
    )

    pose_cb: StringProperty(
        name="Bone CB",
        description='Indicate a constant buffer slot (e.g. "vs-cb2") containing the bone matrices',
        default="",
    )

    pose_cb_off: bpy.props.IntVectorProperty(
        name="Bone CB range",
        description="Indicate start and end offsets (in multiples of 4 component values) to find the matrices in the Bone CB",
        default=[0, 0],
        size=2,
        min=0,
    )

    pose_cb_step: bpy.props.IntProperty(
        name="Vertex group step",
        description="If used vertex groups are 0,1,2,3,etc specify 1. If they are 0,3,6,9,12,etc specify 3",
        default=1,
        min=1,
    )

    semantic_remap: bpy.props.CollectionProperty(type=SemanticRemapItem)
    semantic_remap_idx: bpy.props.IntProperty(
        name="Semantic Remap",
        description="Enter the SemanticName and SemanticIndex the game is using on the left (e.g. TEXCOORD3), and what type of semantic the script should treat it as on the right",
    )  # Needed for template_list

    merge_verts: BoolProperty(
        name="Merge Vertices",
        description="Merge by distance to remove duplicate vertices",
        default=False,
    )
    tris_to_quads: BoolProperty(
        name="Tris to Quads",
        description="Convert all tris to quads",
        default=False,
    )
    clean_loose: BoolProperty(
        name="Clean Loose",
        description="Remove loose geometry",
        default=False,
    )

    def get_vb_ib_paths(self, load_related=None):
        buffer_pattern = re.compile(
            r"""-(?:ib|vb[0-9]+)(?P<hash>=[0-9a-f]+)?(?=[^0-9a-f=])"""
        )
        vb_regex = re.compile(
            r"""^(?P<draw_call>[0-9]+)-vb(?P<slot>[0-9]+)="""
        )  # TODO: Combine with above? (careful not to break hold type frame analysis)

        dirname = os.path.dirname(self.filepath)
        ret = set()
        if load_related is None:
            load_related = self.load_related

        vb_so_map = {}
        if self.load_related_so_vb:
            try:
                fa_log = open_frame_analysis_log_file(dirname)
            except FileNotFoundError:
                self.report(
                    {"WARNING"},
                    "Frame Analysis Log File not found, loading unposed meshes from GPU Stream Output pre-skinning passes will be unavailable",
                )
            else:
                vb_so_map = find_stream_output_vertex_buffers(fa_log)

        files = set()
        if load_related:
            for filename in self.files:
                match = buffer_pattern.search(filename.name)
                if match is None or not match.group("hash"):
                    continue
                paths = glob(
                    os.path.join(
                        dirname, "*%s*.txt" % filename.name[match.start() : match.end()]
                    )
                )
                files.update([os.path.basename(x) for x in paths])
        if not files:
            files = [x.name for x in self.files]
            if files == [""]:
                raise Fatal("No files selected")

        done = set()
        for filename in files:
            if filename in done:
                continue
            match = buffer_pattern.search(filename)
            if match is None:
                if (
                    filename.lower().startswith("log")
                    or filename.lower() == "shaderusage.txt"
                ):
                    # User probably just selected all files including the log
                    continue
                # TODO: Perhaps don't warn about extra files that may have been
                # dumped out that we aren't specifically importing (such as
                # constant buffers dumped with dump_cb or any buffer dumped
                # with dump=), maybe provided we are loading other files from
                # that draw call. Note that this is only applicable if 'load
                # related' is disabled, as that option effectively filters
                # these out above. For now just changed this to an error report
                # rather than a Fatal so other files will still get loaded.
                self.report(
                    {"ERROR"},
                    'Unable to find corresponding buffers from "{}" - filename did not match vertex/index buffer pattern'.format(
                        filename
                    ),
                )
                continue

            use_bin = self.load_buf
            if not match.group("hash") and not use_bin:
                self.report(
                    {"INFO"},
                    "Filename did not contain hash - if Frame Analysis dumped a custom resource the .txt file may be incomplete, Using .buf files instead",
                )
                use_bin = True  # FIXME: Ask

            ib_pattern = filename[: match.start()] + "-ib*" + filename[match.end() :]
            vb_pattern = filename[: match.start()] + "-vb*" + filename[match.end() :]
            ib_paths = glob(os.path.join(dirname, ib_pattern))
            vb_paths = glob(os.path.join(dirname, vb_pattern))
            done.update(map(os.path.basename, itertools.chain(vb_paths, ib_paths)))

            if vb_so_map:
                vb_so_paths = set()
                for vb_path in vb_paths:
                    vb_match = vb_regex.match(os.path.basename(vb_path))
                    if vb_match:
                        draw_call, slot = map(int, vb_match.group("draw_call", "slot"))
                        so = vb_so_map.get(VBSOMapEntry(draw_call, slot))
                        if so:
                            # No particularly good way to determine which input
                            # vertex buffers we need from the stream-output
                            # pass, so for now add them all:
                            vb_so_pattern = f"{so.draw_call:06}-vb*.txt"
                            glob_result = glob(os.path.join(dirname, vb_so_pattern))
                            if not glob_result:
                                self.report(
                                    {"WARNING"},
                                    f"{vb_so_pattern} not found, loading unposed meshes from GPU Stream Output pre-skinning passes will be unavailable",
                                )
                            vb_so_paths.update(glob_result)
                # FIXME: Not sure yet whether the extra vertex buffers from the
                # stream output pre-skinning passes are best lumped in with the
                # existing vb_paths or added as a separate set of paths. Advantages
                # + disadvantages to each, and either way will need work.
                vb_paths.extend(sorted(vb_so_paths))

            if vb_paths and use_bin:
                vb_bin_paths = [os.path.splitext(x)[0] + ".buf" for x in vb_paths]
                ib_bin_paths = [os.path.splitext(x)[0] + ".buf" for x in ib_paths]
                if all(
                    [
                        os.path.exists(x)
                        for x in itertools.chain(vb_bin_paths, ib_bin_paths)
                    ]
                ):
                    # When loading the binary files, we still need to process
                    # the .txt files as well, as they indicate the format:
                    ib_paths = list(zip(ib_bin_paths, ib_paths))
                    vb_paths = list(zip(vb_bin_paths, vb_paths))
                else:
                    self.report(
                        {"WARNING"},
                        "Corresponding .buf files not found - using .txt files",
                    )
                    use_bin = False

            pose_path = None
            if self.pose_cb:
                pose_pattern = (
                    filename[: match.start()] + "*-" + self.pose_cb + "=*.txt"
                )
                try:
                    pose_path = glob(os.path.join(dirname, pose_pattern))[0]
                except IndexError:
                    pass

            if len(ib_paths) > 1:
                raise Fatal("Error: excess index buffers in dump?")
            elif len(ib_paths) == 0:
                if use_bin:
                    name = os.path.basename(vb_paths[0][0])
                    ib_paths = [(None, None)]
                else:
                    name = os.path.basename(vb_paths[0])
                    ib_paths = [None]
                self.report(
                    {"WARNING"},
                    "{}: No index buffer present, support for this case is highly experimental".format(
                        name
                    ),
                )
            ret.add(ImportPaths(tuple(vb_paths), ib_paths[0], use_bin, pose_path))
        return ret

    def execute(self, context):
        if self.load_buf:
            # Is there a way to have the mutual exclusivity reflected in
            # the UI? Grey out options or use radio buttons or whatever?
            if self.merge_meshes or self.load_related:
                self.report(
                    {"INFO"},
                    "Loading .buf files selected: Disabled incompatible options",
                )
            self.merge_meshes = False
            self.load_related = False

        try:
            keywords = self.as_keywords(
                ignore=(
                    "filepath",
                    "files",
                    "filter_glob",
                    "load_related",
                    "load_related_so_vb",
                    "load_buf",
                    "pose_cb",
                    "load_buf_limit_range",
                    "semantic_remap",
                    "semantic_remap_idx",
                )
            )
            paths = self.get_vb_ib_paths()

            import_3dmigoto(self, context, paths, **keywords)
            xxmi: XXMIProperties = context.scene.xxmi
            if not xxmi.dump_path:
                if os.path.exists(
                    os.path.join(os.path.dirname(self.filepath), "hash.json")
                ):
                    xxmi.dump_path = os.path.dirname(self.filepath)
        except Fatal as e:
            self.report({"ERROR"}, str(e))
        return {"FINISHED"}

    def draw(self, context):
        # Overriding the draw method to disable automatically adding operator
        # properties to options panel, so we can define sub-panels to group
        # options and disable grey out mutually exclusive options.
        pass


@orientation_helper(axis_forward="-Z", axis_up="Y")
class Import3DMigotoRaw(Operator, ImportHelper, IOOBJOrientationHelper):
    """Import raw 3DMigoto vertex and index buffers"""

    bl_idname = "import_mesh.migoto_raw_buffers"
    bl_label = "Import 3DMigoto Raw Buffers"
    # bl_options = {'PRESET', 'UNDO'}
    bl_options = {"UNDO"}

    filename_ext = ".vb;.ib"
    filter_glob: StringProperty(
        default="*.vb*;*.ib",
        options={"HIDDEN"},
    )

    files: CollectionProperty(
        name="File Path",
        type=OperatorFileListElement,
    )

    flip_texcoord_v: BoolProperty(
        name="Flip TEXCOORD V",
        description="Flip TEXCOORD V axis during importing",
        default=True,
    )

    flip_winding: BoolProperty(
        name="Flip Winding Order",
        description="Flip winding order (face orientation) during importing. Try if the model doesn't seem to be shading as expected in Blender and enabling the 'Face Orientation' overlay shows **RED** (if it shows BLUE, try 'Flip Normal' instead). Not quite the same as flipping normals within Blender as this only reverses the winding order without flipping the normals. Recommended for Unreal Engine",
        default=False,
    )

    flip_normal: BoolProperty(
        name="Flip Normal",
        description="Flip Normals during importing. Try if the model doesn't seem to be shading as expected in Blender and enabling 'Face Orientation' overlay shows **BLUE** (if it shows RED, try 'Flip Winding Order' instead). Not quite the same as flipping normals within Blender as this won't reverse the winding order",
        default=False,
    )

    def get_vb_ib_paths(self, filename):
        vb_bin_path = glob(glob_escape(os.path.splitext(filename)[0]) + ".vb*")
        ib_bin_path = os.path.splitext(filename)[0] + ".ib"
        fmt_path = os.path.splitext(filename)[0] + ".fmt"
        vgmap_path = os.path.splitext(filename)[0] + ".vgmap"
        if len(vb_bin_path) < 1:
            raise Fatal("Unable to find matching .vb* file(s) for %s" % filename)
        if not os.path.exists(ib_bin_path):
            ib_bin_path = None
        if not os.path.exists(fmt_path):
            fmt_path = None
        if not os.path.exists(vgmap_path):
            vgmap_path = None
        return (vb_bin_path, ib_bin_path, fmt_path, vgmap_path)

    def execute(self, context):
        # I'm not sure how to find the Import3DMigotoReferenceInputFormat
        # instance that Blender instantiated to pass the values from one
        # import dialog to another, but since everything is modal we can
        # just use globals:
        global migoto_raw_import_options
        migoto_raw_import_options = self.as_keywords(
            ignore=("filepath", "files", "filter_glob")
        )

        done = set()
        dirname = os.path.dirname(self.filepath)
        for filename in self.files:
            try:
                (vb_path, ib_path, fmt_path, vgmap_path) = self.get_vb_ib_paths(
                    os.path.join(dirname, filename.name)
                )
                vb_path_norm = set(map(os.path.normcase, vb_path))
                if vb_path_norm.intersection(done) != set():
                    continue
                done.update(vb_path_norm)

                if fmt_path is not None:
                    import_3dmigoto_raw_buffers(
                        self,
                        context,
                        fmt_path,
                        fmt_path,
                        vb_path=vb_path,
                        ib_path=ib_path,
                        vgmap_path=vgmap_path,
                        **migoto_raw_import_options,
                    )
                else:
                    migoto_raw_import_options["vb_path"] = vb_path
                    migoto_raw_import_options["ib_path"] = ib_path
                    bpy.ops.import_mesh.migoto_input_format("INVOKE_DEFAULT")
            except Fatal as e:
                self.report({"ERROR"}, str(e))
        return {"FINISHED"}


class Import3DMigotoReferenceInputFormat(Operator, ImportHelper):
    bl_idname = "import_mesh.migoto_input_format"
    bl_label = "Select a .txt file with matching format"
    bl_options = {"UNDO", "INTERNAL"}

    filename_ext = ".txt;.fmt"
    filter_glob: StringProperty(
        default="*.txt;*.fmt",
        options={"HIDDEN"},
    )

    def get_vb_ib_paths(self):
        if os.path.splitext(self.filepath)[1].lower() == ".fmt":
            return (self.filepath, self.filepath)

        buffer_pattern = re.compile(
            r"""-(?:ib|vb[0-9]+)(?P<hash>=[0-9a-f]+)?(?=[^0-9a-f=])"""
        )

        dirname = os.path.dirname(self.filepath)
        filename = os.path.basename(self.filepath)

        match = buffer_pattern.search(filename)
        if match is None:
            raise Fatal(
                "Reference .txt filename does not look like a 3DMigoto timestamped Frame Analysis Dump"
            )
        ib_pattern = filename[: match.start()] + "-ib*" + filename[match.end() :]
        vb_pattern = filename[: match.start()] + "-vb*" + filename[match.end() :]
        ib_paths = glob(os.path.join(dirname, ib_pattern))
        vb_paths = glob(os.path.join(dirname, vb_pattern))
        if len(ib_paths) < 1 or len(vb_paths) < 1:
            raise Fatal(
                "Unable to locate reference files for both vertex buffer and index buffer format descriptions"
            )
        return (vb_paths[0], ib_paths[0])

    def execute(self, context):
        global migoto_raw_import_options

        try:
            vb_fmt_path, ib_fmt_path = self.get_vb_ib_paths()
            import_3dmigoto_raw_buffers(
                self, context, vb_fmt_path, ib_fmt_path, **migoto_raw_import_options
            )
        except Fatal as e:
            self.report({"ERROR"}, str(e))
        return {"FINISHED"}
