import time
from dataclasses import dataclass, field
from pathlib import Path

import bpy
from bpy_extras.io_utils import axis_conversion

from .data.byte_buffer import AbstractSemantic, MigotoFormat
from .data.data_model import DataModelXXMI
from .data.hash_json import HashJsonData
from .data.numpy_mesh import NumpyMesh
from .datahandling import Fatal
from .datastructures import ImportPaths


@dataclass
class ImporterOptions:
    flip_texcoord_v: bool = False
    flip_winding: bool = False
    flip_mesh: bool = False
    flip_normal: bool = False
    load_related: bool = False
    load_related_so_vb: bool = False
    load_buf: bool = True
    load_buf_limit_range: bool = False
    merge_meshes: bool = False
    pose_cb: str | None = None
    pose_cb_off: tuple[int, int] | None = None
    pose_cb_step: int | None = None
    semantic_remap: dict[str, AbstractSemantic] = field(default_factory=dict)
    semantic_remap_idx: dict[str, int] = field(default_factory=dict)
    merge_verts: bool = False
    tris_to_quads: bool = False
    clean_loose: bool = False
    import_paths: set[ImportPaths] = field(default_factory=set)
    axis_forward: str = "Y"
    axis_up: str = "Z"


# TODO: Add support of import of unhandled semantics into vertex attributes
# TODO: Support multiple vertex buffers and pose data
# semantic remapping
class ObjectImporter:
    def import_object(self, operator, context, cfg):
        import_folder: Path = Path(next(iter(cfg.import_paths))[0][0]).parent
        start_time = time.time()
        print(f"Object import started for '{import_folder.stem}' folder")

        imported_objects = []

        try:
            hash_json_data = HashJsonData(import_folder / "hash.json")
        except FileNotFoundError:
            raise Fatal(
                f"Specified folder is missing hash.json! Expected at: {import_folder / 'hash.json'}",
            )
        except Exception as e:
            raise Fatal(f"Failed to load hash.json:\n{e}")

        for vb_paths, ib_path, use_bin, pose_path in cfg.import_paths:
            vb_path = Path(vb_paths[0])
            ib_path = Path(ib_path)
            fmt_path = vb_path.with_suffix(".fmt")
            obj = self.import_component(
                operator,
                context,
                cfg,
                fmt_path,
                ib_path,
                vb_path,
                hash_json_data,
                axis_forward=cfg.axis_forward,
                axis_up=cfg.axis_up,
            )

            imported_objects.append(obj)

        if len(imported_objects) == 0:
            raise Fatal(
                "Specified folder is missing files for components!",
            )

        assert import_folder is not None

        col = bpy.data.collections.new(import_folder.stem)
        for obj in imported_objects:
            col.objects.link(obj)
            context.scene.collection.objects.link(obj)
            self.cleanup_object(context, obj, cfg)
            # if cfg.skip_empty_vertex_groups and cfg.import_skeleton_type == "MERGED":
            #     remove_unused_vertex_groups(context, obj)
        context.scene.collection.children.link(col)

        print(f"Total import time: {time.time() - start_time:.3f}s")

    def import_component(
        self,
        operator,
        context,
        cfg,
        fmt_path: Path,
        ib_path: Path,
        vb_path: Path,
        hash_json_data: HashJsonData,
        axis_forward="Y",
        axis_up="Z",
    ):
        start_time = time.time()
        migoto_format = MigotoFormat.from_paths(fmt_path, ib_path, vb_path)
        if migoto_format.vb_layout is None or migoto_format.format is None:
            raise Fatal(
                f"Specified .fmt file for {fmt_path.stem} is missing vertex buffer layout!",
            )
        numpy_mesh = NumpyMesh.from_paths(migoto_format, vb_path, ib_path, cfg.load_buf)

        if numpy_mesh.vertex_buffer is None or numpy_mesh.index_buffer is None:
            raise Fatal(
                "object_source_folder",
                f"Specified .fmt file for {fmt_path.stem} is missing vertex or index buffer!",
            )

        vg_remap = None
        # if cfg.import_skeleton_type == "MERGED":
        #     component_pattern = re.compile(r".*component[ -_]*([0-9]+).*")
        #     result = component_pattern.findall(fmt_path.name.lower())
        #     if len(result) == 1:
        #         component = extracted_object.components[int(result[0])]
        #         vg_remap = numpy.array(list(component.vg_map.values()))

        mesh = bpy.data.meshes.new(vb_path.stem)
        obj = bpy.data.objects.new(mesh.name, mesh)
        assert mesh is not None and obj is not None

        global_matrix = axis_conversion(
            from_forward=axis_forward, from_up=axis_up
        ).to_4x4()
        obj.matrix_world = global_matrix

        model = DataModelXXMI()
        model.flip_winding = cfg.flip_winding
        model.flip_texcoord_v = cfg.flip_texcoord_v
        model.legacy_vertex_colors = False

        self.set_custom_properties(obj, migoto_format, cfg)
        model.set_data(
            obj,
            mesh,
            numpy_mesh.index_buffer,
            numpy_mesh.vertex_buffer,
            vg_remap,
            mirror_mesh=cfg.flip_mesh,
        )
        self.set_materials(operator, obj, hash_json_data, vb_path.parent)

        num_shapekeys = (
            0
            if mesh.shape_keys is None
            else len(getattr(mesh.shape_keys, "key_blocks", []))
        )

        print(
            f"{fmt_path.stem} import time: {time.time() - start_time:.3f}s ({len(mesh.vertices)} vertices, {len(mesh.loops)} indices, {num_shapekeys} shapekeys)"
        )
        return obj

    def cleanup_object(self, context, obj, cfg):
        obj.select_set(True)
        context.view_layer.objects.active = obj

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        if cfg.merge_verts:
            bpy.ops.mesh.remove_doubles(use_sharp_edge_from_normals=True)
        if cfg.tris_to_quads:
            bpy.ops.mesh.tris_convert_to_quads(
                uvs=True, vcols=True, seam=True, sharp=True, materials=True
            )
        if cfg.clean_loose:
            bpy.ops.mesh.delete_loose()
        bpy.ops.object.mode_set(mode="OBJECT")
        # if pose_path is not None:
        #     import_pose(
        #         operator,
        #         context,
        #         pose_path,
        #         limit_bones_to_vertex_groups=True,
        #         axis_forward=axis_forward,
        #         axis_up=axis_up,
        #         pose_cb_off=pose_cb_off,
        #         pose_cb_step=pose_cb_step,
        #     )
        #     context.view_layer.objects.active = obj
        #

    def set_custom_properties(
        self, obj, migoto_format: MigotoFormat, cfg: ImporterOptions
    ):
        obj["3DMigoto:VBLayout"] = migoto_format.vb_layout.serialise()
        obj["3DMigoto:Topology"] = migoto_format.topology
        # for raw_vb in vb.vbs:
        #     obj["3DMigoto:VB%iStride" % raw_vb.idx] = raw_vb.stride
        obj["3DMigoto:VB0Stride"] = migoto_format.vb_layout.stride
        obj["3DMigoto:FirstVertex"] = migoto_format.first_vertex
        obj["3DMigoto:FlipWinding"] = cfg.flip_winding
        obj["3DMigoto:FlipNormal"] = cfg.flip_normal
        obj["3DMigoto:FlipMesh"] = cfg.flip_mesh
        obj["3DMigoto:IBFormat"] = migoto_format.format.get_format()
        obj["3DMigoto:FirstIndex"] = migoto_format.first_index

    def set_materials(
        self, operator, obj, hash_json_data: HashJsonData, import_dir: Path
    ):
        name = obj.name.split("-vb")[0]
        part = hash_json_data.get_part_by_fullname(name)
        if part is None:
            operator.report(
                {"WARNING"},
                f"Failed to find diffuse texture for {name} in dump folder!",
            )
            return
        diffuse = next(
            (tex for tex in part.textures if tex.name.lower() == "diffuse"),
            None,
        )
        if diffuse is None:
            operator.report(
                {"WARNING"},
                f"Failed to find diffuse texture for {name} in dump folder!!",
            )
            return
        texture_path = import_dir / (name + diffuse.name + diffuse.extension)

        mat_name = part.fullname
        mat = bpy.data.materials.new(mat_name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            tex_image = mat.node_tree.nodes.new("ShaderNodeTexImage")
            new_image = bpy.data.images.load(str(texture_path))
            new_image.alpha_mode = "CHANNEL_PACKED"
            tex_image.image = new_image
            mat.node_tree.links.new(
                bsdf.inputs["Base Color"],
                tex_image.outputs["Color"],
            )
        # TODO: add proper node support for old versions and fall off color to surface in case of errors
        obj.data.materials.append(mat)
        mat_idx = len(obj.data.materials) - 1
        poly_mat_idx = [mat_idx] * len(obj.data.polygons)
        obj.data.polygons.foreach_set("material_index", poly_mat_idx)
