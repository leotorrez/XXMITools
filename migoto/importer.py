import time
from dataclasses import dataclass, field
from pathlib import Path

import bpy
from bpy_extras.io_utils import axis_conversion

from .data.byte_buffer import AbstractSemantic, MigotoFormat
from .data.data_model import DataModelXXMI
from .data.hash_json import HashJsonData
from .data.numpy_mesh import NumpyMesh, NumpyMeshGroup
from .datahandling import Fatal
from .datastructures import ImportPaths


@dataclass
class ImporterOptions:
    flip_texcoord_v: bool = False
    flip_winding: bool = False
    flip_mesh: bool = False
    flip_normal: bool = False
    create_materials: bool = False
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

        try:
            hash_json_data = HashJsonData(import_folder / "hash.json")
        except FileNotFoundError:
            raise Fatal(
                f"Specified folder is missing hash.json! Expected at: {import_folder / 'hash.json'}",
            )

        imported_objects = self.process_objects(
            operator, context, cfg, hash_json_data, import_folder
        )
        if len(imported_objects) == 0:
            raise Fatal(
                "Specified folder is missing files for components!",
            )

        assert import_folder is not None

        col = bpy.data.collections.new(import_folder.stem)
        context.scene.collection.children.link(col)
        for obj in imported_objects:
            col.objects.link(obj)
            self.cleanup_object(operator, context, obj, cfg)
            # if cfg.skip_empty_vertex_groups and cfg.import_skeleton_type == "MERGED":
            #     remove_unused_vertex_groups(context, obj)

        print(f"Total import time: {time.time() - start_time:.3f}s")

    def process_objects(
        self, operator, context, cfg, hash_json_data, import_folder: Path
    ) -> list[bpy.types.Object]:
        imported_objects = []

        grouped_paths: dict[str, set[ImportPaths]] = {}
        if cfg.merge_meshes:
            for component in reversed(hash_json_data.components):
                fullname = component.fullname
                grouped_paths[fullname] = set(
                    x
                    for x in cfg.import_paths
                    if Path(x.ib_paths).stem.startswith(fullname)
                )
                if len(grouped_paths[fullname]) == 0:
                    del grouped_paths[fullname]
        else:
            for paths in cfg.import_paths:
                ib_path = Path(paths.ib_paths)
                fullname = ib_path.stem.split("-ib")[0]
                if fullname not in grouped_paths:
                    grouped_paths[fullname] = set()
                grouped_paths[fullname].add(paths)

        for fullname, paths in grouped_paths.items():
            obj = self.import_component(
                operator,
                context,
                cfg,
                grouped_paths[fullname],
                hash_json_data,
                fullname,
                axis_forward=cfg.axis_forward,
                axis_up=cfg.axis_up,
            )

            imported_objects.append(obj)

        return imported_objects

    def import_component(
        self,
        operator,
        context,
        cfg,
        paths: set[ImportPaths],
        hash_json_data: HashJsonData,
        name: str,
        axis_forward="Y",
        axis_up="Z",
    ):
        start_time = time.time()
        numpy_mesh_group: NumpyMeshGroup = NumpyMeshGroup()
        migoto_format: MigotoFormat | None = None

        face_counts: dict[ImportPaths, int] = {}
        for p in paths:
            vb_paths, ib_path, _, _ = p
            vb_path = Path(vb_paths[0])
            fmt_path = vb_path.with_suffix(".fmt")
            migoto_format = MigotoFormat.from_paths(fmt_path, ib_path, vb_path)
            if migoto_format.vb_layout is None or migoto_format.format is None:
                raise Fatal(
                    f"Specified .fmt file for {fmt_path.stem} is missing vertex buffer layout!",
                )
            numpy_mesh_group.add_mesh(
                NumpyMesh.from_paths(migoto_format, vb_path, ib_path, fmt_path)
            )

        if migoto_format is None:
            raise Fatal(f"Failed to parse a format for component {name}!")

        vg_remap = None
        # if cfg.import_skeleton_type == "MERGED":
        #     component_pattern = re.compile(r".*component[ -_]*([0-9]+).*")
        #     result = component_pattern.findall(fmt_path.name.lower())
        #     if len(result) == 1:
        #         component = extracted_object.components[int(result[0])]
        #         vg_remap = numpy.array(list(component.vg_map.values()))

        mesh = bpy.data.meshes.new(name)
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
        if cfg.create_materials:
            self.set_materials(operator, obj, name, cfg, hash_json_data)
        model.set_data(
            obj,
            mesh,
            numpy_mesh_group,
            vg_remap,
            mirror_mesh=cfg.flip_mesh,
        )

        num_shapekeys = (
            0
            if mesh.shape_keys is None
            else len(getattr(mesh.shape_keys, "key_blocks", []))
        )

        print(
            f"{name} import time: {time.time() - start_time:.3f}s ({len(mesh.vertices)} vertices, {len(mesh.loops)} indices, {num_shapekeys} shapekeys)"
        )
        return obj

    def cleanup_object(self, operator, context, obj, cfg):
        if cfg.merge_meshes and not cfg.clean_loose:
            operator.report(
                {"WARNING"},
                "Mesh merging enabled without loose geometry cleanup! This may result in floating vertex hidden among your mesh. Consider enabling 'Clean Loose' option to remove them.",
            )
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
        self,
        operator,
        obj: bpy.types.Object,
        name: str,
        cfg: ImporterOptions,
        hash_json_data: HashJsonData,
    ):
        if cfg.merge_meshes:
            poly_mat_idx = []
            component = hash_json_data.get_component_by_fullname(name)
            for part in component.parts:
                diffuse = next(
                    (tex for tex in part.textures if tex.name.lower() == "diffuse"),
                    None,
                )
                if diffuse is None:
                    operator.report(
                        {"WARNING"},
                        f"Failed to find diffuse texture for {part.fullname} in dump folder!",
                    )
                    continue
                if not diffuse.path.exists():
                    operator.report(
                        {"WARNING"},
                        f"Diffuse texture file for {part.fullname} not found at expected path: {diffuse.path}",
                    )
                    continue

                mat_name = part.fullname
                mat = bpy.data.materials.new(mat_name)
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                if bsdf:
                    tex_image = mat.node_tree.nodes.new("ShaderNodeTexImage")

                    for img in bpy.data.images:
                        if img.filepath == str(diffuse.path):
                            material_img = img
                            break
                    else:
                        material_img = bpy.data.images.load(str(diffuse.path))
                        material_img.alpha_mode = "CHANNEL_PACKED"
                    tex_image.image = material_img
                    mat.node_tree.links.new(
                        bsdf.inputs["Base Color"],
                        tex_image.outputs["Color"],
                    )

                obj.data.materials.append(mat)
            return

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

        mat_name = part.fullname
        mat = bpy.data.materials.new(mat_name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            tex_image = mat.node_tree.nodes.new("ShaderNodeTexImage")
            for img in bpy.data.images:
                if img.filepath == str(diffuse.path):
                    material_img = img
                    break
            else:
                material_img = bpy.data.images.load(str(diffuse.path))
                material_img.alpha_mode = "CHANNEL_PACKED"
            material_img.alpha_mode = "CHANNEL_PACKED"
            tex_image.image = material_img
            mat.node_tree.links.new(
                bsdf.inputs["Base Color"],
                tex_image.outputs["Color"],
            )
        # TODO: add proper node support for old versions and fall off color to surface in case of errors
        obj.data.materials.append(mat)
        mat_idx = len(obj.data.materials) - 1
        poly_mat_idx = [mat_idx] * len(obj.data.polygons)
        obj.data.polygons.foreach_set("material_index", poly_mat_idx)
