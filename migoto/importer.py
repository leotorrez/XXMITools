import time
from dataclasses import dataclass, field
from pathlib import Path

import bpy
from bpy.types import Collection, Context, Mesh, Object, Operator
from bpy_extras.io_utils import axis_conversion

from .data.byte_buffer import AbstractSemantic, MigotoFormat
from .data.data_model import DataModelXXMI
from .data.hash_json import Component, HashJsonData
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


def _extract_path(path_or_tuple: str | tuple | list[str]) -> Path:
    """
    Extract the text path from either a string path or a (binary, text) tuple.
    When use_bin=True, paths are tuples of (binary_path, text_path).
    When use_bin=False, paths are simple strings.
    """
    if isinstance(path_or_tuple, (tuple, list)):
        # Return the text path (second element of the tuple)
        found = path_or_tuple[1] if len(path_or_tuple) > 1 else path_or_tuple[0]
    else:
        found = path_or_tuple

    return Path(found) if isinstance(found, str) else found


class ObjectImporter:
    def import_object(self, operator: Operator, context: Context, cfg):
        # Extract the first path, handling both string and tuple formats
        first_import_path = next(iter(cfg.import_paths))
        first_vb_path = _extract_path(first_import_path.vb_paths[0])
        import_folder: Path = Path(first_vb_path).parent

        assert import_folder is not None
        assert context.scene is not None and context.scene.collection is not None

        start_time = time.time()
        print(f"Object import started for '{import_folder.stem}' folder")
        try:
            hash_json_data = HashJsonData(import_folder / "hash.json")
        except FileNotFoundError:
            raise Fatal(
                f"Specified folder is missing hash.json! Expected at: {import_folder / 'hash.json'}",
            )

        imported_objects: list[Object] = self.process_objects(
            operator, cfg, hash_json_data
        )
        if len(imported_objects) == 0:
            raise Fatal(
                "Specified folder is missing files for components!",
            )

        col: Collection = bpy.data.collections.new(import_folder.stem)

        context.scene.collection.children.link(col)
        for obj in imported_objects:
            col.objects.link(obj)
            self.cleanup_object(operator, context, obj, cfg)
            # if cfg.skip_empty_vertex_groups and cfg.import_skeleton_type == "MERGED":
            #     remove_unused_vertex_groups(context, obj)

        print(f"Total import time: {time.time() - start_time:.3f}s")

    def process_objects(self, operator: Operator, cfg, hash_json_data) -> list[Object]:
        imported_objects: list[Object] = []

        # We use lists to ensure the order is kept
        grouped_paths: dict[str, list[ImportPaths]] = {}
        if cfg.merge_meshes:
            for component in hash_json_data.components:
                fullname: str = component.fullname
                result: list[ImportPaths] = []
                for part in component.parts:
                    part_result = next(
                        (
                            paths
                            for paths in cfg.import_paths
                            if Path(_extract_path(paths.ib_paths)).stem.split("-ib")[0]
                            == part.fullname
                        ),
                        None,
                    )
                    if part_result:
                        result.append(part_result)
                if len(result) > 0:
                    grouped_paths[fullname] = result
        else:
            for paths in cfg.import_paths:
                ib_path = Path(_extract_path(paths.ib_paths))
                fullname: str = ib_path.stem.split("-ib")[0]
                if fullname not in grouped_paths:
                    grouped_paths[fullname] = []
                grouped_paths[fullname].append(paths)

        for fullname, paths in grouped_paths.items():
            obj: Object = self.import_component(
                operator,
                cfg,
                paths,
                hash_json_data,
                fullname,
                axis_forward=cfg.axis_forward,
                axis_up=cfg.axis_up,
            )

            imported_objects.append(obj)

        return imported_objects

    def import_component(
        self,
        operator: Operator,
        cfg: ImporterOptions,
        paths: list[ImportPaths],
        hash_json_data: HashJsonData,
        name: str,
        axis_forward="Y",
        axis_up="Z",
    ):
        start_time = time.time()
        numpy_mesh_group: NumpyMeshGroup = NumpyMeshGroup()

        for p in paths:
            vb_paths, ib_path, _, _ = p
            # Extract text path from either string or (binary, text) tuple
            vb_path: Path = _extract_path(vb_paths[0])
            ib_path: Path = _extract_path(ib_path)
            fmt_path: Path = vb_path.with_suffix(".fmt")
            migoto_format: MigotoFormat = MigotoFormat.from_paths(
                fmt_path, ib_path, vb_path
            )
            if migoto_format.vb_layout is None or migoto_format.format is None:
                raise Fatal(
                    f"Specified .fmt file for {fmt_path.stem} is missing vertex buffer layout!",
                )
            numpy_mesh_group.add_mesh(
                NumpyMesh.from_paths(
                    migoto_format, vb_path, ib_path, fmt_path, cfg.load_buf
                )
            )
        if (format := numpy_mesh_group.numpy_mesh.format) is None:
            raise Fatal(f"Failed to determine vertex format for component {name}!")
        vg_remap = None
        # if cfg.import_skeleton_type == "MERGED":
        #     component_pattern = re.compile(r".*component[ -_]*([0-9]+).*")
        #     result = component_pattern.findall(fmt_path.name.lower())
        #     if len(result) == 1:
        #         component = extracted_object.components[int(result[0])]
        #         vg_remap = numpy.array(list(component.vg_map.values()))

        mesh: Mesh = bpy.data.meshes.new(name)
        obj: Object = bpy.data.objects.new(mesh.name, mesh)
        assert mesh is not None and obj is not None

        obj.matrix_world = axis_conversion(axis_forward, axis_up).to_4x4()
        model = DataModelXXMI()
        model.flip_winding = cfg.flip_winding
        model.flip_texcoord_v = cfg.flip_texcoord_v
        model.legacy_vertex_colors = False

        if cfg.create_materials:
            self.set_materials(operator, obj, name, cfg, hash_json_data)
        self.set_custom_properties(obj, format, cfg)
        model.set_data(
            obj,
            mesh,
            numpy_mesh_group,
            vg_remap,
            mirror_mesh=cfg.flip_mesh,
        )

        num_shapekeys: int = (
            0
            if mesh.shape_keys is None
            else len(getattr(mesh.shape_keys, "key_blocks", []))
        )
        print(
            f"{name} import time: {time.time() - start_time:.3f}s ({len(mesh.vertices)} vertices, {len(mesh.loops)} indices, {num_shapekeys} shapekeys)"
        )
        return obj

    def cleanup_object(
        self, operator: Operator, context: Context, obj: Object, cfg: ImporterOptions
    ):
        if cfg.merge_meshes and not cfg.clean_loose:
            operator.report(
                {"WARNING"},
                "Mesh merging enabled without loose geometry cleanup! This may result in floating vertex hidden among your mesh. Consider enabling 'Clean Loose' option to remove them.",
            )
        assert context.view_layer is not None
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

    def set_custom_properties(
        self, obj: Object, migoto_format: MigotoFormat, cfg: ImporterOptions
    ):
        if migoto_format.vb_layout is None or migoto_format.format is None:
            raise Fatal(
                f"Specified .fmt file is missing vertex buffer layout for object {obj.name}!",
            )
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
        operator: Operator,
        obj: Object,
        name: str,
        cfg: ImporterOptions,
        hash_json_data: HashJsonData,
    ):
        def add_material(obj, part, diffuse):
            # TODO: add proper node support for old versions and fall off color to surface in case of errors
            mat_name = part.fullname
            mat = bpy.data.materials.new(mat_name)
            mat.use_nodes = True
            if mat.node_tree is None:
                operator.report(
                    {"WARNING"},
                    f"Failed to create node tree for material {mat_name}! Skipping..",
                )
                return
            if (bsdf := mat.node_tree.nodes.get("Principled BSDF")) is None:
                operator.report(
                    {"WARNING"},
                    f"Failed to find Principled BSDF node for material {mat_name}! Skipping..",
                )
                return
            if (tex_image := mat.node_tree.nodes.new("ShaderNodeTexImage")) is None:
                operator.report(
                    {"WARNING"},
                    f"Failed to create Image Texture node for material {mat_name}! Skipping..",
                )
                return

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
            prop_name = f"3DMigoto:Material_{part.name}"
            obj[prop_name] = mat

            obj.id_properties_ui(prop_name).update(id_type="MATERIAL", description="")

        if cfg.merge_meshes:
            component: Component = hash_json_data.get_component_by_fullname(name)
            for part in component.parts:
                if (diffuse := part.get_texture_by_name("diffuse")) is None:
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
                add_material(obj, part, diffuse)
            return

        if (part := hash_json_data.get_part_by_fullname(name)) is None:
            operator.report(
                {"WARNING"},
                f"Failed to find diffuse texture for {name} in dump folder!",
            )
            return
        if (diffuse := part.get_texture_by_name("diffuse")) is None:
            operator.report(
                {"WARNING"},
                f"Failed to find diffuse texture for {name} in dump folder!!",
            )
            return
        if not diffuse.path.exists():
            operator.report(
                {"WARNING"},
                f"Diffuse texture file for {name} not found at expected path: {diffuse.path}",
            )
            return
        add_material(obj, part, diffuse)
