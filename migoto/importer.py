import time
from pathlib import Path

import bpy
from bpy_extras.io_utils import axis_conversion

from .data.byte_buffer import MigotoFormat
from .data.data_model import DataModelXXMI
from .data.numpy_mesh import NumpyMesh
from .datahandling import Fatal


# TODO: Add support of import of unhandled semantics into vertex attributes
class ObjectImporter:
    def import_object(self, operator, context, cfg):
        import_folder: Path = cfg.import_paths[0].parent
        start_time = time.time()
        print(f"Object import started for '{import_folder.stem}' folder")

        imported_objects = []

        for vb_path, ib_path, use_bin, pose_path in cfg.import_paths:
            fmt_path = vb_path.with_suffix(".fmt")
            obj = self.import_component(
                operator, context, cfg, fmt_path, ib_path, vb_path
            )

            imported_objects.append(obj)

        if len(imported_objects) == 0:
            raise Fatal(
                "object_source_folder",
                "Specified folder is missing files for components!",
            )

        assert import_folder is not None

        col = bpy.data.collections.new(import_folder.stem)
        for obj in imported_objects:
            col.objects.link(obj)
            # if cfg.skip_empty_vertex_groups and cfg.import_skeleton_type == "MERGED":
            #     remove_unused_vertex_groups(context, obj)

        print(f"Total import time: {time.time() - start_time:.3f}s")

    def import_component(
        self,
        operator,
        context,
        cfg,
        fmt_path: Path,
        ib_path: Path,
        vb_path: Path,
        axis_forward="Y",
        axis_up="Z",
    ):
        start_time = time.time()
        import_folder = vb_path.parent
        migoto_format = MigotoFormat.from_paths(fmt_path, ib_path, vb_path)
        numpy_mesh = NumpyMesh.from_paths(
            migoto_format, ib_path, vb_path, cfg.use_binary
        )

        if numpy_mesh.vertex_buffer is None or numpy_mesh.index_buffer is None:
            raise Fatal(
                "object_source_folder",
                f"Specified .fmt file for {fmt_path.stem} is missing vertex or index buffer!",
            )

        try:
            extracted_object = read_metadata(import_folder / "hash.json")
        except FileNotFoundError:
            raise Fatal(
                "object_source_folder", "Specified folder is missing hash.json!"
            )
        except Exception as e:
            raise Fatal("object_source_folder", f"Failed to load hash.json:\n{e}")

        vg_remap = None
        # if cfg.import_skeleton_type == "MERGED":
        #     component_pattern = re.compile(r".*component[ -_]*([0-9]+).*")
        #     result = component_pattern.findall(fmt_path.name.lower())
        #     if len(result) == 1:
        #         component = extracted_object.components[int(result[0])]
        #         vg_remap = numpy.array(list(component.vg_map.values()))

        mesh = bpy.data.meshes.new(fmt_path.stem)
        obj = bpy.data.objects.new(mesh.name, mesh)
        assert mesh is not None and obj is not None

        global_matrix = axis_conversion(
            from_forward=axis_forward, from_up=axis_up
        ).to_4x4()
        obj.matrix_world = global_matrix

        model = DataModelXXMI()
        model.flip_winding = cfg.flip_winding
        model.flip_texcoord_v = cfg.flip_texcoord_v
        model.legacy_vertex_colors = cfg.color_storage == "LEGACY"

        model.set_data(
            obj,
            mesh,
            numpy_mesh.index_buffer,
            numpy_mesh.vertex_buffer,
            vg_remap,
            mirror_mesh=cfg.mirror_mesh,
            mesh_scale=0.01,
            mesh_rotation=(0, 0, 180),
        )

        num_shapekeys = (
            0
            if mesh.shape_keys is None
            else len(getattr(mesh.shape_keys, "key_blocks", []))
        )

        print(
            f"{fmt_path.stem} import time: {time.time() - start_time:.3f}s ({len(mesh.vertices)} vertices, {len(mesh.loops)} indices, {num_shapekeys} shapekeys)"
        )
        # TODO:read metadata and set custom properties
        # add materials
        # post processing cleanup
        # semantic remapping
        return obj


# def blender_import(operator, context, cfg):
#     object_importer = ObjectImporter()
#     object_importer.import_object(operator, context, cfg)
