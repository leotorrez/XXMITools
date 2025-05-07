import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Optional

import addon_utils
import bpy
import numpy
from bpy.types import Collection, Context, Mesh, Object, Depsgraph
from numpy.typing import NDArray

from .. import bl_info
from ..libs.jinja2 import Environment, FileSystemLoader
from .data.data_model import DataModelXXMI
from .datastructures import GameEnum
from .operators import Fatal
from .export_ops import mesh_triangulate
from .data.byte_buffer import Semantic


@dataclass
class SubObj:
    collection_name: str
    depth: int
    name: str
    obj: Object
    mesh: Mesh
    vertex_count: int = 0
    index_count: int = 0
    index_offset: int = 0


@dataclass
class TextureData:
    name: str
    extension: str
    hash: str


@dataclass
class Part:
    fullname: str
    objects: list[SubObj]
    textures: list[TextureData]
    first_index: int
    vertex_count: int = 0


@dataclass
class Component:
    fullname: str
    parts: list[Part]
    root_vs: str
    draw_vb: str
    position_vb: str
    blend_vb: str
    texcoord_vb: str
    ib: str
    vertex_count: int = 0
    strides: dict[str, int] = field(default_factory=dict)


@dataclass
class ModFile:
    name: str
    components: list[Component]
    hash_data: dict[str, str]
    game: GameEnum
    credit: str = ""


@dataclass
class ModExporter:
    # Input
    context: Context
    mod_name: str
    hash_data: dict
    ignore_hidden: bool
    ignore_muted_shape_keys: bool
    apply_modifiers: bool
    only_selected: bool
    copy_textures: bool
    join_meshes: bool
    dump_path: Path
    destination: Path
    credit: str
    game: GameEnum
    # Output
    mod_file: ModFile = field(init=False)
    ini_content: str = field(init=False)
    files_to_write: dict[Path, Union[str, NDArray]] = field(init=False)
    files_to_copy: dict[Path, Path] = field(init=False)

    def __post_init__(self) -> None:
        self.initialize_data()

    def initialize_data(self) -> None:
        print("Initializing data for export...")
        if not self.hash_data:
            raise ValueError("Hash data is empty!")

        candidate_objects: list[Object] = (
            [obj for obj in bpy.context.selected_objects]
            if self.only_selected
            else [obj for obj in self.context.scene.objects]
        )
        if self.ignore_hidden:
            candidate_objects = [obj for obj in candidate_objects if obj.visible_get()]
        self.mod_file = ModFile(
            name=self.mod_name,
            components=[],
            hash_data=self.hash_data,
            game=self.game,
            credit=self.credit,
        )
        for i, component in enumerate(self.hash_data):
            current_name: str = f"{self.mod_name}{component['component_name']}"
            component_entry = Component(
                fullname=current_name,
                parts=[],
                root_vs=component["root_vs"],
                draw_vb=component["draw_vb"],
                position_vb=component["position_vb"],
                blend_vb=component["blend_vb"],
                texcoord_vb=component["texcoord_vb"],
                ib=component["ib"],
                strides={},
            )
            component_vertex_count = 0
            for j, part in enumerate(component["object_classifications"]):
                if component["draw_vb"] == "":
                    continue
                part_name: str = current_name + part
                matching_objs = [
                    obj for obj in candidate_objects if obj.name.startswith(part_name)
                ]
                if not matching_objs:
                    raise Fatal(f"Cannot find object {part_name} in the scene.")
                if len(matching_objs) > 1:
                    raise Fatal(f"Found multiple objects with the name {part_name}.")
                obj: Object = matching_objs[0]
                collection_name = [
                    c
                    for c in bpy.data.collections
                    if c.name.lower().startswith((part_name).lower())
                ]
                if len(collection_name) > 1:
                    raise Fatal(
                        f"ERROR: Found multiple collections with the name {part_name}. Ensure only one collection exists with that name."
                    )
                textures = []
                for entry in component["texture_hashes"][j]:
                    textures.append(TextureData(*entry))

                objects: list[SubObj] = []
                if len(collection_name) == 0:
                    self.obj_from_col(obj, None, objects)
                else:
                    self.obj_from_col(obj, collection_name[0], objects)
                offset = 0
                for entry in objects:
                    entry.index_count = len(entry.mesh.polygons) * 3
                    entry.index_offset = offset
                    offset += entry.index_count
                component_entry.parts.append(
                    Part(
                        fullname=part_name,
                        objects=objects,
                        textures=textures,
                        first_index=component["object_indexes"][j],
                    )
                )
            self.mod_file.components.append(component_entry)

    def obj_from_col(
        self,
        main_obj: Object,
        collection: Optional[Collection],
        destination: list[SubObj],
        depth: int = 0,
    ) -> None:
        """Recursively get all objects from a collection and its sub-collections."""
        depsgraph = bpy.context.evaluated_depsgraph_get()
        if destination == []:
            final_mesh: Mesh = self.process_mesh(main_obj, main_obj, depsgraph)
            destination.append(SubObj("", depth, main_obj.name, main_obj, final_mesh))
        if collection is None:
            return

        objs = [obj for obj in collection.objects if obj.type == "MESH"]
        sorted_objs = sorted(objs, key=lambda x: x.name)
        for obj in sorted_objs:
            final_mesh = self.process_mesh(main_obj, obj, depsgraph)
            destination.append(
                SubObj(collection.name, depth, obj.name, obj, final_mesh)
            )
        for child in collection.children:
            self.obj_from_col(main_obj, child, destination, depth + 1)

    def process_mesh(self, main_obj: Object, obj: Object, depsgraph: Depsgraph) -> Mesh:
        """Process the mesh of the object."""
        # TODO: Add moddifier application for SK'd meshes here
        final_mesh: Mesh = obj.evaluated_get(depsgraph).to_mesh()
        if main_obj != obj:
            # Matrix world seems to be the summatory of all transforms parents included
            # Might need to test for more edge cases and to confirm these suspicious,
            # other available options: matrix_local, matrix_basis, matrix_parent_inverse
            final_mesh.transform(obj.matrix_world)
            final_mesh.transform(main_obj.matrix_world.inverted())
        mesh_triangulate(final_mesh)
        return final_mesh

    def generate_buffers(self) -> None:
        """Generate buffers for the objects."""
        self.files_to_write = {}
        self.files_to_copy = {}
        repeated_textures = {}
        for component in self.mod_file.components:
            output_buffers: dict[str, Optional[NDArray]] = {
                "Position": None,
                "Blend": None,
                "TexCoord": None,
            }
            ib_offset: int = 0
            for part in component.parts:
                print(f"Processing {part.fullname} " + "-" * 10)
                ib_buffer = None
                data_model: DataModelXXMI = DataModelXXMI.from_obj(
                    part.objects[0].obj, self.game
                )
                for entry in part.objects:
                    print(f"Processing {entry.name}...")
                    if len(entry.obj.data.polygons) == 0:
                        print(f"Skipping empty mesh {entry.obj.name}")
                        continue
                    gen_buffers, v_count = data_model.get_data(
                        bpy.context,
                        None,
                        entry.obj,
                        entry.mesh,
                        [],  # type: ignore
                    )
                    entry.vertex_count = v_count
                    part.vertex_count += v_count
                    component.vertex_count += v_count
                    for k in output_buffers:
                        component.strides[k.lower()] = gen_buffers[k].data.itemsize
                        output_buffers[k] = (
                            gen_buffers[k].data
                            if output_buffers[k] is None
                            else numpy.concatenate(
                                (output_buffers[k], gen_buffers[k].data)  # type: ignore
                            )
                        )
                    gen_buffers["IB"].data["INDEX"] += ib_offset
                    ib_buffer = (
                        gen_buffers["IB"].data
                        if ib_buffer is None
                        else numpy.concatenate((ib_buffer, gen_buffers["IB"].data))
                    )
                    ib_offset += v_count
                for t in part.textures:
                    if (
                        t.hash in repeated_textures
                        and self.game != GameEnum.GenshinImpact
                    ):
                        repeated_textures[t.hash].append(t)
                        continue
                    repeated_textures[t.hash] = [t]
                    tex_name = part.fullname + t.name + t.extension
                    self.files_to_copy[self.dump_path / tex_name] = (
                        self.destination / tex_name
                    )
                if ib_buffer is None:
                    print(f"Skipping {part.fullname}.ib due to no index data.")
                    continue
                self.files_to_write[self.destination / (part.fullname + ".ib")] = (
                    ib_buffer
                )
            if (
                output_buffers["Position"] is None
                or output_buffers["Blend"] is None
                or output_buffers["TexCoord"] is None
            ):
                print(f"Skipping {component.fullname} buffers due to no position data.")
                continue
            self.files_to_write[
                self.destination / (component.fullname + "Position.buf")
            ] = output_buffers["Position"]
            self.files_to_write[
                self.destination / (component.fullname + "Blend.buf")
            ] = output_buffers["Blend"]
            self.files_to_write[
                self.destination / (component.fullname + "TexCoord.buf")
            ] = output_buffers["TexCoord"]

    def generate_ini(
        self, template_name: str = "default.ini.j2", user_paths=None
    ) -> None:
        print("Generating .ini file")
        addon_path = None
        for mod in addon_utils.modules():
            if mod.bl_info["name"] == "XXMI_Tools":
                addon_path = Path(mod.__file__).parent
                break
        templates_paths = [addon_path / "templates"]
        if user_paths is not None:
            templates_paths.extend(user_paths)
        env = Environment(
            loader=FileSystemLoader(searchpath=templates_paths),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        template = env.get_template(template_name)
        self.files_to_write[self.destination / (self.mod_name + ".ini")] = (
            template.render(
                version=bl_info["version"],
                mod_file=self.mod_file,
                credit=self.credit,
                game=self.game,
                character_name=self.mod_name,
                join_meshes=self.join_meshes,
            )
        )

    def write_files(self) -> None:
        """Write the files to the destination."""
        print("Writen files: ")
        self.destination.mkdir(parents=True, exist_ok=True)
        try:
            for file_path, content in self.files_to_write.items():
                print(f"{file_path.name}", end=", ")
                if isinstance(content, str):
                    with open(file_path, "w", encoding="utf-8") as file:
                        file.write(content)
                elif isinstance(content, numpy.ndarray):
                    content.tofile(file_path)
        except (OSError, IOError) as e:
            raise Fatal(f"Error writing file {file_path}: {e}")
        if not self.copy_textures:
            return
        try:
            for src, dest in self.files_to_copy.items():
                print(f"{dest.name}", end=", ")
                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dest)
        except (OSError, IOError) as e:
            raise Fatal(f"Error copying file {src} to {dest}: {e}")
        print("")

    def export(self) -> None:
        """Export the mod file."""
        print(f"Exporting {self.mod_name} to {self.destination}")
        self.generate_buffers()
        self.generate_ini()
        self.write_files()

    def cleanup(self) -> None:
        """Cleanup the objects."""
        pass
