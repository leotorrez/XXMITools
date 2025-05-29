import shutil
import time
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import addon_utils
import bpy
import numpy
from bpy.types import Collection, Context, Depsgraph, Mesh, Object, Operator
from numpy.typing import NDArray

from .. import bl_info
from ..libs.jinja2 import Environment, FileSystemLoader
from .data.byte_buffer import (
    BufferLayout,
    BufferSemantic,
    NumpyBuffer,
    Semantic,
)
from .data.data_model import DataModelXXMI
from .data.ini_format import INI_file
from .datastructures import GameEnum
from .export_ops import mesh_triangulate
from .operators import Fatal


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
    hash_data: list[dict]
    game: GameEnum
    credit: str = ""


@dataclass
class ModExporter:
    # Input
    context: Context
    operator: Operator
    dump_path: Path
    destination: Path
    credit: str
    game: GameEnum
    ignore_hidden: bool
    apply_modifiers: bool
    only_selected: bool
    copy_textures: bool
    normalize_weights: bool
    outline_optimization: bool
    no_ramps: bool
    ignore_duplicate_textures: bool
    write_buffers: bool
    write_ini: bool
    template: Optional[Path] = None
    outline_rounding_precision: int = 3
    # Internal / not implemented
    ignore_muted_shape_keys: bool = False
    # Output
    mod_name: str = ""
    hash_data: list[dict] = field(default_factory=list)
    mod_file: ModFile = field(init=False)
    ini_content: str = field(init=False)
    files_to_write: dict[Path, Union[str, NDArray]] = field(init=False)
    files_to_copy: dict[Path, Path] = field(init=False)

    def __post_init__(self) -> None:
        print("Initializing data for export...")
        if self.dump_path == Path(""):
            raise Fatal("Dump path not set")
        if self.dump_path.is_file():
            self.dump_path = self.dump_path.parent

        self.mod_name = self.dump_path.stem

        if self.destination == Path(""):
            self.destination = self.dump_path.parent / f"{self.mod_name}Mod"
            self.operator.report(
                {"WARNING"},
                f"Destination path not set, defaulting to {self.destination}",
            )
        if not self.destination.exists():
            raise Fatal("Destination path does not exist.")
        if self.destination.is_file():
            self.destination = self.destination.parent

        if self.destination == self.dump_path:
            raise Fatal("Destination path can't be the same as Dump path")

        self.hash_data = self.load_hashes(self.dump_path, self.dump_path.stem)
        if not self.hash_data:
            raise Fatal("ERROR", "Hash data is empty or invalid!")

        scene = bpy.context.scene
        if not [
            obj for obj in scene.objects if self.mod_name.lower() in obj.name.lower()
        ] or not [
            file
            for file in self.dump_path.parent.iterdir()
            if self.mod_name.lower() in file.name.lower()
        ]:
            raise Fatal(
                "ERROR: Cannot find match for name. Double check you are exporting as ObjectName.vb to the original data folder, that ObjectName exists in scene and that hash.json exists"
            )

        candidate_objs: list[Object] = (
            [obj for obj in bpy.context.selected_objects]
            if self.only_selected
            else [obj for obj in self.context.scene.objects]
        )
        if self.ignore_hidden:
            candidate_objs = [obj for obj in candidate_objs if obj.visible_get()]
        if self.only_selected:
            selected_objs = [obj for obj in bpy.context.selected_objects]
            candidate_objs = [obj for obj in candidate_objs if obj in selected_objs]
        self.mod_file = ModFile(
            name=self.mod_name,
            components=[],
            hash_data=self.hash_data,
            game=self.game,
            credit=self.credit,
        )
        seen_hashes = set()
        for i, component in enumerate(self.hash_data):
            current_name: str = f"{self.mod_name}{component['component_name']}"
            component_entry: Component = Component(
                fullname=current_name,
                parts=[],
                root_vs=component.get("root_vs", ""),
                draw_vb=component.get("draw_vb", ""),
                position_vb=component.get("position_vb", ""),
                blend_vb=component.get("blend_vb", ""),
                texcoord_vb=component.get("texcoord_vb", ""),
                ib=component.get("ib", ""),
                strides={},
            )
            comp_matching_objs: list[Object] = [
                obj for obj in candidate_objs if obj.name.startswith(current_name)
            ]
            if len(comp_matching_objs) == 0:
                continue
            for j, part in enumerate(component["object_classifications"]):
                part_name: str = current_name + part
                objects: list[SubObj] = []
                textures: list[TextureData] = [
                    TextureData(*e) for e in component["texture_hashes"][j]
                ]
                if self.ignore_duplicate_textures:
                    textures = [
                        t
                        for t in textures
                        if t.hash not in seen_hashes and not seen_hashes.add(t.hash)
                    ]
                matching_objs: list[Collection] = [
                    obj for obj in candidate_objs if obj.name.startswith(part_name)
                ]
                if component["draw_vb"] != "":
                    if not matching_objs:
                        raise Fatal(f"Cannot find object {part_name} in the scene.")
                    if len(matching_objs) > 1:
                        raise Fatal(
                            f"Found multiple objects with the name {part_name}."
                        )
                    obj: Object = matching_objs[0]
                    collection = [
                        c
                        for c in bpy.data.collections
                        if c.name.lower().startswith((part_name).lower())
                    ]
                    if len(collection) > 1:
                        raise Fatal(
                            f"ERROR: Found multiple collections with the name {part_name}. Ensure only one collection exists with that name."
                        )
                    if len(collection) == 0:
                        self.obj_from_col(obj, None, objects)
                    else:
                        self.obj_from_col(obj, collection[0], objects)
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

        objs = [
            obj for obj in collection.objects if obj.type == "MESH" and obj != main_obj
        ]
        if self.ignore_hidden:
            objs = [obj for obj in objs if obj.visible_get()]
        if self.only_selected:
            selected_objs = [obj for obj in bpy.context.selected_objects]
            objs = [obj for obj in objs if obj in selected_objs]
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
            excluded_buffers: list[str] = []
            output_buffers: dict[str, NDArray] = {
                "Position": numpy.empty(0, dtype=numpy.uint8),
                "Blend": numpy.empty(0, dtype=numpy.uint8),
                "TexCoord": numpy.empty(0, dtype=numpy.uint8),
            }
            component_ib: NDArray = numpy.empty(0, dtype=numpy.uint8)
            if self.write_buffers is False:
                for key in output_buffers.keys():
                    excluded_buffers.append(key)
            vb_offset: int = 0
            for part in component.parts:
                print(f"Processing {part.fullname} " + "-" * 10)
                ib_buffer = None
                ib_offset: int = 0
                for t in part.textures:
                    if self.ignore_duplicate_textures:
                        if t.hash in repeated_textures:
                            repeated_textures[t.hash].append(t)
                            continue
                        repeated_textures[t.hash] = [t]
                    if self.no_ramps and t.name in [
                        "ShadowRamp",
                        "MetalMap",
                        "DiffuseGuide",
                    ]:
                        continue
                    tex_name = part.fullname + t.name + t.extension
                    self.files_to_copy[self.dump_path / tex_name] = (
                        self.destination / tex_name
                    )
                if component.draw_vb == "":
                    continue
                data_model: DataModelXXMI = DataModelXXMI.from_obj(
                    part.objects[0].obj,
                    game=self.game,
                    normalize_weights=self.normalize_weights,
                    is_posed_mesh=component.blend_vb != "",
                )
                for entry in part.objects:
                    print(f"Processing {entry.name}...")
                    v_count: int = 0
                    gen_buffers: dict[str, NumpyBuffer] = {
                        key: NumpyBuffer(layout=entry)
                        for key, entry in data_model.buffers_format.items()
                    }
                    if len(entry.obj.data.polygons) == 0:
                        continue
                    self.verify_mesh_requirements(
                        part.objects[0].obj,
                        entry.obj,
                        entry.mesh,
                        data_model.buffers_format,
                        excluded_buffers,
                    )
                    gen_buffers, v_count = data_model.get_data(
                        bpy.context,
                        None,
                        entry.obj,
                        entry.mesh,
                        excluded_buffers,
                        data_model.mirror_mesh,
                    )
                    for k in output_buffers:
                        if k not in gen_buffers:
                            continue
                        output_buffers[k] = (
                            gen_buffers[k].data
                            if len(output_buffers[k]) == 0
                            else numpy.concatenate(
                                (output_buffers[k], gen_buffers[k].data)
                            )
                        )
                    gen_buffers["IB"].data["INDEX"] += vb_offset
                    ib_buffer = (
                        gen_buffers["IB"].data.copy()
                        if ib_buffer is None
                        else numpy.concatenate((ib_buffer, gen_buffers["IB"].data))
                    )
                    vb_offset += v_count
                    entry.vertex_count = v_count
                    part.vertex_count += v_count
                    component.vertex_count += v_count
                    entry.index_count = len(gen_buffers["IB"].data)
                    entry.index_offset = ib_offset
                    ib_offset += entry.index_count
                if ib_buffer is None:
                    print(f"Skipping {part.fullname}.ib due to no index data.")
                    continue
                component_ib = (
                    ib_buffer.copy()
                    if len(component_ib) == 0
                    else numpy.concatenate((component_ib, ib_buffer))
                )
                self.files_to_write[self.destination / (part.fullname + ".ib")] = (
                    ib_buffer
                )
            self.optimize_outlines(output_buffers, component_ib, data_model)
            if component.blend_vb != "":
                self.files_to_write[
                    self.destination / (component.fullname + "Position.buf")
                ] = output_buffers["Position"]
                self.files_to_write[
                    self.destination / (component.fullname + "Blend.buf")
                ] = output_buffers["Blend"]
                self.files_to_write[
                    self.destination / (component.fullname + "Texcoord.buf")
                ] = output_buffers["TexCoord"]
                component.strides = {
                    k.lower(): v.stride
                    for k, v in data_model.buffers_format.items()
                    if k != "IB"
                }
                continue
            self.files_to_write[self.destination / (component.fullname + ".buf")] = (
                output_buffers["Position"]
            )
            component.strides = {"position": output_buffers["Position"].itemsize}

    def verify_mesh_requirements(
        self,
        main_obj: Object,
        obj: Object,
        mesh: Mesh,
        buffers_format: dict[str, BufferLayout],
        excluded_buffers: list[str],
    ) -> None:
        """Checks for format requirements in specific layouts"""
        semantics_to_check: list[BufferSemantic] = [
            semantic
            for key, buffer_layout in buffers_format.items()
            for semantic in buffer_layout.semantics
            if key not in excluded_buffers
        ]
        missing_uvs: list[str] = []
        missing_colors: list[str] = []
        for sem in semantics_to_check:
            abs_enum: Semantic = sem.abstract.enum
            abs_name: str = sem.abstract.get_name()
            if abs_enum == Semantic.Color and mesh.vertex_colors.get(abs_name) is None:
                missing_colors.append(abs_name)
            if abs_enum == Semantic.TexCoord and mesh.uv_layers.get(abs_name) is None:
                missing_uvs.append(abs_name)
            if abs_enum == Semantic.Blendweight:
                if len(mesh.vertices) > 0 and len(obj.vertex_groups) == 0:
                    self.operator.report(
                        {"WARNING"},
                        (
                            f"Mesh({obj.name}) requires vertex groups to be posed. "
                            "Please add vertex groups to the mesh if you intend for it to be rendered. "
                        ),
                    )
                max_groups: int = sem.format.get_num_values()
                for vertex in mesh.vertices:
                    if len(vertex.groups) > max_groups:
                        self.operator.report(
                            {"WARNING"},
                            (
                                f"Mesh({obj.name}) has some vertex with more VGs than the amount supported by the buffer format ({max_groups}). "
                                "Please remove the extra groups from the vertex or use to clean up the weights(limit total plus normalization). "
                                "Alternatively you can enable normalize weights to format(Ignore this warning if you already have it enabled)"
                            ),
                        )
                        break
        # At the moment these errors made the UV layers and vertex colors mandatory to export
        # in the future we might want to make them optional or auto generate them
        if len(missing_uvs) > 0:
            raise Fatal(
                f"Mesh({obj.name}) is missing the following UV layers: {', '.join(missing_uvs)}. "
                f"Please add them to the mesh before exporting."
            )
        if len(missing_colors) > 0:
            raise Fatal(
                f"Mesh({obj.name}) is missing the following vertex colors: {', '.join(missing_colors)}. "
                f"Please add them to the mesh before exporting."
            )

    def generate_ini(
        self,
        template_name: str = "default.ini.j2",
    ) -> None:
        # Extensions handle modifiable paths differently. If we ever move to them we should make modifications in here
        if self.write_ini is False:
            return
        print("Generating .ini file")
        addon_path: Path = Path(__file__).parent.parent
        for mod in addon_utils.modules():
            if mod.bl_info["name"] == "XXMI_Tools":
                addon_path = Path(mod.__file__).parent
                break
        templates_paths: list[Path] = [addon_path / "templates"]
        if (
            self.template != Path("")
            and isinstance(self.template, Path)
            and self.template.exists()
        ):
            templates_paths.insert(0, self.template.parent)
            template_name = self.template.name
        env: Environment = Environment(
            loader=FileSystemLoader(searchpath=templates_paths),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        print(f"Using template {template_name}")
        ini_file: INI_file = INI_file(
            env.get_template(template_name).render(
                version=bl_info["version"],
                mod_file=self.mod_file,
                credit=self.credit,
                game=self.game,
                character_name=self.mod_name,
            )
        )
        ini_file.clean_up_indentation()
        ini_body: str = str(ini_file)
        self.files_to_write[self.destination / (self.mod_name + ".ini")] = ini_body

    def optimize_outlines(
        self,
        output_buffers: dict[str, NDArray],
        ib_buf: NDArray,
        data_model: DataModelXXMI,
    ) -> None:
        """Optimize the outlines of the meshes with angle-weighted normal averaging."""

        def unit_vector(vector: NDArray) -> NDArray:
            norm = numpy.linalg.norm(vector, axis=0, keepdims=True)
            return vector / norm

        def antiparallel_search(normals) -> bool:
            a = numpy.einsum("ij,kj->ik", normals, normals)
            return bool(numpy.any((a > -1.000001) & (a < -0.999999)))

        def calc_angle(edge_a: NDArray, edge_b: NDArray) -> NDArray:
            return numpy.arccos(
                numpy.clip(
                    numpy.einsum(
                        "ij, ij->i",
                        unit_vector(edge_a),
                        unit_vector(edge_b),
                    ),
                    -1.0,
                    1.0,
                )
            )

        pos_buf: NDArray = output_buffers["Position"]
        ib_data: NDArray = ib_buf["INDEX"]
        if len(pos_buf) == 0:
            return

        start_time: int | float = time.time()

        loops_coord: NDArray = pos_buf["POSITION"][ib_data, 0:3]
        triangles: NDArray = loops_coord.reshape(-1, 3, 3)
        edge0: NDArray = triangles[:, 1] - triangles[:, 2]
        edge1: NDArray = triangles[:, 2] - triangles[:, 0]
        edge2: NDArray = triangles[:, 0] - triangles[:, 1]
        angle0: NDArray = calc_angle(edge2, edge1)
        angle1: NDArray = calc_angle(edge2, edge0)
        angle2: NDArray = calc_angle(edge1, edge0)
        loops_angle: NDArray = numpy.zeros((len(triangles), 3), dtype=numpy.float32)
        loops_angle[:, 0] = angle0
        loops_angle[:, 1] = angle1
        loops_angle[:, 2] = angle2
        faces_normal: NDArray = numpy.cross(edge1, edge2)
        faces_normal /= numpy.linalg.norm(faces_normal, axis=1)[:, numpy.newaxis]

        loops_face_normal: NDArray = faces_normal.repeat(3, axis=0)
        loops_outline_vector: NDArray = loops_face_normal.copy()

        verts_outline_vector: NDArray = numpy.zeros(
            (len(pos_buf), 3), dtype=numpy.float32
        )

        if self.outline_optimization == "EXPERIMENTAL":
            loops_round_coord: NDArray = numpy.round(
                loops_coord, self.outline_rounding_precision
            )
            loops_angle = loops_angle.flatten()
            loops_weighted_normal = loops_face_normal * loops_angle[:, None]

            u, u_idx, u_inverse = numpy.unique(
                loops_round_coord,
                axis=0,
                return_index=True,
                return_inverse=True,
            )

            accumulated_normals: NDArray = numpy.zeros((len(u), 3), dtype=numpy.float32)
            # Use numpy.add.at to efficiently sum weighted normals for each unique vertex
            numpy.add.at(accumulated_normals, u_inverse, loops_weighted_normal)
            accumulated_normals = numpy.where(
                accumulated_normals == 0,
                loops_face_normal[u_idx],
                accumulated_normals,
            )
            loops_outline_vector = accumulated_normals[u_inverse]

            norm = numpy.linalg.norm(loops_outline_vector, axis=1, keepdims=True)
            loops_outline_vector /= norm
            verts_outline_vector[ib_data] = loops_outline_vector
        elif self.outline_optimization == "ON":
            verts_outline_vector[ib_data] = loops_outline_vector
            face_verts = ib_data.reshape(-1, 3)
            vertex_round_pos: NDArray = numpy.round(
                pos_buf["POSITION"][:, 0:3], self.outline_rounding_precision
            )
            connected_faces: dict = {}
            pos_same_vertices: dict[tuple, set] = {}

            for i_poly in range(len(faces_normal)):
                for vert in face_verts[i_poly]:
                    connected_faces.setdefault(vert, []).append(i_poly)

                    vert_position: NDArray = pos_buf["POSITION"][vert, 0:3]
                    pos_same_vertices.setdefault(
                        tuple(
                            round(coord, self.outline_rounding_precision)
                            for coord in vert_position
                        ),
                        {vert},
                    ).add(vert)

            Connected_Faces_bySameVertex = {}
            for key, value in pos_same_vertices.items():
                for vertex in value:
                    Connected_Faces_bySameVertex.setdefault(key, set()).update(
                        connected_faces[vertex]
                    )

            ################# CALCULATIONS #####################

            IteratedValues: set = set()
            for key, vertex_group in pos_same_vertices.items():
                if key in IteratedValues:
                    continue
                FacesConnectedbySameVertex: list = list(
                    Connected_Faces_bySameVertex[key]
                )
                row: int = len(FacesConnectedbySameVertex)

                ConnectedFaceNormals = numpy.empty(shape=(row, 3))
                for i_normal, x in enumerate(FacesConnectedbySameVertex):
                    ConnectedFaceNormals[i_normal] = faces_normal[x]
                if antiparallel_search(ConnectedFaceNormals):
                    continue

                VectorMatrix0 = numpy.empty(shape=(row, 3))
                VectorMatrix1 = numpy.empty(shape=(row, 3))

                ConnectedWeightedNormal = numpy.empty(shape=(row, 3))

                for i, facei in enumerate(FacesConnectedbySameVertex):
                    vlist = face_verts[facei]

                    vert0p = set(vlist) & vertex_group

                    for vert0 in vert0p:
                        v0 = vertex_round_pos[vert0]
                        vn = [vertex_round_pos[x] for x in vlist if x != vert0]
                        VectorMatrix0[i] = vn[0] - v0
                        VectorMatrix1[i] = vn[1] - v0
                    ConnectedWeightedNormal[i] = faces_normal[facei]

                    influence_restriction: int = len(vert0p)
                    if influence_restriction > 1:
                        numpy.multiply(
                            ConnectedWeightedNormal[i],
                            0.5 ** (1 - influence_restriction),
                        )

                angle = calc_angle(VectorMatrix0, VectorMatrix1)
                ConnectedWeightedNormal *= angle[:, None]

                wSum = unit_vector(
                    numpy.nansum(ConnectedWeightedNormal, axis=0)
                ).tolist()

                if numpy.all(wSum == 0.0):
                    continue
                for vertexf in vertex_group:
                    verts_outline_vector[vertexf] = wSum
                    IteratedValues.add(vertexf)
        if self.game in [
            GameEnum.GenshinImpact,
            GameEnum.HonkaiStarRail,
            GameEnum.HonkaiImpact3rd,
        ]:
            pos_buf["TANGENT"][:, 0:3] = verts_outline_vector[:, 0:3]
        elif self.game == GameEnum.HonkaiImpactPart2:
            pos_buf["COLOR"][:, 0:3] = verts_outline_vector[:, 0:3]
        elif self.game == GameEnum.ZenlessZoneZero:
            output_buffers["TexCoord"]["TEXCOORD2"] = verts_outline_vector[:, 1:3]

        print(f"Optimized outlines in {time.time() - start_time:.4f} seconds")

    def write_files(self) -> None:
        """Write the files to the destination."""
        self.destination.mkdir(parents=True, exist_ok=True)
        print("Writen files: ")
        try:
            for file_path, content in self.files_to_write.items():
                print(f" - {file_path.name}")
                if isinstance(content, str) and self.write_ini:
                    with open(file_path, "w", encoding="utf-8") as file:
                        file.write(content)
                elif isinstance(content, numpy.ndarray) and self.write_buffers:
                    content.tofile(file_path)
        except (OSError, IOError) as e:
            raise Fatal(f"Error writing file {file_path}: {e}")
        if not self.copy_textures:
            return
        try:
            for src, dest in self.files_to_copy.items():
                print(f" - {dest.name}")
                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dest)
        except (OSError, IOError) as e:
            raise Fatal(f"Error copying file {src} to {dest}: {e}")

    def export(self) -> None:
        """Export the mod file."""
        start: float = time.time()
        print(f"Exporting {self.mod_name} to {self.destination}")
        self.generate_buffers()
        self.generate_ini()
        self.write_files()
        print()
        self.operator.report(
            {"INFO"},
            f"Exported {self.mod_name} to {self.destination} in {(time.time() - start):2f} seconds",
        )

    def cleanup(self) -> None:
        """Cleanup the objects."""
        pass

    def load_hashes(
        self, path: Path, name: str, hashfile: str = "hash.json"
    ) -> list[dict]:
        with open(path / hashfile, "r") as f:
            char_hashes = json.load(f)
        # TODO: Check for hash.json integrity
        return char_hashes
