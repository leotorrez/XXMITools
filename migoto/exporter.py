import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import bpy
import numpy
from bpy.types import Collection, Context, Depsgraph, Mesh, Object, Operator, Scene
from numpy.typing import NDArray

from .. import bl_info
from ..libs.jinja2 import Environment, FileSystemLoader
from .data.byte_buffer import (
    AbstractSemantic,
    BufferLayout,
    BufferSemantic,
    NumpyBuffer,
    Semantic,
)
from .data.data_model import DataModelXXMI
from .data.hash_json import Component, HashJsonData, SubObj
from .data.ini_format import INI_file
from .datahandling import mesh_triangulate
from .datastructures import Fatal, GameEnum


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
    template: Path | None = None
    outline_rounding_precision: int = 8
    outline_gate_divergence: float = 14.0
    outline_custom_normals: bool = False
    # Internal / not implemented
    ignore_muted_shape_keys: bool = False
    # Output
    mod_name: str = ""
    hash_data: list[dict] = field(default_factory=list)
    mod_file: ModFile = field(init=False)
    ini_content: str = field(init=False)
    files_to_write: dict[Path, str | NDArray] = field(init=False)
    files_to_copy: list[tuple[Path, Path]] = field(init=False)

    def __post_init__(self) -> None:
        print("Initializing data for export...")
        self.__objs_to_cleanup: list[Object] = []
        self.__depsgraph: Depsgraph = bpy.context.evaluated_depsgraph_get()
        if self.dump_path == Path(""):
            raise Fatal("Dump path not set")
        if not self.dump_path.is_dir() or self.dump_path.suffix != "":
            self.dump_path = self.dump_path.parent

        self.mod_name = self.dump_path.stem

        if self.destination == Path(""):
            self.destination = self.dump_path.parent / f"{self.mod_name}Mod"
            self.operator.report(
                {"WARNING"},
                f"Destination path not set, defaulting to {self.destination}",
            )
        if self.destination.is_file():
            self.destination = self.destination.parent
        if self.destination == self.dump_path:
            raise Fatal("Destination path can't be the same as Dump path")
        self.destination.mkdir(parents=True, exist_ok=True)

        self.hash_data = self.load_hashes(self.dump_path / "hash.json")
        if not self.hash_data:
            raise Fatal("ERROR", "Hash data is empty or invalid!")

        scene: Scene = bpy.context.scene
        if not [
            obj for obj in scene.objects if obj.name.startswith(self.mod_name)
        ] or not [
            file
            for file in self.dump_path.parent.iterdir()
            if self.mod_name.lower() in file.name.lower()
        ]:
            raise Fatal(
                "ERROR: Cannot find match for name. Make sure your dump and objects have matching names and that hash.json exists in the dump folder."
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
        hash_json_data = HashJsonData(self.dump_path / "hash.json")
        for component in hash_json_data.components:
            comp_matching_objs: list[Object] = [
                obj for obj in candidate_objs if obj.name.startswith(component.fullname)
            ]
            if len(comp_matching_objs) == 0 and component.draw_vb != "":
                continue
            for part in component.parts:
                if not self.copy_textures:
                    part.textures = []
                if component.draw_vb == "":
                    continue

                objects: list[SubObj] = []
                matching_objs: list[Object] = [
                    obj for obj in candidate_objs if obj.name.startswith(part.fullname)
                ]
                if not matching_objs:
                    raise Fatal(f"Cannot find object {part.fullname} in the scene.")
                if len(matching_objs) > 1:
                    raise Fatal(
                        f"Found multiple objects with the name {part.fullname}."
                    )
                obj: Object = matching_objs[0]
                collection = [
                    c
                    for c in bpy.data.collections
                    if c.name.lower().startswith((part.fullname).lower())
                ]
                if len(collection) > 1:
                    raise Fatal(
                        f"ERROR: Found multiple collections with the name {part.fullname}. Ensure only one collection exists with that name."
                    )
                if len(collection) == 0:
                    self.obj_from_col(obj, None, objects)
                else:
                    self.obj_from_col(obj, collection[0], objects)
                part.objects = objects
            self.mod_file.components.append(component)

    def obj_from_col(
        self,
        main_obj: Object,
        collection: Collection | None,
        destination: list[SubObj],
        depth: int = 0,
    ) -> None:
        """Recursively get all objects from a collection and its sub-collections."""
        if destination == []:
            final_mesh: Mesh = self.process_mesh(main_obj, main_obj)
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
            final_mesh = self.process_mesh(main_obj, obj)
            destination.append(
                SubObj(collection.name, depth, obj.name, obj, final_mesh)
            )
        for child in collection.children:
            self.obj_from_col(main_obj, child, destination, depth + 1)

    def process_mesh(self, main_obj: Object, obj: Object) -> Mesh:
        """Process the mesh of the object."""
        # TODO: Add moddifier application for SK'd meshes here
        final_mesh: Mesh = (
            obj.evaluated_get(self.__depsgraph).to_mesh()
            if self.apply_modifiers
            else obj.to_mesh()
        )
        if main_obj != obj:
            # Matrix world seems to be the summatory of all transforms parents included
            # Might need to test for more edge cases and to confirm these suspicious,
            # other available options: matrix_local, matrix_basis, matrix_parent_inverse
            final_mesh.transform(obj.matrix_world)
            final_mesh.transform(main_obj.matrix_world.inverted())
        mesh_triangulate(final_mesh)
        masked_vgs = [
            vg.index for vg in obj.vertex_groups if vg.name.startswith("MASK")
        ]
        _ = [
            vg.__setattr__("weight", 0.0)
            for vert in final_mesh.vertices
            for vg in vert.groups
            if vg.group in masked_vgs
        ]
        self.__objs_to_cleanup.append(obj)
        return final_mesh

    def generate_buffers(self) -> None:
        """Generate buffers for the objects."""
        self.files_to_write = {}
        self.files_to_copy = []
        for component in self.mod_file.components:
            data_model: DataModelXXMI = DataModelXXMI.from_obj(
                (
                    component.parts[0].objects[0].obj
                    if len(component.parts[0].objects)
                    else None
                ),
                self.game,
                self.normalize_weights,
                component.blend_vb,
                component.texcoord_vb,
            )
            excluded_buffers: list[str] = []
            out_buffers: dict[str, NumpyBuffer] = {
                key: NumpyBuffer(layout=entry)
                for key, entry in data_model.buffers_format.items()
            }
            if self.write_buffers is False:
                for key in out_buffers:
                    excluded_buffers.append(key)
            vb_offset: int = 0
            for part in component.parts:
                print(f"Processing {part.fullname} " + "-" * 10)
                # XXX:: This system duplicates textures when 2 parts share them.
                # We should probably imitate that link on the export folder and lock it behind a toggle
                for t in part.textures:
                    source = t.path
                    dest = self.destination / (t.fullname + t.extension)
                    self.files_to_copy.append((source, dest))

                if component.draw_vb == "":
                    continue
                part_ib: NumpyBuffer = NumpyBuffer(data_model.buffers_format["IB"])
                ib_offset: int = 0
                for entry in part.objects:
                    print(f"Processing {entry.name}...")
                    v_count: int = 0
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
                    if gen_buffers["IB"].data is not None:
                        gen_buffers["IB"].data["INDEX"] += vb_offset
                        entry.index_count = len(gen_buffers["IB"].data)
                        part_ib.append(gen_buffers["IB"])
                    for k, v in out_buffers.items():
                        if k not in gen_buffers:
                            continue
                        v.append(gen_buffers[k])
                    vb_offset += v_count
                    entry.vertex_count = v_count
                    part.vertex_count += v_count
                    component.vertex_count += v_count
                    entry.index_offset = ib_offset
                    ib_offset += entry.index_count
                if part_ib.data is None or len(part_ib) == 0:
                    print(f"Skipping {part.fullname}.ib due to no index data.")
                    continue
                self.files_to_write[self.destination / (part.fullname + ".ib")] = (
                    part_ib.data
                )
            if self.outline_optimization and len(out_buffers) > 0:
                self.optimize_outlines(out_buffers)
            for key, buffer in out_buffers.items():
                if key == "IB":
                    continue
                self.files_to_write[
                    self.destination / (component.fullname + key + ".buf")
                ] = buffer.data
            component.strides = {
                k.lower(): v.stride
                for k, v in data_model.buffers_format.items()
                if k != "IB"
            }

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
            if (
                abs_enum == Semantic.Color
                and mesh.vertex_colors.get(abs_name) is None
                and mesh.color_attributes.get(abs_name) is None
            ):
                missing_colors.append(abs_name)
            if abs_enum == Semantic.TexCoord and mesh.uv_layers.get(abs_name) is None:
                missing_uvs.append(abs_name)
            if abs_enum == Semantic.Blendweights:
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
                scene=self.context.scene,
            )
        )
        ini_file.clean_up_indentation()
        ini_body: str = str(ini_file)
        self.files_to_write[self.destination / (self.mod_name + ".ini")] = ini_body

    def optimize_outlines(self, output_buffs: dict[str, NumpyBuffer]) -> None:
        """Optimize the outlines of the meshes with angle-weighted normal averaging."""

        def unit_vector(vector: NDArray) -> NDArray:
            """Normalize the input vector to unit length."""
            norm = numpy.linalg.norm(vector, axis=1, keepdims=True)
            norm = numpy.where(norm == 0, 1, norm)
            return vector / norm

        pos_buf: NumpyBuffer = output_buffs["Position"]
        if len(pos_buf) == 0:
            return

        tex_buf: NumpyBuffer = output_buffs["Texcoord"]
        ib_data: NDArray = output_buffs["IB"].data["INDEX"]

        start_time: int | float = time.time()

        loops_coord: NDArray = pos_buf.data["POSITION"][ib_data, 0:3]
        triangles: NDArray = loops_coord.reshape(-1, 3, 3)
        edge0: NDArray = triangles[:, 1] - triangles[:, 2]
        edge1: NDArray = triangles[:, 2] - triangles[:, 0]
        edge2: NDArray = triangles[:, 0] - triangles[:, 1]

        # Precompute pairwise cross vectors and their magnitudes (reuse for face normals and angles)
        cross01 = numpy.cross(edge0, edge1)  # edge0 x edge1
        cross12 = numpy.cross(edge1, edge2)  # edge1 x edge2
        cross20 = numpy.cross(edge2, edge0)  # edge2 x edge0

        cross01_mag = numpy.linalg.norm(cross01, axis=1)
        cross12_mag = numpy.linalg.norm(cross12, axis=1)
        cross20_mag = numpy.linalg.norm(cross20, axis=1)

        # Precompute pairwise dot products
        dot01 = numpy.einsum("ij,ij->i", edge0, edge1)
        dot12 = numpy.einsum("ij,ij->i", edge1, edge2)
        dot20 = numpy.einsum("ij,ij->i", edge2, edge0)

        # Interior angle at each loop corner via atan2(||a x b||, a.b); avoids the
        # sign/abs ambiguity of a plain arccos on the unit edges. Measured to match
        # the game's weighting better than the previously used abs-based calc_angle.
        angle0: NDArray = numpy.arctan2(cross12_mag, -dot12).astype(numpy.float32)
        angle1: NDArray = numpy.arctan2(cross20_mag, -dot20).astype(numpy.float32)
        angle2: NDArray = numpy.arctan2(cross01_mag, -dot01).astype(numpy.float32)

        loops_angle: NDArray = numpy.zeros((len(triangles), 3), dtype=numpy.float32)
        loops_angle[:, 0] = angle0
        loops_angle[:, 1] = angle1
        loops_angle[:, 2] = angle2

        # Loop normals for the weld. The outline must be derived from the
        # mesh's pure geometry (per-triangle face normals), not from the
        # buffer's NORMAL semantic: users customize the displayed normals,
        # which must never corrupt the outline. On the faithful Ramielle
        # mesh, the geometry weld reproduces the game's TEXCOORD1 mask at
        # >99.8% agreement (best of any variant). Only when the user
        # explicitly enables custom normals do we weld with the authored
        # split normals instead, as a deliberate opt-in.
        use_loop_normals: bool = self.outline_custom_normals
        if use_loop_normals and "NORMAL" in pos_buf.data.dtype.names:
            loops_face_normal: NDArray = unit_vector(
                pos_buf.data["NORMAL"][ib_data, 0:3]
            )
        else:
            if use_loop_normals:
                print(
                    "WARNING: outline normals require a NORMAL semantic; falling back to geometry face normals."
                )
            faces_normal: NDArray = unit_vector(cross01)
            loops_face_normal: NDArray = faces_normal.repeat(3, axis=0)

        verts_outline_vector: NDArray = numpy.zeros(
            (len(pos_buf), 3), dtype=numpy.float32
        )

        loops_angle = loops_angle.flatten()
        loops_weighted_normal = loops_face_normal * loops_angle[:, None]

        # Smooth-normal weld. Group loops by position (merging split-seam
        # duplicates that share a surface point even across small positional
        # drift), then angle-average the weighted normals per group. The
        # averaged normal is always used -- no crease/raw-face fallback: a
        # coherence-based fallback snaps mildly curved smooth areas to a raw
        # face normal, which shows up as high-frequency speckle on the outline.
        # Loop normals come from geometry by default (matching the game) or
        # from the buffer's authored split normals when custom normals are on.
        loops_round_coord: NDArray = numpy.round(
            loops_coord, self.outline_rounding_precision
        )
        unique_groups, _, u_inverse = numpy.unique(
            loops_round_coord,
            axis=0,
            return_index=True,
            return_inverse=True,
        )

        accumulated_normals: NDArray = numpy.zeros(
            (len(unique_groups), 3), dtype=numpy.float32
        )
        # Use numpy.add.at to efficiently sum weighted normals per group
        numpy.add.at(accumulated_normals, u_inverse, loops_weighted_normal)
        verts_outline_vector[ib_data] = unit_vector(accumulated_normals[u_inverse])

        if self.game == GameEnum.HonkaiImpactPart2:
            # HI3 Part 2 stores the outline in the vertex COLOR: RGB is the
            # object-space outline normal (angle-weighted weld), alpha is a
            # per-vertex outline weight capped at 0.5, tapered off at sharp
            # creases and mostly zero on non-outline components (eyes/mouth).
            # The game encodes the signed direction as UNORM with 2c-1 (and
            # the shader decodes it back), so re-centre the signed weld into
            # 0..1 and re-scale the stored alpha bytes to 0..1 before the
            # UNORM8 encoder converts them; passing signed vectors straight
            # through would wrap the negative channels into garbage.
            filled_outline = numpy.zeros_like(pos_buf.data["COLOR"], numpy.float32)
            filled_outline[:, 0:3] = (verts_outline_vector + 1.0) * 0.5

            result_abstract = AbstractSemantic(Semantic.Color)
            result_buf = pos_buf
            result_element: BufferSemantic | None = result_buf.layout.get_element(
                result_abstract
            )
            filled_outline[:, 3] = result_element.format.type_decoder(
                pos_buf.data["COLOR"][:, 3]
            )
            result_data = filled_outline

        elif self.game == GameEnum.ZenlessZoneZero:
            # Outlines for ZZZ are stored in TEXCOORD1 as the projected
            # outline normal in tangent space. The per-vertex angle-weighted
            # loop-normal weld (accumulated over rounding groups, matching the
            # game's own construction) is projected through the mesh's smooth
            # TBN basis (smooth NORMAL/TANGENT/BITANGENTSIGN), written raw.
            outline_vert: NDArray = verts_outline_vector
            tan: NDArray = unit_vector(pos_buf.data["TANGENT"][:, 0:3])
            bitansign: NDArray = pos_buf.data["BITANGENTSIGN"][:, None]
            bitan: NDArray = (
                numpy.cross(pos_buf.data["NORMAL"][:, 0:3], tan) * bitansign
            )
            dot_prods: NDArray = numpy.zeros(
                (len(outline_vert), 2), dtype=numpy.float32
            )
            dot_prods[:, 0] = numpy.einsum("ij,ij->i", tan, outline_vert)
            dot_prods[:, 1] = numpy.einsum("ij,ij->i", bitan, outline_vert)
            # The game writes TEXCOORD1 only where the angle-weighted
            # outline normal actually bends away from the vertex normal
            # (real creases/silhouettes). Where the weld hugs the smooth
            # normal the outline is null. The game applies a hard cut with
            # no fade band - above it the magnitude is the raw tangent-space
            # projection (|sin(divergence)| rising across the whole range),
            # so smoothing the gate would distort correct high-divergence
            # values. The cutoff is configurable in the UI; the default is
            # tuned to best reproduce the game's ZZZ mask with the
            # geometry weld.
            aligned = numpy.einsum(
                "ij,ij->i",
                outline_vert,
                pos_buf.data["NORMAL"][:, 0:3],
            )
            weld_divergence = numpy.degrees(numpy.arccos(numpy.clip(aligned, -1, 1)))
            fade = (weld_divergence >= self.outline_gate_divergence).astype(
                numpy.float32
            )
            dot_prods[:, 0] *= fade
            dot_prods[:, 1] *= fade

            result_abstract = AbstractSemantic(Semantic.TexCoord, 1)
            result_buf = tex_buf
            result_data = dot_prods
        else:
            result_abstract = AbstractSemantic(Semantic.Tangent)
            result_data = verts_outline_vector
            result_buf = pos_buf

        result_element: BufferSemantic | None = result_buf.layout.get_element(
            result_abstract
        )
        if result_element is None:
            # TODO: might want to force add anyways
            self.operator.report(
                {"WARNING"},
                "Semantic not found in the buffer layout. Skipping outline optimization.",
            )
        else:
            result_buf.import_semantic_data(
                result_data,
                result_element,
                [result_element.format.type_encoder],
            )
        print(f"Optimized outlines in {time.time() - start_time:.4f} seconds")

    def write_files(self) -> None:
        """Write the files to the destination."""
        self.destination.mkdir(parents=True, exist_ok=True)
        print("Writen files: ")
        for file_path, content in self.files_to_write.items():
            try:
                print(f" - {file_path.name}")
                if isinstance(content, str) and self.write_ini:
                    with open(file_path, "w", encoding="utf-8") as file:
                        file.write(content)
                elif isinstance(content, numpy.ndarray) and self.write_buffers:
                    content.tofile(file_path)
            except OSError as e:
                raise Fatal(f"Error writing file {file_path}: {e}")
        if not self.copy_textures:
            return
        for src, dest in self.files_to_copy:
            try:
                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    print(f" - {dest.name} skipped.")
                    continue
                shutil.copy(src, dest)
                print(f" - {dest.name}")
            except Exception as e:
                raise Fatal(f"Error copying file {src} to {dest}: {e}")

    def cleanup(self) -> None:
        """Cleanup after the exporter."""
        for obj in self.__objs_to_cleanup:
            obj.to_mesh_clear()
            if not isinstance(obj.data, Mesh):
                continue
            obj.data.update()

    def export(self) -> None:
        """Export the mod file."""
        start: float = time.time()
        if len(self.mod_file.components) == 0:
            raise Fatal("No components found to export. Aborting export.")
        print(f"Exporting {self.mod_name} to {self.destination}")
        self.generate_buffers()
        self.generate_ini()
        self.write_files()
        self.cleanup()
        print()
        self.operator.report(
            {"INFO"},
            f"Exported {self.mod_name} to {self.destination} in {(time.time() - start):2f} seconds",
        )

    def load_hashes(self, path: Path) -> list[dict]:
        """Load the hash data from the hash.json file."""
        if not path.exists() or not path.is_file():
            raise Fatal(f"Hash file {path} does not exist or is not a file.")
        with open(path, "r") as f:
            char_hashes = json.load(f)
        # TODO: Check for hash.json integrity
        return char_hashes
